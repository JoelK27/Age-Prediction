"""
Training-Skript für Age-Prediction (Regression oder Klassifikation).

Hauptfunktionen:
- Lädt UTKFace-ähnliche Daten (Alter wird aus Dateinamen geparst) über AgeDataset.
- Erzeugt Train/Val-Splits stratifiziert nach Alters-Dekaden (0–9, 10–19, …).
- Wendet konfigurierbare Augmentierungen an (simple/strong).
- Baut ein konfigurierbares CNN (CustomAgeCNN) für Regression oder Klassifikation.
- Trainiert mit AdamW, Mixed Precision (AMP), optional EMA und TTA in Val.
- Unterstützt LR-Scheduler: OneCycle, Cosine, ReduceLROnPlateau.
- Checkpointing: Bestes Modell je nach Metrik (MAE bei Regression, Acc bei Klassifikation) + last.pt.
- Visualisiert Trainingskurven als training_curves.png (und optional Samples als samples.png).
"""

import os, argparse, math, random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from torch.amp import autocast, GradScaler  # Mixed Precision (statt cuda.amp)

from src.data.dataset import AgeDataset
from src.models.custom_cnn import CustomAgeCNN
from src.utils.metrics import mae, mse


class IndexedDataset(Dataset):
    """
    Wrapper-Dataset, um Teilmengen (per Indexliste) mit eigenen Transforms zu nutzen.
    Nutzt die (path, age)-Samples aus AgeDataset und erzeugt Targets je nach Modus.

    - base: Basissamples (AgeDataset, enthält Pfade + integer Alter)
    - indices: Auswahl der Indizes für diesen Split (Train/Val)
    - transform: torchvision-Transform-Pipeline für Bilder
    - regression: True -> float-Ziel (Alter), False -> Klassenindex (Dekaden-Bucket)
    - age_bins: Liste von (lo, hi) Intervallen zur Bucket-Zuordnung in Klassifikation
    """
    def __init__(self, base: AgeDataset, indices, transform, regression: bool, age_bins):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.regression = regression
        self.age_bins = age_bins

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, i):
        """
        Lädt Bild, wendet Transform an und erzeugt Target.
        Rückgabe:
        - img: Tensor [C,H,W] (float, normalisiert)
        - target: float32 (Regression) oder long (Klassifikation)
        """
        idx = self.indices[i]
        path, age = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.regression:
            target = torch.tensor(age, dtype=torch.float32)
        else:
            # Altersbereich in Dekadenklasse überführen (0..9, 10..19, ..., 90..120)
            cls = 0
            for k,(lo,hi) in enumerate(self.age_bins):
                if lo <= age <= hi:
                    cls = k; break
            target = torch.tensor(cls, dtype=torch.long)
        return img, target


def get_args():
    """
    CLI-Argumente für Training konfigurieren.
    Hinweise:
    - Ohne --regression wird Klassifikation (CrossEntropyLoss) trainiert.
    - --balanced_sampler gleicht die Klassen-/Dekadenverteilung im Training aus (Sampling-Gewichte).
    - --tta_val aktiviert H-Flip-TTA nur für die Validierung.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Ordner mit Bildern (nicht rekursiv)")
    ap.add_argument("--epochs", type=int, default=20, help="Anzahl Trainingsepochen")
    ap.add_argument("--batch_size", type=int, default=32, help="Batchgröße")
    ap.add_argument("--lr", type=float, default=1e-3, help="Lernrate für AdamW")
    ap.add_argument("--val_split", type=float, default=0.2, help="Anteil Validation (0..1)")
    ap.add_argument("--regression", action="store_true", help="Regression statt Klassifikation")
    ap.add_argument("--num_classes", type=int, default=10, help="Klassenanzahl (Dekaden) für Klassifikation")
    ap.add_argument("--save_dir", default="checkpoints", help="Ausgabeordner für Checkpoints und Plots")
    ap.add_argument("--plot_samples", action="store_true", help="Speichert Grid mit Trainingssamples")

    # Speed/Qualität:
    ap.add_argument("--img_size", type=int, default=160, help="Resize-Zielauflösung (quadratisch)")
    ap.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 2)), help="DataLoader-Worker")
    ap.add_argument("--no_amp", action="store_true", help="Mixed Precision deaktivieren (nur float32)")
    ap.add_argument("--model_width", type=int, default=32, help="Breite des CNN (Kanal-Multiplikator)")
    ap.add_argument("--classifier_dropout", type=float, default=0.35, help="Dropout im Klassifikations-Head")

    # Stabilität/Training:
    ap.add_argument("--seed", type=int, default=42, help="Random-Seed für Reproduzierbarkeit")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="L2-Regularisierung für AdamW")
    ap.add_argument("--loss", choices=["huber", "mae"], default="huber", help="Loss für Regression")
    ap.add_argument("--huber_delta", type=float, default=5.0, help="Delta-Parameter für HuberLoss")
    ap.add_argument("--scheduler", choices=["onecycle", "cosine", "plateau", "none"], default="onecycle", help="LR-Scheduler")
    ap.add_argument("--early_stopping", type=int, default=10, help="Patience für Early Stopping")
    ap.add_argument("--ema", action="store_true", help="EMA der Gewichte (stabilere Eval)")
    ap.add_argument("--ema_decay", type=float, default=0.999, help="EMA-Decay [0,1)")
    ap.add_argument("--tta_val", action="store_true", help="Horizontal-Flip TTA in Val")
    ap.add_argument("--aug_strength", choices=["simple","strong"], default="simple", help="Augmentierungsstärke")
    ap.add_argument("--mixup", type=float, default=0.0, help="Mixup-Alpha (nur Regression)")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label Smoothing (nur Klassifikation)")
    ap.add_argument("--balanced_sampler", action="store_true", help="Ausgeglichenes Sampling nach Alters-Dekaden")
    return ap.parse_args()


def main():
    """
    Hauptablauf:
    - Seeds, Gerätewahl, Augmentierungen
    - Datensatz laden, stratifizierter Split, optionale Balancierung
    - DataLoader, optionales Sample-Grid
    - Modell, Loss, Optimizer, Scheduler, AMP/EMA
    - Trainings-/Validierungsschleife mit Checkpointing und Kurven
    """
    args = get_args()
    # Reproduzierbarkeit (sofern möglich)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Ausgabenordner vorbereiten, Device wählen
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Aktiviert CUDNN-Autotuner für schnellere Convs bei konstanten Input-Shapes
        torch.backends.cudnn.benchmark = True

    # Augmentationen
    # strong: includes RandomResizedCrop + RandomErasing (teurer, robuster)
    # simple: Resize + moderate ColorJitter/Rotation (schneller)
    if args.aug_strength == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])
    # Eval-Transform: strikt deterministisch (Resize + Normalize)
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    # Dekaden-Buckets (für Klassifikation und Stratified-Split)
    age_bins = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]

    # Dataset laden (AgeDataset parst Alter aus Dateinamen). transform=None: wir transformieren erst in IndexedDataset.
    base_ds = AgeDataset(args.data_dir, transform=None, regression=args.regression, age_bins=age_bins)

    # Stratified Split: nach Dekaden verteilen, dann je Bucket train/val splitten
    ages = [age for _, age in base_ds.samples]
    def age_to_bin(a): return min(a // 10, 9)  # 0..9 (90+ zusammengefasst)
    bins = {}
    for idx, a in enumerate(ages):
        b = age_to_bin(a)
        bins.setdefault(b, []).append(idx)
    val_idx, train_idx = [], []
    for _, idxs in bins.items():
        n = len(idxs)
        n_val = max(1, int(n * args.val_split))
        random.shuffle(idxs)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    # Split-Datasets mit jeweiligen Transforms erstellen
    train_ds = IndexedDataset(base_ds, train_idx, train_tf, args.regression, age_bins)
    val_ds   = IndexedDataset(base_ds, val_idx,   eval_tf,  args.regression, age_bins)

    pin = device == "cuda"
    nw = int(args.num_workers)

    # Class-Balancing über WeightedRandomSampler (pro Dekaden-Bucket)
    sampler = None
    if args.balanced_sampler:
        counts = [0]*10
        for i in train_idx:
            counts[age_to_bin(ages[i])] += 1
        weights = []
        for i in train_idx:
            b = age_to_bin(ages[i])
            # Gewicht = invers zur Bucket-Häufigkeit (seltene Buckets werden öfter gesampelt)
            w = 1.0 / max(1, counts[b])
            weights.append(w)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # DataLoader erstellen
    # Hinweis: persistent_workers/prefetch_factor beschleunigen Pipelines mit vielen Epochen
    if nw > 0:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=(sampler is None), sampler=sampler,
                                   num_workers=nw, pin_memory=pin, persistent_workers=True,
                                   prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=nw, pin_memory=pin, persistent_workers=True,
                                 prefetch_factor=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=(sampler is None), sampler=sampler,
                                   num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=pin)

    # Beispielbilder speichern (nur optische Kontrolle der Augmentierung)
    if args.plot_samples:
        imgs, _ = next(iter(train_loader))
        grid = utils.make_grid(imgs[:16], nrow=8, normalize=True, value_range=(-1,1))
        plt.figure(figsize=(12,4))
        plt.imshow(grid.permute(1,2,0).cpu()); plt.axis("off")
        plt.title("Train Samples")
        plt.savefig(os.path.join(args.save_dir, "samples.png")); plt.close()

    # Modell instanziieren
    model = CustomAgeCNN(
        regression=args.regression,
        num_classes=args.num_classes,
        width=args.model_width,
        classifier_dropout=args.classifier_dropout
    ).to(device)

    # Loss auswählen (abhängig vom Modus)
    if args.regression:
        # Huber ist robuster gegen Ausreißer, MAE optimiert direkt die Zielmetrik
        if args.loss == "huber":
            criterion = nn.HuberLoss(delta=args.huber_delta)
        else:
            criterion = nn.L1Loss()
    else:
        # Klassifikation: CrossEntropy mit optionalem Label Smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "onecycle":
        # OneCycle: schnelleres Konvergieren, gute Standardwahl
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1,len(train_loader)),
            pct_start=0.15, div_factor=10.0, final_div_factor=100.0
        )
    elif args.scheduler == "cosine":
        # CosineAnnealing über Epochen (einfach, stabil)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        # Plateau: LR senken, wenn sich Val-Metrik nicht verbessert
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5, verbose=True
            )
        except TypeError:
            # Fallback für ältere Torch-Versionen ohne 'verbose'
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
            )
    else:
        scheduler = None

    # AMP (Mixed Precision) + optional EMA (Exponential Moving Average) der Gewichte
    amp_enabled = (device == "cuda" and not args.no_amp)
    device_type = "cuda" if device == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=amp_enabled)

    ema_model = None
    if args.ema:
        # EMA als eval-Modell (nicht trainierbar)
        ema_model = deepcopy(model).to(device)
        for p in ema_model.parameters(): p.requires_grad_(False)

    def ema_update(ema_m, src_m, decay):
        """
        EMA-Update: ema = decay*ema + (1-decay)*src (für Parameter und Buffer).
        Wird typischerweise nach jedem Optimizer-Step aufgerufen.
        """
        with torch.no_grad():
            for (_, p_ema), (_, p) in zip(ema_m.named_parameters(), src_m.named_parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
            for (_, b_ema), (_, b) in zip(ema_m.named_buffers(), src_m.named_buffers()):
                b_ema.copy_(b)

    # Mixup (nur für Regression sinnvoll: Targets sind reell und lassen sich mischen)
    def apply_mixup(x, y, alpha: float):
        """
        Standard-Mixup: konvexes Mischen von Bildern und Targets mittels Beta(alpha,alpha).
        alpha=0 -> kein Mixup.
        """
        if alpha <= 0.0:
            return x, y
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
        return x, y

    # Trainingsbuchhaltung
    best_metric = math.inf            # Minimierungsziel: MAE (Regression) bzw. (1-Acc) in Klassifikation
    patience = 0                      # Early-Stopping-Zähler
    history = {"train_loss": [], "val_loss": []}
    if args.regression: history["val_mae"] = []
    else: history["val_acc"] = []

    # Epochen-Schleife
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            # Daten auf Device schieben (non_blocking mit pin_memory)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Vorwärts + Loss in AMP-Context
            with autocast(device_type=device_type, enabled=amp_enabled):
                if args.regression and args.mixup > 0.0:
                    xb_m, yb_m = apply_mixup(xb, yb, args.mixup)
                    out = model(xb_m)
                    loss = criterion(out, yb_m)
                else:
                    out = model(xb)
                    loss = criterion(out, yb)
            # Backward + Optimizer-Step (skaliert durch GradScaler)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update()
            # OneCycle: Step pro Batch
            if scheduler and args.scheduler == "onecycle":
                scheduler.step()
            # EMA-Update (nach Optimizer)
            if ema_model is not None:
                ema_update(ema_model, model, args.ema_decay)
            train_loss += loss.item()*xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Cosine: ein Step pro Epoche
        if scheduler and args.scheduler == "cosine":
            scheduler.step()

        # Evaluation (wahlweise mit EMA-Weights und TTA)
        def evaluate(m):
            """
            Evaluiert das Modell m auf dem Val-Loader.
            Rückgabe:
            - Regression: (val_loss, val_mae, val_mse, None)
            - Klassifikation: (val_loss, None, None, val_acc)
            """
            m.eval()
            v_loss = v_mae = v_mse = 0.0
            correct = total = 0
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    with autocast(device_type=device_type, enabled=amp_enabled):
                        # Optional: TTA via H-Flip (Vorhersagen mitteln)
                        if args.tta_val:
                            out1 = m(xb)
                            out2 = m(torch.flip(xb, dims=[3]))
                            out = (out1 + out2) / 2
                        else:
                            out = m(xb)
                        loss = criterion(out, yb)
                    v_loss += loss.item()*xb.size(0)
                    if args.regression:
                        v_mae += mae(out, yb).item()*xb.size(0)
                        v_mse += mse(out, yb).item()*xb.size(0)
                    else:
                        preds = out.argmax(1)
                        correct += (preds==yb).sum().item()
                        total += yb.size(0)
            v_loss /= len(val_loader.dataset)
            if args.regression:
                v_mae /= len(val_loader.dataset); v_mse /= len(val_loader.dataset)
                return v_loss, v_mae, v_mse, None
            else:
                acc = correct/total if total>0 else 0.0
                return v_loss, None, None, acc

        # Wähle EMA-Modell zur Evaluation, wenn vorhanden (glättet die Gewichte)
        eval_model = ema_model if ema_model is not None else model
        val_loss, val_mae, val_mse, val_acc = evaluate(eval_model)

        # Plateau-Scheduler: Step mit Metrik (Regression: MAE, Klassifikation: ValLoss)
        if scheduler and args.scheduler == "plateau":
            sched_metric = val_mae if args.regression else val_loss
            scheduler.step(sched_metric)

        # Fortschritt loggen
        if args.regression:
            metric = val_mae
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f} MAE={val_mae:.2f} MSE={val_mse:.2f}")
        else:
            acc = val_acc or 0.0
            metric = 1 - acc  # Minimieren (je kleiner, desto besser)
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f} Acc={acc:.3f}")

        # History für Plots
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        if args.regression: history["val_mae"].append(val_mae)
        else: history["val_acc"].append(val_acc)

        # Checkpointing: speichere Bestes nach Metrik
        improved = metric < best_metric - 1e-6
        if improved:
            best_metric = metric; patience = 0
            ckpt_path = os.path.join(args.save_dir, f"best_epoch_{epoch}.pt")
            # Speichere EMA-Gewichte, falls aktiv; sonst aktuelle Modellgewichte
            to_save = eval_model.state_dict() if ema_model is not None else model.state_dict()
            torch.save({"model": to_save, "args": vars(args)}, ckpt_path)
            print("Gespeichert:", ckpt_path)
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping (patience {args.early_stopping})")
                break

    # Trainingskurven speichern (Loss + MAE/Acc)
    epochs = range(1, len(history["train_loss"])+1)
    if args.regression:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"], label="Val Loss"); ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(epochs, history["val_mae"], label="Val MAE", color="orange"); ax2.set_title("Validation MAE"); ax2.set_xlabel("Epoch"); ax2.legend()
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6,4))
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"], label="Val Loss")
        ax1.plot(epochs, history["val_acc"], label="Val Acc"); ax1.set_title("Training Curves"); ax1.set_xlabel("Epoch"); ax1.legend()
    fig.tight_layout(); plt.savefig(os.path.join(args.save_dir, "training_curves.png")); plt.close()

    # Letztes Modell (finaler Zustand) speichern
    last_path = os.path.join(args.save_dir, "last.pt")
    last_state = (ema_model.state_dict() if ema_model is not None else model.state_dict())
    torch.save({"model": last_state, "args": vars(args)}, last_path)
    print("Letztes Modell gespeichert:", last_path)


if __name__ == "__main__":
    main()