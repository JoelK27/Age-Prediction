import os, argparse, math, random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from torch.amp import autocast, GradScaler  # statt cuda.amp

from src.data.dataset import AgeDataset
from src.models.custom_cnn import CustomAgeCNN
from src.utils.metrics import mae, mse

class IndexedDataset(Dataset):
    def __init__(self, base: AgeDataset, indices, transform, regression: bool, age_bins):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.regression = regression
        self.age_bins = age_bins
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        path, age = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.regression:
            target = torch.tensor(age, dtype=torch.float32)
        else:
            cls = 0
            for k,(lo,hi) in enumerate(self.age_bins):
                if lo <= age <= hi:
                    cls = k; break
            target = torch.tensor(cls, dtype=torch.long)
        return img, target

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--plot_samples", action="store_true")
    # Speed/Qualität:
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 2)))
    ap.add_argument("--no_amp", action="store_true", help="Mixed Precision deaktivieren")
    ap.add_argument("--model_width", type=int, default=32)
    ap.add_argument("--classifier_dropout", type=float, default=0.35)
    # Stabilität/Training:
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["huber", "mae"], default="huber")
    ap.add_argument("--huber_delta", type=float, default=5.0)
    ap.add_argument("--scheduler", choices=["onecycle", "cosine", "plateau", "none"], default="onecycle")
    ap.add_argument("--early_stopping", type=int, default=10, help="Patience")
    ap.add_argument("--ema", action="store_true", help="EMA der Gewichte nutzen")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--tta_val", action="store_true", help="Horizontal-Flip TTA in Val")
    ap.add_argument("--aug_strength", choices=["simple","strong"], default="simple")
    ap.add_argument("--mixup", type=float, default=0.0, help="Mixup-Alpha (nur Regression)")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Nur Klassifikation")
    return ap.parse_args()

def main():
    args = get_args()
    # Seeds
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Augmentationen
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
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    age_bins = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]

    # Dataset laden (ohne Transform) und stratified splitten
    base_ds = AgeDataset(args.data_dir, transform=None, regression=args.regression, age_bins=age_bins)

    ages = [age for _, age in base_ds.samples]
    def age_to_bin(a): return min(a // 10, 9)
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

    train_ds = IndexedDataset(base_ds, train_idx, train_tf, args.regression, age_bins)
    val_ds   = IndexedDataset(base_ds, val_idx,   eval_tf,  args.regression, age_bins)

    pin = device == "cuda"
    nw = int(args.num_workers)

    # DataLoader erstellen (prefetch_factor nur setzen, wenn Worker > 0)
    if nw > 0:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=pin, persistent_workers=True,
                                  prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=nw, pin_memory=pin, persistent_workers=True,
                                prefetch_factor=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=pin)

    if args.plot_samples:
        imgs, _ = next(iter(train_loader))
        grid = utils.make_grid(imgs[:16], nrow=8, normalize=True, value_range=(-1,1))
        plt.figure(figsize=(12,4))
        plt.imshow(grid.permute(1,2,0).cpu()); plt.axis("off")
        plt.title("Train Samples")
        plt.savefig(os.path.join(args.save_dir, "samples.png")); plt.close()

    model = CustomAgeCNN(
        regression=args.regression,
        num_classes=args.num_classes,
        width=args.model_width,
        classifier_dropout=args.classifier_dropout
    ).to(device)

    # Loss
    if args.regression:
        if args.loss == "huber":
            criterion = nn.HuberLoss(delta=args.huber_delta)
        else:
            criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1,len(train_loader)),
            pct_start=0.15, div_factor=10.0, final_div_factor=100.0
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5, verbose=True
        )
    else:
        scheduler = None

    # AMP + EMA
    amp_enabled = (device == "cuda" and not args.no_amp)
    device_type = "cuda" if device == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=amp_enabled)

    ema_model = None
    if args.ema:
        ema_model = deepcopy(model).to(device)
        for p in ema_model.parameters(): p.requires_grad_(False)

    def ema_update(ema_m, src_m, decay):
        with torch.no_grad():
            for (_, p_ema), (_, p) in zip(ema_m.named_parameters(), src_m.named_parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
            for (_, b_ema), (_, b) in zip(ema_m.named_buffers(), src_m.named_buffers()):
                b_ema.copy_(b)

    # Mixup (nur Regression)
    def apply_mixup(x, y, alpha: float):
        if alpha <= 0.0:
            return x, y
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
        return x, y

    best_metric = math.inf
    patience = 0
    history = {"train_loss": [], "val_loss": []}
    if args.regression: history["val_mae"] = []
    else: history["val_acc"] = []

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=amp_enabled):
                if args.regression and args.mixup > 0.0:
                    xb_m, yb_m = apply_mixup(xb, yb, args.mixup)
                    out = model(xb_m)
                    loss = criterion(out, yb_m)
                else:
                    out = model(xb)
                    loss = criterion(out, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update()
            if scheduler and args.scheduler == "onecycle":
                scheduler.step()
            if ema_model is not None:
                ema_update(ema_model, model, args.ema_decay)
            train_loss += loss.item()*xb.size(0)
        train_loss /= len(train_loader.dataset)

        if scheduler and args.scheduler == "cosine":
            scheduler.step()

        # Evaluation (wahlweise mit EMA und TTA)
        def evaluate(m):
            m.eval()
            v_loss = v_mae = v_mse = 0.0
            correct = total = 0
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    with autocast(device_type=device_type, enabled=amp_enabled):
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

        eval_model = ema_model if ema_model is not None else model
        val_loss, val_mae, val_mse, val_acc = evaluate(eval_model)

        if args.scheduler == "plateau":
            sched_metric = val_mae if args.regression else val_loss
            scheduler.step(sched_metric)

        if args.regression:
            metric = val_mae
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f} MAE={val_mae:.2f} MSE={val_mse:.2f}")
        else:
            acc = val_acc or 0.0
            metric = 1 - acc
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f} Acc={acc:.3f}")

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        if args.regression: history["val_mae"].append(val_mae)
        else: history["val_acc"].append(val_acc)

        improved = metric < best_metric - 1e-6
        if improved:
            best_metric = metric; patience = 0
            ckpt_path = os.path.join(args.save_dir, f"best_epoch_{epoch}.pt")
            to_save = eval_model.state_dict() if ema_model is not None else model.state_dict()
            torch.save({"model": to_save, "args": vars(args)}, ckpt_path)
            print("Gespeichert:", ckpt_path)
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping (patience {args.early_stopping})")
                break

    # Trainingskurven speichern
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

    # Letztes Modell speichern
    last_path = os.path.join(args.save_dir, "last.pt")
    last_state = (ema_model.state_dict() if ema_model is not None else model.state_dict())
    torch.save({"model": last_state, "args": vars(args)}, last_path)
    print("Letztes Modell gespeichert:", last_path)

if __name__ == "__main__":
    main()