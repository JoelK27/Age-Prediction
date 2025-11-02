import os
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

from src.models.custom_cnn import CustomAgeCNN  # Modellarchitektur aus deinem Projekt

# Standard-Preprocessing wie im Training (Resize + Normalize auf [-1,1])
def make_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)  # falls nur state_dict gespeichert wurde
    cfg = ckpt.get("args", {})            # Trainings-Argumente, wenn vorhanden
    return state_dict, cfg

def build_model_from_ckpt(cfg: dict, override_regression: Optional[bool], num_classes: Optional[int], model_width: Optional[int], device: torch.device):
    # Defaults aus Checkpoint ziehen, CLI kann überschreiben
    regression = cfg.get("regression", True) if override_regression is None else override_regression
    nc = cfg.get("num_classes", 10)
    if num_classes is not None:
        nc = num_classes
    width = cfg.get("model_width", 32)
    if model_width is not None:
        width = model_width

    model = CustomAgeCNN(regression=regression, num_classes=nc, width=width).to(device)
    model.eval()
    return model, regression, nc, width

def predict_tensor(model: torch.nn.Module, x: torch.Tensor, regression: bool, tta: bool) -> Tuple[float, Optional[float], Optional[int]]:
    # x: (1,3,H,W) auf device
    with torch.no_grad():
        if not regression:
            # Klassifikation: logits -> softmax-Probs
            if tta:
                logits1 = model(x)
                logits2 = model(torch.flip(x, dims=[3]))
                logits = (logits1 + logits2) / 2
            else:
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            return float(pred.item()), float(conf.item()), int(pred.item())
        else:
            # Regression: Mittelwert über TTA
            if tta:
                y1 = model(x)
                y2 = model(torch.flip(x, dims=[3]))
                y = (y1 + y2) / 2
            else:
                y = model(x)
            age = float(y.squeeze(0).item())
            # plausible Grenzen einklammern
            age = max(0.0, min(120.0, age))
            return age, None, None

def default_bins(num_classes: int) -> Optional[List[Tuple[int, int]]]:
    # Standard: 10 Dekaden-Bins wie im Training (0–9, 10–19, …, 90–120)
    if num_classes == 10:
        bins = [(i*10, i*10 + 9) for i in range(9)]
        bins.append((90, 120))
        return bins
    return None

def describe_bucket(idx: int, bins: Optional[List[Tuple[int,int]]]) -> str:
    if bins is None or idx < 0 or idx >= len(bins):
        return f"class {idx}"
    lo, hi = bins[idx]
    return f"{lo}-{hi}"

def is_image_file(path: str) -> bool:
    # Robuster als nur auf Endungen zu prüfen
    try:
        with Image.open(path) as _:
            return True
    except Exception:
        return false_like  # noqa

# da Python kein 'false_like' kennt, definieren wir False direkt:
false_like = False

def infer_image(path: str, model: torch.nn.Module, regression: bool, tf, device: torch.device, tta: bool, num_classes: int):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    pred, conf, cls_idx = predict_tensor(model, x, regression, tta)
    if regression:
        return {"path": path, "pred_age": round(pred, 2)}
    else:
        bins = default_bins(num_classes)
        label = describe_bucket(int(pred), bins)
        out = {"path": path, "pred_class": int(pred), "label": label}
        if conf is not None:
            out["confidence"] = round(conf, 4)
        return out

def run_single(image_path: str, model, regression: bool, tf, device, tta: bool, num_classes: int):
    result = infer_image(image_path, model, regression, tf, device, tta, num_classes)
    if regression:
        print(f"{result['path']}: Predicted Age = {result['pred_age']}")
    else:
        conf_str = f" (p={result['confidence']})" if "confidence" in result else ""
        print(f"{result['path']}: Predicted Bucket = {result['pred_class']} [{result['label']}]"+conf_str)

def run_folder(folder: str, model, regression: bool, tf, device, tta: bool, num_classes: int, out_csv: Optional[str]):
    paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    paths = [p for p in paths if os.path.isfile(p) and is_image_file(p)]
    paths.sort()

    rows = []
    for p in paths:
        try:
            rows.append(infer_image(p, model, regression, tf, device, tta, num_classes))
        except Exception as e:
            print(f"Skip {p}: {e}")

    # Ausgabe
    if not rows:
        print("Keine validen Bilder gefunden.")
        return
    for r in rows[:10]:
        # ersten Zeilen zur Kontrolle ausgeben
        if regression:
            print(f"{r['path']}: {r['pred_age']}")
        else:
            conf_str = f" (p={r.get('confidence')})" if "confidence" in r else ""
            print(f"{r['path']}: {r['label']}{conf_str}")
    print(f"{len(rows)} Bilder inferiert.")

    if out_csv:
        try:
            import csv
            keys = sorted({k for r in rows for k in r.keys()})
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"Ergebnisse gespeichert: {out_csv}")
        except Exception as e:
            print(f"Konnte CSV nicht speichern: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="Inference für Age-Prediction")
    ap.add_argument("--checkpoint", required=True, help="Pfad zum .pt Checkpoint (best_epoch_X.pt)")
    ap.add_argument("--image", help="Pfad zu einem Bild")
    ap.add_argument("--input_dir", help="Ordner mit Bildern für Batch-Inferenz")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--tta", action="store_true", help="Test-Time Augmentation (H-Flip)")
    # Optionale Overrides (falls Checkpoint-Args fehlen/überschrieben werden sollen)
    ap.add_argument("--img_size", type=int, help="Resize-Größe, Default aus Checkpoint")
    ap.add_argument("--model_width", type=int, help="Modellbreite, Default aus Checkpoint")
    ap.add_argument("--num_classes", type=int, help="Nur für Klassifikation, Default aus Checkpoint")
    ap.add_argument("--regression", type=lambda v: v.lower() in ("1","true","yes"), help="True/False zum Überschreiben")
    ap.add_argument("--out_csv", help="CSV-Ziel für Batch-Inferenz")
    return ap.parse_args()

def main():
    args = parse_args()

    if not args.image and not args.input_dir:
        print("Bitte --image oder --input_dir angeben.")
        return

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    # Checkpoint laden
    state_dict, cfg = load_checkpoint(args.checkpoint, device)

    # Bildgröße aus Checkpoint oder CLI
    img_size = args.img_size if args.img_size is not None else int(cfg.get("img_size", 160))
    tf = make_transform(img_size)

    # Modell erstellen
    model, regression, num_classes, width = build_model_from_ckpt(
        cfg=cfg,
        override_regression=args.regression,
        num_classes=args.num_classes,
        model_width=args.model_width,
        device=device
    )
    # Gewichte laden
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warnung: state_dict Unterschiede. missing={len(missing)} unexpected={len(unexpected)}")

    # Inferenz
    if args.image:
        run_single(args.image, model, regression, tf, device, args.tta, num_classes)
    if args.input_dir:
        run_folder(args.input_dir, model, regression, tf, device, args.tta, num_classes, args.out_csv)

if __name__ == "__main__":
    main()