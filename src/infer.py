"""
Inferenz-Skript:
- Lädt ein gespeichertes Checkpoint (state_dict + Trainings-Args)
- Baut das Modell gemäß gespeicherten/überschriebenen Parametern
- Preprocessing entspricht der Eval-Pipeline im Training (Resize + Normalize)
- Unterstützt Einzelbild (--image) und Batch (--input_dir) Inferenz
- Optional: TTA (Horizontal-Flip) und MAE/MSE-Ausgabe bei Ordner-Inferenz

Beispiele:
- Einzelbild (Regression):
  python -m src.infer --checkpoint checkpoints\\best_epoch_18.pt --image "path\\to.jpg" --regression true --tta
- Ordner (Regression, CSV speichern):
  python -m src.infer --checkpoint checkpoints\\best_epoch_18.pt --input_dir "src\\data\\UTKface_inthewild\\part3" --regression true --tta --out_csv infer_results.csv

Hinweise:
- Der Modus (Regression vs. Klassifikation) wird primär aus den im Checkpoint gespeicherten Args gelesen.
  Mit --regression kann der Modus optional überschrieben werden (nur nutzen, wenn sicher).
- Im Klassifikationsmodus wird aktuell der Klassenindex/Bucket (und Label-Text) ausgegeben.
  Falls ein kontinuierliches Alter aus Klassifikation benötigt wird, kann man den Erwartungswert aus Softmax*Bin-Zentren berechnen (nicht implementiert, nur Hinweis).
"""

import os
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

from src.models.custom_cnn import CustomAgeCNN  # Modellarchitektur aus deinem Projekt
from src.data.dataset import _extract_age  # Label aus Dateiname

# Standard-Preprocessing wie im Training (Resize + Normalize auf [-1,1])
def make_transform(img_size: int):
    """
    Erzeugt die Eval-Transform-Pipeline:
    - Resize auf (img_size, img_size)
    - ToTensor (Skalierung 0..1)
    - Normalize (Mittel=0.5, Std=0.5) -> Pixelbereich [-1, 1]
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Lädt ein Checkpoint (.pt) und gibt Tupel (state_dict, cfg) zurück.

    Parameter:
    - checkpoint_path: Pfad zum gespeicherten Modell
    - device: Zielgerät (cpu/cuda) für das Laden

    Rückgabe:
    - state_dict: Modellgewichte (dict der Tensoren)
    - cfg: gespeicherte Trainingsargumente (dict) oder {} falls nicht vorhanden
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Unterstützt sowohl "komplettes Paket" (inkl. args) als auch reines state_dict
    state_dict = ckpt.get("model", ckpt)  # falls nur state_dict gespeichert wurde
    cfg = ckpt.get("args", {})            # Trainings-Argumente, wenn vorhanden
    return state_dict, cfg

def build_model_from_ckpt(cfg: dict, override_regression: Optional[bool], num_classes: Optional[int], model_width: Optional[int], classifier_dropout: Optional[float], device: torch.device):
    """
    Baut das Modell aus den Checkpoint-Args, mit optionalen CLI-Overrides.

    Parameter:
    - cfg: im Checkpoint gespeicherte Trainingsargumente
    - override_regression: True/False um Modus zu erzwingen, None = aus cfg übernehmen
    - num_classes: optional Anzahl Klassen (Klassifikation)
    - model_width: optional Breite des CNN (Kanal-Multiplikator)
    - classifier_dropout: optional Dropout im Klassifikations-Head
    - device: Zielgerät

    Rückgabe:
    - model: initialisiertes Modell im eval()-Modus auf device
    - regression: effektiver Modus (bool)
    - nc: effektive Anzahl Klassen
    - width: effektive Modellbreite
    """
    # Defaults aus Checkpoint lesen, dann mit CLI überschreiben (falls gesetzt)
    regression = cfg.get("regression", True) if override_regression is None else override_regression
    nc = cfg.get("num_classes", 10)
    if num_classes is not None:
         nc = num_classes
    width = cfg.get("model_width", 32)
    if model_width is not None:
        width = model_width

    drop = cfg.get("classifier_dropout", 0.35)
    if classifier_dropout is not None:
        drop = classifier_dropout

    model = CustomAgeCNN(regression=regression, num_classes=nc, width=width, classifier_dropout=drop).to(device)
    model.eval()  # Inferenzmodus (deaktiviert Dropout/BatchNorm-Update)
    return model, regression, nc, width

def predict_tensor(model: torch.nn.Module, x: torch.Tensor, regression: bool, tta: bool) -> Tuple[float, Optional[float], Optional[int]]:
    """
    Führt eine Vorhersage für einen (1,3,H,W)-Batch durch.

    Parameter:
    - model: geladenes Modell
    - x: Tensor der Größe (1,3,H,W) auf dem korrekten device
    - regression: True = kontinuierliche Altersschätzung, False = Klassifikation
    - tta: Test-Time-Augmentation (horizontaler Flip und Mittelung)

    Rückgabe:
    - Wenn Klassifikation: (pred_class_idx als float, confidence, pred_class_idx)
    - Wenn Regression: (alter_float, None, None); Alter wird auf [0,120] geklemmt
    """
    # x: (1,3,H,W) auf device
    with torch.inference_mode():
        if not regression:
            # Klassifikation: logits -> softmax-Probs -> Top-1
            if tta:
                logits1 = model(x)
                logits2 = model(torch.flip(x, dims=[3]))
                logits = (logits1 + logits2) / 2
            else:
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            # Hinweis: Hier wird nur der Bucket zurückgegeben.
            # Ein kontinuierliches Alter könnte als Erwartungswert der Bin-Zentren gerechnet werden.
            return float(pred.item()), float(conf.item()), int(pred.item())
        else:
            # Regression: optional TTA (Flip), dann Mittelwert der Vorhersagen
            if tta:
                y1 = model(x)
                y2 = model(torch.flip(x, dims=[3]))
                y = (y1 + y2) / 2
            else:
                y = model(x)
            age = float(y.squeeze(0).item())
            # Plausible Grenzen einklammern
            age = max(0.0, min(120.0, age))
            return age, None, None

def default_bins(num_classes: int) -> Optional[List[Tuple[int, int]]]:
    """
    Liefert Standard-Altersintervalle (Bins) für Klassifikation.

    Für 10 Klassen: Dekaden-Bins (0–9, ..., 90–120).
    Für andere Klassenzahlen: nicht definiert (None), d. h. nur Label-Text für 10-Klassen-Fall.
    """
    # Standard: 10 Dekaden-Bins wie im Training (0–9, 10–19, …, 90–120)
    if num_classes == 10:
        bins = [(i*10, i*10 + 9) for i in range(9)]
        bins.append((90, 120))
        return bins
    return None

def describe_bucket(idx: int, bins: Optional[List[Tuple[int,int]]]) -> str:
    """
    Formatiert einen Klassenindex als lesbares Intervall (z. B. '20-29').

    Fallback: 'class {idx}', wenn kein Mapping existiert.
    """
    if bins is None or idx < 0 or idx >= len(bins):
        return f"class {idx}"
    lo, hi = bins[idx]
    return f"{lo}-{hi}"

def is_image_file(path: str) -> bool:
    """
    Prüft anhand des Öffnens via PIL, ob eine Datei als Bild gelesen werden kann.
    Robuster als reine Endungsprüfung.
    """
    # Robuster als nur auf Endungen zu prüfen
    try:
        with Image.open(path) as _:
            return True
    except Exception:
        return false_like  # noqa

# da Python kein 'false_like' kennt, definieren wir False direkt:
false_like = False

def infer_image(path: str, model: torch.nn.Module, regression: bool, tf, device: torch.device, tta: bool, num_classes: int):
    """
    Führt Inferenz für ein einzelnes Bild aus und gibt ein Ergebnis-Dict zurück.

    Parameter:
    - path: Bildpfad
    - model: geladenes Modell
    - regression: Modus (True=Regression, False=Klassifikation)
    - tf: Transform-Pipeline (eval)
    - device: cpu/cuda
    - tta: Test-Time-Augmentation (Flip)
    - num_classes: Anzahl Klassen (für Label-Text in Klassifikation)

    Rückgabe:
    - Regression: {"path", "pred_age"}
    - Klassifikation: {"path", "pred_class", "label", optional "confidence"}
    """
    img = Image.open(path).convert("RGB")
    # Transform + Batch-Dimension
    x = tf(img).unsqueeze(0).to(device)

    # Rohe Vorhersage
    pred, conf, cls_idx = predict_tensor(model, x, regression, tta)

    if regression:
        # Alter als float (schon geklemmt) runden und zurückgeben
        return {"path": path, "pred_age": round(pred, 2)}
    else:
        # Klassifikation: Bucket-Label auflösen (nur für 10-Klassen standardisiert)
        bins = default_bins(num_classes)
        label = describe_bucket(int(pred), bins)
        out = {"path": path, "pred_class": int(pred), "label": label}
        if conf is not None:
            out["confidence"] = round(conf, 4)
        return out

def run_single(image_path: str, model, regression: bool, tf, device, tta: bool, num_classes: int):
    """
    Hilfsfunktion: Inferenz für genau ein Bild mit schöner Konsolenausgabe.
    """
    result = infer_image(image_path, model, regression, tf, device, tta, num_classes)
    if regression:
        print(f"{result['path']}: Predicted Age = {result['pred_age']}")
    else:
        conf_str = f" (p={result['confidence']})" if "confidence" in result else ""
        print(f"{result['path']}: Predicted Bucket = {result['pred_class']} [{result['label']}]"+conf_str)

def run_folder(folder: str, model, regression: bool, tf, device, tta: bool, num_classes: int, out_csv: Optional[str]):
    """
    Batch-Inferenz für alle Bilder in einem Ordner (nicht rekursiv).
    Optional: CSV mit Ergebnissen schreiben; bei Regression MAE/MSE gegen GT (aus Dateinamen) ausgeben.

    CSV-Felder:
    - Regression: path, pred_age
    - Klassifikation: path, pred_class, label, (optional) confidence
    """
    # Dateiliste erstellen und nur validierbare Bilddateien behalten
    paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    paths = [p for p in paths if os.path.isfile(p) and is_image_file(p)]
    paths.sort()

    rows = []
    for p in paths:
        try:
            rows.append(infer_image(p, model, regression, tf, device, tta, num_classes))
        except Exception as e:
            # Einzelne fehlerhafte Dateien nicht abbrechen lassen
            print(f"Skip {p}: {e}")

    if not rows:
        print("Keine validen Bilder gefunden."); return

    # Kurzer Vorschau-Print
    for r in rows[:10]:
        if regression:
            print(f"{r['path']}: {r['pred_age']}")
        else:
            conf_str = f" (p={r.get('confidence')})" if "confidence" in r else ""
            print(f"{r['path']}: {r['label']}{conf_str}")
    print(f"{len(rows)} Bilder inferiert.")

    # Optional: Ground-Truth aus Dateinamen und MAE/MSE berechnen (nur Regression)
    if regression:
        gts, preds = [], []
        for r in rows:
            gt = _extract_age(os.path.basename(r["path"]))
            if gt is not None:
                gts.append(float(gt)); preds.append(float(r["pred_age"]))
        if gts:
            import numpy as np
            mae = float(np.mean(np.abs(np.array(preds) - np.array(gts))))
            mse = float(np.mean((np.array(preds) - np.array(gts))**2))
            print(f"Eval (aus Dateinamen): MAE={mae:.2f} MSE={mse:.2f} auf {len(gts)} Bildern")

    # Optional CSV schreiben
    if out_csv:
        try:
            import csv
            # Feldnamen dynamisch aus den Keys der Ergebnis-Dicts ableiten
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
    """
    CLI-Argumente:
    - --checkpoint: .pt-Datei (erforderlich)
    - --image / --input_dir: Einzelbild oder Batch-Ordner
    - --device: 'auto' (Default), 'cpu' oder 'cuda'
    - --tta: aktiviert horizontales Flip-TTA
    - Overrides (nur wenn erforderlich):
      --img_size, --model_width, --num_classes, --regression, --classifier_dropout
    - --out_csv: optionaler Pfad zum Schreiben der Ergebnisse (Batch)
    """
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
    ap.add_argument("--classifier_dropout", type=float, help="Dropout im Classifier (nur Form, eval deaktiviert)")
    ap.add_argument("--out_csv", help="CSV-Ziel für Batch-Inferenz")
    return ap.parse_args()

def main():
    """
    Programm-Einstieg:
    - Argumente parsen
    - Checkpoint laden und Modell konstruieren
    - Entweder Einzelbild- oder Batch-Inferenz ausführen
    """
    args = parse_args()

    if not args.image and not args.input_dir:
        print("Bitte --image oder --input_dir angeben.")
        return

    # Device automatisch wählen, wenn 'auto' (CUDA falls verfügbar)
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    # Checkpoint laden (Gewichte + gespeicherte Args)
    state_dict, cfg = load_checkpoint(args.checkpoint, device)

    # Bildgröße aus Checkpoint oder CLI
    img_size = args.img_size if args.img_size is not None else int(cfg.get("img_size", 160))
    tf = make_transform(img_size)

    # Modell erstellen (Checkpoint-Args + optionale Overrides)
    model, regression, num_classes, width = build_model_from_ckpt(
        cfg=cfg,
        override_regression=args.regression,
        num_classes=args.num_classes,
        model_width=args.model_width,
        classifier_dropout=args.classifier_dropout,
        device=device
    )
    # Gewichte laden; bei Warnungen können Layer-Formate nicht exakt passen (z. B. Modus geändert)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warnung: state_dict Unterschiede. missing={len(missing)} unexpected={len(unexpected)}")

    # Inferenz ausführen
    if args.image:
        run_single(args.image, model, regression, tf, device, args.tta, num_classes)
    if args.input_dir:
        run_folder(args.input_dir, model, regression, tf, device, args.tta, num_classes, args.out_csv)

if __name__ == "__main__":
    main()