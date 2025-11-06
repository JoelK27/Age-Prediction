# Age-Prediction

Eigenes CNN zur Altersschätzung auf UTKFace-ähnlichen Dateinamen.

Wichtige Dateien
- Training: [`src/training.py`](src/training.py)
- Inferenz: [`src/infer.py`](src/infer.py)
- Analyse: [`src/analysis.py`](src/analysis.py)
- Dataset-Lader: [`src/data/dataset.py`](src/data/dataset.py)
- Modell: [`src/models/custom_cnn.py`](src/models/custom_cnn.py)
- Metriken: [`src/utils/metrics.py`](src/utils/metrics.py)
- Beispiel-Check: [`src/datacheck.py`](src/datacheck.py)
- Projektreport: [`Report.md`](Report.md)
- Abhängigkeiten: [`requirements.txt`](requirements.txt)

## 1) Setup

Terminal im Projektordner öffnen und Umgebung einrichten:
```
cd "C:\Users\Startklar\Desktop\5. Semester\CVNLP\Age-Prediction"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Daten vorbereiten

- Lege alle Bilder direkt in einen Ordner, z. B.:
  - src/data/UTKface_inthewild/part3
- Erwartetes Namensschema (UTKFace): age_gender_race_timestamp.ext
  - Beispiel: 23_0_0_201701161745.jpg
  - Dateien, die auf “jpg” enden, aber keinen Punkt haben (…25357jpg), werden toleriert.
- Nicht rekursiv: Loader liest nur Dateien direkt in --data_dir.

Schnelltest des Loaders:
```
python -m src.datacheck
```

## 3) Datensatz analysieren

Erzeugt Histogramm und Basisstatistiken.
```
python -m src.analysis --data_dir "src/data/UTKface_inthewild/part3" --out_dir reports
```
Output:
- Konsole: Anzahl, Min/Max/Ø, Perzentile, Dateiendungen
- Plot: reports/age_distribution.png

## 4) Trainieren

Allgemein
- Regression: kontinuierliches Alter (Ziel-Metrik MAE).
- Klassifikation: Altersbuckets (z. B. 10 Dekaden). Für Fairness vergleiche später MAE via Inferenz auf Ordner.

Regression (robust und schnell, gute Basis)
```
python -m src.training --data_dir "src/data/UTKface_inthewild/part3" ^
  --epochs 20 --batch_size 64 --regression ^
  --img_size 160 --model_width 32 --num_workers 8 ^
  --save_dir checkpoints
```

Klassifikation (10 Bins, schnelle Accuracy-Baseline)
```
python -m src.training --data_dir "src/data/UTKface_inthewild/part3" ^
  --epochs 20 --batch_size 64 ^
  --num_classes 10 --img_size 160 --model_width 32 ^
  --num_workers 8 --save_dir checkpoints
```

Ausgaben in checkpoints/
- best_epoch_X.pt: bestes Modell nach Validierungsmetrik (Regression: MAE, Klassifikation: Accuracy)
- training_curves.png: Trainings-/Validierungskurven
- samples.png: Beispielgrid (nur bei --plot_samples)

Tipps für Tempo/Stabilität
- --img_size 160, --model_width 32/48
- GPU + AMP (Standard auf CUDA); CPU: ggf. --no_amp
- --num_workers 4–8
- --val_split 0.2 (Default), --save_dir anpassen

## 5) Inferenz

Einzelbild (Regression)
```
python -m src.infer --checkpoint checkpoints\best_epoch_X.pt ^
  --image "C:\pfad\zu\bild.jpg" --regression --tta
```

Ordner (Regression, MAE/MSE aus Dateinamen, CSV speichern)
```
python -m src.infer --checkpoint checkpoints\best_epoch_X.pt ^
  --input_dir "src\data\UTKface_inthewild\part3" --regression --tta ^
  --out_csv infer_results.csv
```

Einzelbild (Klassifikation)
```
python -m src.infer --checkpoint checkpoints\best_epoch_cls.pt ^
  --image "C:\pfad\zu\bild.jpg" --tta
```
Hinweis: Im Klassifikationsmodus gibt [`src/infer.py`](src/infer.py) den Klassenindex und das Bucket-Label (z. B. 20–29) aus. Eine kontinuierliche Altersschätzung aus Klassifikation (Erwartungswert über Bin-Zentren) ist als Idee im Code kommentiert, aber nicht aktiviert.

## 6) Projekt-Workflow (Kurz)

1) Analyse: Verteilung prüfen
- python -m src.analysis --data_dir "src/data/UTKface_inthewild/part3"

2) Training starten (Regression oder Klassifikation)
- Siehe Befehle in Abschnitt 4.

3) Bestes Modell auswählen
- checkpoints/best_epoch_X.pt

4) Inferenz und Evaluation
- Einzelbild/Ordner mit [`src/infer.py`](src/infer.py); bei Ordner zeigt die Konsole MAE/MSE (Regression).

## 7) Dateien im Detail

- [`src/data/dataset.py`](src/data/dataset.py)
  - [`AgeDataset`](src/data/dataset.py): Liest Bilder und erzeugt Targets (Regression: float Alter; Klassifikation: Bucket-Index).
  - [`_extract_age`](src/data/dataset.py): Liest Alter aus Dateiname (erstes Token vor „_“).
  - Toleriert Dateien ohne echten Suffix, die auf „jpg“ enden (…25357jpg).

- [`src/models/custom_cnn.py`](src/models/custom_cnn.py)
  - [`CustomAgeCNN`](src/models/custom_cnn.py): Kompaktes CNN mit 4 Conv-Blöcken, Global Average Pooling, MLP-Head.
  - regression=True: Ausgabe [B] (Alter), sonst [B, C] (Logits für Klassen).

- [`src/utils/metrics.py`](src/utils/metrics.py)
  - [`mae`](src/utils/metrics.py), [`mse`](src/utils/metrics.py): Fehlermaße für Regression.

- [`src/training.py`](src/training.py)
  - CLI-Training für Regression/Klassifikation.
  - Augmentierungen (Train) vs. deterministisches Eval-Preprocessing (Val).
  - Optimizer: AdamW; optional Scheduler; Checkpointing des besten Modells.
  - Plots: training_curves.png, optional samples.png.

- [`src/infer.py`](src/infer.py)
  - Lädt Checkpoint, baut Modell, führt Inferenz aus.
  - --image für Einzelbild, --input_dir für Batch.
  - --tta aktiviert H-Flip-Mittelung.
  - Bei Regression mit Ordner-Eval: MAE/MSE aus Dateinamen.

- [`src/analysis.py`](src/analysis.py)
  - Basisanalyse und Alters-Histogramm (reports/age_distribution.png).

- [`src/datacheck.py`](src/datacheck.py)
  - Mini-Skript zum schnellen Sichten der ersten Samples (Pfad/Alter).

- Reports/Checkpoints
  - reports/: Analyseplots
  - checkpoints/: Modelle und Trainingskurven

## 8) Häufige Probleme

- “Pfad nicht gefunden”: Absoluten Pfad nutzen oder im Projektroot starten.
- “Keine gültigen Dateien”: Dateinamen müssen mit Alter beginnen (z. B. 23_…).
- PIL-Fehler beim Öffnen: Datei hat keine echte Endung; in .jpg/.png umbenennen/konvertieren.
- Langsames Training: Kleinere --img_size, moderates --model_width, mehr --num_workers, AMP aktiv lassen.

## 9) Hinweise

- Zielmetrik für Regression ist MAE. Vergleiche Modelle konsistent mit Ordner-Eval in der Inferenz.
- Der Loader ist nicht rekursiv; bei tieferen Strukturen ein Manifest oder rekursives Listing ergänzen.
- Für mehr Details, Ergebnisse und Ausblick siehe [`Report.md`](Report.md).