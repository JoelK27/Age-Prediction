# Age-Prediction

Dieses Projekt trainiert ein eigenes CNN zur Altersschätzung auf UTKFace-ähnlichen Dateinamen.

Wichtige Dateien:
- Training: [src/training.py](src/training.py)
- Analyse: [src/analysis.py](src/analysis.py)
- Inferenz: [src/infer.py](src/infer.py)
- Dataset-Lader: [src/data/dataset.py](src/data/dataset.py)
- Modell: [src/models/custom_cnn.py](src/models/custom_cnn.py)
- Metriken: [src/utils/metrics.py](src/utils/metrics.py)
- Abhängigkeiten: [requirements.txt](requirements.txt)

## 1) Setup

1. Terminal im Projektordner öffnen:
```
cd "C:\Users\Startklar\Desktop\5. Semester\CVNLP\Age-Prediction"
```

2. Virtuelle Umgebung erstellen und aktivieren:
```
python -m venv .venv
.venv\Scripts\activate
```

3. Abhängigkeiten installieren:
```
pip install -r requirements.txt
```

Hinweis: Windows-Backslashes (\) und Slashes (/) funktionieren beide in den Beispielen.

## 2) Daten vorbereiten

- Lege deine Bilder direkt in einen Ordner, z. B.:
  - src/data/UTKface_inthewild/part3
- Der Loader erwartet UTKFace-ähnliche Dateinamen: age_gender_race_timestamp.ext
  - Beispiel: 23_0_0_201701161745.jpg
  - Dateien, die auf “jpg” enden, aber keinen Punkt haben (…25357jpg), werden unterstützt.
- Der Loader ist nicht rekursiv: Er liest nur Dateien, die direkt in --data_dir liegen.

Der Datensatzordner ist per .gitignore von Git ausgeschlossen.

## 3) Datensatz analysieren

Erzeugt Verteilungsplot und Basisstatistiken.
```
python -m src.analysis --data_dir "src/data/UTKface_inthewild/part3" --out_dir reports
```
Ausgaben:
- Konsole: Anzahl, Min/Max/Ø, Perzentile, Endungen
- Plot: reports/age_distribution.png

Fehlersuche:
- “Pfad nicht gefunden”: absoluten Pfad angeben:
  "C:\Users\Startklar\Desktop\5. Semester\CVNLP\Age-Prediction\src\data\UTKface_inthewild\part3"

## 4) Training

Regression (Alter als Zahl):
```
python -m src.training --data_dir "src/data/UTKface_inthewild/part3" ^
  --epochs 10 --batch_size 64 --regression --plot_samples ^
  --img_size 160 --model_width 32 --num_workers 8
```

Klassifikation (Altersbuckets):
```
python -m src.training --data_dir "src/data/UTKface_inthewild/part3" ^
  --epochs 10 --batch_size 64 --num_classes 10 --num_workers 8
```

Ausgaben in checkpoints/:
- best_epoch_X.pt: bestes Modell
- samples.png: Grid mit Trainingssamples (nur bei --plot_samples)
- training_curves.png: Trainings-/Validierungskurven

Speed-Tipps:
- --img_size 128 oder 160 statt 256 verwenden
- --num_workers 4–8 testen
- GPU nutzen (AMP ist automatisch aktiv), sonst --no_amp
- --model_width 32 (kleineres Modell)

## 5) Inferenz

Nutzt ein trainiertes Checkpoint zur Vorhersage.
```
python -m src.infer --image "pfad\zu\bild.jpg" --checkpoint checkpoints\best_epoch_X.pt --regression
```
Hinweise:
- Das Modell akzeptiert beliebige Auflösungen (AdaptiveAvgPool).
- Für Bilder ohne Datei-Endung (…jpg) kann PIL Probleme haben; ein echtes .jpg/.png verwenden.

## 6) Wichtige Optionen (Auszug)

- --data_dir: Ordner mit den Bildern (nicht rekursiv)
- --epochs, --batch_size, --lr: Trainingsparameter
- --val_split: Anteil Validierung (Default 0.2)
- --regression: Aktiviert Regression (ohne Flag = Klassifikation)
- --num_classes: Anzahl Altersbuckets (bei Klassifikation)
- --img_size: Zielauflösung (Default 160)
- --model_width: Modellbreite (Default 32)
- --num_workers: DataLoader-Worker (Default CPU-Kerne)
- --no_amp: Deaktiviert Mixed Precision auf CUDA
- --save_dir: Ausgabeverzeichnis (Default checkpoints)
- --plot_samples: Speichert Grid der Trainingsbilder

## 7) Häufige Probleme

- Pfadfehler: Absoluten Pfad nutzen oder im Projektroot starten.
- “Keine gültigen Dateien”: Dateinamen müssen mit Alter beginnen (z. B. 23_…).
- PIL kann Bild nicht öffnen: Datei hat keine echte Endung; in .jpg/.png umbenennen/konvertieren.
- Training sehr langsam: Siehe Speed-Tipps (img_size, workers, AMP).

## 8) Architektur, Augmentation und Metriken

- Eigenes CNN: [src/models/custom_cnn.py](src/models/custom_cnn.py) mit Global Average Pooling und konfigurierbarer Breite.
- Augmentation: Resize, Horizontal Flip, leichte ColorJitter/Rotation ([src/training.py](src/training.py)).
- Metriken: MAE/MSE für Regression, Accuracy für Klassifikation ([src/utils/metrics.py](src/utils/metrics.py)).
