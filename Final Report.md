# Project Report – Age Prediction

1) Dataset-Beschreibung
- Quelle/Typ: UTKFace-ähnlicher „in the wild“ Datensatz mit Dateinamensschema age_gender_race_timestamp.ext. Geladen über [`src/data/dataset.py`](src/data/dataset.py) (Klasse `AgeDataset`).
- Struktur: Bilder liegen flach in einem Ordner (nicht rekursiv). Beispiele: 23_0_0_201701161745.jpg, inkl. Sonderfall-Dateien, die auf „jpg“ enden ohne Punkt (…25357jpg), die vom Loader unterstützt werden.
- Größe/Balance:
  - Analyse-Skript: [`src/analysis.py`](src/analysis.py). Erzeugt Reports unter reports/ (z. B. age_distribution.png).
  - Beobachtung: Das Histogramm der Altersverteilung (reports/age_distribution.png) zeigt typischerweise eine Schieflage hin zu mittleren Altersklassen (Imbalance).
- Imaging Source: „In the wild“ (heterogene Beleuchtung, Posen, Hintergründe), dadurch hoher Variabilitätsgrad und realistisches Rauschen.

2) Bezug zu realen Anwendungen
- Altersschätzung für:
  - Zugangsalterskontrollen (Altersverifikation).
  - Demographische Analysen, Marketing/Personalisierung.
  - Vorfilterung in Fotoverwaltung.
- Risiken/Hinweise:
  - Bias/Fairness: Ungleich verteilte Alters-, Geschlechts- oder Ethnizitätsgruppen können systematische Fehler erzeugen.
  - Datenschutz/Ethik: Gesichtsdaten sind sensibel; Anonymisierung und konforme Nutzung sind erforderlich.

3) Aktuelle Probleme und Lösungen (Stand dieses Projekts)
- Val-Loss/MAE stagnierte zeitweise früh
  - Ursache: Zu hohe oder zu konstante LR, Datensatz-Imbalance, Val mit Augmentierungen o. Ä.
  - Lösungen:
    - Striktes Eval-Preprocessing (nur Resize+Normalize) in [`src/training.py`](src/training.py).
    - Geeignete Scheduler wählen: OneCycle/Cosine; bei Stagnation `ReduceLROnPlateau` (LR halbieren).
    - Balanced Sampling nach Dekaden via `--balanced_sampler` (reduziert Bias zu mittleren Altern).
    - Early Stopping nutzen (z. B. `--early_stopping 6–10`).
- Falscher Loss im Klassifikationsmodus
  - Symptom: HuberLoss-Fehler mit Klassifikations-Targets (Warnung/RuntimeError).
  - Fix: Klassifikation verwendet CrossEntropyLoss (CE) mit optionalem Label Smoothing; Regression nutzt MAE/Huber. Implementiert in [`src/training.py`](src/training.py).
- Scheduler-Kompatibilität
  - Symptom: `ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`.
  - Fix: Fallback ohne `verbose` für ältere Torch-Versionen in [`src/training.py`](src/training.py).
- `AttributeError: 'NoneType' object has no attribute 'step'`
  - Ursache: Scheduler-Zweig wurde nicht korrekt instanziiert; späterer `step()` auf `None`.
  - Fix: Korrekte Verzweigung und Schutzabfragen (`if scheduler and ...`) in [`src/training.py`](src/training.py).
- Inferenz-Bias: Kinder/Ältere teils stark fehlgeschätzt (Richtung mittleres Alter)
  - Beobachtung: Einzelbilder mit 9–15 Jahren werden auf 30–50 geschätzt; Gesamt-MAE bleibt dennoch konsistent.
  - Ursachen: Imbalanced Data, Regressions-Bias zur Mitte.
  - Maßnahmen:
    - `--balanced_sampler` im Training.
    - EMA im Training (`--ema`) + TTA in Inferenz (`--tta`); Inferenz-Skript [`src/infer.py`](src/infer.py) mittelt optional Flip/TTA.
    - Optional (außerhalb dieses Abschnitts): Lineare Kalibrierung a·pred+b (unterstützt in [`src/infer.py`](src/infer.py)).
- Trainingsdauer zu hoch bei großem Setup
  - Beobachtung: `--img_size 192` + `--model_width 64` + starke Augmentierung verlangsamen Epochen deutlich.
  - Speed-Uplifts:
    - Kleinere Auflösung (160) und moderate Breite (48) bei minimalem MAE-Verlust.
    - OneCycleLR, AMP (aktiv per CUDA), erhöhte `--num_workers`, `pin_memory=True`.
    - `--aug_strength simple` im Training (ohne teure RandomResizedCrop/Erasing); `--tta_val` weglassen (halbiert Val-Zeit).
- Inferenz-Verbesserungen ohne Kalibrierung
  - Nutzung von TTA (H-Flip; optional Multiscale) im [`src/infer.py`](src/infer.py), konsistentes Preprocessing (Resize+Normalize), korrektes Checkpoint laden (Regression vs. Klassifikation).

4) Vorgehen und Begründungen (What/Why)
- Laden/Label-Parsing:
  - What: `AgeDataset` extrahiert das Alter aus dem Dateinamen (erstes Unterstrich-separiertes Token), filtert nach gültigen Endungen und Altersbereich.
  - Why: UTKFace-Standardkonvention, schnell und ohne separate Labeldatei.
- Preprocessing & Augmentation:
  - What: Resize, Normalize, RandomHorizontalFlip, leichte Rotation/ColorJitter in [`src/training.py`](src/training.py).
  - Why: Robuster gegen Pose/Helligkeit, reduziert Overfitting bei kleinem/heterogenem Datensatz.
- Split:
  - What: Stratifizierter Split nach Dekaden (stabilere Verteilung in Val).
  - Why: Repräsentative Validierung trotz Imbalance.
- Eigenes Modell:
  - What: `CustomAgeCNN` mit konfigurierbarer Breite, Conv-Blöcken, Global Average Pooling und kleinem MLP-Head ([`src/models/custom_cnn.py`](src/models/custom_cnn.py)).
  - Why: Eigenes CNN (Auflagen-konform), gute Balance aus Kapazität/Speed; GAP reduziert Parameter und macht Eingabegröße flexibel.
- Trainings-Setup:
  - What: AdamW, AMP (CUDA), OneCycle/Cosine/Plateau-Scheduler; optional EMA/TTA. Regression: MAE/Huber; Klassifikation: CrossEntropy mit Label Smoothing.
  - Why: Stabil und schnell; direkte Optimierung der Zielmetrik (MAE) bei Regression, robuste CE bei Klassifikation.
- Evaluation/Visualisierung:
  - What: Logs + Kurven (training_curves.png), Beispielsamples (samples.png), Ordner-Inferenz mit MAE/MSE (aus Dateinamen) in [`src/infer.py`](src/infer.py).
  - Why: Transparentes Monitoring, Validierung der Daten-/Labelqualität.

5) Ergebnisse und Interpretation
- Finaler Regressionslauf:
  - Bestes Val-MAE: 11.59 (Epoche ~18).
  - Interpretation: Durchschnittlicher Fehler ~11.6 Jahre. Für ein eigenes, kompaktes CNN ohne Transfer Learning auf UTKFace-artigen Daten solide.
  - Train/Val-Gap: moderat → akzeptable Generalisierung; bei großem Gap greifen Augmentierung, Weight Decay, Dropout.
- Batch-Inferenz (Ordner, 3252 Bilder, TTA):
  - Eval in [`src/infer.py`](src/infer.py): MAE=11.47, MSE=229.47 → konsistent zum Val-MAE.
  - Fehlercharakteristik:
    - Kinder/Jugendliche werden tendenziell zu alt geschätzt; sehr alte Personen etwas jünger.
    - Mittlere Altersbereiche mit geringerem Fehler.
- Klassifikation (10 Bins, erwarteter Alterswert):
  - Bei Klassifikation wird der Erwartungswert über Klassen-Bin-Zentren als Alter berechnet (in `infer.py` umgesetzt). Das reduziert Quantisierungsfehler gegenüber Hart-Labels, bleibt aber durch Bin-Breite limitiert.
- Praxis-Hinweis:
  - TTA in Val erhöht Laufzeit deutlich; für Training meist aus, zur finalen Inferenz an.

6) Limitierungen und Ausblick (Erweiterungsmöglichkeiten)
- Daten/Preprocessing:
  - Balanced Sampling weiter konsequent nutzen; optional gezieltes Oversampling seltener Dekaden.
  - Gesichtserkennung/Alignment vor dem Training (stabilere Gesichts-Crops, weniger Hintergrundrauschen).
  - Saubere Trennung: dediziertes Test-Set oder k-fold Cross-Validation.
- Modell/Learning:
  - Kapazität/Details erhöhen: `--model_width 64`, `--img_size 192–224` (mit Early Stopping/Regularisierung).
  - Loss-Tuning: Huber mit kleineren Deltas (3–5) vs. MAE; ggf. Kombinationen.
  - Ordinales Lernen (z. B. CORAL) oder Label-Distribution-Learning für glattere Altersverteilungen (reduziert Mittelwert-Bias).
  - Fortgeschrittene Schedules (Cosine mit Restarts, Plateau-Feintuning/Low-LR-Phasen).
  - SWA oder Snapshot-Ensembles (robustere Minima, +1–2 MAE-Punkte möglich).
- Inferenz:
  - Standardisierte TTA (H-Flip + optional Multiscale) ohne Kalibrierung; EMA-Gewichte für Deployment nutzen.
  - Optional lineare Kalibrierung a·pred+b, wenn auf Zielverteilung ein konstanter Bias bleibt (Flag `--calibrate` in [`src/infer.py`](src/infer.py)).
- Evaluation/Transparenz:
  - Pro-Dekade-MAE zusätzlich zum Gesamt-MAE reporten; Scatter GT vs. Pred visualisieren (Notebook/Script).
  - Bias-/Fairness-Analysen (falls Labels vorhanden).
- Transfer Learning (größter Hebel, falls erlaubt):
  - Vortrainierte Backbones (ResNet/MobileNet/EfficientNet) liefern auf UTKFace oft MAE ~6–9; im Vergleich zum eigenen CNN eine starke Baseline.

Referenzen auf Code
- Dataset: [`AgeDataset`](src/data/dataset.py)
- Modell: [`CustomAgeCNN`](src/models/custom_cnn.py)
- Training/Eval: [`src/training.py`](src/training.py)
- Metriken: [`mae`, `mse`](src/utils/metrics.py)
- Analyse: [`src/analysis.py`](src/analysis.py)
- Inferenz: [`src/infer.py`](src/infer.py)