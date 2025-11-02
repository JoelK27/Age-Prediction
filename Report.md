# Project Report – Age Prediction

1) Dataset-Beschreibung
- Quelle/Typ: UTKFace-ähnlicher „in the wild“ Datensatz mit Dateinamensschema age_gender_race_timestamp.ext. Geladen über [`src/data/dataset.py`](src/data/dataset.py) (Klasse `AgeDataset`).
- Struktur: Bilder liegen flach in einem Ordner (nicht rekursiv). Beispiele: 23_0_0_201701161745.jpg, inkl. Sonderfall-Dateien, die auf „jpg“ enden ohne Punkt (…25357jpg), die vom Loader unterstützt werden.
- Größe/Balance:
  - Analyse-Skript: [`src/analysis.py`](src/analysis.py). Erzeugt Reports unter reports/ (z. B. age_distribution.png).
  - Beobachtung: Das Histogramm der Altersverteilung (reports/age_distribution.png) zeigt typischerweise eine Schieflage hin zu mittleren Altersklassen. Bitte die konkrete Grafik aus deinem Lauf referenzieren.
- Imaging Source: „In the wild“ (heterogene Beleuchtung, Posen, Hintergründe), dadurch hoher Variabilitätsgrad und realistisches Rauschen.

2) Bezug zu realen Anwendungen
- Altersschätzung für:
  - Zugangsalterskontrollen (Altersverifikation).
  - Demographische Analysen, Marketing/Personalisierung.
  - Vorfilterung in Fotoverwaltung.
- Risiken/Hinweise:
  - Bias/ Fairness: Ungleich verteilte Alters-, Geschlechts- oder Ethnizitätsgruppen können systematische Fehler erzeugen.
  - Datenschutz/Ethik: Gesichtsdaten sind sensibel; Anonymisierung und konforme Nutzung sind erforderlich.

3) Probleme und Lösungen
- Pfadfehler bei der Analyse/ dem Laden:
  - Problem: „Pfad nicht gefunden“ bei relativen Pfaden.
  - Lösung: Verbesserte Fehlermeldung in [`src/analysis.py`](src/analysis.py); Nutzung absoluter Pfade in README dokumentiert.
- Inkonsistente Dateiendungen:
  - Problem: Dateien ohne „.jpg“ (…jpg) führten zu Parsing-/Filterproblemen.
  - Lösung: Normalisierung/Erweiterung im Loader (`_normalize_name`) und erweiterter Filter in [`src/data/dataset.py`](src/data/dataset.py).
- Langsames Training:
  - Problem: Lange Epochen bei hoher Auflösung/ großem Modell.
  - Lösungen: 
    - Kleinere Eingabegröße (--img_size, z. B. 160), leichteres Modell mit Global Average Pooling in [`src/models/custom_cnn.py`](src/models/custom_cnn.py).
    - Mixed Precision (AMP) und DataLoader-Tuning (--num_workers, pin_memory) in [`src/training.py`](src/training.py).
- Deprecation-Warnung AMP:
  - Problem: GradScaler-API veraltet.
  - Lösung: Umstellung auf `torch.amp.GradScaler` in [`src/training.py`](src/training.py).
- Versionskontrolle großer Daten:
  - Problem: Datasets im Git.
  - Lösung: Explizites Ignorieren des Datenordners in [.gitignore](.gitignore), Reports/Plots bleiben versionierbar.

4) Vorgehen und Begründungen (What/Why)
- Laden/Label-Parsing:
  - What: `AgeDataset` extrahiert das Alter aus dem Dateinamen (erstes Unterstrich-separiertes Token), filtert nach gültigen Endungen und Altersbereich.
  - Why: UTKFace-Standardkonvention, schnell und ohne separate Labeldatei.
- Preprocessing & Augmentation:
  - What: Resize, Normalize, RandomHorizontalFlip, leichte Rotation/ColorJitter in [`src/training.py`](src/training.py).
  - Why: Robuster gegen Pose/Helligkeit, reduziert Overfitting bei kleinem/ heterogenem Datensatz.
- Split:
  - What: random 80/20 Train/Val (`--val_split` einstellbar).
  - Why: Einfache, schnelle Validierung; klare Trennung zur Überwachung der Generalisierung.
- Eigenes Modell:
  - What: `CustomAgeCNN` mit konfigurierbarer Breite, Conv-Blöcken und Global Average Pooling, kleiner MLP-Head ([`src/models/custom_cnn.py`](src/models/custom_cnn.py)).
  - Why: Eigenes CNN (Auflagen-konform), gute Balance aus Kapazität/Speed; GAP reduziert Parameter und macht die Eingabegröße flexibel.
- Trainings-Setup:
  - What: Adam (`--lr 1e-3`), MAE (L1) für Regression, CrossEntropy für Klassifikation, AMP auf CUDA, Best-Checkpoint-Speicherung nach Validierungsmetrik ([`src/training.py`](src/training.py)).
  - Why: 
    - MAE spiegelt das Ziel (Durchschnittsfehler in Jahren) direkt wider und ist robust gegen Ausreißer.
    - AMP beschleunigt Training auf GPU, Checkpointing verhindert Überanpassung.
- Evaluation/Visualisierung:
  - What: Konsole-Logs, Speicherung von Trainingskurven (training_curves.png) und Beispielsamples (samples.png).
  - Why: Nachvollziehbarkeit des Lernverlaufs, visuelle Prüfung der Datenqualität und Augmentierungen.
- Inferenz:
  - What: [`src/infer.py`](src/infer.py) lädt Checkpoint und gibt vorhergesagtes Alter bzw. Klassenindex aus.
  - Why: Reproduzierbare Testbarkeit am Einzelbild.

5) Ergebnisse (kurz)
- Artefakte:
  - reports/age_distribution.png: Altersverteilung.
  - checkpoints/training_curves.png: Verlust/ MAE bzw. Accuracy über Epochen.
  - checkpoints/best_epoch_X.pt: Bestes Modell nach Validierungsmetrik.
- Interpretation:
  - Nutze MAE zur Beurteilung der Regression. Typisch liegt die anfängliche MAE höher und fällt über die Epochen; prüfe, ob der Abstand zwischen Train/Val moderat bleibt (kein starkes Overfitting).

6) Limitierungen und Ausblick
- Limitierungen:
  - Keine explizite Test-Set-Abspaltung/keine k-fold Validierung.
  - Keine Fairness-/Bias-Metriken.
  - Loader liest nicht rekursiv.
- Ausblick:
  - Transfer Learning (z. B. ResNet-Backbone) als starker Baseline-Vergleich.
  - Separates Test-Set oder k-fold Cross-Validation.
  - Uncertainty-Schätzung (z. B. MC-Dropout).
  - Fairness-/Bias-Analyse nach Demografie (falls Labels vorhanden).
  - Rekursives Laden oder DataFrame-basiertes Manifest.

Referenzen auf Code
- Dataset: [`AgeDataset`](src/data/dataset.py)
- Modell: [`CustomAgeCNN`](src/models/custom_cnn.py)
- Training/Eval: [`src/training.py`](src/training.py)
- Metriken: [`mae`, `mse`](src/utils/metrics.py)
- Analyse: [`src/analysis.py`](src/analysis.py)
- Inferenz: [`src/infer.py`](src/infer.py)