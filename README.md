# Age-Prediction

## Schritte
1. Dataset Analyse:
```
python -m src.analysis --data_dir UTKface_inthewild/part3
```
2. Training (Regression):
```
python -m src.training --data_dir UTKface_inthewild/part3 --epochs 15 --batch_size 32 --regression --plot_samples
```
3. Inferenz:
```
python -m src.inference --image data/sample/example.jpg --checkpoint checkpoints/best_epoch_X.pt --regression
```

## Report Leitfragen
- Datenquelle & Struktur (Dateinamensschema, Anzahl, Altersverteilung).
- Balance (Histogramm age_distribution.png).
- Augmentation: Warum diese Operationen?
- Eigenes CNN: Architekturentscheidungen (Tiefe, Kernel 3x3, Dropout).
- Metriken: MAE & MSE für Regression (Begründung).
- Probleme & Lösungen (fehlende Dateiendungen, inkonsistente Extensions).
- Real-World: Zugangskontrolle, demographische Analysen (Bias erwähnen).