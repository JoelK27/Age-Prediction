"""
Einfaches, kompaktes CNN für Age-Prediction.

Aufbau:
- Feature-Extractor mit 4 Conv-Blöcken:
  - Conv2d + BatchNorm + SiLU Aktivierung
  - Zwischenstufen mit MaxPool zum Downsampling (insgesamt /8)
  - Abschluss mit AdaptiveAvgPool2d(1), damit die Eingabegröße flexibel bleibt
- Classifier-Head (MLP):
  - Flatten -> Linear(c4->128) -> SiLU -> Dropout -> Linear(->1 oder ->num_classes)

Modi:
- regression=True: kontinuierliche Altersschätzung (1-D Ausgang), Rückgabe: [B]
  (Der Wert wird in der Inferenz auf [0,120] geklemmt.)
- regression=False: Klassifikation in num_classes Buckets, Rückgabe: [B, C]

Hinweis:
- Die Breite der Feature-Kanäle wird über 'width' skaliert (c1=width, c2=2*width, ...).
- SiLU ist eine robuste Aktivierung und oft genauer/stabiler als ReLU für solche Aufgaben.
"""

import torch
import torch.nn as nn

class CustomAgeCNN(nn.Module):
    """
    Kompakt-CNN für Altersschätzung.

    Parameter:
    - regression (bool): True für Regression (Alter in Jahren), False für Klassifikation (Buckets)
    - num_classes (int): Anzahl Klassen bei Klassifikation (z. B. 10 Dekaden-Buckets)
    - width (int): Kanalbreite des ersten Blocks; spätere Blöcke skalieren mit 2x, 4x, 8x
    - classifier_dropout (float): Dropout-Rate im MLP-Head (nur bei Klassifikation wirksam)
    """
    def __init__(self, regression=True, num_classes=10, width=32, classifier_dropout=0.35):
        super().__init__()
        self.regression = regression

        # Kanalbreiten pro Block (wird mit 'width' skaliert)
        c1, c2, c3, c4 = width, width*2, width*4, width*8
        act = nn.SiLU  # Aktivierung: gut für Regressionsaufgaben und stabil in der Praxis

        # Feature-Extractor:
        # - Zwei Convs im ersten und zweiten Block (mehr Repräsentationskraft bei geringer Rechenlast)
        # - MaxPool reduziert Auflösung (Downsampling) und erhöht Rezeption
        # - AdaptiveAvgPool2d(1) macht das Netz eingabegrößenunabhängig
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), act(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), act(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), act(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), act(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), act(),
            nn.MaxPool2d(2),

            nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), act(),
            nn.AdaptiveAvgPool2d(1)  # Ausgabeform: [B, c4, 1, 1]
        )

        # Classifier-Head:
        # - Reduktion auf 128 Kanäle + Aktivierung
        # - Dropout als Regularisierung (wirkt v. a. im Klassifikationsmodus)
        # - Finale Linear-Schicht: 1 (Regression) oder num_classes (Klassifikation)
        self.classifier = nn.Sequential(
            nn.Flatten(),              # [B, c4, 1, 1] -> [B, c4]
            nn.Linear(c4, 128),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1 if regression else num_classes)
        )

    def forward(self, x):
        """
        Vorwärtsdurchlauf.

        Eingabe:
        - x: Tensor [B, 3, H, W], H und W beliebig (werden über GAP gehandhabt)

        Rückgabe:
        - Regression: Tensor [B] mit kontinuierlichen Alterswerten (unkalibriert, unclamped)
        - Klassifikation: Tensor [B, C] mit Logits für Softmax/CE
        """
        x = self.features(x)   # Feature-Extraktion
        x = self.classifier(x) # MLP-Head

        if self.regression:
            # [B, 1] -> [B]; Clamping/Kalibrierung erfolgt in der Inferenz-Pipeline
            return x.squeeze(1)
        return x