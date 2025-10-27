import torch
import torch.nn as nn

class CustomAgeCNN(nn.Module):
    def __init__(self, regression=True, num_classes=10, width=32):
        super().__init__()
        self.regression = regression
        c1, c2, c3, c4 = width, width*2, width*4, width*8
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # macht Inputgröße flexibel und klein
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1 if regression else num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.regression:
            return x.squeeze(1)
        return x