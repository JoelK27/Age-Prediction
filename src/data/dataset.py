"""
Dataset-Loader für UTKFace-ähnliche Dateinamen (age_gender_race_timestamp.ext).

Wichtig:
- _extract_age liest das Alter aus dem Dateinamen (erstes Token vor dem ersten Unterstrich).
- Dateien ohne echten Suffix (…jpg statt … .jpg) werden toleriert.
- Regression: Ziel ist float (Alter), Klassifikation: Bucket-Index (0..num_classes-1).
"""

import os
from typing import Callable, Optional, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torch

# Zugelassene Endungen; .pg ist im Dump enthalten und wird hier toleriert
VALID_EXT = {".jpg", ".jpeg", ".png", ".pg"}  # .pg in deinem Dump

def _normalize_name(fn: str) -> str:
    """
    Ergänzt ".jpg", falls der Dateiname nur auf 'jpg' (ohne Punkt) endet.
    Das hilft nur beim Parsen – der echte Dateiname bleibt unverändert.
    """
    lower = fn.lower()
    if lower.endswith("jpg") and not lower.endswith(".jpg"):
        return fn[:-3] + ".jpg"
    return fn

def _extract_age(fn: str) -> Optional[int]:
    """
    Extrahiert das Alter aus UTKFace-Pattern: age_gender_race_timestamp.ext
    Gibt None zurück, wenn das Parsing fehlschlägt.
    """
    base = os.path.basename(fn)
    base = _normalize_name(base)
    core = base.split(".")[0]
    parts = core.split("_")
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None

class AgeDataset(Dataset):
    """
    Liest alle gültigen Bilder in root_dir und liefert (Tensor, Target).
    - transform: torchvision-Transform-Pipeline
    - regression=True: Target ist float (Alter)
    - regression=False: Target ist long (Klassenindex via age_bins)
    - min_age/max_age: optionaler Altersfilter (z. B. zum Entfernen von Ausreißern)
    """
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        regression: bool = True,
        age_bins: Optional[List[Tuple[int,int]]] = None,
        min_age: int = 0,
        max_age: int = 120
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.regression = regression
        self.age_bins = age_bins
        self.samples = []

        # Dateiliste einmalig einlesen und filtern
        for fn in sorted(os.listdir(root_dir)):
            age = _extract_age(fn)
            # Überspringe Dateien ohne valides Alter oder außerhalb des erlaubten Bereichs
            if age is None or not (min_age <= age <= max_age):
                continue

            name_lower = fn.lower()
            ext = "." + fn.split(".")[-1].lower() if "." in fn else ""
            # Akzeptiere Standard-Endungen sowie Dateien ohne Punkt, die auf "jpg" enden
            if ext in VALID_EXT or (ext == "" and name_lower.endswith("jpg")):
                path = os.path.join(root_dir, fn)
                self.samples.append((path, age))

        if not self.samples:
            raise RuntimeError(f"Keine gültigen Dateien in {root_dir}")

    def _age_to_class(self, age: int) -> int:
        """
        Weist ein Alter einem definierten Intervall (age_bins) zu.
        Fällt Alter nicht in ein Intervall (sollte nicht passieren), nutze letztes Bucket.
        """
        for idx, (lo, hi) in enumerate(self.age_bins):
            if lo <= age <= hi:
                return idx
        return len(self.age_bins) - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Lädt Bild als RGB, wendet Transform an und erzeugt Target je nach Modus.
        Rückgabe:
        - img: Tensor [C,H,W] (float, normalisiert)
        - target: float (Regression) oder long (Klassifikation)
        """
        path, age = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.regression:
            target = torch.tensor(age, dtype=torch.float32)
        else:
            # age_bins muss im Klassifikationsmodus gesetzt sein
            target = torch.tensor(self._age_to_class(age), dtype=torch.long)
        return img, target