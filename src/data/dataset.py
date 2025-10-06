import os
from typing import Callable, Optional, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torch

VALID_EXT = {".jpg", ".jpeg", ".png", ".pg"}  # .pg in deinem Dump

def _normalize_name(fn: str) -> str:
    # Fälle wie 55_0_0_...25357jpg -> künstlich ".jpg" einfügen falls fehlt
    if fn.lower().endswith("jpg") and not fn.lower().endswith(".jpg"):
        return fn[:-3] + ".jpg"
    return fn

def _extract_age(fn: str) -> Optional[int]:
    # UTKFace Pattern: age_gender_race_timestamp.jpg
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
        for fn in os.listdir(root_dir):
            age = _extract_age(fn)
            if age is None or not (min_age <= age <= max_age):
                continue
            ext = "." + fn.split(".")[-1].lower() if "." in fn else ""
            if ext in VALID_EXT:
                path = os.path.join(root_dir, fn)
                self.samples.append((path, age))
        if not self.samples:
            raise RuntimeError(f"Keine gültigen Dateien in {root_dir}")

    def _age_to_class(self, age: int) -> int:
        for idx, (lo, hi) in enumerate(self.age_bins):
            if lo <= age <= hi:
                return idx
        return len(self.age_bins) - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age = self.samples[idx]
        norm_name = _normalize_name(path)
        if norm_name != path and os.path.exists(path):
            # optional: rename on the fly (nicht zwingend)
            pass
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.regression:
            target = torch.tensor(age, dtype=torch.float32)
        else:
            target = torch.tensor(self._age_to_class(age), dtype=torch.long)
        return img, target