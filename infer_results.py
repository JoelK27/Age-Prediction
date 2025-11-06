import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.dataset import _extract_age

CSV = r"infer_results.csv"  # Pfad bei Bedarf anpassen
OUT = r"reports"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
df["file"] = df["path"].apply(os.path.basename)
df["gt"] = df["file"].apply(_extract_age)
df = df.dropna(subset=["gt", "pred_age"]).copy()
df["gt"] = df["gt"].astype(float)
df["pred_age"] = df["pred_age"].astype(float)

# Scatter: GT vs. Pred
plt.figure(figsize=(5,5))
plt.scatter(df["gt"], df["pred_age"], s=8, alpha=0.4)
lims = [0, 120]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Ground Truth (Jahre)")
plt.ylabel("Prediction (Jahre)")
plt.title("Predicted vs. Ground Truth")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pred_vs_gt_scatter.png"), dpi=200)
plt.close()

# MAE pro Dekade
bins = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]
labels = [f"{lo}-{hi}" for lo,hi in bins]
def bucket(a):
    for i,(lo,hi) in enumerate(bins):
        if lo <= a <= hi:
            return labels[i]
    return labels[-1]

df["bucket"] = df["gt"].apply(bucket)
df["ae"] = (df["pred_age"] - df["gt"]).abs()
mae_per_bucket = df.groupby("bucket")["ae"].mean().reindex(labels)

plt.figure(figsize=(8,3))
mae_per_bucket.plot(kind="bar", color="#4472c4")
plt.ylabel("MAE (Jahre)")
plt.title("MAE pro Dekade")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "mae_per_decade.png"), dpi=200)
plt.close()

print("Gespeichert:", os.path.join(OUT, "pred_vs_gt_scatter.png"))
print("Gespeichert:", os.path.join(OUT, "mae_per_decade.png"))