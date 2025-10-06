import os, collections, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.data.dataset import _extract_age, _normalize_name

def analyze(root: str, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    ages = []
    exts = collections.Counter()
    for fn in os.listdir(root):
        age = _extract_age(fn)
        if age is not None:
            ages.append(age)
        if "." in fn:
            exts["." + fn.split(".")[-1].lower()] += 1

    if not ages:
        print("Keine Alterslabels gefunden.")
        return
    s = pd.Series(ages)
    print("Anzahl Bilder:", len(s))
    print("Alter Min/Max/Ø:", s.min(), s.max(), round(s.mean(),2))
    print("Perzentile (25/50/75):", s.quantile([0.25,0.5,0.75]).to_dict())
    print("Extensions:", exts)

    plt.figure(figsize= (8,4))
    sns.histplot(s, bins=30, kde=True)
    plt.title("Altersverteilung")
    plt.xlabel("Alter")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "age_distribution.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    analyze(args.data_dir, args.out_dir)