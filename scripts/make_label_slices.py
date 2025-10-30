# scripts/make_label_slices.py
import pandas as pd, numpy as np
from pathlib import Path
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("--train_csv", default="data/labels/allcuts_train.csv")
ap.add_argument("--out_dir", default="data/labels/slices")
ap.add_argument("--fractions", nargs="+", default=["0.01","0.05","0.10","0.25"])
args=ap.parse_args()

df=pd.read_csv(args.train_csv)
out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

for f in args.fractions:
    frac=float(f)
    df_slice=df.groupby("species", group_keys=False).apply(lambda x: x.sample(max(1,int(len(x)*frac)), random_state=1337))
    p=out/f"train_{int(frac*100)}.csv"
    df_slice.to_csv(p, index=False)
    print("saved", p, "rows:", len(df_slice))
