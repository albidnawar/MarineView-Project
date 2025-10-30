import pandas as pd, random
from pathlib import Path

FRACS = [0.01, 0.05, 0.10, 0.25]  # 1%, 5%, 10%, 25%
SRC = Path("data/labels/allcuts_train.csv")
OUT = Path("data/labels/fracs"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)

for f in FRACS:
    df_frac = df.groupby("species", group_keys=False).apply(
        lambda x: x.sample(max(1, int(len(x)*f)), random_state=1337)
    )
    df_frac.to_csv(OUT / f"train_frac_{int(f*100)}.csv", index=False)
    print(f"Saved {len(df_frac)} samples for {int(f*100)}%")
