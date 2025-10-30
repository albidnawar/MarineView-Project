import os, csv, random
from pathlib import Path

SRC = Path("data/raw/full_cuts_2s")  # hierarchical folders -> species
OUT = Path("data/labels"); OUT.mkdir(parents=True, exist_ok=True)

rows = []
for sp_dir in SRC.iterdir():
    if not sp_dir.is_dir(): continue
    species = sp_dir.name.replace("_"," ").replace("-"," ")
    for wav in sp_dir.rglob("*.wav"):
        rows.append((str(wav.resolve()), species))

if not rows:
    raise SystemExit("No WAV files found under data/raw/full_cuts_2s")

paths, labels = zip(*rows)

# Count per-class occurrences
counts = {}
for s in labels:
    counts[s] = counts.get(s, 0) + 1

# Indices for classes with at least 2 samples and singletons
eligible_idx = [i for i, s in enumerate(labels) if counts[s] >= 2]
singleton_idx = [i for i, s in enumerate(labels) if counts[s] == 1]

def stratified_split(X, y, test_size: float, seed: int):
    rnd = random.Random(seed)
    by_class = {}
    for idx, cls in enumerate(y):
        by_class.setdefault(cls, []).append(idx)
    test_indices = []
    train_indices = []
    for cls, idxs in by_class.items():
        idxs = list(idxs)
        rnd.shuffle(idxs)
        n = len(idxs)
        # Ensure at least one sample in both splits when possible
        n_test = max(1, int(round(n * test_size)))
        if n_test >= n:
            n_test = n - 1
        test_indices.extend(idxs[:n_test])
        train_indices.extend(idxs[n_test:])
    # Keep global order randomized but reproducible
    rnd.shuffle(test_indices)
    rnd.shuffle(train_indices)
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test  = [X[i] for i in test_indices]
    y_test  = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test

def subset(arr, idxs):
    return [arr[i] for i in idxs]

X_tr, y_tr = [], []

if eligible_idx:
    X_elig = subset(paths, eligible_idx)
    y_elig = subset(labels, eligible_idx)
    X_tr_e, X_tmp_e, y_tr_e, y_tmp_e = stratified_split(
        X_elig, y_elig, test_size=0.25, seed=1337
    )
    # Add singletons to train
    X_tr = list(X_tr_e) + subset(paths, singleton_idx)
    y_tr = list(y_tr_e) + subset(labels, singleton_idx)

    # Second split on the remaining (tmp) set, but only stratify labels with >=2
    if X_tmp_e:
        tmp_counts = {}
        for s in y_tmp_e:
            tmp_counts[s] = tmp_counts.get(s, 0) + 1
        elig2_idx = [i for i, s in enumerate(y_tmp_e) if tmp_counts[s] >= 2]
        single2_idx = [i for i, s in enumerate(y_tmp_e) if tmp_counts[s] == 1]

        X_va, X_te, y_va, y_te = [], [], [], []
        if elig2_idx:
            X_tmp2 = subset(X_tmp_e, elig2_idx)
            y_tmp2 = subset(y_tmp_e, elig2_idx)
            X_va_e, X_te_e, y_va_e, y_te_e = stratified_split(
                X_tmp2, y_tmp2, test_size=0.4, seed=1337
            )
            X_va = list(X_va_e)
            y_va = list(y_va_e)
            X_te = list(X_te_e)
            y_te = list(y_te_e)
        # Put the remaining singletons into validation set
        if single2_idx:
            X_va += subset(X_tmp_e, single2_idx)
            y_va += subset(y_tmp_e, single2_idx)
    else:
        X_va, X_te, y_va, y_te = [], [], [], []
else:
    # No class has >=2 samples; put everything in train
    X_tr = list(paths)
    y_tr = list(labels)
    X_va, X_te, y_va, y_te = [], [], [], []

def write_csv(name, X, Y):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filepath","species"])
        for p, s in zip(X, Y): w.writerow([p, s])

write_csv("allcuts_train.csv", X_tr, y_tr)
write_csv("allcuts_val.csv",   X_va, y_va)
write_csv("allcuts_test.csv",  X_te, y_te)

print("splits:", len(X_tr), "train,", len(X_va), "val,", len(X_te), "test")
