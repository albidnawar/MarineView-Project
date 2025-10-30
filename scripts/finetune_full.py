# scripts/finetune_full.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18
import torchaudio, soundfile as sf


# ---------------- Utils ----------------
def load_wav_safe(path, target_sr=16000):
    """Robust loader: torchaudio -> soundfile; mono; resample to target_sr."""
    try:
        wav, sr = torchaudio.load(path)  # [C,T]
        wav = wav.mean(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav
    except Exception:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if isinstance(x, np.ndarray) and x.ndim == 2:
            x = x.mean(axis=1)
        wav = torch.from_numpy(x if isinstance(x, np.ndarray) else np.array(x, dtype=np.float32))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav


# --------------- Dataset ---------------
class MelDS(Dataset):
    def __init__(self, csv_path, label_list=None, sr=16000, n_mels=64, hop_ms=10, train=False):
        """
        If label_list is provided, use it to build a fixed mapping for both
        train and val sets (prevents index mismatch).
        """
        self.df = pd.read_csv(csv_path)
        self.train = train
        self.sr = sr

        hop = int(sr * hop_ms / 1000)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=n_mels, hop_length=hop, n_fft=1024,
            center=True, pad_mode="constant"
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.target_T = int(round((2.0 * sr) / hop))
        self.need = int(2.0 * sr)

        if label_list is None:
            self.labels = sorted(self.df["species"].unique())
        else:
            self.labels = list(label_list)
            # drop rows with labels not in label_list (e.g., unseen in train)
            self.df = self.df[self.df["species"].isin(self.labels)].reset_index(drop=True)
        self.map = {c: i for i, c in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        p = self.df.filepath[i]
        s = self.df.species[i]
        y = self.map[s]

        wav = load_wav_safe(p, self.sr)
        if wav.numel() < self.need:
            wav = F.pad(wav, (0, self.need - wav.numel()))

        M = self.db(self.mel(wav) + 1e-10)   # [n_mels, T]
        M = (M - M.mean()) / (M.std() + 1e-6)

        # Simple SpecAugment on train
        if self.train:
            t = M.size(1)
            if t > 20:
                t0 = torch.randint(0, t - 20, (1,)).item()
                M[:, t0:t0 + 20] = M.min()
            f = M.size(0)
            if f > 8:
                f0 = torch.randint(0, f - 8, (1,)).item()
                M[f0:f0 + 8, :] = M.min()

        # pad/crop time to fixed frames
        T = M.size(1)
        if T < self.target_T:
            M = F.pad(M, (0, self.target_T - T))
        elif T > self.target_T:
            start = max(0, (T - self.target_T) // 2)
            M = M[:, start:start + self.target_T]

        return M.unsqueeze(0), torch.tensor(y, dtype=torch.long)


# --------------- Model -----------------
class SupModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Identity()
        self.encoder = base
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# --------------- Losses ----------------
def focal_loss(logits, y, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p[torch.arange(y.size(0), device=y.device), y]
    return (-(1 - pt) ** gamma * logp[torch.arange(y.size(0), device=y.device), y]).mean()


def class_balanced_ce(logits, y, class_counts, beta=0.9999, device="cpu"):
    eff = 1.0 - np.power(beta, class_counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(class_counts)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    return F.cross_entropy(logits, y, weight=w)


# --------------- Train -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/labels/allcuts_train.csv")
    ap.add_argument("--val_csv",   default="data/labels/allcuts_val.csv")
    ap.add_argument("--epochs",    type=int, default=40)
    ap.add_argument("--batch",     type=int, default=128)
    ap.add_argument("--loss",      type=str, default="focal", choices=["focal", "class_balanced"])
    ap.add_argument("--head_lr",   type=float, default=1e-3)
    ap.add_argument("--enc_lr",    type=float, default=1e-4)
    ap.add_argument("--ckpt",      type=str, default="checkpoints/finetune_full.pt",
                    help="where to save the best checkpoint")
    ap.add_argument("--no_ssl",    action="store_true", help="train from scratch (do not load ssl_encoder.pt)")
    ap.add_argument("--workers",   type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    # Shared label list from TRAIN
    df_train_full = pd.read_csv(args.train_csv)
    label_list = sorted(df_train_full["species"].unique())
    num_classes = len(label_list)

    # Datasets / loaders
    tr = MelDS(args.train_csv, label_list=label_list, train=True)
    va = MelDS(args.val_csv,   label_list=label_list, train=False)

    # class-balanced sampler for train
    class_counts_series = tr.df["species"].value_counts().reindex(label_list, fill_value=0)
    counts_per_item = tr.df["species"].map(class_counts_series.to_dict()).astype(float).values
    weights = torch.tensor(1.0 / counts_per_item, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(tr, batch_size=args.batch, sampler=sampler, num_workers=args.workers)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # Model
    model = SupModel(num_classes).to(device)

    # Optionally load SSL backbone
    if not args.no_ssl:
        ssl_path = "checkpoints/ssl_encoder.pt"
        if Path(ssl_path).exists():
            ssl = torch.load(ssl_path, map_location="cpu")
            backbone_state = {k.replace("backbone.", ""): v
                              for k, v in ssl.items() if k.startswith("backbone.")}
            print(f"[info] Loaded SSL weights: {len(backbone_state)} params")
            model.encoder.load_state_dict(backbone_state, strict=False)
        else:
            print(f"[warn] {ssl_path} not found → training from scratch.")
    else:
        print("[info] --no_ssl set → training from scratch.")

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.enc_lr},
        {"params": model.head.parameters(),    "lr": args.head_lr},
    ], weight_decay=1e-4)

    counts_array = class_counts_series.values
    best = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        for M, y in dl_tr:
            M, y = M.to(device), y.to(device)
            logits = model(M)
            if args.loss == "focal":
                loss = focal_loss(logits, y, gamma=2.0)
            else:
                loss = class_balanced_ce(logits, y, counts_array, beta=0.9999, device=device)
            opt.zero_grad(); loss.backward(); opt.step()

        # Validate
        model.eval(); correct = seen = 0
        with torch.no_grad():
            for M, y in dl_va:
                M, y = M.to(device), y.to(device)
                pred = model(M).argmax(1)
                correct += (pred == y).sum().item()
                seen += y.size(0)
        acc = correct / max(1, seen)
        print(f"epoch {ep}: val_acc={acc:.3f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.ckpt)
    print("best val acc:", best)


if __name__ == "__main__":
    main()
