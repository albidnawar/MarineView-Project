# src/eval_report.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# -----------------------
# Robust audio loader
# -----------------------
def load_wav_safe(path, target_sr=16000):
    """Load WAV/FLAC robustly, mono, resampled to target_sr."""
    try:
        wav, sr = torchaudio.load(path)  # [C, T]
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


# -----------------------
# Dataset (fixed label list from TRAIN)
# -----------------------
class MelDS(Dataset):
    def __init__(self, csv_path, label_list, sr=16000, n_mels=64, hop_ms=10, window_s=2.0):
        """
        csv_path: path to CSV with columns [filepath, species]
        label_list: list of species strings (from TRAIN) that define class indices
        """
        df = pd.read_csv(csv_path)
        # Keep only rows with labels seen during training
        df = df[df["species"].isin(label_list)].reset_index(drop=True)
        self.filtered_out = None
        self.filtered_out = None  # kept for compatibility
        self.dropped = None
        # (we'll report dropped count in main)

        self.df = df
        self.sr = sr
        self.label_list = list(label_list)
        self.map = {c: i for i, c in enumerate(self.label_list)}

        hop = int(sr * hop_ms / 1000)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            hop_length=hop,
            n_fft=1024,
            center=True,
            pad_mode="constant",
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.target_T = int(round((window_s * sr) / hop))
        self.need_len = int(window_s * sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        p = self.df.filepath[i]
        s = self.df.species[i]
        y = self.map[s]  # guaranteed to exist

        # waveform -> log-mel
        wav = load_wav_safe(p, self.sr)
        if wav.numel() < self.need_len:
            wav = F.pad(wav, (0, self.need_len - wav.numel()))
        M = self.db(self.mel(wav) + 1e-10)
        M = (M - M.mean()) / (M.std() + 1e-6)

        # pad/crop time to fixed frames
        T = M.size(1)
        if T < self.target_T:
            M = F.pad(M, (0, self.target_T - T))
        elif T > self.target_T:
            start = max(0, (T - self.target_T) // 2)
            M = M[:, start : start + self.target_T]

        return M.unsqueeze(0), y


# -----------------------
# Model
# -----------------------
class SupModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        base.fc = nn.Identity()
        self.encoder = base
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# -----------------------
# Plot confusion matrix
# -----------------------
def plot_confusion(cm, labels, out_png):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (e.g., checkpoints\\finetune_full_focal.pt)")
    ap.add_argument("--train_csv", required=True, help="data\\labels\\allcuts_train.csv")
    ap.add_argument("--test_csv", required=True, help="data\\labels\\allcuts_test.csv")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the class list from TRAIN ONLY (what the model learned)
    df_train = pd.read_csv(args.train_csv)
    label_list = sorted(df_train["species"].unique())
    num_classes = len(label_list)

    # Build test dataset but DROP any samples with unseen labels
    df_test = pd.read_csv(args.test_csv)
    n_test_total = len(df_test)
    n_test_kept = (df_test["species"].isin(label_list)).sum()
    n_test_dropped = n_test_total - n_test_kept
    if n_test_dropped > 0:
        print(f"[info] Test samples dropped (unseen labels): {n_test_dropped} / {n_test_total}")

    ds_te = MelDS(args.test_csv, label_list=label_list)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    # Build model with the SAME number of classes as TRAIN
    model = SupModel(num_classes=num_classes).to(device)

    # Load checkpoint (strict=False is fine if there are extra keys)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for M, y in dl_te:
            M = M.to(device)
            logits = model(M)
            pred = logits.argmax(1).cpu().numpy().tolist()
            ps.extend(pred)
            ys.extend(y if isinstance(y, list) else y.numpy().tolist())

    # Classification report using the TRAIN label set
    rep = classification_report(
        ys,
        ps,
        labels=list(range(num_classes)),
        target_names=label_list,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))

    Path("reports").mkdir(exist_ok=True)
    base = Path(args.ckpt).stem
    txt_path = Path("reports") / f"{base}_report.txt"
    png_path = Path("reports") / f"{base}_cm.png"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(rep)

    print(rep)
    print(f"[saved] {txt_path}")

    plot_confusion(cm, label_list, png_path)
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
