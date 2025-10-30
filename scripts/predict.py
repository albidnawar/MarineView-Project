# src/predict.py
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torchvision.models import resnet18

# ---------- constants (match your training) ----------
SR = 16000
N_MELS = 64
HOP_MS = 10
N_FFT = 1024
WINDOW_S = 2.0  # fixed 2s crops
# ----------------------------------------------------

def load_wav_safe(path, target_sr=SR):
    """Robust loader: torchaudio -> soundfile, mono, resampled."""
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

class SupModel(nn.Module):
    """ResNet18 encoder for 1-channel spectrograms + linear head."""
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Identity()
        self.encoder = base
        self.head = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.head(self.encoder(x))

def build_mel():
    hop = int(SR * HOP_MS / 1000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_mels=N_MELS, hop_length=hop, n_fft=N_FFT,
        center=True, pad_mode="constant"
    )
    db = torchaudio.transforms.AmplitudeToDB()
    target_T = int(round((WINDOW_S * SR) / hop))
    need_len = int(WINDOW_S * SR)
    return mel, db, target_T, need_len

def wav_to_input(wav: torch.Tensor, mel, db, target_T, need_len) -> torch.Tensor:
    # pad waveform to at least 2s for stability
    if wav.numel() < need_len:
        wav = F.pad(wav, (0, need_len - wav.numel()))
    # log-mel
    M = db(mel(wav) + 1e-10)           # [n_mels, T]
    M = (M - M.mean()) / (M.std() + 1e-6)
    # center crop/pad to fixed T
    T = M.size(1)
    if T < target_T:
        M = F.pad(M, (0, target_T - T))
    elif T > target_T:
        start = max(0, (T - target_T)//2)
        M = M[:, start:start+target_T]
    return M.unsqueeze(0).unsqueeze(0)  # [1,1,n_mels,T]

def topk_from_logits(logits: torch.Tensor, labels: list[str], k: int = 5):
    probs = logits.softmax(dim=1).squeeze(0)   # [C]
    val, idx = torch.topk(probs, k=min(k, probs.numel()))
    val = val.detach().cpu().numpy().tolist()
    idx = idx.detach().cpu().numpy().tolist()
    return [(labels[i], float(p)) for i, p in zip(idx, val)]

def expand_audio_pattern(pattern: str):
    # Accept single file, folder, or glob pattern; return list of files
    p = Path(pattern)
    if p.is_file():
        return [str(p)]
    # glob (supports **)
    files = glob.glob(pattern, recursive=True)
    # if a directory was passed, default to *.wav under it
    if p.exists() and p.is_dir() and not files:
        files = glob.glob(str(p / "**" / "*.wav"), recursive=True)
    # filter common audio extensions
    keep_ext = {".wav", ".flac", ".aif", ".aiff", ".mp3"}  # mp3 will be handled by soundfile if available
    files = [f for f in files if Path(f).suffix.lower() in keep_ext]
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--train_csv", required=True, help="CSV used for training labels (data\\labels\\allcuts_train.csv)")
    ap.add_argument("--audio", required=True, help="Path to file/folder/glob (e.g., D:\\clips\\**\\*.wav)")
    ap.add_argument("--topk", type=int, default=5, help="How many predictions to show")
    ap.add_argument("--out", type=str, default="", help="Optional CSV to save predictions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load label list from TRAIN CSV (shared mapping)
    df_tr = pd.read_csv(args.train_csv)
    label_list = sorted(df_tr["species"].unique())
    num_classes = len(label_list)

    # 2) Build model + load weights
    model = SupModel(num_classes=num_classes).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3) Audio list
    files = expand_audio_pattern(args.audio)
    if not files:
        print(f"[error] No audio found for pattern: {args.audio}")
        return
    print(f"[info] Found {len(files)} file(s).")

    # 4) Preprocessors
    mel, db, target_T, need_len = build_mel()

    results = []
    for i, fpath in enumerate(files, 1):
        try:
            wav = load_wav_safe(fpath, SR)
            x = wav_to_input(wav, mel, db, target_T, need_len).to(device)
            with torch.no_grad():
                logits = model(x)
            topk = topk_from_logits(logits, label_list, k=args.topk)
            # print pretty
            print(f"\n[{i}/{len(files)}] {fpath}")
            for rank, (lab, prob) in enumerate(topk, 1):
                print(f"  {rank}. {lab:30s}  {prob:.3f}")
            # save raw top-1 to table
            top1_label, top1_prob = topk[0]
            results.append({"filepath": fpath, "pred_label": top1_label, "pred_prob": round(top1_prob, 4)})
        except Exception as e:
            print(f"[skip] {fpath}: {e}")

    # 5) Optional CSV
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\n[saved] {out_path}")

if __name__ == "__main__":
    main()
