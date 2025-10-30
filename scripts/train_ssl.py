# scripts/train_ssl.py
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


# ---------------------------
# Robust audio loading
# ---------------------------

def load_wav_safe(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio as mono float32 at target_sr.
    1) Try torchaudio
    2) Fallback to soundfile (libsndfile)
    Returns: (wav[T], sr)
    """
    # 1) torchaudio
    try:
        wav, sr = torchaudio.load(path)  # [C, T]
        wav = wav.mean(0)                # mono [T]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav, target_sr
    except Exception:
        pass

    # 2) soundfile fallback
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x is None:
        raise RuntimeError(f"Unreadable audio: {path}")
    if x.ndim == 2:  # [T, C]
        x = x.mean(axis=1)
    wav = torch.from_numpy(x)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


# ---------------------------
# Dataset (fixed-length spectrograms)
# ---------------------------

class SSLDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        sr: int = 16000,
        n_mels: int = 64,
        hop_ms: int = 10,
        window_s: float = 2.0,
        n_fft: int = 1024,
    ):
        self.paths = [p.strip() for p in open(list_file, "r", encoding="utf-8") if p.strip()]
        if len(self.paths) == 0:
            raise RuntimeError(f"No files listed in: {list_file}")

        self.sr = sr
        self.n_mels = n_mels
        self.hop = int(sr * hop_ms / 1000)          # samples per hop
        self.window_s = window_s
        self.n_fft = n_fft

        # Target time-frames for ~window_s seconds (e.g., 2.0s / 10ms ≈ 200 frames)
        self.target_T = max(1, int(round((window_s * sr) / self.hop)))

        # MelSpectrogram with safe padding
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            hop_length=self.hop,
            n_fft=n_fft,
            center=True,
            pad_mode="constant",  # <- crucial so tiny clips don't crash
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    # light augmentations for SSL
    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        # small gain jitter
        if random.random() < 0.9:
            g_db = random.uniform(-6.0, 6.0)
            wav = wav * (10 ** (g_db / 20.0))
        # light Gaussian noise
        if random.random() < 0.9:
            wav = wav + 0.003 * torch.randn_like(wav)
        return wav

    def _ensure_len(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Ensure waveform length is at least:
        - n_fft (for STFT safety)
        - window_s seconds (so we can get target_T frames)
        """
        min_len = max(self.n_fft, int(self.window_s * self.sr))
        if wav.numel() < min_len:
            pad = min_len - wav.numel()
            wav = F.pad(wav, (0, pad))
        return wav

    def _to_logmel(self, wav: torch.Tensor, random_crop: bool) -> torch.Tensor:
        """
        Convert waveform to log-mel and pad/crop time axis to fixed target_T.
        """
        # Mel spectrogram
        M = self.mel(wav)                # [n_mels, T]
        M = self.db(M + 1e-10)
        # per-example normalization
        M = (M - M.mean()) / (M.std() + 1e-6)

        # Fix time length to target_T
        T = M.size(1)
        if T < self.target_T:
            M = F.pad(M, (0, self.target_T - T))  # right pad
        elif T > self.target_T:
            start = random.randint(0, T - self.target_T) if random_crop else max(0, (T - self.target_T) // 2)
            M = M[:, start:start + self.target_T]

        return M.unsqueeze(0)            # [1, n_mels, target_T]

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        wav, _ = load_wav_safe(path, self.sr)
        wav = self._ensure_len(wav)

        # two views with augment + random crops
        x1 = self._to_logmel(self._augment(wav), random_crop=True)
        x2 = self._to_logmel(self._augment(wav), random_crop=True)
        return x1, x2

    def __len__(self) -> int:
        return len(self.paths)


# ---------------------------
# SimCLR bits
# ---------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hid=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out),
        )
    def forward(self, z): return self.net(z)

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    Z = torch.cat([z1, z2], dim=0)            # [2N, D]
    S = torch.matmul(Z, Z.t()) / temperature  # [2N, 2N]
    mask = torch.eye(2 * N, device=Z.device, dtype=torch.bool)
    S = S.masked_fill(mask, -9e15)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(Z.device)
    return F.cross_entropy(S, targets)


# ---------------------------
# Encoder (ResNet18, 1-channel input)
# ---------------------------

class Encoder(nn.Module):
    def __init__(self, proj_out=128):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Identity()
        self.backbone = base   # 512-d features
        self.proj = ProjectionHead(512, 256, proj_out)

    def forward(self, x):
        z = self.backbone(x)
        return self.proj(z)


# ---------------------------
# Training loop
# ---------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=str, default="data/ssl_files.txt", help="path to txt list of wavs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0, help="Windows: start with 0; try 2–4 later")
    ap.add_argument("--out", type=str, default="checkpoints/ssl_encoder.pt")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--hop_ms", type=int, default=10)
    ap.add_argument("--window_s", type=float, default=2.0)
    ap.add_argument("--n_fft", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    ds = SSLDataset(
        args.list,
        sr=16000,
        n_mels=args.n_mels,
        hop_ms=args.hop_ms,
        window_s=args.window_s,
        n_fft=args.n_fft,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    model = Encoder(proj_out=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x1, x2 in dl:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent(z1, z2, temperature=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        avg = running / max(1, len(dl))
        print(f"epoch {ep}/{args.epochs}  loss={avg:.4f}")
        torch.save(model.state_dict(), args.out)  # save each epoch

    print(f"[done] saved encoder to: {args.out}")


if __name__ == "__main__":
    main()
