import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, soundfile as sf
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from pathlib import Path
import torch.nn.functional as nnF

# robust loader (torchaudio -> soundfile)
def load_wav_safe(path, target_sr=16000):
    try:
        wav, sr = torchaudio.load(path); wav = wav.mean(0)
        if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav
    except Exception:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim == 2: x = x.mean(axis=1)
        wav = torch.from_numpy(x)
        if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav

class MelDS(Dataset):
    def __init__(self, csv_path, sr=16000, n_mels=64, hop_ms=10, window_s=2.0, train=False):
        self.df = pd.read_csv(csv_path)
        self.sr, self.train = sr, train
        hop = int(sr*hop_ms/1000)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=n_mels, hop_length=hop, n_fft=1024,
            center=True, pad_mode="constant"
        )
        self.db  = torchaudio.transforms.AmplitudeToDB()
        self.labels = sorted(self.df["species"].unique())
        self.map = {c:i for i,c in enumerate(self.labels)}
        self.target_T = int(round((window_s*sr)/hop))

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        p, s = self.df.filepath[i], self.df.species[i]
        y = self.map[s]
        wav = load_wav_safe(p, self.sr)
        need = int(2.0*self.sr)
        if wav.numel() < need: wav = nnF.pad(wav, (0, need - wav.numel()))
        M = self.db(self.mel(wav) + 1e-10)
        M = (M - M.mean())/(M.std()+1e-6)
        T = M.size(1)
        if T < self.target_T: M = nnF.pad(M, (0, self.target_T - T))
        elif T > self.target_T:
            start = max(0, (T - self.target_T)//2)
            M = M[:, start:start+self.target_T]
        return M.unsqueeze(0), torch.tensor(y)

class LinearProbe(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        base.fc = nn.Identity()
        self.encoder = base
        self.head = nn.Linear(512, num_classes)
    def forward(self, x):
        with torch.no_grad():      # freeze encoder
            z = self.encoder(x)
        return self.head(z)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/labels/allcuts_train.csv")
    ap.add_argument("--val_csv",   default="data/labels/allcuts_val.csv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch",  type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_tr, ds_va = MelDS(args.train_csv, train=True), MelDS(args.val_csv, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    num_classes = len(ds_tr.labels)
    model = LinearProbe(num_classes).to(device)

    # load SSL encoder weights
    ssl = torch.load("checkpoints/ssl_encoder.pt", map_location="cpu")
    backbone_state = {k.replace("backbone.",""): v for k,v in ssl.items() if k.startswith("backbone.")}
    model.encoder.load_state_dict(backbone_state, strict=False)

    opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)  # only head trains
    best = 0.0
    Path("checkpoints").mkdir(exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        for M,y in dl_tr:
            M,y = M.to(device), y.to(device)
            loss = F.cross_entropy(model(M), y)
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval(); correct=seen=0
        with torch.no_grad():
            for M,y in dl_va:
                M,y = M.to(device), y.to(device)
                pred = model(M).argmax(1)
                correct += (pred==y).sum().item(); seen += y.size(0)
        acc = correct/seen
        print(f"epoch {ep}: val_acc={acc:.3f}")
        if acc>best:
            best=acc
            torch.save(model.state_dict(), "checkpoints/linear_probe.pt")
    print("best val acc:", best)

if __name__ == "__main__":
    main()
