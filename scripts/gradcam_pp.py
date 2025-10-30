# scripts/gradcam_pp.py
import argparse, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, soundfile as sf
import numpy as np, matplotlib.pyplot as plt
from torchvision.models import resnet18
from pathlib import Path
import difflib

# ---------- Audio -> log-Mel (matches training) ----------
SR=16000
N_MELS=64
HOP_MS=10          # 10 ms hop
WIN_MS=25          # 25 ms window
N_FFT=1024         # FFT size (>= win_length)
WIN_S=2.0          # 2 seconds target window

def load_wav(path, sr=SR):
    try:
        x, s = torchaudio.load(path); x = x.mean(0)
    except:
        y, s = sf.read(path, dtype="float32", always_2d=False)
        x = torch.from_numpy(y if y.ndim==1 else y.mean(1))
    if s != sr:
        x = torchaudio.functional.resample(x, s, sr)
    return x

def to_mel(wav):
    hop = int(SR * HOP_MS / 1000)      # 160
    win = int(SR * WIN_MS / 1000)      # 400
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,                    # 1024
        win_length=win,                 # 400
        hop_length=hop,                 # 160
        n_mels=N_MELS,                  # 64
        center=True,
        pad_mode="constant",
        power=2.0
    )
    db  = torchaudio.transforms.AmplitudeToDB()
    need = int(WIN_S*SR)               # 2*16000 = 32000
    tgt  = int(round(need / hop))      # ~200 frames

    if wav.numel() < need:
        wav = F.pad(wav, (0, need - wav.numel()))
    M = db(mel(wav) + 1e-10)           # [n_mels, T]
    M = (M - M.mean())/(M.std()+1e-6)

    T = M.size(1)
    if T < tgt: M = F.pad(M, (0, tgt - T))
    elif T > tgt:
        st = max(0, (T - tgt)//2)
        M = M[:, st:st+tgt]
    return M.unsqueeze(0)              # [1, n_mels, T]

# ---------- Model (same as training) ----------
class SupModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        b = resnet18(weights=None)
        b.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        b.fc = nn.Identity()
        self.encoder = b
        self.head = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.head(self.encoder(x))

# ---------- Grad-CAM++ ----------
class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.fmap = None; self.grad = None
        def f_hook(m, i, o): self.fmap = o.detach()          # [B,C,H,W]
        def b_hook(m, gi, go): self.grad = go[0].detach()     # [B,C,H,W]
        target_layer.register_forward_hook(f_hook)
        target_layer.register_full_backward_hook(b_hook)

    def __call__(self, logits, class_idx):
        self.model.zero_grad(set_to_none=True)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        A   = self.fmap[0]         # [C,H,W]
        dY  = self.grad[0]         # [C,H,W]
        eps = 1e-7

        d2 = dY**2
        d3 = d2 * dY
        Apos = torch.relu(A)
        sumA = Apos.view(A.size(0), -1).sum(dim=1) + eps

        alpha = d2 / (2*d2 + (sumA.view(-1,1,1))*d3 + eps)
        weights = (alpha * torch.relu(dY)).view(A.size(0), -1).sum(dim=1)

        cam = torch.relu((weights.view(-1,1,1) * A).sum(dim=0))
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap(mel_img, cam, out_png, title=""):
    M = mel_img.cpu().numpy()
    plt.figure(figsize=(7,3))
    plt.imshow(M, aspect="auto", origin="lower", cmap="magma")
    plt.imshow(cam, cmap="jet", alpha=0.35,
               extent=[0, M.shape[1], 0, M.shape[0]],
               origin="lower", aspect="auto")
    plt.title(title); plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160); plt.close()

def normalize_name(s: str) -> str:
    """Normalize species name for matching."""
    return " ".join(s.replace("_", " ").replace("-", " ").split()).strip().casefold()

def fuzzy_match_class(class_name: str, labels: list) -> int:
    """Find class index with fuzzy matching."""
    normalized_input = normalize_name(class_name)
    normalized_labels = [normalize_name(label) for label in labels]
    
    # Exact match first
    if normalized_input in normalized_labels:
        return normalized_labels.index(normalized_input)
    
    # Fuzzy match
    matches = difflib.get_close_matches(normalized_input, normalized_labels, n=1, cutoff=0.75)
    if matches:
        matched_label = matches[0]
        idx = normalized_labels.index(matched_label)
        print(f"[info] Fuzzy-matched '{class_name}' → '{labels[idx]}'")
        return idx
    
    return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_csv", required=True)   # to recover label order
    ap.add_argument("--audio", required=True)       # one wav path
    ap.add_argument("--class_name", default="")     # optional override
    args = ap.parse_args()

    import pandas as pd
    labels = sorted(pd.read_csv(args.train_csv)["species"].unique())
    num_classes = len(labels)

    model = SupModel(num_classes)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()

    # target layer = last conv
    target_layer = model.encoder.layer4[-1].conv2
    campp = GradCAMPP(model, target_layer)

    wav = load_wav(args.audio)
    M = to_mel(wav)                  # [1,n_mels,T]
    x = M.unsqueeze(0)               # [B,1,n_mels,T]
    
    # Ensure input requires gradients for Grad-CAM++
    x = x.requires_grad_(True)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(1).item())

    if args.class_name:
        # Use fuzzy matching for class names
        normalized_input = " ".join(args.class_name.replace("_", " ").replace("-", " ").split()).strip().casefold()
        normalized_labels = [" ".join(label.replace("_", " ").replace("-", " ").split()).strip().casefold() for label in labels]
        
        if normalized_input in normalized_labels:
            target_idx = normalized_labels.index(normalized_input)
        else:
            # Fuzzy match
            import difflib
            matches = difflib.get_close_matches(normalized_input, normalized_labels, n=1, cutoff=0.75)
            if matches:
                matched_label = matches[0]
                target_idx = normalized_labels.index(matched_label)
                print(f"[info] Fuzzy-matched '{args.class_name}' → '{labels[target_idx]}'")
            else:
                raise SystemExit(f"class_name '{args.class_name}' not found in training labels")
    else:
        target_idx = pred_idx

    # Forward pass with gradients enabled
    logits = model(x)
    cam = campp(logits, target_idx)
    title = f"pred={labels[pred_idx]}  target={labels[target_idx]}"
    out = f"reports/gradcampp_{Path(args.audio).stem}.png"
    overlay_heatmap(M.squeeze(0), cam, out, title=title)
    print("saved:", out)

if __name__ == "__main__":
    main()
