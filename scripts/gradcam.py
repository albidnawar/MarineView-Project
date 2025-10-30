# scripts/gradcam.py
import argparse, glob, difflib
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, soundfile as sf
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# ----- constants (match your training) -----
SR = 16000
N_MELS = 64
HOP_MS = 10
N_FFT = 1024
WINDOW_S = 2.0
# -------------------------------------------

def expand_audio(pattern: str):
    """Accept single file, folder, or glob; return list of real files."""
    p = Path(pattern)
    if p.is_file():
        return [str(p)]
    files = glob.glob(pattern, recursive=True)
    if p.exists() and p.is_dir() and not files:
        files = glob.glob(str(p / "*.wav"))
    keep = {".wav", ".flac", ".aif", ".aiff"}
    return sorted(f for f in files if Path(f).suffix.lower() in keep)

def load_wav(path, sr=SR):
    """Robust audio loader: torchaudio → soundfile, mono, resampled."""
    try:
        x, s = torchaudio.load(path)   # [C,T]
        x = x.mean(0)
        if s != sr:
            x = torchaudio.functional.resample(x, s, sr)
        return x
    except Exception:
        y, s = sf.read(path, dtype="float32", always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        x = torch.from_numpy(y if isinstance(y, np.ndarray) else np.array(y, dtype=np.float32))
        if s != sr:
            x = torchaudio.functional.resample(x, s, sr)
        return x

def mel_and_norm(wav):
    """Waveform → log-Mel (normalized) → fixed time length."""
    hop = int(SR * HOP_MS / 1000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_mels=N_MELS, hop_length=hop, n_fft=N_FFT,
        center=True, pad_mode="constant")
    db = torchaudio.transforms.AmplitudeToDB()
    need = int(WINDOW_S * SR)
    tgt = int(round((WINDOW_S * SR) / hop))

    if wav.numel() < need:
        wav = F.pad(wav, (0, need - wav.numel()))
    M = db(mel(wav) + 1e-10)           # [n_mels, T]
    M = (M - M.mean()) / (M.std() + 1e-6)

    T = M.size(1)
    if T < tgt:
        M = F.pad(M, (0, tgt - T))
    elif T > tgt:
        st = max(0, (T - tgt) // 2)
        M = M[:, st:st + tgt]
    return M

class Net(nn.Module):
    """ResNet18 backbone (1-ch input) + linear head."""
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(512, num_classes)
    def forward(self, x):  # x: [B,1,n_mels,T]
        z = self.backbone(x)        # [B,512]
        return self.head(z)         # [B,num_classes]

def normalize_name(s: str) -> str:
    return " ".join(s.replace("_", " ").replace("-", " ").split()).strip().casefold()

def fuzzy_to_idx(name: str, name_to_idx: dict[str,int]) -> int | None:
    """Map possibly-typo'd name to class idx via exact or fuzzy matching."""
    key = normalize_name(name)
    if key in name_to_idx:
        return name_to_idx[key]
    cand = difflib.get_close_matches(key, list(name_to_idx.keys()), n=1, cutoff=0.75)
    if cand:
        print(f"[info] fuzzy-matched '{name}' → '{cand[0]}'")
        return name_to_idx[cand[0]]
    return None

def plot_gradcam_overlay(mel_img, cam_t, out_png, title="Grad-CAM"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(mel_img, aspect='auto', origin='lower')
    ax.imshow(cam_t[None, :], aspect='auto', origin='lower', alpha=0.35)
    ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("mel bins")
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def run_one(audio_path, model, labels, target_idx=None, true_label=None):
    """Compute Grad-CAM for one file. If target_idx is None, use top-1 prediction."""
    wav = load_wav(audio_path)
    M = mel_and_norm(wav)                 # [n_mels, T]
    x = M.unsqueeze(0).unsqueeze(0)       # [1,1,n_mels,T]
    x = x.requires_grad_(True)

    # capture last conv block features via forward hook
    feats_holder = {}
    def fwd_hook(_m, _i, o):
        feats_holder["feats"] = o        # [B,C,H,W]
    h = model.backbone.layer4.register_forward_hook(fwd_hook)

    # forward
    logits = model(x)                     # [1,C]
    probs = logits.softmax(dim=1)[0]
    pred_idx = int(probs.argmax().item())
    pred_prob = float(probs[pred_idx].item())
    pred_name = labels[pred_idx]

    # pick target (class to explain)
    if target_idx is None:
        target_idx = pred_idx
    if target_idx < 0 or target_idx >= len(labels):
        h.remove(); raise RuntimeError(f"target_idx out of range: {target_idx}")

    # grab feature map
    if "feats" not in feats_holder:
        h.remove(); raise RuntimeError("Grad-CAM: layer4 forward hook did not fire.")
    feats = feats_holder["feats"]        # [1,C,H,W]
    if not feats.requires_grad:
        feats.requires_grad_(True)

    # gradient of target score wrt feature map
    score = logits[:, target_idx]        # [1]
    grads = torch.autograd.grad(outputs=score, inputs=feats,
                                grad_outputs=torch.ones_like(score),
                                retain_graph=False, create_graph=False, allow_unused=False)[0]
    h.remove()

    A = feats.detach()[0]                # [C,H,W]
    G = grads.detach()[0]                # [C,H,W]
    w = G.mean(dim=(1, 2))               # [C]
    cam = (w[:, None, None] * A).sum(0)  # [H,W]
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam_t = cam.mean(0).cpu().numpy()
    mel_np = M.numpy()

    title = f"Grad-CAM (pred={pred_name} {pred_prob:.2f}, target={labels[target_idx]})"
    if true_label is not None:
        title = f"Grad-CAM (pred={pred_name} {pred_prob:.2f}, true={true_label}, target={labels[target_idx]})"

    out = Path("reports") / f"gradcam_{Path(audio_path).stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_gradcam_overlay(mel_np, cam_t, out, title=title)

    return pred_name, pred_prob, labels[target_idx], out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model (e.g., checkpoints\\best_model.pt)")
    ap.add_argument("--train_csv", required=True, help="data\\labels\\allcuts_train.csv")
    ap.add_argument("--audio", required=True, help="file path or glob, e.g., demo\\*.wav")
    ap.add_argument("--true_from", choices=["none","parent","csv"], default="none",
                    help="derive ground truth from parent folder or a CSV")
    ap.add_argument("--true_csv", default="", help="CSV with [filepath,species] if --true_from csv")
    ap.add_argument("--target", choices=["pred","true"], default="pred",
                    help="class to explain: model prediction or the true class")
    ap.add_argument("--class_name", default="",
                    help="force Grad-CAM to this class (name must match training labels). Overrides --target.")
    args = ap.parse_args()

    # labels from TRAIN (shared mapping)
    labels = sorted(pd.read_csv(args.train_csv)["species"].unique())
    name_to_idx = {normalize_name(n): i for i, n in enumerate(labels)}

    # optional truth lookup CSV
    df_true = None
    if args.true_from == "csv" and args.true_csv:
        df_true = pd.read_csv(args.true_csv)
        df_true["filepath_norm"] = df_true["filepath"].apply(lambda p: str(Path(p).resolve()).casefold())

    # model
    model = Net(num_classes=len(labels))
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    files = expand_audio(args.audio)
    if not files:
        print(f"[error] No audio matched: {args.audio}")
        return

    # optional forced class idx
    forced_idx = None
    if args.class_name:
        forced_idx = fuzzy_to_idx(args.class_name, name_to_idx)
        if forced_idx is None:
            print(f"[warn] class_name '{args.class_name}' not found in training labels; ignoring.")

    print(f"[info] Running Grad-CAM on {len(files)} file(s)...")
    for i, f in enumerate(files, 1):
        # derive true label (optional)
        true_label = None
        if args.true_from == "parent":
            true_label = Path(f).parent.name.replace("_", " ").replace("-", " ")
        elif args.true_from == "csv" and df_true is not None:
            f_norm = str(Path(f).resolve()).casefold()
            rows = df_true[df_true["filepath_norm"] == f_norm]
            if len(rows) > 0:
                true_label = rows.iloc[0]["species"]

        # choose target index
        target_idx = None
        if forced_idx is not None:
            target_idx = forced_idx
        elif args.target == "true" and true_label is not None:
            target_idx = fuzzy_to_idx(true_label, name_to_idx)
            if target_idx is None:
                print(f"[warn] True label '{true_label}' not in training labels; falling back to prediction.")
                target_idx = None  # use prediction

        try:
            pred, prob, target, out = run_one(f, model, labels, target_idx=target_idx, true_label=true_label)
            if true_label:
                print(f"[{i}/{len(files)}] {f} → pred: {pred} ({prob:.2f}) | true: {true_label} | target: {target} | saved: {out}")
            else:
                print(f"[{i}/{len(files)}] {f} → pred: {pred} ({prob:.2f}) | target: {target} | saved: {out}")
        except Exception as e:
            print(f"[skip] {f}: {e}")

if __name__ == "__main__":
    main()
