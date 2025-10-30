
#script for spliting audios in 2sec
# scripts/segment_and_vad.py

import os, glob
from pathlib import Path

import torch
import torchaudio

# -------- SETTINGS (works with many subfolders) --------
SRC_ROOT = "data/raw"               # <— scan EVERYTHING under data/raw
DST_ROOT = "data/raw/full_cuts_2s"  # output root (mirrors subfolders)
SR       = 16000
WIN_S    = 2.0
OVERLAP  = 0.5        # 0.5 => 50% overlap; use 0.0 for non-overlap
TOP_DB   = 35.0       # energy gate (lower to keep more)
EXTS     = ("*.wav","*.WAV","*.flac","*.FLAC","*.aif","*.AIF","*.aiff","*.AIFF")
# -------------------------------------------------------

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def loud_enough(x: torch.Tensor, top_db: float = TOP_DB) -> bool:
    db = 20 * torch.log10(x.abs().mean() + 1e-9)
    return db > -top_db

def save_wav(dst_path: Path, wav: torch.Tensor, sr: int = SR):
    ensure_dir(dst_path)
    torchaudio.save(str(dst_path), wav.unsqueeze(0), sr)

def list_audio_files(root: str):
    files = []
    for pat in EXTS:
        files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    return sorted(files)

def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    win = int(SR * WIN_S)
    hop = max(1, int(win * (1.0 - OVERLAP)))

    wav_paths = list_audio_files(SRC_ROOT)
    print(f"[info] Searching recursively in: {Path(SRC_ROOT).resolve()}")
    print(f"[info] Found {len(wav_paths)} audio files (extensions={EXTS})")
    if not wav_paths:
        print("[hint] Put your audio anywhere under data/raw/, or change SRC_ROOT.")
        return
    print("[info] First few files:")
    for p in wav_paths[:5]:
        print("   ", p)

    kept_short = kept_long = skipped_quiet = forced_keep = 0

    for idx, src_path in enumerate(wav_paths, 1):
        try:
            wav, sr = torchaudio.load(src_path)   # [C,T]
            wav = wav.mean(0)                      # mono
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)

            # Mirror full relative path under DST_ROOT
            rel = Path(src_path).relative_to(SRC_ROOT)     # e.g., sub1/sub2/file.wav
            dst_base = (Path(DST_ROOT) / rel).with_suffix(".wav")

            T = wav.numel()

            # Case A: short (<2s) — keep as-is if loud, otherwise keep with "_quiet" tag
            if T < win:
                if loud_enough(wav):
                    save_wav(dst_base, wav)
                    kept_short += 1
                else:
                    save_wav(dst_base.with_stem(dst_base.stem + "_quiet"), wav)
                    forced_keep += 1
            else:
                # Case B: long (>=2s) — split into windows
                seg_idx = 0
                any_saved = False
                for i in range(0, max(0, T - win + 1), hop):
                    seg = wav[i:i + win]
                    if seg.numel() < win:
                        break
                    if loud_enough(seg):
                        out = dst_base.with_stem(dst_base.stem + f"_{seg_idx:04d}")
                        save_wav(out, seg)
                        seg_idx += 1
                        any_saved = True
                    else:
                        skipped_quiet += 1
                if not any_saved:
                    # keep first window anyway so every file yields something
                    out = dst_base.with_stem(dst_base.stem + "_0000_forced")
                    save_wav(out, wav[:win])
                    forced_keep += 1
                kept_long += seg_idx

            if idx % 100 == 0:
                print(f"[prog] processed={idx}  short_kept={kept_short}  long_kept={kept_long} "
                      f"quiet_skipped={skipped_quiet}  forced_keep={forced_keep}")

        except Exception as e:
            print(f"[skip] {src_path}: {e}")

    print(
        "\n[done]\n"
        f"- Short clips kept:           {kept_short}\n"
        f"- Long segments kept:         {kept_long}\n"
        f"- Segments skipped (quiet):   {skipped_quiet}\n"
        f"- Forced keeps (quiet/all):   {forced_keep}\n"
        f"Output in: {Path(DST_ROOT).resolve()}"
    )

if __name__ == "__main__":
    main()
