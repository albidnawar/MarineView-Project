#make the audios in one folder named full_cuts_2s_flat

import shutil
from pathlib import Path

SRC = Path("data/raw/full_cuts_2s")
DST = Path("data/raw/full_cuts_2s_flat")
DST.mkdir(parents=True, exist_ok=True)

count = 0
for wav in list(SRC.rglob("*.wav")) + list(SRC.rglob("*.WAV")):
    # build a unique name from relative path (subfolder_file.wav)
    rel = wav.relative_to(SRC)
    flat_name = "_".join(rel.parts).replace(" ", "_")
    out = DST / flat_name

    # if collision (same name twice), add index
    if out.exists():
        stem = out.stem
        i = 1
        while (DST / f"{stem}_{i}{out.suffix}").exists():
            i += 1
        out = DST / f"{stem}_{i}{out.suffix}"

    shutil.copy2(wav, out)
    count += 1

print(f"[done] Copied {count} wavs to {DST}")
