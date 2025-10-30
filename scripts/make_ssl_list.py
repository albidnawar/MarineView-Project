#make the unlabled list for ssl
from pathlib import Path

DATA_DIR = Path("data/raw/full_cuts_2s_flat")
OUT_FILE = Path("data/ssl_files.txt")

DATA_DIR.mkdir(parents=True, exist_ok=True)
paths = sorted(p.resolve() for p in DATA_DIR.glob("*.wav"))

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for p in paths:
        f.write(str(p) + "\n")

print(f"[done] wrote {OUT_FILE} with {len(paths)} files")
