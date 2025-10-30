# Marine SSL CNN

Self-supervised and supervised CNN pipeline for marine mammal audio classification using log-Mel spectrograms (ResNet18 backbone). Includes dataset preparation (segmentation + VAD), labeled splits, SSL training utilities, finetuning, evaluation, prediction, Grad-CAM, Grad-CAM++, and LIME explanations.

## Quick start (Windows PowerShell)

```powershell
# 1) Clone repo and cd
cd D:\marine_ssl_cnn

# 2) Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Upgrade pip and install core deps
python -m pip install --upgrade pip
# Torch/Torchaudio CPU (or select CUDA build you need)
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# Vision utils
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu124
# Others commonly used
python -m pip install pandas matplotlib scikit-learn soundfile
```

If you have a `requirement.text` (typo) file, please rename it to `requirements.txt` and install with:
```powershell
python -m pip install -r requirements.txt
```

## Data layout

Raw WAVs are expected under `data/raw/` in species subfolders. Example (truncated):
```
marine_ssl_cnn/
  data/
    raw/
      DuskyDolphin/1964/64124001.wav
      SpermWhale/1966/...
```

### 2-second segmentation with simple VAD
Writes 2s segments preserving hierarchical subfolders into `data/raw/full_cuts_2s`.
```powershell
# args: --src --dst --sr --win_s --top_db --overlap --backend
python scripts/segment_and_vad.py --src data/raw --dst data/raw/full_cuts_2s --sr 16000 --win_s 2.0 --top_db 35 --overlap 0.5 --backend soundfile
```

## Labeled splits

Create stratified train/val/test CSVs under `data/labels/`. Singletons are kept in train, and splitting is done with an internal stratified routine (no sklearn dependency required).
```powershell
python scripts\make_labeled_splits.py
# Outputs:
#   data/labels/allcuts_train.csv
#   data/labels/allcuts_val.csv
#   data/labels/allcuts_test.csv
```

## Training

There are multiple training scripts; a typical flow is:
- Self-supervised (optional): `scripts/train_ssl.py` + `scripts/make_ssl_list.py`
- Supervised or finetuning: `scripts/finetune_full.py` (and related variants)

Example (finetune, arguments may vary inside the script):
```powershell
python scripts\finetune_full.py --train_csv data\labels\allcuts_train.csv --val_csv data\labels\allcuts_val.csv --out_ckpt checkpoints\best_model.pt
```

Checkpoints are stored in `checkpoints/`.

## Evaluation and reports

Classification report and confusion matrix:
```powershell
python scripts\eval_report.py --ckpt checkpoints\best_model.pt --train_csv data\labels\allcuts_train.csv --test_csv data\labels\allcuts_test.csv
# Saves reports/<ckpt_stem>_report.txt and reports/<ckpt_stem>_cm.png
```

## Inference (single files or folders)
```powershell
python scripts\predict.py --ckpt checkpoints\best_model.pt --train_csv data\labels\allcuts_train.csv --audio "D:\clips\**\*.wav" --topk 3 --out predictions.csv
```

## Visual explanations

### Grad-CAM (time overlay)
```powershell
python scripts\gradcam.py --ckpt checkpoints\best_model.pt --train_csv data\labels\allcuts_train.csv --audio data\raw\DuskyDolphin\1964\64124001.wav
# Output: reports/gradcam_<stem>.png
```

### Grad-CAM++ (variant)
```powershell
python scripts\gradcam_pp.py --ckpt checkpoints\best_model.pt --train_csv data\labels\allcuts_train.csv --audio data\raw\DuskyDolphin\1964\64124001.wav --class_name "Dusky Dolphin"
# Output: reports/gradcampp_<stem>.png
```
Notes:
- `--class_name` supports fuzzy matching (spaces/underscores/hyphens handled).
- Some versions require inputs to require gradients; this is handled by the script.

### LIME
Requires `lime` and `scikit-image`.
```powershell
python -m pip install lime scikit-image matplotlib
python scripts\lime_explain.py --ckpt checkpoints\best_model.pt --train_csv data\labels\allcuts_train.csv --audio data\raw\DuskyDolphin\1964\64124001.wav
# Output: reports/lime_<stem>.png
```

## Common issues

- Torch/Torchaudio backend errors loading WAV:
  - Install `numpy soundfile` and prefer soundfile backend in scripts when available.
- CUDA not visible:
  - Ensure correct CUDA build of torch/torchvision and that your GPU driver supports it.
  - Quick check: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- Label mismatches between checkpoints and CSVs:
  - Ensure the training label list (sorted unique species from train CSV) matches the checkpoint’s class order. Visualization scripts include fuzzy matching and options to override class targets.

## Project structure (key files)
```
checkpoints/            # model checkpoints
reports/                # generated reports & figures
scripts/
  segment_and_vad.py    # 2s segmentation + simple VAD
  make_labeled_splits.py# create train/val/test CSVs
  train_ssl.py          # self-supervised training (optional)
  finetune_full.py      # supervised finetuning
  eval_report.py        # classification report + confusion matrix
  predict.py            # batch predictions
  gradcam.py            # Grad-CAM visualization
  gradcam_pp.py         # Grad-CAM++ visualization
  lime_explain.py       # LIME explanation (image-based)
```

## Tips
- Use PowerShell `.\.venv\Scripts\Activate.ps1` to ensure you’re using the correct environment.
- Large installs (torch/torchvision) can be configured to use a D:\ pip cache via pip.ini if space is limited on C:\.
- For long jobs, prefer running scripts without interactive prompts and write outputs to `reports/` or `outputs/`.

## License
Provide your project’s license here (e.g., MIT).


