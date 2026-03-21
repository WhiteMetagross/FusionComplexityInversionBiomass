"""
B3: DINOv2 + ConvNeXt Zero-Shot Ensemble — 5-Fold CV (Local)

No training — uses pretrained models for inference only.
Uses torch.compile for accelerated inference.

Usage (WSL):
    conda activate mambahar
    python local_training_cv/baselines/B3_DINOv2_Giant_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import os, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch.amp import autocast
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import timm
from timm.data import resolve_model_data_config

# ── Config ─────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 17
N_FOLDS = 5
NUM_TARGETS = 5
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
COMP_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])

DINO_MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'
CONV_MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k_384'
W_DINO, W_CONV = 0.55, 0.45

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print(f'Device: {DEVICE}')

# ── Load & pivot ───────────────────────────────────────────────────
df_long = pd.read_csv(DATA_DIR / 'train.csv')
df_long['image_id'] = df_long['sample_id'].str.split('__').str[0]
df_wide = df_long.pivot_table(
    index=['image_id', 'image_path'], columns='target_name',
    values='target', aggfunc='first'
).reset_index()
for col in TARGET_NAMES:
    if col not in df_wide.columns:
        df_wide[col] = 0.0

# ── Folds ──────────────────────────────────────────────────────────
df_wide['total_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(df_wide, df_wide['total_bin'], groups=df_wide['image_id'])):
    df_wide.loc[val_idx, 'fold'] = fold
print(f'Training images: {len(df_wide)}')

TRAIN_IMG_DIR = DATA_DIR / 'train'


def weighted_r2_score(y_true, y_pred):
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    return np.sum(r2_scores * COMP_WEIGHTS) / np.sum(COMP_WEIGHTS), r2_scores


def run_model_inference(model_name, img_ids, img_dir):
    """Zero-shot inference with a compiled timm model."""
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_TARGETS)
    model = model.to(DEVICE).eval()
    model = torch.compile(model, backend='inductor')  # triton acceleration

    data_cfg = resolve_model_data_config(model)
    img_size = data_cfg['input_size'][-1]
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preds = np.zeros((len(img_ids), NUM_TARGETS), dtype=np.float32)
    with torch.no_grad():
        for idx, img_id in enumerate(tqdm(img_ids, desc=f'{model_name}')):
            img_path = img_dir / f'{img_id}.jpg'
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            img_t = val_tfm(img).unsqueeze(0).to(DEVICE)
            with autocast(device_type=DEVICE):
                out = model(img_t)
            preds[idx] = out.cpu().numpy()[0]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds


# ── Inference ──────────────────────────────────────────────────────
all_img_ids = df_wide['image_id'].values
print(f'\nRunning DINOv2 ({DINO_MODEL_NAME})...')
preds_dino = run_model_inference(DINO_MODEL_NAME, all_img_ids, TRAIN_IMG_DIR)
print(f'Running ConvNeXt ({CONV_MODEL_NAME})...')
preds_conv = run_model_inference(CONV_MODEL_NAME, all_img_ids, TRAIN_IMG_DIR)
preds_ensemble = W_DINO * preds_dino + W_CONV * preds_conv

# ── Per-fold evaluation ────────────────────────────────────────────
y_true_all = df_wide[TARGET_NAMES].values
fold_results = []

for fold in range(N_FOLDS):
    val_mask = (df_wide['fold'] == fold).values
    w_r2, per_target = weighted_r2_score(y_true_all[val_mask], preds_ensemble[val_mask])
    fold_results.append({'fold': fold, 'weighted_r2': w_r2, 'per_target_r2': per_target})
    print(f'Fold {fold}: Weighted R² = {w_r2:.6f}')
    for j, col in enumerate(TARGET_NAMES):
        print(f'  {col}: R² = {per_target[j]:.6f}')

# ── Summary ────────────────────────────────────────────────────────
w_r2_all = [r['weighted_r2'] for r in fold_results]
per_target_arr = np.array([r['per_target_r2'] for r in fold_results])
print(f'\n{"="*60}')
print(f'B3: DINOv2 + ConvNeXt Zero-Shot Ensemble — {N_FOLDS}-Fold CV')
print(f'{"="*60}')
print(f'Mean Weighted R²: {np.mean(w_r2_all):.6f} ± {np.std(w_r2_all):.6f}')
for j, col in enumerate(TARGET_NAMES):
    print(f'  {col}: {np.mean(per_target_arr[:, j]):.6f} ± {np.std(per_target_arr[:, j]):.6f}')

w_r2_dino, _ = weighted_r2_score(y_true_all, preds_dino)
w_r2_conv, _ = weighted_r2_score(y_true_all, preds_conv)
print(f'\nDINOv2 only:  Weighted R² = {w_r2_dino:.6f}')
print(f'ConvNeXt only: Weighted R² = {w_r2_conv:.6f}')
print(f'Ensemble:      Weighted R² = {np.mean(w_r2_all):.6f}')
