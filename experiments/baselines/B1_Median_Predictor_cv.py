"""
B1: Median Predictor Baseline — 5-Fold Cross-Validation (Local)

CPU-only statistical baseline. Computes median of each target from
the training fold and uses it as prediction for all validation samples.

Usage (WSL):
    conda activate mambahar
    python experiments/baselines/B1_Median_Predictor_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ── Paths ──────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'

SEED = 17
N_FOLDS = 5
TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
COMP_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])

random.seed(SEED)
np.random.seed(SEED)
print(f'Seed set to {SEED}')

# ── Load & pivot ───────────────────────────────────────────────────
df_long = pd.read_csv(DATA_DIR / 'train.csv')
df_long['image_id'] = df_long['sample_id'].str.split('__').str[0]

df_wide = df_long.pivot_table(
    index=['image_id', 'image_path'],
    columns='target_name',
    values='target',
    aggfunc='first'
).reset_index()

for col in TARGET_COLS:
    if col not in df_wide.columns:
        df_wide[col] = 0.0

print(f'Training images: {len(df_wide)}')
print(f'Target columns: {TARGET_COLS}')
print(df_wide.head())

# ── Folds ──────────────────────────────────────────────────────────
df_wide['total_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(df_wide, df_wide['total_bin'], groups=df_wide['image_id'])):
    df_wide.loc[val_idx, 'fold'] = fold
print(f'Fold distribution:\n{df_wide["fold"].value_counts().sort_index()}')


def weighted_r2_score(y_true, y_pred):
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * COMP_WEIGHTS) / np.sum(COMP_WEIGHTS)
    return weighted, r2_scores


# ── 5-Fold CV: Median Predictor ───────────────────────────────────
fold_results = []
for fold in range(N_FOLDS):
    train_fold = df_wide[df_wide['fold'] != fold]
    val_fold = df_wide[df_wide['fold'] == fold]
    medians = train_fold[TARGET_COLS].median().values
    y_true = val_fold[TARGET_COLS].values
    y_pred = np.tile(medians, (len(val_fold), 1))
    w_r2, per_target = weighted_r2_score(y_true, y_pred)
    fold_results.append({'fold': fold, 'weighted_r2': w_r2, 'per_target_r2': per_target})
    print(f'Fold {fold}: Weighted R² = {w_r2:.6f}')
    for j, col in enumerate(TARGET_COLS):
        print(f'  {col}: R² = {per_target[j]:.6f}')

# ── Summary ────────────────────────────────────────────────────────
w_r2_all = [r['weighted_r2'] for r in fold_results]
per_target_all = np.array([r['per_target_r2'] for r in fold_results])

print(f'\n{"="*60}')
print(f'B1: Median Predictor — {N_FOLDS}-Fold CV Results')
print(f'{"="*60}')
print(f'Mean Weighted R²: {np.mean(w_r2_all):.6f} ± {np.std(w_r2_all):.6f}')
for j, col in enumerate(TARGET_COLS):
    print(f'  {col}: {np.mean(per_target_all[:, j]):.6f} ± {np.std(per_target_all[:, j]):.6f}')
