"""
B2: EfficientNet-B3 — 5-Fold Cross-Validation (Local)

Single-image EfficientNet-B3 backbone with weighted MSE loss.
Uses torch.compile for acceleration.

Usage (WSL):
    conda activate mambahar
    python local_training_cv/baselines/B2_EfficientNet_B3_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import os, gc, random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ── Paths & Config ─────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'
OUTPUT_DIR = PROJ_ROOT / 'local_training_cv' / 'output' / 'B2_EfficientNet_B3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 17
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 384
LR = 1e-4
NUM_WORKERS = 4
N_FOLDS = 5
EARLY_STOPPING_PATIENCE = 10

TARGET_NAMES = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
COMP_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load & Pivot ───────────────────────────────────────────────────
train_long = pd.read_csv(DATA_DIR / 'train.csv')
train_long['image_id'] = train_long['sample_id'].str.split('__').str[0]

pivot = train_long.pivot_table(
    index=['image_id', 'image_path'],
    columns='target_name',
    values='target',
    aggfunc='first'
).reset_index()

for col in TARGET_NAMES:
    if col not in pivot.columns:
        pivot[col] = 0.0

def make_full_path(p):
    if os.path.isabs(p):
        return p
    return str(DATA_DIR / p)

pivot["image_full_path"] = pivot["image_path"].apply(make_full_path)
exists_mask = pivot["image_full_path"].apply(os.path.exists)
if not exists_mask.all():
    print(f"Warning: {(~exists_mask).sum()} missing files dropped.")
    pivot = pivot.loc[exists_mask].reset_index(drop=True)

# ── Folds ──────────────────────────────────────────────────────────
pivot['total_bin'] = pd.qcut(pivot['Dry_Total_g'], q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pivot['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(pivot, pivot['total_bin'], groups=pivot['image_id'])):
    pivot.loc[val_idx, 'fold'] = fold

print(f"Training images: {len(pivot)}")
print(f"Fold distribution:\n{pivot['fold'].value_counts().sort_index()}")

# ── Dataset & Transforms ──────────────────────────────────────────
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class BiomassImageDataset(Dataset):
    def __init__(self, df, target_cols=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_cols = target_cols or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_full_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        targets = row[self.target_cols].values.astype(np.float32)
        return img, torch.tensor(targets, dtype=torch.float32)

# ── Model ──────────────────────────────────────────────────────────
class BiomassModel(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_targets)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ── Loss & Metrics ─────────────────────────────────────────────────
mse_loss = nn.MSELoss(reduction="none")
weights_tensor = torch.tensor(COMP_WEIGHTS, dtype=torch.float32).to(DEVICE)

def weighted_mse_loss(preds, targets):
    loss_per_elem = mse_loss(preds, targets)
    return (loss_per_elem.mean(dim=0) * weights_tensor).sum()

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

# ── Train / Validate ───────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    running_loss, n = 0.0, 0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model(imgs)
            loss = weighted_mse_loss(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return running_loss / n

def validate(model, loader):
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            out = model(imgs)
            preds_list.append(out.cpu().numpy())
            trues_list.append(targets.numpy())
    return weighted_r2_score(np.vstack(trues_list), np.vstack(preds_list))

# ── 5-Fold CV ──────────────────────────────────────────────────────
fold_results = []

for fold in range(N_FOLDS):
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
    train_fold = pivot[pivot['fold'] != fold].reset_index(drop=True)
    val_fold = pivot[pivot['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_fold)}, Val: {len(val_fold)}")

    train_ds = BiomassImageDataset(train_fold, TARGET_NAMES, train_tfm)
    val_ds = BiomassImageDataset(val_fold, TARGET_NAMES, val_tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = BiomassModel().to(DEVICE)
    model = torch.compile(model, backend='inductor')  # triton acceleration
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    best_w_r2, patience_ctr = -1e9, 0
    save_path = str(OUTPUT_DIR / f"best_model_fold{fold}.pth")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        w_r2, per_r2 = validate(model, val_loader)

        improved = ""
        if w_r2 > best_w_r2:
            best_w_r2 = w_r2
            sd = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save(sd, save_path)
            patience_ctr = 0
            improved = " *"
        else:
            patience_ctr += 1

        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.6f} | W-R²: {w_r2:.6f}{improved}")
        if patience_ctr >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Evaluate best checkpoint
    raw_model = BiomassModel().to(DEVICE)
    raw_model.load_state_dict(torch.load(save_path, weights_only=True))
    w_r2, per_r2 = validate(raw_model, val_loader)
    fold_results.append({'fold': fold, 'weighted_r2': w_r2, 'per_target_r2': per_r2})
    print(f"\n  Fold {fold} Best Weighted R²: {w_r2:.6f}")
    for j, col in enumerate(TARGET_NAMES):
        print(f"    {col}: R² = {per_r2[j]:.6f}")

    del model, raw_model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

# ── Summary ────────────────────────────────────────────────────────
w_r2_all = [r['weighted_r2'] for r in fold_results]
per_target_all = np.array([r['per_target_r2'] for r in fold_results])

print(f"\n{'='*60}")
print(f"B2: EfficientNet-B3 — {N_FOLDS}-Fold CV Results")
print(f"{'='*60}")
print(f"Mean Weighted R²: {np.mean(w_r2_all):.6f} ± {np.std(w_r2_all):.6f}")
for j, col in enumerate(TARGET_NAMES):
    print(f"  {col}: {np.mean(per_target_all[:, j]):.6f} ± {np.std(per_target_all[:, j]):.6f}")
