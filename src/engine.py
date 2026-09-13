"""
Shared training engine for CSIRO Biomass cross-validation experiments.

This module provides the complete training pipeline for all experiments
in the Fusion Complexity Inversion study. It handles data loading,
augmentation, cross-validation splitting, training loops with mixed
precision, validation with optional test-time augmentation, and
checkpoint management with automatic resume support.

Used by baselines B1-B6, ablation studies E1-E8 and A1-A6, and the
proposed GatedDepthwiseConv model.

Author: Mridankan Mandal
Paper: Fusion Complexity Inversion: Why Simpler Cross-View Modules
       Outperform SSMs and Cross-View Attention Transformers for
       Pasture Biomass Regression (arXiv:2603.07819).

Usage (through WSL conda environment):
    conda activate mambahar
    python experiments/baselines/B4_DINOv2_Metadata_cv.py
"""

import os
import gc
import math
import random
import json
import warnings
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold, KFold

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Image caching — eliminates WSL cross-filesystem I/O bottleneck.
# ---------------------------------------------------------------------------

def prepare_image_cache(img_dir, img_size, df):
    """Pre-split and pre-resize all images to native Linux filesystem for fast loading.

    Reads each JPEG once from img_dir (possibly on slow /mnt/c/), splits into
    left/right halves, resizes to (img_size, img_size), and saves as .npy files
    in /tmp. Subsequent epochs and runs load small numpy arrays from ext4
    instead of decoding large JPEGs from NTFS-over-9P.

    Returns cache_dir path, or None if caching fails.
    """
    cache_dir = f"/tmp/biomass_cache_{img_size}"
    manifest_path = os.path.join(cache_dir, "_manifest.json")

    unique_images = df['image_path'].unique()
    needed = len(unique_images)

    # Fast path — cache already complete.
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get('count') == needed and manifest.get('img_size') == img_size:
                print(f"[Cache] Ready: {cache_dir} ({needed} images, {img_size}px)")
                return cache_dir
        except (json.JSONDecodeError, OSError):
            pass  # Rebuild the cache.

    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Cache] Building image cache ({needed} images → {img_size}×{img_size}) ...")

    cached = 0
    for img_rel in tqdm(unique_images, desc='Caching images'):
        img_name = os.path.basename(img_rel)
        base = os.path.splitext(img_name)[0]
        left_file = os.path.join(cache_dir, f"{base}_L.npy")
        right_file = os.path.join(cache_dir, f"{base}_R.npy")

        if os.path.exists(left_file) and os.path.exists(right_file):
            cached += 1
            continue

        full_path = os.path.join(img_dir, img_name)
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        mid = w // 2
        left = cv2.resize(img[:, :mid], (img_size, img_size), interpolation=cv2.INTER_AREA)
        right = cv2.resize(img[:, mid:], (img_size, img_size), interpolation=cv2.INTER_AREA)

        np.save(left_file, left)
        np.save(right_file, right)

    with open(manifest_path, 'w') as f:
        json.dump({'count': needed, 'img_size': img_size}, f)

    newly = needed - cached
    print(f"[Cache] Done — {cached} reused, {newly} newly cached → {cache_dir}")
    return cache_dir


def get_project_root():
    """Get the project root from this file's location."""
    return Path(__file__).resolve().parents[1]


def setup_environment(seed=17):
    """Set seeds, CUDA flags, and return device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'config'):
        torch._inductor.config.compile_threads = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def _encode_metadata(df, df_wide):
    """Encode metadata columns into numeric features for model input.

    Returns (df_wide with metadata columns, list of metadata column names).
    Features:
      - State: one-hot (4 dims)
      - Species: one-hot (15 dims)
      - Pre_GSHH_NDVI: continuous, already ~[0,1]
      - Height_Ave_cm: continuous, normalized /100
      - Sampling month: sin/cos cyclical (2 dims)
    Total is approximately 23 dimensions (exact count depends on unique categories in the data).
    """
    # Get unique metadata per image (same for all target rows).
    meta_df = df.drop_duplicates('image_id')[[
        'image_id', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm', 'Sampling_Date'
    ]].copy()

    # One-hot encode categorical features.
    state_dummies = pd.get_dummies(meta_df['State'], prefix='meta_state').astype(np.float32)
    species_dummies = pd.get_dummies(meta_df['Species'], prefix='meta_species').astype(np.float32)

    # Continuous features.
    meta_df['meta_ndvi'] = meta_df['Pre_GSHH_NDVI'].astype(np.float32)
    meta_df['meta_height'] = (meta_df['Height_Ave_cm'] / 100.0).astype(np.float32)

    # Cyclical month encoding.
    month = pd.to_datetime(meta_df['Sampling_Date']).dt.month
    meta_df['meta_month_sin'] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    meta_df['meta_month_cos'] = np.cos(2 * np.pi * month / 12).astype(np.float32)

    cont_cols = ['meta_ndvi', 'meta_height', 'meta_month_sin', 'meta_month_cos']
    meta_df = pd.concat([meta_df[['image_id'] + cont_cols], state_dummies, species_dummies], axis=1)

    meta_feature_cols = [c for c in meta_df.columns if c != 'image_id']
    df_wide = df_wide.merge(meta_df, on='image_id', how='left')
    df_wide[meta_feature_cols] = df_wide[meta_feature_cols].fillna(0.0)

    print(f"Metadata: {len(meta_feature_cols)} features encoded")
    return df_wide, meta_feature_cols


def load_train_data(cfg):
    """Load and pivot training CSV, create K-fold splits.

    Supports the following cfg.CV_STRATEGY values:
      - 'stratified_group' (default): StratifiedGroupKFold grouped by image_id
      - 'standard': plain KFold (random, no stratification or grouping).
    """
    df = pd.read_csv(cfg.TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    df_wide = df.pivot_table(
        index=['image_id', 'image_path'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    for col in cfg.TARGET_COLS:
        if col not in df_wide.columns:
            df_wide[col] = 0.0

    # Encode metadata if requested.
    if getattr(cfg, 'USE_METADATA', False):
        df_wide, meta_cols = _encode_metadata(df, df_wide)
        cfg._META_COLS = meta_cols
        cfg._META_DIM = len(meta_cols)
    else:
        cfg._META_COLS = None
        cfg._META_DIM = 0

    cv_strategy = getattr(cfg, 'CV_STRATEGY', 'stratified_group')
    df_wide['fold'] = -1

    if cv_strategy == 'standard':
        kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
        for fold, (_, val_idx) in enumerate(kf.split(df_wide)):
            df_wide.loc[val_idx, 'fold'] = fold
        print(f"CV strategy: standard KFold (no stratification)")
    else:
        df_wide['total_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=5, labels=False, duplicates='drop')
        sgkf = StratifiedGroupKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
        for fold, (_, val_idx) in enumerate(sgkf.split(df_wide, df_wide['total_bin'], groups=df_wide['image_id'])):
            df_wide.loc[val_idx, 'fold'] = fold
        print(f"CV strategy: StratifiedGroupKFold (grouped by image_id)")

    print(f"Loaded {len(df_wide)} training images")
    print(f"Fold distribution:\n{df_wide['fold'].value_counts().sort_index()}")
    return df_wide


def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class BiomassDataset(Dataset):
    """Dual-view biomass dataset that splits each image into left and right halves."""
    def __init__(self, df, img_dir, target_cols, transform=None, meta_cols=None, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.paths = df['image_path'].values
        self.labels = df[target_cols].values.astype(np.float32)
        self.meta = df[meta_cols].values.astype(np.float32) if meta_cols else None
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.paths[idx])
        base = os.path.splitext(img_name)[0]

        if self.cache_dir:
            # Fast path: pre-split, pre-resized numpy arrays on native Linux filesystem.
            left = np.load(os.path.join(self.cache_dir, f"{base}_L.npy"))
            right = np.load(os.path.join(self.cache_dir, f"{base}_R.npy"))
        else:
            # Slow path: full JPEG decode and split (used if caching is disabled).
            path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((1000, 2000, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid:]

        if self.transform:
            left = self.transform(image=left)['image']
            right = self.transform(image=right)['image']

        label = torch.from_numpy(self.labels[idx])
        if self.meta is not None:
            meta = torch.from_numpy(self.meta[idx])
            return left, right, meta, label
        return left, right, label


def biomass_loss(preds, labels, loss_fn='huber', huber_beta=5.0):
    if loss_fn == 'mse':
        return nn.MSELoss()(preds, labels)
    return nn.SmoothL1Loss(beta=huber_beta)(preds, labels)


def weighted_r2_score(y_true, y_pred):
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores


def build_optimizer(model, lr_backbone, lr_head, wd):
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=wd, fused=True)


def build_scheduler(optimizer, total_steps, warmup_epochs, epochs):
    def lr_lambda(step):
        warmup_steps = warmup_epochs * (total_steps // epochs)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    loss_fn = getattr(cfg, 'LOSS_FN', 'huber')
    huber_beta = getattr(cfg, 'HUBER_BETA', 5.0)
    accum_steps = getattr(cfg, 'GRAD_ACCUM_STEPS', 1)
    use_metadata = getattr(cfg, 'USE_METADATA', False)
    meta_dropout = getattr(cfg, 'META_DROPOUT', 0.2)

    pbar = tqdm(loader, desc='Training')
    for i, batch in enumerate(pbar):
        if use_metadata:
            left, right, meta, labels = batch
            meta = meta.to(device, non_blocking=True)
            # Random metadata dropout — trains the model to work without metadata.
            if meta_dropout > 0:
                mask = (torch.rand(meta.size(0), 1, device=device) > meta_dropout).float()
                meta = meta * mask
        else:
            left, right, labels = batch
            meta = None

        left = left.to(device, non_blocking=True, memory_format=torch.channels_last)
        right = right.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with autocast('cuda'):
            preds = model(left, right, metadata=meta)
            loss = biomass_loss(preds, labels, loss_fn, huber_beta) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})

        # Free GPU tensors immediately to avoid overlapping with the next batch.
        del preds, loss

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, use_tta=False, use_metadata=False):
    model = getattr(model, '_orig_mod', model)
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc='Validating'):
        if use_metadata:
            left, right, meta, labels = batch
            meta = meta.to(device, non_blocking=True)
        else:
            left, right, labels = batch
            meta = None

        left = left.to(device, non_blocking=True, memory_format=torch.channels_last)
        right = right.to(device, non_blocking=True, memory_format=torch.channels_last)
        with autocast('cuda'):
            preds = model(left, right, metadata=meta)
            if use_tta:
                p_hf = model(left.flip(-1), right.flip(-1), metadata=meta)
                p_vf = model(left.flip(-2), right.flip(-2), metadata=meta)
                preds = (preds + p_hf + p_vf) / 3.0
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return weighted_r2_score(all_labels, all_preds)


def _get_model_sd(model):
    """Extract state_dict, handling the torch.compile wrapper."""
    return model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()


def _load_model_sd(model, state_dict):
    """Load state_dict into a model, handling the torch.compile wrapper."""
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    target.load_state_dict(state_dict)


def _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_r2, best_epoch,
                     patience_ctr, fold, cfg):
    """Save full training checkpoint for resumption."""
    ckpt = {
        'model_state_dict': _get_model_sd(model),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'best_r2': best_r2,
        'best_epoch': best_epoch,
        'patience_ctr': patience_ctr,
    }
    ckpt_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_checkpoint.pth")
    torch.save(ckpt, ckpt_path)


def _load_checkpoint(fold, cfg):
    """Load a checkpoint if it exists. Returns dict or None."""
    ckpt_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_checkpoint.pth")
    if os.path.exists(ckpt_path):
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return None


def train_fold(fold, train_df, cfg, model_factory, device, use_compile=True):
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold}")
    print(f"{'='*60}")

    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    meta_cols = getattr(cfg, '_META_COLS', None)
    cache_dir = getattr(cfg, '_CACHE_DIR', None)
    train_ds = BiomassDataset(train_data, cfg.TRAIN_IMAGE_DIR, cfg.TARGET_COLS,
                              get_train_transforms(cfg.IMG_SIZE), meta_cols=meta_cols,
                              cache_dir=cache_dir)
    val_ds = BiomassDataset(val_data, cfg.TRAIN_IMAGE_DIR, cfg.TARGET_COLS,
                            get_val_transforms(cfg.IMG_SIZE), meta_cols=meta_cols,
                            cache_dir=cache_dir)

    nw = cfg.NUM_WORKERS
    persist = nw > 0
    prefetch = 2 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=persist, prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=persist, prefetch_factor=prefetch)

    model = model_factory(cfg).to(device)
    model = model.to(memory_format=torch.channels_last)

    # Check if the model uses Mamba SSM (has @torch.compiler.disable graph breaks).
    # torch.compile hurts these models: 80s compilation overhead and graph break penalties.
    has_mamba_ssm = any(
        hasattr(m, 'mamba') for m in model.modules()
    )
    if use_compile and hasattr(torch, 'compile') and not has_mamba_ssm:
        print("Compiling model with torch.compile (inductor/triton backend)...")
        model = torch.compile(model, backend='inductor')
    elif has_mamba_ssm:
        print("Skipping torch.compile (Mamba SSM blocks have incompatible CUDA kernels)")

    optimizer = build_optimizer(model, cfg.LR_BACKBONE, cfg.LR_HEAD, cfg.WD)
    accum_steps = getattr(cfg, 'GRAD_ACCUM_STEPS', 1)
    steps_per_epoch = -(-len(train_loader) // accum_steps)  # ceil division
    total_steps = steps_per_epoch * cfg.EPOCHS
    scheduler = build_scheduler(optimizer, total_steps, cfg.WARMUP_EPOCHS, cfg.EPOCHS)
    scaler = GradScaler('cuda')

    best_r2 = -float('inf')
    best_epoch = 0
    patience_ctr = 0
    start_epoch = 0

    # --- Resume from checkpoint if available ---
    ckpt = _load_checkpoint(fold, cfg)
    if ckpt is not None:
        _load_model_sd(model, ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_r2 = ckpt['best_r2']
        best_epoch = ckpt['best_epoch']
        patience_ctr = ckpt['patience_ctr']
        print(f"Resumed from checkpoint: epoch {start_epoch}, best R²={best_r2:.4f} "
              f"(epoch {best_epoch}), patience={patience_ctr}")
        del ckpt  # Free CPU memory holding full model and optimizer state copies.

    for epoch in range(start_epoch, cfg.EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg)

        # Clear GPU memory between training and validation phases.
        gc.collect()
        torch.cuda.empty_cache()

        # Validate every N epochs (and always on last epoch) to save time
        val_every = getattr(cfg, 'VAL_EVERY_N_EPOCHS', 1)
        is_last_epoch = (epoch + 1 == cfg.EPOCHS)
        should_validate = ((epoch + 1) % val_every == 0) or is_last_epoch

        print(f"Train Loss: {train_loss:.4f}")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            resrv = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"VRAM: {alloc:.2f}GB alloc, {resrv:.2f}GB reserved, {peak:.2f}GB peak")

        if should_validate:
            use_tta = getattr(cfg, 'USE_TTA', False)
            use_metadata = getattr(cfg, 'USE_METADATA', False)
            val_r2, per_r2 = validate(model, val_loader, device, use_tta=use_tta,
                                      use_metadata=use_metadata)

            print(f"Val R²: {val_r2:.4f}")
            print(f"Per-target: Green={per_r2[0]:.3f}, Dead={per_r2[1]:.3f}, "
                  f"Clover={per_r2[2]:.3f}, GDM={per_r2[3]:.3f}, Total={per_r2[4]:.3f}")

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_epoch = epoch + 1
                patience_ctr = 0
                save_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_best.pth")
                torch.save(_get_model_sd(model), save_path)
                print(f"Saved best model (R²={best_r2:.4f})")
            else:
                patience_ctr += 1
                print(f"No improvement for {patience_ctr} epoch(s)")
                if patience_ctr >= cfg.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    _save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                                     best_r2, best_epoch, patience_ctr, fold, cfg)
                    break

        # Save checkpoint after every epoch for resumption
        _save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                         best_r2, best_epoch, patience_ctr, fold, cfg)

        # Free CUDA cached memory between epochs to prevent fragmentation out-of-memory errors.
        # (Critical for 8 GB GPUs with Mamba fp32 kernels.)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFold {fold} Best: R²={best_r2:.4f} at epoch {best_epoch}")

    # Mark fold as fully completed.
    done_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_done.json")
    with open(done_path, 'w') as f:
        json.dump({'fold': fold, 'best_r2': float(best_r2), 'best_epoch': best_epoch}, f)

    # Delete resume checkpoint after fold completion — only best.pth is needed.
    ckpt_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_checkpoint.pth")
    if os.path.exists(ckpt_path):
        ckpt_size = os.path.getsize(ckpt_path) / 1e9
        os.remove(ckpt_path)
        print(f"Deleted resume checkpoint ({ckpt_size:.1f} GB): {ckpt_path}")

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_r2


def run_cv(cfg, model_factory, use_compile=True):
    """Run full K-fold cross-validation with automatic resume support.

    Resume logic per fold:
      1. fold{n}_done.json exists → skip training, validate best checkpoint to recover R².
      2. fold{n}_checkpoint.pth exists → resume training from last saved epoch.
      3. Neither exists → train from scratch.
    """
    device = setup_environment(cfg.SEED)
    cfg.DEVICE = device

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    for k in sorted(vars(cfg)):
        if not k.startswith('_'):
            print(f"  {k}: {getattr(cfg, k)}")

    train_df = load_train_data(cfg)

    # Build image cache on native Linux filesystem (eliminates WSL /mnt/c/ I/O bottleneck).
    cache_dir = prepare_image_cache(cfg.TRAIN_IMAGE_DIR, cfg.IMG_SIZE, train_df)
    cfg._CACHE_DIR = cache_dir

    fold_scores = []
    for fold in cfg.FOLDS_TO_TRAIN:
        done_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_done.json")
        best_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_best.pth")

        if os.path.exists(done_path) and os.path.exists(best_path):
            # Fold already completed — validate to recover the R-squared score.
            with open(done_path) as f:
                done_info = json.load(f)
            print(f"\n{'='*60}")
            print(f"FOLD {fold}: Already completed (best epoch {done_info['best_epoch']}) — validating")
            print(f"{'='*60}")

            val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
            meta_cols = getattr(cfg, '_META_COLS', None)
            val_ds = BiomassDataset(val_data, cfg.TRAIN_IMAGE_DIR, cfg.TARGET_COLS,
                                    get_val_transforms(cfg.IMG_SIZE), meta_cols=meta_cols,
                                    cache_dir=cache_dir)
            nw = cfg.NUM_WORKERS
            val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                    num_workers=nw, pin_memory=True,
                                    persistent_workers=nw > 0,
                                    prefetch_factor=3 if nw > 0 else None)

            model = model_factory(cfg).to(device)
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))

            use_tta = getattr(cfg, 'USE_TTA', False)
            use_metadata = getattr(cfg, 'USE_METADATA', False)
            val_r2, per_r2 = validate(model, val_loader, device, use_tta=use_tta,
                                      use_metadata=use_metadata)
            print(f"Fold {fold} R²: {val_r2:.4f}")
            print(f"Per-target: Green={per_r2[0]:.3f}, Dead={per_r2[1]:.3f}, "
                  f"Clover={per_r2[2]:.3f}, GDM={per_r2[3]:.3f}, Total={per_r2[4]:.3f}")
            fold_scores.append(val_r2)

            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # Train (or resume training) this fold.
            ckpt_path = os.path.join(cfg.MODEL_DIR, f"fold{fold}_checkpoint.pth")
            if os.path.exists(ckpt_path):
                print(f"\n[Resume] Found checkpoint for fold {fold}")
            score = train_fold(fold, train_df, cfg, model_factory, device, use_compile)
            fold_scores.append(score)

        print(f"Running mean CV R²: {np.mean(fold_scores):.4f}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Fold scores: {fold_scores}")
    print(f"Mean CV R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    summary = {
        'model': cfg.MODEL_NAME,
        'folds': cfg.N_FOLDS,
        'epochs': cfg.EPOCHS,
        'batch_size': cfg.BATCH_SIZE,
        'fold_scores': [float(s) for s in fold_scores],
        'mean_cv': float(np.mean(fold_scores)),
        'std_cv': float(np.std(fold_scores))
    }
    summary_path = os.path.join(cfg.OUTPUT_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return fold_scores
