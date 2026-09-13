# Usage Instructions:

**Author:** Mridankan Mandal

This document outlines how to execute experiments both on Kaggle and locally through WSL.

---

## A. Kaggle Execution:

### 1. DINOv3-ViT-L Model:

Kaggle can interact with the Hugging Face hub as long as the **Internet** toggle is enabled in the kernel settings. The B5 notebook integrates the image encoder using the Hugging Face ID `facebook/dinov3-vitl16-pretrain-lvd1689m`.

If you prefer offline execution or find the download unstable, you must perform the following steps.

1. Search Hugging Face for `facebook/dinov3-vitl16-pretrain-lvd1689m`.
2. Download the PyTorch `pytorch_model.bin` or `model.safetensors` alongside its configuration files.
3. Upload this directory as a Private Dataset on Kaggle through the `Add Data -> Upload Process` workflow.
4. Modify the `MODEL_NAME` internal path to point to your new bound dataset.

### 2. VMamba-Base Model:

The VMamba architecture requires explicit pre-trained visual state space model (VSSM) weights.

1. Navigate to the official VMamba repository on GitHub (`MzeroMiko/VMamba`).
2. Download the VMamba-Base v2 (s2l15) ImageNet-1K weights from the **GitHub Releases** tag `#v2cls`, that is, the file `vssm_base_0229_ckpt_epoch_237.pth` (338 MB).
3. Upload this `.pth` file as a Kaggle Dataset.
4. Update the variable `VMAMBA_CKPT` in the proposed model to the exact absolute path.

### 3. Data Inputs:

Ensure that you attach the CSIRO Image2Biomass competition dataset to every notebook payload. The dataset is expected to be loaded in `./train/` and `./test/` directories tied through `train.csv`.

### 4. Execution Order:

The notebooks are independent, but it is scientifically rigorous to run `B4_DINOv2_Metadata_cv.ipynb` first to guarantee identical stratification mappings before running the ablation studies in order A1 through A6.

---

## B. Local Execution (WSL + conda):

### Prerequisites:

- WSL (Ubuntu) with conda environment `mambahar` (Python 3.11, and PyTorch 2.5.1+cu121).
- NVIDIA GPU with CUDA support (tested on RTX 4060 Laptop with 8 GB VRAM).
- Required packages: `mamba_ssm`, `timm`, `scikit-learn`, `pandas`, `albumentations`, and `tqdm`.

### One-Time Setup:

Open WSL, enter the cloned repository, and run:

```bash
cd /mnt/c/path/to/FusionComplexityInversionBiomass
conda run --no-capture-output -n mambahar bash src/utils/setup_deps.sh
```

This command installs `vmamba.py` into the conda site-packages, downloads four VMamba pretrained weights to `pretrained/`, and pre-caches all timm backbone models.

### Running Individual Experiments:

Run experiments from the cloned repository root. Replace `<script>` with a path from the tables below.

```bash
cd /mnt/c/path/to/FusionComplexityInversionBiomass
conda run --no-capture-output -n mambahar python <script>
```

Training data must exist at `csiro-biomass/train.csv` and `csiro-biomass/train/`. Checkpoints and summaries are written below `output/`. VMamba weights are read from `pretrained/`.

Run the dependency free path smoke tests before training:

```bash
conda run --no-capture-output -n mambahar python tests/test_repository_paths.py -v
```

**Baselines (B1-B6):**

| Script | Description |
|--------|-------------|
| `experiments/baselines/B1_Median_Predictor_cv.py` | Median predictor (CPU only, no GPU required). |
| `experiments/baselines/B2_EfficientNet_B3_cv.py` | EfficientNet-B3 CNN baseline. |
| `experiments/baselines/B3_DINOv2_Giant_cv.py` | DINOv2-Giant zero-shot feature probing. |
| `experiments/baselines/B4_DINOv2_Metadata_cv.py` | DINOv2-Large with GatedDepthwiseConv fusion. |
| `experiments/baselines/B5_DINOv3_ViT_L_Metadata_cv.py` | DINOv3-ViT-L with GatedDepthwiseConv fusion. |
| `experiments/baselines/B6_VMamba_Base_Metadata_cv.py` | VMamba-Base with Mamba SSM fusion. |

**Proposed Models:**

| Script | Description |
|--------|-------------|
| `experiments/proposed/Proposed_DINOv3_ViT_L_BabyMamba_cv.py` | DINOv3-ViT-L with BidirMamba fusion (proposed model). |
| `experiments/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.py` | DINOv3-ViT-L with Cross-View Gated Attention fusion. |

**Ablation Studies (E1-E8, fusion and metadata factorial):**

| Script | Description |
|--------|-------------|
| `experiments/ablation/E1_BabyMamba_NoMeta_cv.py` | BabyMamba fusion without metadata. |
| `experiments/ablation/E2_CVGA_NoMeta_cv.py` | CVGA fusion without metadata. |
| `experiments/ablation/E3_GatedDWConv_Meta_cv.py` | GatedDepthwiseConv fusion with metadata. |
| `experiments/ablation/E4_NoFusion_cv.py` | No fusion module (mean pooling only). |
| `experiments/ablation/E5_FullMamba_NoMeta_cv.py` | Full Mamba SSM fusion without metadata. |
| `experiments/ablation/E6_SingleBlock_cv.py` | Single fusion block ablation. |
| `experiments/ablation/E7_QuadBlock_cv.py` | Quad (four) fusion block ablation. |
| `experiments/ablation/E8_NoFusion_Meta_cv.py` | No fusion with metadata features. |

**Ablation Studies (A1-A6, backbone and training):**

| Script | Description |
|--------|-------------|
| `experiments/ablation/ablation_a1_ssm_scale_cv.py` | SSM backbone scale comparison. |
| `experiments/ablation/ablation_a2_metadata_cv.py` | With and without metadata. |
| `experiments/ablation/ablation_a3_cross_validation_cv.py` | Cross-validation strategy comparison. |
| `experiments/ablation/ablation_a4_loss_function_cv.py` | MSE compared with Huber loss. |
| `experiments/ablation/ablation_a5_tta_cv.py` | Test-time augmentation impact. |
| `experiments/ablation/ablation_a6_vmamba_vs_dinov2_cv.py` | VMamba compared with DINOv2. |

### Cross-Validation Strategy:

All experiments use **Stratified Group K-Fold** (5 folds) through `sklearn.model_selection.StratifiedGroupKFold`.

- **Stratification:** Samples are binned into 5 quantiles by `Dry_Total_g` to ensure balanced target distribution across folds.
- **Grouping:** Grouped by `image_id` so that left and right views of the same sample never leak across train and validation splits.
- **Seed:** Fixed at 17 for full reproducibility.

This is neither purely random nor geo-stratified. It is **target-stratified with image-level grouping**.

### VRAM Notes (8 GB GPU):

- VMamba-based scripts (Proposed, A1, and A6) use `BATCH_SIZE=4` with `GRAD_ACCUM_STEPS=2` (that is, effective batch size 8), and gradient checkpointing is enabled.
- DINOv2-Large (B4) uses `BATCH_SIZE=4`, and DINOv3-ViT-L (B5) uses `BATCH_SIZE=5`.
- All models use mixed precision through `torch.amp.autocast('cuda')`, and `torch.compile` with the inductor backend is enabled where compatible.
- VMamba backbone forward passes are wrapped with `@torch.compiler.disable` to avoid conflicts with custom triton kernels.

---

## C. Parameter Counting:

To verify model parameter counts reported in the paper, run the counting utility.

```bash
python src/utils/count_params.py
```

This script instantiates all model variants and prints detailed parameter breakdowns for each fusion block, backbone, regression head, and metadata MLP.
