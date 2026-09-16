# Usage Instructions:

This document describes how to execute experiments, evaluations, and notebooks locally inside WSL and on Kaggle.

## Scope:

Python experiment files under `experiments/` and source modules under `src/` provide the canonical reference implementation. Notebooks under `notebooks/` contain complete standalone training or evaluation code for Kaggle and local interactive sessions without importing repository source modules.

To control drift between reference scripts and standalone notebooks, every training notebook records source parity metadata:
- Matching reference script path.
- Reference Git commit hash (`fc6f5e3`).
- SHA256 digest of the reference Python script.
- Protocol version and model configuration constants.

Static parity between notebook metadata and reference scripts is enforced by `tests/test_notebooks.py`.

## Supported execution contexts:

### 1. Local execution (WSL + Conda):
- Operating system: Ubuntu inside WSL2.
- Environment: Project Conda environment named `mambahar`.
- Hardware: NVIDIA GPU with CUDA acceleration (validated on NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB memory).
- Purpose: Full cross validation training, classical pipelines, checkpoint evaluation, and automated unit testing.

### 2. Kaggle execution:
- Runtime: Kaggle GPU kernel (NVIDIA T4 or P100).
- Input binding: Attach the competition dataset `csiro-biomass` to `/kaggle/input/csiro-biomass`.
- Internet mode: Enable Internet for models downloading weights from Hugging Face (`vit_large_patch14_dinov2.lvd142m` or `vit_large_patch16_dinov3.lvd1689m`), or attach uploaded weight datasets for offline runs.
- Purpose: Standalone interactive exploration, standalone training, and test inference.

## Clone and environment check:

Run all research commands inside WSL using the project Conda environment `mambahar`. Never use the Conda `base` environment, Windows Python, or global `pip`.

```bash
cd /mnt/c/Users/Xeron/Desktop/IIIT-ACourseWork/IMLProject/FusionComplexityInversionBiomass
conda run --no-capture-output -n mambahar python -c "
import sys, torch, timm, sklearn, skimage, xgboost
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('timm:', timm.__version__)
print('scikit-learn:', sklearn.__version__)
print('scikit-image:', skimage.__version__)
print('XGBoost:', xgboost.__version__)
"
```

### Validated local snapshot (verified 2026-09-16):
- Python: 3.11.15.
- PyTorch: 2.9.1+cu130.
- CUDA runtime: 13.0.
- CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB).
- NVIDIA driver: 581.57.
- timm: 1.0.29.
- scikit-learn: 1.8.0.
- scikit-image: 0.26.0.
- XGBoost: 3.2.0.

The script `src/utils/setup_deps.sh` installs floating dependencies and copies `vmamba.py` into the environment. It is a setup helper, not a locked dependency specification.

## Dataset setup:

Acquire the CSIRO Image2Biomass dataset from Kaggle. The expected directory structure is:

```text
csiro-biomass/
├── train.csv
├── train/
│   ├── ID1001187975.jpg
│   └── ...
├── test.csv
└── test/
    ├── ID1001187975.jpg
    └── ...
```

Point experiment scripts to the dataset directory by setting the environment variable `BIOMASS_DATA_DIR`:

```bash
export BIOMASS_DATA_DIR=/mnt/c/path/to/csiro-biomass
```

If `BIOMASS_DATA_DIR` is not set, scripts check the default repository sibling or child path `csiro-biomass`. Never commit image files, archives, or full dataset directories to Git.

## Preflight checks:

Run repository verification tests inside WSL before launching experiments:

```bash
# Static path integrity test (requires no data or GPU):
conda run --no-capture-output -n mambahar python tests/test_repository_paths.py -v

# Protocol consistency test (requires no external weights):
conda run --no-capture-output -n mambahar python -m pytest tests/test_rerun_protocols.py -v

# Notebook ecosystem validation test (verifies all 19 notebook gates):
conda run --no-capture-output -n mambahar python tests/test_notebooks.py -v

# Classical pipeline artifact-independent unit tests:
conda run --no-capture-output -n mambahar python tests/test_classical_baselines.py TestClassicalBaselines.test_12_classical_pipeline_scaler_import
```

Full classical tests require `train.csv` and existing fold files:

```bash
conda run --no-capture-output -n mambahar python tests/test_classical_baselines.py -v
```

## Experimental protocol:

All models follow a strictly aligned protocol:
- Partitioning: Target stratified five fold cross validation with seed 17.
- Grouping: Image level grouping. Because pivoting yields exactly one row per unique image identifier, grouping by image identifier is redundant after pivoting.
- Primary target variables: Evaluated on raw gram targets:
  - `Dry_Green_g` (weight 0.1)
  - `Dry_Dead_g` (weight 0.1)
  - `Dry_Clover_g` (weight 0.1)
  - `GDM_g` (weight 0.2, defined compositionally as Green + Clover)
  - `Dry_Total_g` (weight 0.5, defined compositionally as GDM + Dead)
- Primary metric: Raw weighted R squared across five targets with weights `[0.1, 0.1, 0.1, 0.2, 0.5]`. Log R squared is a diagnostic metric, not the primary ranking metric.
- Primitive head parameterization: Models train regression heads on three non negative primitive targets (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`) and derive `GDM_g` and `Dry_Total_g` by exact summation.
- Loss function: Compositional Huber loss with beta 5.0 weighted by the target evaluation weights.
- Single seed limitation: Reported fold standard deviation represents fold dispersion across the five validation splits from seed 17, not multi seed stochastic uncertainty.

## Neural training commands:

Execute maintained reference neural training experiments using `conda run`:

| Experiment ID | Reference Script Path | Backbone | Fusion Module | Metadata State | Hardware | Output Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | `experiments/baselines/B1_Median_Predictor_cv.py` | None (Median) | None | False | CPU | `output/reruns_2026_09_13/B1_Median_Predictor` |
| **B2** | `experiments/baselines/B2_EfficientNet_B3_cv.py` | `efficientnet_b3.ra2_in1k` | GatedDepthwiseConv | False | GPU | `output/reruns_2026_09_13/B2_EfficientNet_B3_matched` |
| **B3** | `experiments/baselines/B3_DINOv2_Giant_cv.py` | `facebook/dinov2-base` | Supervised Probe | False | GPU | `output/reruns_2026_09_13/B3_DINOv2_Base_probe` |
| **B4** | `experiments/baselines/B4_DINOv2_Metadata_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | GPU | `output/reruns_2026_09_13/B4_DINOv2_Large_matched` |
| **B5** | `experiments/baselines/B5_DINOv3_ViT_L_Metadata_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | GatedDepthwiseConv | False | GPU | `output/reruns_2026_09_13/B5_DINOv3_ViT_L_matched` |
| **B6** | `experiments/baselines/B6_VMamba_Base_Metadata_cv.py` | `vmamba_base` | Mamba SSM | True | GPU + CUDA | `output/reruns_2026_09_13/B6_VMamba_Base_Metadata` |
| **B9** | `experiments/baselines/B9_VMamba_Base_NoMetadata_cv.py` | `vmamba_base` | Mamba SSM | False | GPU + CUDA | `output/reruns_2026_09_13/B9_VMamba_Base_NoMetadata` |
| **B10** | `experiments/baselines/B10_DINOv2_Giant_cv.py` | `facebook/dinov2-giant` | Supervised Probe | False | GPU | `output/reruns_2026_09_13/B10_DINOv2_Giant_probe` |
| **A1** | `experiments/ablation/ablation_a1_ssm_scale_cv.py` | `vmamba_small` | Mamba SSM | True | GPU + CUDA | `output/reruns_2026_09_13/ablation_a1_ssm_scale` |
| **A2** | `experiments/ablation/ablation_a2_metadata_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | True | GPU | `output/reruns_2026_09_13/ablation_a2_metadata` |
| **A3** | `experiments/ablation/ablation_a3_cross_validation_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | GPU | `output/reruns_2026_09_13/ablation_a3_cv` |
| **A4** | `experiments/ablation/ablation_a4_loss_function_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | GPU | `output/reruns_2026_09_13/ablation_a4_loss` |
| **A5** | `experiments/ablation/ablation_a5_tta_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | GPU | `output/reruns_2026_09_13/ablation_a5_tta` |
| **A6** | `experiments/ablation/ablation_a6_vmamba_vs_dinov2_cv.py` | `vmamba_base` | Mamba SSM | False | GPU + CUDA | `output/reruns_2026_09_13/ablation_a6_vmamba` |
| **A7** | `experiments/ablation/ablation_a7_metadata_only_mlp_cv.py` | Tabular MLP | None | True | CPU / GPU | `output/metadata_repair_2026_09_13/A7_Metadata_Only_MLP` |
| **E1** | `experiments/ablation/E1_BabyMamba_NoMeta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | BabyMamba | False | GPU | `output/reruns_2026_09_13/E1_BabyMamba_NoMeta` |
| **E2** | `experiments/ablation/E2_CVGA_NoMeta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | CVGA | False | GPU | `output/reruns_2026_09_13/E2_CVGA_NoMeta` |
| **E3** | `experiments/ablation/E3_GatedDWConv_Meta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | GatedDepthwiseConv | True | GPU | `output/reruns_2026_09_13/E3_GatedDWConv_Meta` |
| **E4** | `experiments/ablation/E4_NoFusion_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | NoFusion | False | GPU | `output/reruns_2026_09_13/E4_NoFusion` |
| **E5** | `experiments/ablation/E5_FullMamba_NoMeta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | FullMamba | False | GPU | `output/reruns_2026_09_13/E5_FullMamba_NoMeta` |
| **E6** | `experiments/ablation/E6_SingleBlock_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | SingleBlock | False | GPU | `output/reruns_2026_09_13/E6_SingleBlock` |
| **E7** | `experiments/ablation/E7_QuadBlock_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | QuadBlock | False | GPU | `output/reruns_2026_09_13/E7_QuadBlock` |
| **E8** | `experiments/ablation/E8_NoFusion_Meta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | NoFusion | True | GPU | `output/reruns_2026_09_13/E8_NoFusion_Meta` |
| **Proposed BabyMamba** | `experiments/proposed/Proposed_DINOv3_ViT_L_BabyMamba_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | BidirMamba | True | GPU | `output/reruns_2026_09_13/Proposed_BabyMamba` |
| **Proposed CVGA** | `experiments/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | CVGA | True | GPU | `output/reruns_2026_09_13/Proposed_CVGA` |

Example command to launch training:

```bash
conda run --no-capture-output -n mambahar python experiments/baselines/B2_EfficientNet_B3_cv.py
```

## Classical baseline commands:

The script `experiments/classical/run_c_series_cv.py` is the unified runner for all eight classical baseline configurations (C1 through C8).

Run all classical baselines with inner model selection:

```bash
conda run --no-capture-output -n mambahar python experiments/classical/run_c_series_cv.py --models C1 C2 C3 C4 C5 C6 C7 C8
```

Run fast tabular and handcrafted models (CPU only):

```bash
conda run --no-capture-output -n mambahar python experiments/classical/run_c_series_cv.py --models C1 C2 C3 C4 C5
```

Features and model families:
- C1: 23 tabular metadata features with Ridge regression.
- C2: 23 tabular metadata features with XGBoost.
- C3: 652 handcrafted cross view features with Ridge regression and StandardScaler.
- C4: 652 handcrafted cross view features with XGBoost.
- C5: Handcrafted features plus tabular metadata with XGBoost.
- C6: Frozen DINOv2 Base representations with PCA and Ridge regression.
- C7: Frozen DINOv3 ViT-L representations with PCA and Ridge regression.
- C8: Frozen DINOv3 ViT-L representations with PCA and XGBoost.

Feature caching:
- Handcrafted features are cached to `output/classical_baselines_2026_09_15/handcrafted_cache.npz`.
- DINOv2 features are loaded from `checkpoints/B3_DINOv2_Base_probe`.
- DINOv3 features are extracted once on GPU and cached to `output/classical_baselines_2026_09_15/dinov3_cache.npz` to serve both C7 and C8.

## Checkpoint only evaluation:

The unified checkpoint evaluator performs strictly inference without training, retraining, or optimizer state updates.

Run checkpoint suites through the canonical entry point:

```bash
# 1. Standard checkpoint integrity evaluation:
conda run --no-capture-output -n mambahar python experiments/evaluate_checkpoints.py --suite integrity

# 2. Metadata counterfactual modes (present, zero, shuffled) and matched control analysis:
conda run --no-capture-output -n mambahar python experiments/evaluate_checkpoints.py --suite metadata

# 3. Additional ablation checkpoint evaluation:
conda run --no-capture-output -n mambahar python experiments/evaluate_checkpoints.py --suite additional

# 4. Run all three suites sequentially:
conda run --no-capture-output -n mambahar python experiments/evaluate_checkpoints.py --suite all
```

Standalone evaluation notebooks:
- Standard evaluation notebook: `notebooks/evaluation/unified_checkpoint_evaluation.ipynb`.
- Metadata counterfactual notebook: `notebooks/evaluation/unified_metadata_evaluation.ipynb`.

Normal model evaluation computes validation scores using true validation inputs under each model's declared metadata state. Metadata mode evaluation perturbs tabular metadata vectors into `present`, `zero`, and `shuffled` conditions to assess metadata sensitivity and visual predictor damage against matched no metadata controls.

## Notebook use:

All training notebooks default to safe validation mode:

```python
RUN_FULL = False
```

- Safe mode (`RUN_FULL = False`): Validates local paths, package imports, configuration parameters, and execution devices. Does not download large foundation backbones, train weights, or overwrite checkpoints.
- Full mode (`RUN_FULL = True`): Executes complete training loops inside the notebook and writes artifacts below `output/`. Does not delegate execution to external experiment scripts.

Repository and data discovery inside notebooks checks candidate roots automatically (`REPO_ROOT`, `/kaggle/working`, and sibling directories). For full notebook details, see the manifest in [notebooks/README.md](file:///c:/Users/Xeron/Desktop/IIIT-ACourseWork/IMLProject/FusionComplexityInversionBiomass/notebooks/README.md).

## Submission limitations:

The official Kaggle test CSV (`test.csv`) contains only `sample_id`, `image_path`, and `target_name`. It does not contain training metadata attributes (`State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`, `Sampling_Date`).

Consequently:
- Vision only models (B1, B2, B4, B5) generate valid Kaggle test submissions directly from test images.
- Models trained with metadata (B6, Proposed BabyMamba, Proposed CVGA, A2, C1, C2, C5) are metadata blocked by the official competition schema.
- Submission notebooks for metadata trained models include fail fast checks that halt execution early with a descriptive error. Do not invent synthetic metadata or fill zeroes silently for competition submissions.

## Outputs and provenance:

Outputs from training and evaluation are stored below `output/`:
- Checkpoints: `output/reruns_2026_09_13/<EXP_ID>/fold_<F>_best.pth`.
- Out of fold predictions: `output/reruns_2026_09_13/<EXP_ID>/oof_predictions.csv`.
- Fold and aggregate metrics: `output/reruns_2026_09_13/<EXP_ID>/metrics.json`.
- Run metadata: `output/reruns_2026_09_13/<EXP_ID>/run_metadata.json` recording configuration, Git commit, package versions, and hardware information.

Git ignored directories:
- `output/`
- `checkpoints/`
- `csiro-biomass/`
- `pretrained/`
- `results/`
- `third_party/`

## Troubleshooting:

- CUDA out of memory: Reduce `BATCH_SIZE` in the configuration cell or script. Ensure PyTorch AMP mixed precision is enabled (`autocast('cuda')`).
- Missing Hugging Face weights: If downloading `facebook/dinov3-vitl16-pretrain-lvd1689m` fails due to network constraints, download the safetensors file offline and place it under `pretrained/` or attach it as a Kaggle dataset.
- VMamba extension ABI mismatch: If `mamba_ssm` or `vmamba` fails to import due to CUDA or PyTorch symbol mismatches, run standard GatedDepthwiseConv models (B2, B4, B5, E3) which do not require specialized CUDA extensions.
- Missing fold file: If `output/reruns_2026_09_13/folds_seed17.csv` is missing, scripts reconstruct identical splits using StratifiedGroupKFold on target quintiles with seed 17.
- Missing test metadata: If a submission script fails with `KeyError: 'State'`, verify whether the model was trained with metadata. Official competition test rows contain images only.
