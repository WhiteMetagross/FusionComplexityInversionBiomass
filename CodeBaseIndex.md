# Codebase Index:

This document provides a component-level inventory of tracked repository source files, experiments, evaluators, notebooks, tests, and analysis tools.

## 1. Core source code (`src/`):

### Shared library modules:

- **`src/engine.py`:**
  - Purpose: Shared training engine for neural dual view cross validation experiments.
  - Key classes and functions: `load_train_data` (pivots long to wide format and partitions folds), `BiomassDataset` (dual view image dataset with optional metadata vector), `prepare_image_cache` (caches split images to local filesystem), `train_one_epoch` (mixed precision forward pass with gradient accumulation), `validate` (evaluates validation split with optional test time augmentation), `weighted_r2_score` (computes raw weighted R squared), `CompositionalHuberLoss` (weighted Huber loss on primitive predictions), and `run_cv` (full five fold cross validation orchestrator with checkpoint resume).
  - Consumers: Neural baseline scripts (B2, B4, B5, B6, B9), ablation scripts (A1 through A6, E1 through E8), and proposed model scripts.
  - Limits: Requires `train.csv` and dual view images formatted with identical height and width.

- **`src/models.py`:**
  - Purpose: Definitions of neural network architectures, cross view fusion blocks, and regression heads.
  - Key classes:
    - `GatedDepthwiseConvBlock`: Gated depthwise convolution for local cross view spatial mixing.
    - `MambaFusionBlock`: Selective state space model cross view fusion block using `mamba_ssm`.
    - `BidirMamba`: Bidirectional Mamba fusion block with gating, depthwise convolution, and weight tied bidirectional scanning.
    - `BabyMambaFusionBlock`: Weight tied bidirectional Mamba block with shared parameters across forward and backward passes.
    - `CVGABlock`: Cross View Gated Attention block implementing bidirectional cross attention via `torch.nn.functional.scaled_dot_product_attention`.
    - `BiomassModelTimm`: Model wrapper accepting any `timm` visual backbone, any fusion block, and optional metadata projection MLP.
    - `VMambaBackbone`: Visual State Space Model backbone wrapping `vmamba.VSSM`.
    - `BiomassModelVMamba`: Dual view model combining `VMambaBackbone` and `MambaFusionBlock`.
  - Consumers: All neural training experiments and checkpoint evaluators.
  - Limits: Mamba fusion blocks require CUDA runtime and custom extensions. When running without extensions, fallback execution is disabled by default.

- **`src/frozen_probe.py`:**
  - Purpose: Supervised frozen dual view probe harness for foundation vision representations.
  - Key classes and functions: `DualViewFrozenProbe` (two view linear projection probe with compositional Softplus heads), `ProbeConfig` (dataclass for extraction and probe hyperparameters), `load_fold_data` (dataset loader for probing), and `weighted_r2` (weighted R squared evaluation).
  - Consumers: Baseline experiments B3 and B10.
  - Limits: Extracts CLS token embeddings in offline passes. Requires sufficient storage for embedding caches.

- **`src/classical_features.py`:**
  - Purpose: Feature extraction, preprocessing, and caching routines for classical baseline models (C1 through C8).
  - Key classes and functions: `MetadataPreprocessor` (encodes categoricals and scales continuous attributes with `StandardScaler`), `extract_handcrafted_pair` (computes 163 color moments, texture, and vegetation features per view and 652 cross view difference/ratio features), `get_handcrafted_feature_names`, `build_or_load_handcrafted_cache`, `load_dinov2_b3_cache`, and `build_or_load_dinov3_cache`.
  - Consumers: `experiments/classical/run_c_series_cv.py` and `tests/test_classical_baselines.py`.
  - Limits: DINOv3 feature extraction requires GPU acceleration.

### Utility scripts (`src/utils/`):

- **`src/utils/count_params.py`:**
  - Purpose: Programmatically counts and breaks down parameters across backbones, fusion blocks, regression heads, and metadata MLPs.
  - Consumers: Model verification and paper parameter reporting.

- **`src/utils/setup_deps.sh`:**
  - Purpose: Convenience shell script to download VMamba weights into `pretrained/` and install `vmamba.py` into the environment.
  - Limits: Installs unpinned floating packages. It is not an environment lock file.

## 2. Experiment entry points (`experiments/`):

### Baselines (`experiments/baselines/`):

| Experiment ID | Script Path | Actual Model | Fusion Type | Metadata State | Action | Output Root |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | `experiments/baselines/B1_Median_Predictor_cv.py` | Global Median | None | False | Training / CV | `output/reruns_2026_09_13/B1_Median_Predictor` |
| **B2** | `experiments/baselines/B2_EfficientNet_B3_cv.py` | `efficientnet_b3.ra2_in1k` | GatedDepthwiseConv | False | Training / CV | `output/reruns_2026_09_13/B2_EfficientNet_B3_matched` |
| **B3** | `experiments/baselines/B3_DINOv2_Giant_cv.py` | `facebook/dinov2-base` | Supervised Probe | False | Probe Training / CV | `output/reruns_2026_09_13/B3_DINOv2_Base_probe` |
| **B4** | `experiments/baselines/B4_DINOv2_Metadata_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | Training / CV | `output/reruns_2026_09_13/B4_DINOv2_Large_matched` |
| **B5** | `experiments/baselines/B5_DINOv3_ViT_L_Metadata_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | GatedDepthwiseConv | False | Training / CV | `output/reruns_2026_09_13/B5_DINOv3_ViT_L_matched` |
| **B6** | `experiments/baselines/B6_VMamba_Base_Metadata_cv.py` | `vmamba_base` | Mamba SSM | True | Training / CV | `output/reruns_2026_09_13/B6_VMamba_Base_Metadata` |
| **B9** | `experiments/baselines/B9_VMamba_Base_NoMetadata_cv.py` | `vmamba_base` | Mamba SSM | False | Training / CV | `output/reruns_2026_09_13/B9_VMamba_Base_NoMetadata` |
| **B10** | `experiments/baselines/B10_DINOv2_Giant_cv.py` | `facebook/dinov2-giant` | Supervised Probe | False | Probe Training / CV | `output/reruns_2026_09_13/B10_DINOv2_Giant_probe` |

*Notes on legacy names and released checkpoints:*
- `B3_DINOv2_Giant_cv.py` is a legacy name. The actual model is `facebook/dinov2-base` supervised probe. It is not Giant and not zero-shot.
- `B4_DINOv2_Metadata_cv.py` and `B5_DINOv3_ViT_L_Metadata_cv.py` set `USE_METADATA = False`. They are vision only models.
- Baselines B7 (CVGA with metadata) and B8 (BidirMamba with metadata) have released checkpoints and evaluation harness support, but do not have maintained training scripts in this repository.

### Proposed models (`experiments/proposed/`):

| Experiment ID | Script Path | Actual Model | Fusion Type | Metadata State | Action | Output Root |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed BabyMamba** | `experiments/proposed/Proposed_DINOv3_ViT_L_BabyMamba_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | BidirMamba | True | Training / CV | `output/reruns_2026_09_13/Proposed_BabyMamba` |
| **Proposed CVGA** | `experiments/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | CVGA | True | Training / CV | `output/reruns_2026_09_13/Proposed_CVGA` |

### Ablation studies (`experiments/ablation/`):

| Experiment ID | Script Path | Backbone | Fusion Module | Metadata State | Action | Output Root |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | `experiments/ablation/ablation_a1_ssm_scale_cv.py` | `vmamba_small` | Mamba SSM | True | Training / CV | `output/reruns_2026_09_13/ablation_a1_ssm_scale` |
| **A2** | `experiments/ablation/ablation_a2_metadata_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | True | Training / CV | `output/reruns_2026_09_13/ablation_a2_metadata` |
| **A3** | `experiments/ablation/ablation_a3_cross_validation_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | Training / CV | `output/reruns_2026_09_13/ablation_a3_cv` |
| **A4** | `experiments/ablation/ablation_a4_loss_function_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | Training / CV | `output/reruns_2026_09_13/ablation_a4_loss` |
| **A5** | `experiments/ablation/ablation_a5_tta_cv.py` | `vit_large_patch14_dinov2.lvd142m` | GatedDepthwiseConv | False | Training / CV | `output/reruns_2026_09_13/ablation_a5_tta` |
| **A6** | `experiments/ablation/ablation_a6_vmamba_vs_dinov2_cv.py` | `vmamba_base` | Mamba SSM | False | Training / CV | `output/reruns_2026_09_13/ablation_a6_vmamba` |
| **A7** | `experiments/ablation/ablation_a7_metadata_only_mlp_cv.py` | Tabular MLP | None | True | Training / CV | `output/metadata_repair_2026_09_13/A7_Metadata_Only_MLP` |
| **E1** | `experiments/ablation/E1_BabyMamba_NoMeta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | BabyMamba | False | Training / CV | `output/reruns_2026_09_13/E1_BabyMamba_NoMeta` |
| **E2** | `experiments/ablation/E2_CVGA_NoMeta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | CVGA | False | Training / CV | `output/reruns_2026_09_13/E2_CVGA_NoMeta` |
| **E3** | `experiments/ablation/E3_GatedDWConv_Meta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | GatedDepthwiseConv | True | Training / CV | `output/reruns_2026_09_13/E3_GatedDWConv_Meta` |
| **E4** | `experiments/ablation/E4_NoFusion_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | NoFusion | False | Training / CV | `output/reruns_2026_09_13/E4_NoFusion` |
| **E5** | `experiments/ablation/E5_FullMamba_NoMeta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | FullMamba | False | Training / CV | `output/reruns_2026_09_13/E5_FullMamba_NoMeta` |
| **E6** | `experiments/ablation/E6_SingleBlock_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | SingleBlock | False | Training / CV | `output/reruns_2026_09_13/E6_SingleBlock` |
| **E7** | `experiments/ablation/E7_QuadBlock_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | QuadBlock | False | Training / CV | `output/reruns_2026_09_13/E7_QuadBlock` |
| **E8** | `experiments/ablation/E8_NoFusion_Meta_cv.py` | `vit_large_patch16_dinov3.lvd1689m` | NoFusion | True | Training / CV | `output/reruns_2026_09_13/E8_NoFusion_Meta` |

### Classical C series (`experiments/classical/`):

- **`experiments/classical/run_c_series_cv.py`:** Single unified entry point for classical baseline models C1 through C8. Performs inner three fold model selection strictly inside each outer training partition, followed by locked outer five fold cross validation.

## 3. Evaluation tools:

- **`experiments/evaluate_checkpoints.py`:** Canonical CLI entry point exposing `--suite integrity`, `--suite metadata`, `--suite additional`, and `--suite all`. Executes inference without training.
- **`experiments/evaluate_checkpoint_integrity.py`:** Verifies checkpoint loading (`strict=True`), computes raw per target R squared, raw weighted R squared, and diagnostic log R squared across all obtainable models.
- **`experiments/ablation/evaluate_metadata_modes.py`:** Evaluates models under `present`, `zero`, and `shuffled` metadata modes. Computes metadata information use, missing metadata sensitivity, and visual predictor damage against matched no metadata controls.
- **`experiments/ablation/evaluate_additional_kaggle_models.py`:** Evaluates supplementary ablation checkpoints.

Primary metric is raw weighted R squared with weights `[0.1, 0.1, 0.1, 0.2, 0.5]`. Log R squared metrics are emitted solely for diagnostic comparison.

## 4. Notebooks (`notebooks/`):

All notebook paths, metadata specifications, execution modes, and submission statuses are indexed in [notebooks/README.md](file:///c:/Users/Xeron/Desktop/IIIT-ACourseWork/IMLProject/FusionComplexityInversionBiomass/notebooks/README.md).

- **Training notebooks:** 26 complete standalone implementations covering B1-B6, B9, B10, A1-A7, E1-E8, Proposed BabyMamba, Proposed CVGA, and C series. All default to safe mode (`RUN_FULL = False`).
- **Evaluation notebooks:**
  - `notebooks/evaluation/unified_checkpoint_evaluation.ipynb`: Standalone normal checkpoint evaluation under strict loading.
  - `notebooks/evaluation/unified_metadata_evaluation.ipynb`: Standalone metadata mode evaluation (`present`, `zero`, `shuffled`) with matched controls.
- **Submission notebooks:** 14 submission notebooks audited under `notebooks/submission/`. Metadata trained models carry fail fast schema checks because the official test CSV lacks tabular metadata.

## 5. Tests (`tests/`):

All test suites run inside WSL using Conda environment `mambahar`:

| Test Script | Purpose | Prerequisites |
| :--- | :--- | :--- |
| **`tests/test_repository_paths.py`** | Validates internal import paths, runtime directory conventions, and absence of author personal paths. | Standard library only. No data or GPU required. |
| **`tests/test_rerun_protocols.py`** | Verifies non negative compositional outputs, weighted R squared calculation, fold counts, and backbone token adaptation. | PyTorch, NumPy. No external checkpoints required. |
| **`tests/test_classical_baselines.py`** | 12 verification gates for classical C series models, including data hashing, handcrafted feature bitwise equality, leak free selection, and `StandardScaler` integration. | Requires `train.csv` and fold split files. |
| **`tests/test_notebooks.py`** | 19 static verification gates for all repository notebooks: JSON validity, nbformat 4, python3 kernelspec, null execution counts, empty outputs, no personal paths, no credentials, manifest consistency, source parity metadata, and fail fast checks. | Standard library only. No data or GPU required. |

## 6. Analysis and figures (`analysis/`):

Tracked analysis files:
- **`analysis/analyze_tsne.py`, `analysis/generate_tsne_figure.py`:** t-SNE projection generation from extracted backbone features.
- **`analysis/dataset_analysis.py`, `analysis/dataset_analysis_v2.py`, `analysis/dataset_analysis_v3.py`:** Exploratory analysis of target distributions, spatial patterns, and feature correlations.
- **`analysis/extract_feature_maps.py`:** Feature map visualization utilities for DINOv2 and VMamba backbones.
- **`analysis/generate_architecture_diagram.py`:** Architectural diagram generator.
- **`analysis/generate_paper_figures.py`:** Compiles exploratory and preliminary paper figures.
- **`analysis/export_paper_figures.py`:** Canonical exporter that generates publication ready figures matching manuscript specifications.
- **`analysis/paper_figure_manifest.json`:** Manifest mapping figure numbers, file names, titles, and data sources.

## 7. Local and external artifacts:

The following directories are excluded from Git version control:
- `checkpoints/`: Local and downloaded model checkpoints (`.pth`).
- `csiro-biomass/`: Extracted dataset containing `train.csv`, `test.csv`, and image folders.
- `output/`: Generated training logs, out of fold predictions, metrics, and run metadata.
- `pretrained/`: Pretrained backbone checkpoints (DINOv2, VMamba VSSM weights).
- `results/`: Adjudicated tables, notes, and local audit documents.
- `third_party/`: External source checkouts.
- `zips/`: Packaged archives for distribution.

These directories must be acquired, created, or downloaded locally.

## 8. Known limitations:

- **Single seed evaluation:** All neural training experiments evaluate five fold cross validation over seed 17. Standard deviations denote fold dispersion across splits from one seed, not multi seed stochastic uncertainty.
- **Redundant image grouping:** Pivoting long format competition rows yields exactly one row per unique image identifier. Grouping by image identifier during StratifiedGroupKFold is therefore functionally equivalent to standard StratifiedKFold.
- **Missing test metadata:** Official Kaggle `test.csv` rows provide image paths only. Models utilizing tabular metadata attributes (`State`, `Species`, `NDVI`, `Height`, `Sampling_Date`) cannot be deployed directly on competition test data.
- **Hardware dependencies:** VMamba selective state space scanning requires compiled `mamba_ssm` and `vmamba` CUDA extensions.
- **Legacy file names:** `B3_DINOv2_Giant_cv.py` implements DINOv2 Base; `B4_DINOv2_Metadata_cv.py` and `B5_DINOv3_ViT_L_Metadata_cv.py` do not use metadata. File names are preserved for backward compatibility.
