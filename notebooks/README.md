# Notebooks Manifest:

This directory contains executable Jupyter notebooks for training, evaluation, and competition submissions.

## Maintenance Policy:

- Python scripts under `experiments/` and `src/` remain the primary reference implementation.
- All training notebooks contain complete standalone training implementations without importing from `src` or `experiments`.
- All training notebooks default to safe mode (`RUN_FULL = False`) to validate paths, configuration, and environment without downloading large weights or starting compute-intensive jobs.
- Notebooks follow a strict 15-section ordering.
- Official Kaggle `test.csv` does not contain training metadata (`State`, `Species`, `NDVI`, `Height`, `Sampling_Date`). Models requiring metadata are marked as metadata blocked.

## Notebook Inventory:

| Notebook Path | Experiment ID | Source Script | Action | Data Requirement | Checkpoint Requirement | Hardware | Internet Mode | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `notebooks/training/baselines/B1_Median_Predictor_cv.ipynb` | B1 | `experiments/baselines/B1_Median_Predictor_cv.py` | Training / CV | `train.csv` | None | CPU | Offline | Standalone Validated |
| `notebooks/training/baselines/B2_EfficientNet_B3_cv.ipynb` | B2 | `experiments/baselines/B2_EfficientNet_B3_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/baselines/B3_DINOv2_Giant_cv.ipynb` | B3 (DINOv2 Base Probe) | `experiments/baselines/B3_DINOv2_Giant_cv.py` | Training / CV | `train.csv`, `train/` | Hugging Face weights | GPU | Needed for first run | Standalone Validated (Legacy filename) |
| `notebooks/training/baselines/B4_DINOv2_Metadata_cv.ipynb` | B4 (DINOv2 Large No-Meta) | `experiments/baselines/B4_DINOv2_Metadata_cv.py` | Training / CV | `train.csv`, `train/` | Hugging Face weights | GPU | Needed for first run | Standalone Validated (Legacy filename) |
| `notebooks/training/baselines/B5_DINOv3_ViT_L_Metadata_cv.ipynb` | B5 (DINOv3 ViT-L No-Meta) | `experiments/baselines/B5_DINOv3_ViT_L_Metadata_cv.py` | Training / CV | `train.csv`, `train/` | Hugging Face weights | GPU | Needed for first run | Standalone Validated (Legacy filename) |
| `notebooks/training/baselines/B6_VMamba_Base_Metadata_cv.ipynb` | B6 | `experiments/baselines/B6_VMamba_Base_Metadata_cv.py` | Training / CV | `train.csv`, `train/` | VMamba pretrained weights | GPU + CUDA | Offline with weights | Standalone Validated |
| `notebooks/training/baselines/B9_VMamba_Base_NoMetadata_cv.ipynb` | B9 | `experiments/baselines/B9_VMamba_Base_NoMetadata_cv.py` | Training / CV | `train.csv`, `train/` | VMamba pretrained weights | GPU + CUDA | Offline with weights | Standalone Validated |
| `notebooks/training/baselines/B10_DINOv2_Giant_cv.ipynb` | B10 | `experiments/baselines/B10_DINOv2_Giant_cv.py` | Training / CV | `train.csv`, `train/` | DINOv2 Giant / Large | GPU | Needed for first run | Standalone Validated |
| `notebooks/training/ablation/ablation-a1-ssm-scale_cv.ipynb` | A1 | `experiments/ablation/ablation_a1_ssm_scale_cv.py` | Training / CV | `train.csv`, `train/` | VMamba checkpoints | GPU + CUDA | Offline with weights | Standalone Validated |
| `notebooks/training/ablation/ablation-a2-metadata_cv.ipynb` | A2 | `experiments/ablation/ablation_a2_metadata_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/ablation/ablation-a3-cross-validation_cv.ipynb` | A3 | `experiments/ablation/ablation_a3_cross_validation_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/ablation/ablation-a4-loss-function_cv.ipynb` | A4 | `experiments/ablation/ablation_a4_loss_function_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/ablation/ablation-a5-tta_cv.ipynb` | A5 | `experiments/ablation/ablation_a5_tta_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/ablation/ablation-a6-vmamba-vs-dinov2_cv.ipynb` | A6 | `experiments/ablation/ablation_a6_vmamba_vs_dinov2_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/ablation/ablation-a7-metadata-only-mlp_cv.ipynb` | A7 | `experiments/ablation/ablation_a7_metadata_only_mlp_cv.py` | Training / CV | `train.csv` | None | CPU / GPU | Offline | Standalone Validated |
| `notebooks/training/fusion/E1_BabyMamba_NoMeta_cv.ipynb` | E1 | `experiments/ablation/E1_BabyMamba_NoMeta_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E2_CVGA_NoMeta_cv.ipynb` | E2 | `experiments/ablation/E2_CVGA_NoMeta_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E3_GatedDWConv_Meta_cv.ipynb` | E3 | `experiments/ablation/E3_GatedDWConv_Meta_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E4_NoFusion_cv.ipynb` | E4 | `experiments/ablation/E4_NoFusion_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E5_FullMamba_NoMeta_cv.ipynb` | E5 | `experiments/ablation/E5_FullMamba_NoMeta_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E6_SingleBlock_cv.ipynb` | E6 | `experiments/ablation/E6_SingleBlock_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E7_QuadBlock_cv.ipynb` | E7 | `experiments/ablation/E7_QuadBlock_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/fusion/E8_NoFusion_Meta_cv.ipynb` | E8 | `experiments/ablation/E8_NoFusion_Meta_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Optional | Standalone Validated |
| `notebooks/training/proposed/Proposed_DINOv3_ViT_L_BabyMamba_cv.ipynb` | Proposed BabyMamba | `experiments/proposed/Proposed_DINOv3_ViT_L_BabyMamba_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Needed for weights | Standalone Validated |
| `notebooks/training/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.ipynb` | Proposed CVGA | `experiments/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.py` | Training / CV | `train.csv`, `train/` | None | GPU | Needed for weights | Standalone Validated |
| `notebooks/training/classical/C_series_cv.ipynb` | C1-C8 | `experiments/classical/run_c_series_cv.py` | Training / CV | `train.csv`, `train/` | Optional DINO cache | CPU / GPU | Offline | Standalone Validated |
| `notebooks/evaluation/unified_checkpoint_evaluation.ipynb` | Unified Normal Eval | `experiments/evaluate_checkpoints.py` | Inference Only | `train.csv`, `train/` | Fold checkpoints | GPU / CPU | Offline | Standalone Validated |
| `notebooks/evaluation/unified_metadata_evaluation.ipynb` | Unified Metadata Eval | `experiments/ablation/evaluate_metadata_modes.py` | Inference Only | `train.csv`, `train/` | Fold checkpoints | GPU / CPU | Offline | Standalone Validated |

## Submission Notebooks Status:

| Submission Notebook | Architecture | Status | Reason |
| :--- | :--- | :--- | :--- |
| `notebooks/submission/baselines/B1_Median_Predictor_submission.ipynb` | Median | Supported and updated | Image/metadata agnostic. |
| `notebooks/submission/baselines/B2_EfficientNet_B3_submission.ipynb` | EfficientNet-B3 | Supported and updated | Matched dual-view vision model; no metadata required. |
| `notebooks/submission/baselines/B3_DINOv2_Giant_submission.ipynb` | DINOv2-Base Probe | Checkpoint dependent and not locally validated | Requires trained linear probe checkpoint. |
| `notebooks/submission/baselines/B4_DINOv2_Metadata_submission.ipynb` | DINOv2-Large | Supported and updated | No metadata model despite legacy filename. |
| `notebooks/submission/baselines/B5_DINOv3_ViT_L_Metadata_submission.ipynb` | DINOv3-ViT-L | Supported and updated | No metadata model despite legacy filename. |
| `notebooks/submission/baselines/B6_VMamba_Base_Metadata_submission.ipynb` | VMamba-Base | Metadata blocked by official test schema | Official test.csv lacks metadata columns. |
| `notebooks/submission/proposed/Proposed_BabyMamba_submission.ipynb` | DINOv3 + BabyMamba | Metadata blocked by official test schema | Model trained with 23 metadata features; official test rows lack metadata. |
| `notebooks/submission/proposed/Proposed_CVGA_submission.ipynb` | DINOv3 + CVGA | Metadata blocked by official test schema | Model trained with 23 metadata features; official test rows lack metadata. |
| `notebooks/submission/ablation/ablation-a1-ssm-scale_submission.ipynb` | VMamba Scale | Historical and not a maintained reproduction path | Multi-condition ablation. |
| `notebooks/submission/ablation/ablation-a2-metadata_submission.ipynb` | Metadata Ablation | Metadata blocked by official test schema | Metadata model branch requires missing test columns. |
| `notebooks/submission/ablation/ablation-a3-cross-validation_submission.ipynb` | CV Strategy | Historical and not a maintained reproduction path | Training split ablation. |
| `notebooks/submission/ablation/ablation-a4-loss-function_submission.ipynb` | Loss Function | Historical and not a maintained reproduction path | Objective function ablation. |
| `notebooks/submission/ablation/ablation-a5-tta_submission.ipynb` | TTA | Historical and not a maintained reproduction path | Inference ablation. |
| `notebooks/submission/ablation/ablation-a6-vmamba-vs-dinov2_submission.ipynb` | Backbone | Historical and not a maintained reproduction path | Backbone comparison study. |
