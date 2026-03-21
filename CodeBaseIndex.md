# Code Base Index:

**Author:** Mridankan Mandal

This document provides a file-level index of every component in the repository, organized by directory.

---

## 1. Core Source Code (`src/`):

### Shared Modules:

- **`engine.py`:** Complete training engine providing `setup_environment()`, `load_train_data()` (StratifiedGroupKFold), `prepare_image_cache()` (WSL filesystem optimization), `BiomassDataset` (dual-view dataset with optional metadata), `train_one_epoch()` (with gradient accumulation and metadata dropout), `validate()` (with optional test-time augmentation), `train_fold()` (with checkpoint resume), and `run_cv()` (full K-fold cross-validation with automatic resume support). Uses mixed precision through `torch.amp.autocast`, `torch.compile` with inductor backend, and `GradScaler`.

- **`models.py`:** All neural network architectures used in the study.
  - `GatedDepthwiseConvBlock`: Gated depthwise-conv fusion block for local spatial feature mixing (baselines B4, and B5).
  - `MambaFusionBlock`: Hardware-accelerated Mamba SSM fusion block using `mamba_ssm` with custom CUDA kernels (B6, and ablations A1-A6). Decorated with `@torch.compiler.disable`.
  - `BidirMamba`: Bidirectional Mamba fusion block with sigmoid gating, depthwise conv, and weight-tied forward and backward SSM scans.
  - `BabyMambaFusionBlock`: Weight-tied bidirectional Mamba block where forward and backward passes share the exact same parameters.
  - `CVGABlock`: Cross-View Gated Attention block with bidirectional cross-attention between left and right views. Uses `F.scaled_dot_product_attention` for automatic flash dispatch.
  - `BiomassModelTimm`: Dual-view model wrapper accepting any timm backbone, any fusion block, and optional metadata MLP. Gradient checkpointing is enabled.
  - `VMambaBackbone`: VMamba-Base v2 (s2l15) backbone wrapping `vmamba.VSSM`. Auto-resolves cached weights from `pretrained/`. Forward is decorated with `@torch.compiler.disable`.
  - `BiomassModelVMamba`: Dual-view model combining `VMambaBackbone` and `MambaFusionBlock` fusion.

### Utility Scripts (`src/utils/`):

- **`count_params.py`:** Parameter counting utility that instantiates all model variants. Prints detailed breakdowns for each fusion block (1-block and 2-block totals), backbone, regression head, metadata MLP, and full model totals. Used to verify parameter counts reported in the paper.

- **`setup_deps.sh`:** One-time setup script for local execution. Installs the `vmamba.py` module from the MzeroMiko/VMamba GitHub repository into the conda site-packages, downloads four VMamba pretrained weights from GitHub releases, and pre-caches all timm backbone models.

---

## 2. Dataset Analysis (`analysis/`):

- **`dataset_analysis.py`, `dataset_analysis_v2.py`, `dataset_analysis_v3.py`:** Exploratory Data Analysis (EDA) scripts used to evaluate distributions, relationships, and feature importance.
- **`extract_feature_maps.py`, `analyze_tsne.py`, `generate_tsne_figure.py`:** Utilities to extract DINOv2 and VMamba backbone features and generate high-quality t-SNE mapping visualizations.
- **`generate_paper_figures.py`, `generate_architecture_diagram.py`:** Scripts to programmatically compile and layout the visualizations and architecture diagrams used in the study.

---

## 3. Experiment Configurations (`experiments/`):

### Baselines (`experiments/baselines/`):

- **`B1_Median_Predictor_cv.py`:** Median predictor baseline (CPU only). Computes the global training median as the prediction for all validation samples.
- **`B2_EfficientNet_B3_cv.py`:** EfficientNet-B3 CNN baseline using timm pretrained weights (batch_size=12).
- **`B3_DINOv2_Giant_cv.py`:** DINOv2-Giant zero-shot feature probing baseline (batch_size=3).
- **`B4_DINOv2_Metadata_cv.py`:** DINOv2-Large backbone with GatedDepthwiseConv fusion (batch_size=4, and image size 518px). Resumes from Kaggle checkpoints for folds 0-1.
- **`B5_DINOv3_ViT_L_Metadata_cv.py`:** DINOv3-ViT-L backbone with GatedDepthwiseConv fusion (batch_size=4, and image size 512px).
- **`B6_VMamba_Base_Metadata_cv.py`:** VMamba-Base v2 backbone with MambaFusionBlock fusion (batch_size=4, grad_accum=2).

### Proposed Models (`experiments/proposed/`):

- **`Proposed_DINOv3_ViT_L_BabyMamba_cv.py`:** Primary proposed model using DINOv3-ViT-L backbone with BidirMamba fusion (batch_size=5). Uses metadata features for final regression.
- **`Proposed_DINOv3_ViT_L_CVGA_cv.py`:** Alternative proposed model using DINOv3-ViT-L backbone with Cross-View Gated Attention fusion (batch_size=5).

### Ablation Studies (E-Series, `experiments/ablation/`):

These experiments isolate the effect of fusion complexity and metadata on the DINOv3-ViT-L backbone.

- **`E1_BabyMamba_NoMeta_cv.py`:** BidirMamba fusion without metadata. Isolates the fusion effect independent of metadata.
- **`E2_CVGA_NoMeta_cv.py`:** Cross-View Gated Attention fusion without metadata. Tests attention-based fusion without metadata.
- **`E3_GatedDWConv_Meta_cv.py`:** GatedDepthwiseConv fusion with metadata. Completes the 2x2 factorial (fusion type multiplied by metadata).
- **`E4_NoFusion_cv.py`:** No fusion blocks. Simply concatenates left and right backbone features and pools directly.
- **`E5_FullMamba_NoMeta_cv.py`:** Full-scale Mamba SSM fusion without metadata. Tests the unidirectional full Mamba block.
- **`E6_SingleBlock_cv.py`:** Single fusion block instead of the default two. Tests the depth of the fusion module.
- **`E7_QuadBlock_cv.py`:** Four fusion blocks. Tests whether more fusion depth improves performance.
- **`E8_NoFusion_Meta_cv.py`:** No fusion blocks with metadata features injected. Tests whether metadata alone provides sufficient cross-view information.

### Ablation Studies (A-Series, `experiments/ablation/`):

These experiments evaluate backbone scale, training strategy, and augmentation choices.

- **`ablation_a1_ssm_scale_cv.py`:** SSM backbone scale comparison (VMamba-Tiny, Small, and Base with batch_size=4, and grad_accum=2).
- **`ablation_a2_metadata_cv.py`:** Metadata ablation. Tests model accuracy with and without metadata feature injection.
- **`ablation_a3_cross_validation_cv.py`:** Cross-validation strategy ablation. Compares standard KFold with site-stratified group splits.
- **`ablation_a4_loss_function_cv.py`:** Loss function ablation. Compares MSE loss with Huber (SmoothL1) loss.
- **`ablation_a5_tta_cv.py`:** Test-time augmentation ablation. Measures the impact of horizontal flip, vertical flip, and 90-degree rotation TTA.
- **`ablation_a6_vmamba_vs_dinov2_cv.py`:** Backbone comparison. Compares VMamba linear scanning with DINOv2 ViT quadratic attention (batch_size=4, and grad_accum=2).

---

## 3. Kaggle Notebooks (`notebooks/`):

### Training Notebooks (`notebooks/training/`):

#### Baselines (`notebooks/training/baselines/`):

- **`B1_Median_Predictor_cv.ipynb`:** Kaggle notebook for B1 median predictor baseline.
- **`B2_EfficientNet_B3_cv.ipynb`:** Kaggle notebook for B2 EfficientNet-B3 baseline.
- **`B3_DINOv2_Giant_cv.ipynb`:** Kaggle notebook for B3 DINOv2-Giant zero-shot probing.
- **`B4_DINOv2_Metadata_cv.ipynb`:** Kaggle notebook for B4 DINOv2-Large with GatedDepthwiseConv fusion.
- **`B5_DINOv3_ViT_L_Metadata_cv.ipynb`:** Kaggle notebook for B5 DINOv3-ViT-L with GatedDepthwiseConv fusion.
- **`B6_VMamba_Base_Metadata_cv.ipynb`:** Kaggle notebook for B6 VMamba-Base with Mamba SSM fusion.

#### Ablation Studies (`notebooks/training/ablation/`):

- **`ablation-a1-ssm-scale_cv.ipynb`:** Kaggle notebook for A1 SSM scale comparison.
- **`ablation-a2-metadata_cv.ipynb`:** Kaggle notebook for A2 metadata ablation.
- **`ablation-a3-cross-validation_cv.ipynb`:** Kaggle notebook for A3 cross-validation strategy comparison.
- **`ablation-a4-loss-function_cv.ipynb`:** Kaggle notebook for A4 loss function comparison.
- **`ablation-a5-tta_cv.ipynb`:** Kaggle notebook for A5 test-time augmentation impact.
- **`ablation-a6-vmamba-vs-dinov2_cv.ipynb`:** Kaggle notebook for A6 VMamba vs DINOv2 comparison.

### Submission Notebooks (`notebooks/submission/`):

These notebooks generate inference predictions and submission files for the Kaggle competition.

#### Baselines (`notebooks/submission/baselines/`):

- **`B1_Median_Predictor_submission.ipynb`:** Submission notebook for B1 median predictor.
- **`B2_EfficientNet_B3_submission.ipynb`:** Submission notebook for B2 EfficientNet-B3.
- **`B3_DINOv2_Giant_submission.ipynb`:** Submission notebook for B3 DINOv2-Giant.
- **`B4_DINOv2_Metadata_submission.ipynb`:** Submission notebook for B4 DINOv2-Large.
- **`B5_DINOv3_ViT_L_Metadata_submission.ipynb`:** Submission notebook for B5 DINOv3-ViT-L.
- **`B6_VMamba_Base_Metadata_submission.ipynb`:** Submission notebook for B6 VMamba-Base.

#### Ablation Studies (`notebooks/submission/ablation/`):

- **`ablation-a1-ssm-scale_submission.ipynb`:** Submission notebook for A1 SSM scale comparison.
- **`ablation-a2-metadata_submission.ipynb`:** Submission notebook for A2 metadata ablation.
- **`ablation-a3-cross-validation_submission.ipynb`:** Submission notebook for A3 CV strategy.
- **`ablation-a4-loss-function_submission.ipynb`:** Submission notebook for A4 loss function.
- **`ablation-a5-tta_submission.ipynb`:** Submission notebook for A5 TTA impact.
- **`ablation-a6-vmamba-vs-dinov2_submission.ipynb`:** Submission notebook for A6 VMamba vs DINOv2.

#### Proposed Models (`notebooks/submission/proposed/`):

- **`Proposed_BabyMamba_submission.ipynb`:** Submission notebook for the proposed BabyMamba model.
- **`Proposed_CVGA_submission.ipynb`:** Submission notebook for the proposed CVGA model.

---

## 4. Figures (`img/`):

All figures used in the research paper are stored here.

- **`fig_architecture.png`:** Overall architecture diagram of the dual-view biomass regression framework.
- **`fig_main_results.png`:** Main results table comparing all 17 configurations.
- **`fig_ablation_studies.png`:** Ablation study results visualization.
- **`fig_fold_analysis.png`:** Per-fold CV performance analysis.
- **`fig02_target_distributions.png`:** Distribution of five biomass target variables.
- **`fig03_correlation_heatmap.png`:** Correlation heatmap between biomass components.
- **`fig04_ndvi_height_biomass.png`:** Relationship between NDVI, canopy height, and biomass.
- **`fig05_biomass_by_state.png`:** Biomass variation across Australian states.
- **`fig06_seasonal_dynamics.png`:** Temporal dynamics of biomass across sampling months.
- **`fig07_species_analysis.png`:** Species-level biomass analysis.
- **`fig08_feature_space_analysis.png`:** Feature space visualization of backbone representations.
- **`fig09_evaluation_quality.png`:** Model prediction quality evaluation plots.
- **`fig11_backbone_feature_maps.png`:** Backbone feature map visualizations across architectures.
