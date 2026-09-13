# Paper Figure Reproducibility Audit:

## Scope:

- Paper: `2603.07819v5.pdf`.
- Paper SHA256: `162aff659721ce26f14e4b932a9be8499f4dbe98894c860e5f38e99e91a6bc6e`.
- Repository commit inspected: `676705374f2ad24bf11dcd75db145b14c383a743` plus the working tree changes listed below.
- Canonical paper raster assets: `paper/img/`.
- Repository figure assets: `FusionComplexityInversionBiomass/img/`.
- Runtime: WSL, Conda environment `mambahar`, Python 3.11, Matplotlib 3.10.8, Seaborn 0.13.2, Pillow 12.1.1, NumPy 2.4.4, and SciPy 1.17.1.

## Evidence:

- LaTeX references 13 PNG figures at `paper/MandalBiomassPaper.tex:127,228,259,268,279,286,293,363,402,477,484,548,562`.
- The 13 repository PNG files and the 13 `paper/img/` PNG files have equal dimensions, equal SHA256 hashes, and equal pixels.
- The 13 raster images extracted directly from the nine page arXiv v5 PDF have equal dimensions and equal RGBA pixels to `paper/img/`. PDF recompression changes file hashes but not pixels.
- All 13 canonical PNG files were inspected visually. No clipped labels, missing panels, blank images, or raster corruption were found.
- All 13 SVG files parse as valid XML.

## Generator Map:

- `analysis/dataset_analysis_v3.py` covers `fig02` through `fig09`.
- `analysis/generate_paper_figures.py` covers `fig_main_results`, `fig_ablation_studies`, and `fig_fold_analysis`.
- `analysis/generate_architecture_diagram.py` covers `fig_architecture`.
- `analysis/extract_feature_maps.py` covers `fig11_backbone_feature_maps`.

## Rerun Results:

- `dataset_analysis_v3.py` reproduced seven of its eight paper PNG files byte for byte.
- Its original `fig05_biomass_by_state` output had equal dimensions and all equal scientific values, but different strip point positions. Cause: Seaborn jitter was unseeded. The default save path now exports the exact reviewed PNG and SVG master. Set `PAPER_FIGURE_RECOMPUTE_UNSEEDED_JITTER=1` only for an exploratory redraw.
- `generate_paper_figures.py` reproduced all three PNG files byte for byte.
- The Matplotlib architecture draft was not the published layout. It produced `7878 x 3534`, while the paper uses `7878 x 3533`, and had materially smaller text and blocks. The exact final Inkscape SVG master was recovered from `C:\Users\Xeron\Desktop\ProjectBioMass\figures\svg\fig_architecture.svg`. Its paired PNG is byte identical to the paper PNG. The architecture script now exports these exact masters by default. Use `--recompute-draft` only to redraw the pre Inkscape draft.
- The historical `fig08_feature_space_analysis.svg` was unrelated. It contained t SNE backbone plots and had a 21.2 percent aspect ratio mismatch. It was replaced by the SVG produced in the same run that reproduced the paper PNG exactly.
- `extract_feature_maps.py` was left unchanged. Per user direction, no fresh model forward pass, model download, or feature map replacement was performed. The existing PNG stayed unchanged. Its matching archival SVG was recovered and hash verified. This is artifact verification, not independent model recomputation.

## Repairs:

- Added the exact SVG masters beside the canonical PNG files under `img/`.
- Added `analysis/paper_figure_manifest.json` with dimensions, generator provenance, and PNG and SVG SHA256 hashes.
- Added `analysis/export_paper_figures.py` for exact, hash checked export of all 13 PNG and SVG pairs.
- Added clone safe `BIOMASS_DATA_DIR`, `PAPER_FIGURE_OUTPUT_DIR`, and `PAPER_FIGURE_ASSET_DIR` path controls to the verified generators.
- Preserved paper text and LaTeX without modification.

## Exact Export Command:

```bash
cd /mnt/c/Users/Xeron/Desktop/IIIT-ACourseWork/IMLProject/FusionComplexityInversionBiomass
/home/xeron/miniconda3/bin/conda run --no-capture-output -n mambahar \
  python analysis/export_paper_figures.py
```

The command writes the verified files to `output/paper_figures/png/` and `output/paper_figures/svg/`.

## Scientific Recompute Commands:

```bash
cd /mnt/c/Users/Xeron/Desktop/IIIT-ACourseWork/IMLProject/FusionComplexityInversionBiomass
BIOMASS_DATA_DIR=/mnt/c/Users/Xeron/Desktop/IIIT-ACourseWork/IMLProject/csiro-biomass \
PAPER_FIGURE_OUTPUT_DIR=output/analysis \
/home/xeron/miniconda3/bin/conda run --no-capture-output -n mambahar \
  python analysis/dataset_analysis_v3.py

PAPER_FIGURE_OUTPUT_DIR=output/analysis \
/home/xeron/miniconda3/bin/conda run --no-capture-output -n mambahar \
  python analysis/generate_paper_figures.py

PAPER_FIGURE_OUTPUT_DIR=output/analysis \
/home/xeron/miniconda3/bin/conda run --no-capture-output -n mambahar \
  python analysis/generate_architecture_diagram.py
```

## Interpretation:

- Exact export reproduces the reviewed paper artifacts byte for byte. It does not recompute model activations or statistical values.
- Scientific recomputation verifies data and plotting logic. It may not reproduce SVG bytes because SVG metadata and element identifiers depend on the plotting runtime.
- The feature map model computation remains unverified by explicit user choice. Its frozen paper artifact is verified exactly and remains unchanged.
