# Figure generation entry points:

Each Python file regenerates the PNG and SVG pair with the same base name. Run the files from the repository root inside the `mambahar` Conda environment.

Figures 3 through 6 read the verified aggregate and fold files under `output/checkpoint_integrity_2026_09_14` and `output/classical_baselines_2026_09_15`. Create those files with the checkpoint and classical evaluation commands in `Usage.md`. Appendix figures A1 through A6 read the CSIRO data from `csiro-biomass` by default. Set `BIOMASS_DATA_DIR` to use another local dataset path.

Figure 1 needs the original SVG, original PNG and a rasterized revised SVG. Figure 2 needs the checkpoints, split file and dataset paths accepted by `analysis/generate_model_attributions.py`. Both scripts expose their required command line arguments through `--help`.

Example:

```bash
conda run --no-capture-output -n mambahar python img/codes/fig03_benchmark_overview.py
conda run --no-capture-output -n mambahar python img/codes/figA1_correlation_heatmap.py
```

Use `analysis/export_paper_figures.py --verify-only` after generation to compare the assets with the reviewed manifest.
