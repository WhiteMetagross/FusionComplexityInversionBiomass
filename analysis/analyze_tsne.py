# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""Quantitative analysis of t-SNE embeddings for paper verification."""
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull
import pandas as pd, os

BASE = "/mnt/c/Users/Xeron/Desktop/ProjectBioMass/csiro-biomass"
df = pd.read_csv(os.path.join(BASE, "train.csv"))
pivot = df.pivot_table(
    index=["image_path","Sampling_Date","State","Species","Pre_GSHH_NDVI","Height_Ave_cm"],
    columns="target_name", values="target"
).reset_index()
quintiles = pd.qcut(pivot["Dry_Total_g"], q=5, labels=False).values
dry_total = pivot["Dry_Total_g"].values

TSNE_PARAMS = dict(n_components=2, perplexity=30, random_state=42,
                   max_iter=1000, learning_rate="auto", init="pca")

backbones = [
    ("EfficientNet-B3", np.load("/tmp/tsne_cache_EfficientNet-B3.npy")),
    ("VMamba-Base", np.load("/tmp/tsne_cache_VMamba-Base.npy")),
    ("DINOv2-ViT-L", np.load("/tmp/tsne_cache_DINOv2-ViT-L.npy")),
    ("DINOv3-ViT-L", np.load("/tmp/tsne_cache_DINOv3-ViT-L.npy")),
]

print("=" * 100)
print(f"{'Backbone':20s} | {'Silhouette':>10s} | {'Rho_dim1':>12s} | {'Rho_dim2':>12s} | {'MaxRho':>8s} | {'Var Ratio':>10s}")
print("-" * 100)

for name, feats in backbones:
    tsne = TSNE(**TSNE_PARAMS)
    emb = tsne.fit_transform(feats.astype(np.float64))

    sil = silhouette_score(emb, quintiles)
    rho1, p1 = spearmanr(emb[:, 0], dry_total)
    rho2, p2 = spearmanr(emb[:, 1], dry_total)
    max_rho = max(abs(rho1), abs(rho2))

    # Between-class / total variance ratio
    centroids = np.array([emb[quintiles == q].mean(axis=0) for q in range(5)])
    within_var = sum(np.var(emb[quintiles == q], axis=0).sum() for q in range(5)) / 5
    between_var = np.var(centroids, axis=0).sum()
    var_ratio = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0

    print(f"{name:20s} | {sil:+10.4f} | {rho1:+.3f}(p{p1:.0e}) | {rho2:+.3f}(p{p2:.0e}) | {max_rho:8.3f} | {var_ratio:10.4f}")

    # Describe what we actually see
    descs = []
    if sil < 0.02:
        descs.append("no quintile clustering")
    elif sil < 0.10:
        descs.append("weak quintile clustering")
    else:
        descs.append("moderate quintile clustering")

    if max_rho < 0.15:
        descs.append("no biomass gradient")
    elif max_rho < 0.30:
        descs.append("weak biomass gradient")
    elif max_rho < 0.50:
        descs.append("moderate biomass gradient")
    else:
        descs.append("strong biomass gradient")

    if var_ratio < 0.02:
        descs.append("disorganized cloud")
    elif var_ratio < 0.08:
        descs.append("loosely structured")
    elif var_ratio < 0.20:
        descs.append("moderately structured")
    else:
        descs.append("well-structured manifold")

    print(f"{'':20s}   -> Observation: {', '.join(descs)}")

print("=" * 100)
print("\nPaper claims:")
print("  EfficientNet-B3: 'disorganized clouds'")
print("  VMamba-Base: (implied intermediate)")
print("  DINOv2-ViT-L: (implied near-structured)")
print("  DINOv3-ViT-L: 'structured manifolds with biomass correlated gradients'")
print("\nAbstract claim: 'progressive organization from disorganized clusters")
print("  (EfficientNet-B3) to smooth biomass correlated manifolds (DINOv3)'")
