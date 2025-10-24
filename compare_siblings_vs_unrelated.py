import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
import re

# Load all distance matrices
genetic_full = np.load(config.GENETIC_DISTANCE_NPY)
embedding_euclid_full = np.load(config.EMBEDDING_EUCLIDEAN_DISTANCE_NPY)
embedding_corr_full = np.load(config.EMBEDDING_CORRELATION_DISTANCE_NPY)
image_ssim = np.load(config.IMAGE_DISTANCE_NPY)
image_rgb = np.load(config.IMAGE_HIST_DISTANCE_NPY)
image_chi2 = np.load(config.IMAGE_CV_HIST_CHISQR_NPY)
image_dense = np.load(config.IMAGE_DENSE_COSINE_NPY)

# Detect subjects with images
image_dir = config.IMAGE_SAMPLE_DIR
pattern = re.compile(r"subject_(\d+)_sample_\d+\.png")
image_files = list(image_dir.glob("subject_*_sample_*.png"))
image_indices = sorted({
    int(pattern.match(f.name).group(1).lstrip("0"))
    for f in image_files if pattern.match(f.name)
})
print(f"Detected {len(image_indices)} subject images.")

# Subset full matrices to image subjects
genetic = genetic_full[np.ix_(image_indices, image_indices)]
embedding_euclid = embedding_euclid_full[np.ix_(image_indices, image_indices)]
embedding_corr = embedding_corr_full[np.ix_(image_indices, image_indices)]

# Get top 20 most similar and dissimilar pairs
N = len(image_indices)
i_upper = np.triu_indices(N, k=1)
pairwise_genetic = list(zip(i_upper[0], i_upper[1], genetic[i_upper]))
sorted_pairs = sorted(pairwise_genetic, key=lambda x: x[2])

most_similar = [(i, j) for i, j, _ in sorted_pairs[:20]]
most_dissimilar = [(i, j) for i, j, _ in sorted_pairs[-20:]]

# Prepare all metric matrices for plotting
metric_matrices = {
    "Embedding Euclidean": embedding_euclid,
    "Embedding Correlation": embedding_corr,
    "SSIM": image_ssim,
    "RGB Histogram": image_rgb,
    "Chi2 Histogram": image_chi2,
    "Dense Cosine": image_dense,
}

# Plot multi-panel scatterplot
ncols = 3
nplots = len(metric_matrices)
nrows = (nplots + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
axes = axes.flatten()

for ax, (label, matrix) in zip(axes, metric_matrices.items()):
    sim_vals = [matrix[i, j] for i, j in most_similar]
    dis_vals = [matrix[i, j] for i, j in most_dissimilar]

    u_stat, pval = mannwhitneyu(sim_vals, dis_vals, alternative="two-sided")

    # Compute Rank-Biserial Correlation (effect size)
    n1 = len(sim_vals)
    n2 = len(dis_vals)
    rbc = 1 - (2 * u_stat) / (n1 * n2)

    # Print results
    print("Top Similar:", sorted(sim_vals))
    print("Top Dissimilar:", sorted(dis_vals))
    print(f"\n=== {label} ===")
    print(f"  Similar     → mean = {np.mean(sim_vals):.4f}, SD = {np.std(sim_vals):.4f}")
    print(f"  Dissimilar  → mean = {np.mean(dis_vals):.4f}, SD = {np.std(dis_vals):.4f}")
    print(f"  Mann–Whitney U: {u_stat:.4f}, p = {pval:.6f}")
    print(f"  Rank-Biserial Correlation = {rbc:.4f}")
