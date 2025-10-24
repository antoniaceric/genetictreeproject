# compare genetic, embedding and image distances

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config
from plot_func import plot_multiple_distance_correlations
import seaborn as sns
import pandas as pd
from plot_func import plot_sankey_agreement, plot_multi_sankey_grid


# Load distance matrices
gen_dist = np.load(config.GENETIC_DISTANCE_NPY)
img_dist = np.load(config.IMAGE_DISTANCE_NPY)              # SSIM-based
rgb_dist = np.load(config.IMAGE_HIST_DISTANCE_NPY)         # RGB histogram-based
chi2_dist = np.load(config.IMAGE_CV_HIST_CHISQR_NPY)       # histogram chi squared based
dense_dist = np.load(config.IMAGE_DENSE_COSINE_NPY)        # dense cosine similarity based
emb_euclid = np.load(config.EMBEDDING_EUCLIDEAN_DISTANCE_NPY) # embedding distances euclidean distance based
emb_corr   = np.load(config.EMBEDDING_CORRELATION_DISTANCE_NPY) # embedding distance correlation distance based
sample_ids = np.load(config.SAMPLE_ORDER_NPY)

# Compute correlation on the FULL dataset (before subsetting)
print("\nCorrelation Genetic vs Embeddings (FULL SAMPLE)")
full_gen_dist = np.load(config.GENETIC_DISTANCE_NPY)
full_emb_euclid = np.load(config.EMBEDDING_EUCLIDEAN_DISTANCE_NPY)
full_emb_corr = np.load(config.EMBEDDING_CORRELATION_DISTANCE_NPY)

# Flatten upper triangles
i_upper = np.triu_indices_from(full_gen_dist, k=1)
flat_full_gen = full_gen_dist[i_upper]
flat_full_euclid = full_emb_euclid[i_upper]
flat_full_corr = full_emb_corr[i_upper]

# Compute Spearman correlations
rho_full_euclid, pval_full_euclid = spearmanr(flat_full_gen, flat_full_euclid)
rho_full_corr, pval_full_corr = spearmanr(flat_full_gen, flat_full_corr)

print(f"Spearman (FULL) Genetic vs Embedding Euclidean:  rho = {rho_full_euclid:.4f}, p = {pval_full_euclid:.4g}")
print(f"Spearman (FULL) Genetic vs Embedding Correlation: rho = {rho_full_corr:.4f}, p = {pval_full_corr:.4g}")

# Permutation test for statistical significance
def permutation_test(reference_labels, test_labels, score_func, n_permutations=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    observed_score = score_func(reference_labels, test_labels)
    permuted_scores = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(test_labels)
        permuted_scores[i] = score_func(reference_labels, permuted)
    p_val = (np.sum(permuted_scores >= observed_score) + 1) / (n_permutations + 1)
    return observed_score, permuted_scores, p_val

# Cluster agreement on FULL SAMPLE (with permutation tests)
def compare_clusters(dist_matrix, n_clusters=9):
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    return model.fit_predict(dist_matrix)

print("\n=== Cluster Agreement (FULL SAMPLE) ===")
labels_gen_full = compare_clusters(full_gen_dist)
labels_emb_euclid_full = compare_clusters(full_emb_euclid)
labels_emb_corr_full = compare_clusters(full_emb_corr)

# Euclidean
ari_full_euclid, _, p_ari_full_euclid = permutation_test(labels_gen_full, labels_emb_euclid_full, adjusted_rand_score)
nmi_full_euclid, _, p_nmi_full_euclid = permutation_test(labels_gen_full, labels_emb_euclid_full, normalized_mutual_info_score)
print(f"FULL Cluster agreement (Embedding Euclidean): ARI = {ari_full_euclid:.4f}, p = {p_ari_full_euclid:.4g}")
print(f"FULL Cluster agreement (Embedding Euclidean): NMI = {nmi_full_euclid:.4f}, p = {p_nmi_full_euclid:.4g}")

# Correlation
ari_full_corr, _, p_ari_full_corr = permutation_test(labels_gen_full, labels_emb_corr_full, adjusted_rand_score)
nmi_full_corr, _, p_nmi_full_corr = permutation_test(labels_gen_full, labels_emb_corr_full, normalized_mutual_info_score)
print(f"FULL Cluster agreement (Embedding Correlation): ARI = {ari_full_corr:.4f}, p = {p_ari_full_corr:.4g}")
print(f"FULL Cluster agreement (Embedding Correlation): NMI = {nmi_full_corr:.4f}, p = {p_nmi_full_corr:.4g}")


# Match subjects based on image filenames
image_dir = config.IMAGE_SAMPLE_DIR
image_files = sorted([
    f for f in os.listdir(image_dir) if f.startswith("subject_") and f.endswith(".png")
])

# Extract subject indices (e.g., subject_009 → 9)
subject_indices = sorted(set(int(f.split("_")[1]) for f in image_files))

# Sanity check: filter out invalid indices
max_index = gen_dist.shape[0]
subject_indices = [i for i in subject_indices if i < max_index]

print(f"Found {len(subject_indices)} subject images → indices: {subject_indices[:5]}...")

# Subset only the genetic and embedding distance matrices (image matrices are already subsetted and ordered)
gen_dist = gen_dist[np.ix_(subject_indices, subject_indices)]
emb_euclid = emb_euclid[np.ix_(subject_indices, subject_indices)]
emb_corr   = emb_corr[np.ix_(subject_indices, subject_indices)]

# Assert shapes match
assert gen_dist.shape == img_dist.shape == rgb_dist.shape == chi2_dist.shape == dense_dist.shape, \
    "Matrix shapes do not align. Check that image distance matrices match number of image subjects."
assert emb_euclid.shape == gen_dist.shape == emb_corr.shape, "Embedding distance shape mismatch."

# Flatten upper triangles for Spearman correlation
flat_gen = gen_dist[np.triu_indices_from(gen_dist, k=1)]
flat_img = img_dist[np.triu_indices_from(img_dist, k=1)]
flat_rgb = rgb_dist[np.triu_indices_from(rgb_dist, k=1)]
flat_chi2 = chi2_dist[np.triu_indices_from(chi2_dist, k=1)]
flat_dense = dense_dist[np.triu_indices_from(dense_dist, k=1)]
flat_emb_euclid = emb_euclid[np.triu_indices_from(emb_euclid, k=1)]
flat_emb_corr   = emb_corr[np.triu_indices_from(emb_corr, k=1)]

# Spearman correlations
rho, pval = spearmanr(flat_gen, flat_img)
print(f"Spearman correlation (SSIM): rho = {rho:.4f}, p = {pval:.4g}")

rho_rgb, pval_rgb = spearmanr(flat_gen, flat_rgb)
print(f"Spearman correlation (RGB hist): rho = {rho_rgb:.4f}, p = {pval_rgb:.4g}")

rho_chi2, pval_chi2 = spearmanr(flat_gen, flat_chi2)
print(f"Spearman correlation (CHI2): rho = {rho_chi2:.4f}, p = {pval_chi2:.4g}")

rho_dense, pval_dense = spearmanr(flat_gen, flat_dense)
print(f"Spearman correlation (DENSE COSINE): rho = {rho_dense:.4f}, p = {pval_dense:.4g}")

rho_emb_euclid, pval_emb_euclid = spearmanr(flat_gen, flat_emb_euclid)
print(f"Spearman (genetic vs embedding - Euclidean):  rho = {rho_emb_euclid:.4f}, p = {pval_emb_euclid:.4g}")

rho_emb_corr,   pval_emb_corr   = spearmanr(flat_gen, flat_emb_corr)
print(f"Spearman (genetic vs embedding - Correlation): rho = {rho_emb_corr:.4f}, p = {pval_emb_corr:.4g}")


print("\n=== Cluster Agreement (FULL SAMPLE) for varying k ===")
for k in range(2, 15):
    # Genetic clustering as reference
    labels_gen_k = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average').fit_predict(full_gen_dist)
    
    # Embedding Euclidean
    labels_emb_euclid_k = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average').fit_predict(full_emb_euclid)
    ari_euclid = adjusted_rand_score(labels_gen_k, labels_emb_euclid_k)
    nmi_euclid = normalized_mutual_info_score(labels_gen_k, labels_emb_euclid_k)

    # Embedding Correlation
    labels_emb_corr_k = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average').fit_predict(full_emb_corr)
    ari_corr = adjusted_rand_score(labels_gen_k, labels_emb_corr_k)
    nmi_corr = normalized_mutual_info_score(labels_gen_k, labels_emb_corr_k)

    print(f"k = {k} → Euclid ARI: {ari_euclid:.4f}, NMI: {nmi_euclid:.4f} | Corr ARI: {ari_corr:.4f}, NMI: {nmi_corr:.4f}")


# Clustering comparison
def compare_clusters(dist_matrix, name):
    model = AgglomerativeClustering(n_clusters=9, affinity='precomputed', linkage='average')
    labels = model.fit_predict(dist_matrix)
    return labels

labels_gen = compare_clusters(gen_dist, "genetic")
labels_img = compare_clusters(img_dist, "image (SSIM)")
labels_rgb = compare_clusters(rgb_dist, "image (RGB HIST)")
labels_chi2 = compare_clusters(chi2_dist, "image (CHI2)")
labels_dense = compare_clusters(dense_dist, "image (DENSE COSINE)")

# SSIM-based cluster agreement
ari_obs, _, ari_p = permutation_test(labels_gen, labels_img, adjusted_rand_score)
nmi_obs, _, nmi_p = permutation_test(labels_gen, labels_img, normalized_mutual_info_score)

print(f"Cluster agreement (SSIM): ARI = {ari_obs:.4f}, p = {ari_p:.4g}")
print(f"Cluster agreement (SSIM): NMI = {nmi_obs:.4f}, p = {nmi_p:.4g}")

# RGB hist-based cluster agreement
ari_rgb_obs, _, ari_rgb_p = permutation_test(labels_gen, labels_rgb, adjusted_rand_score)
nmi_rgb_obs, _, nmi_rgb_p = permutation_test(labels_gen, labels_rgb, normalized_mutual_info_score)

print(f"Cluster agreement (RGB hist): ARI = {ari_rgb_obs:.4f}, p = {ari_rgb_p:.4g}")
print(f"Cluster agreement (RGB hist): NMI = {nmi_rgb_obs:.4f}, p = {nmi_rgb_p:.4g}")

# CHI2-based cluster agreement
ari_chi2_obs, _, ari_chi2_p = permutation_test(labels_gen, labels_chi2, adjusted_rand_score)
nmi_chi2_obs, _, nmi_chi2_p = permutation_test(labels_gen, labels_chi2, normalized_mutual_info_score)

print(f"Cluster agreement (CHI2): ARI = {ari_chi2_obs:.4f}, p = {ari_chi2_p:.4g}")
print(f"Cluster agreement (CHI2): NMI = {nmi_chi2_obs:.4f}, p = {nmi_chi2_p:.4g}")

# DENSE COSINE hist-based cluster agreement
ari_dense_obs, _, ari_dense_p = permutation_test(labels_gen, labels_dense, adjusted_rand_score)
nmi_dense_obs, _, nmi_dense_p = permutation_test(labels_gen, labels_dense, normalized_mutual_info_score)

print(f"Cluster agreement (DENSE COSINE): ARI = {ari_dense_obs:.4f}, p = {ari_dense_p:.4g}")
print(f"Cluster agreement (DENSE COSINE): NMI = {nmi_dense_obs:.4f}, p = {nmi_dense_p:.4g}")

# EMBEDDING EUCLIDEAN-based cluster agreement
labels_emb_euclid = compare_clusters(emb_euclid, "embedding-euclidean")
ari_emb_euclid, _, ari_emb_euclid_p = permutation_test(labels_gen, labels_emb_euclid, adjusted_rand_score)
nmi_emb_euclid, _, nmi_emb_euclid_p = permutation_test(labels_gen, labels_emb_euclid, normalized_mutual_info_score)

print(f"Cluster agreement (embedding - Euclidean): ARI = {ari_emb_euclid:.4f}, p = {ari_emb_euclid_p:.4g}")
print(f"Cluster agreement (embedding - Euclidean): NMI = {nmi_emb_euclid:.4f}, p = {nmi_emb_euclid_p:.4g}")

# EMBEDDING CORRELATION-based cluster agreement
labels_emb_corr = compare_clusters(emb_corr, "embedding-correlation")
ari_emb_corr, _, ari_emb_corr_p = permutation_test(labels_gen, labels_emb_corr, adjusted_rand_score)
nmi_emb_corr, _, nmi_emb_corr_p = permutation_test(labels_gen, labels_emb_corr, normalized_mutual_info_score)

print(f"Cluster agreement (embedding - Correlation): ARI = {ari_emb_corr:.4f}, p = {ari_emb_corr_p:.4g}")
print(f"Cluster agreement (embedding - Correlation): NMI = {nmi_emb_corr:.4f}, p = {nmi_emb_corr_p:.4g}")

## PLOTS

# Figure: Scatterplot all distance metrics
PLOT_DIR = config.FINAL_IMPUTED_DIR / "distance_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

distances_to_plot = {
    "SSIM": img_dist,
    "RGB hist": rgb_dist,
    "CHI2 hist": chi2_dist,
    "Dense cosine": dense_dist,
    "Embedding Euclidean": emb_euclid,
    "Embedding Correlation": emb_corr,
}

plot_multiple_distance_correlations(
    gen_dist,
    distances_to_plot,
    ncols=3,
    save_path=PLOT_DIR / "genetic_vs_all_distances.png"
)

plot_multiple_distance_correlations(
    gen_dist,
    distances_to_plot,
    ncols=3,
    save_path=PLOT_DIR / "genetic_vs_all_distances_HIGHGEN.png",
    filter_high_gen_dist=True
)

# Figure: Clustering label correlation heatmap

cluster_df = pd.DataFrame({
    "Genetic": labels_gen,
    "SSIM": labels_img,
    "RGB hist": labels_rgb,
    "CHI2 hist": labels_chi2,
    "Dense cosine": labels_dense,
    "Embed Euclidean": labels_emb_euclid,
    "Embed Correlation": labels_emb_corr
})

label_corr = cluster_df.corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(label_corr, annot=True, cmap="coolwarm", vmin=0, vmax=1, square=True)
plt.title("Clustering Label Correlations")
plt.tight_layout()
heatmap_path = PLOT_DIR / "clustering_label_correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Heatmap saved to: {heatmap_path}")

import numpy as np
import matplotlib.pyplot as plt

# Output directory for plots
from pathlib import Path
import config

PLOT_DIR = config.FINAL_IMPUTED_DIR / "embedding_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Sankey chart clustering agreements
plot_sankey_agreement(labels_gen, labels_img, "SSIM",save_dir=config.FINAL_IMPUTED_DIR / "distance_plots" )
plot_sankey_agreement(labels_gen, labels_rgb, "RGB hist", save_dir=config.FINAL_IMPUTED_DIR / "distance_plots")
plot_sankey_agreement(labels_gen, labels_chi2, "CHI2 hist", save_dir=config.FINAL_IMPUTED_DIR / "distance_plots")
plot_sankey_agreement(labels_gen, labels_dense, "Dense cosine", save_dir=config.FINAL_IMPUTED_DIR / "distance_plots")
plot_sankey_agreement(labels_gen, labels_emb_euclid, "Embedding Euclidean", save_dir=config.FINAL_IMPUTED_DIR / "embedding_plots")
plot_sankey_agreement(labels_gen, labels_emb_corr, "Embedding Correlation", save_dir=config.FINAL_IMPUTED_DIR / "embedding_plots")

# Multi-sankey grid
label_dict = {
    "SSIM": labels_img,
    "RGB hist": labels_rgb,
    "CHI2 hist": labels_chi2,
    "Dense cosine": labels_dense,
    "Embedding Euclidean": labels_emb_euclid,
    "Embedding Correlation": labels_emb_corr,
}

method_order = list(label_dict.keys())
sankey_save_path = Path(config.FINAL_IMPUTED_DIR) / "distance_plots" / "sankey_grid.png"
sankey_save_path.parent.mkdir(parents=True, exist_ok=True)

plot_multi_sankey_grid(labels_gen, label_dict, method_order, sankey_save_path)
