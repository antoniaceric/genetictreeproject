# Compute pairwise distances between SNP embeddings
# using Euclidean distance and correlation

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config
import os

# Load embeddings (PCA or similarity-based)
embedding_path = config.ZSCORE_VECTOR_NPY  # or config.SNP_IMAGE_NPY for PCA
Z = np.load(embedding_path)  # Shape: (N, d)

print(f"Loaded embeddings from: {embedding_path}")
print(f"Embedding shape: {Z.shape}")

# Euclidean distance
euclid_dist = euclidean_distances(Z, Z)  # (N, N)
euclid_out_path = config.EMBEDDING_DIR / "embedding_euclidean_distances.npy"
np.save(euclid_out_path, euclid_dist)
print(f"Saved Euclidean distances to: {euclid_out_path}")

# Correlation distance
# pdist gives condensed (1D) format, squareform converts to square
corr_dist = squareform(pdist(Z, metric="correlation"))  # (N, N)
corr_out_path = config.EMBEDDING_DIR / "embedding_correlation_distances.npy"
np.save(corr_out_path, corr_dist)
print(f"Saved Correlation distances to: {corr_out_path}")
