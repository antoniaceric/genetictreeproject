"""
PCA-reduce each SNP vector to 4 × 16 × 16 = 1024 components and
save a *flat* (N, 1024) array.  `prep_ldm.py` will reshape it to
(N, 4, 16, 16) for the diffusion model.

UNDER CONSTRUCTION; NOT FUNCTIONING!

"""

import numpy as np
from sklearn.decomposition import PCA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config


# Paths
input_path   = config.SNP_VECTOR_NPY
output_path  = config.SNP_IMAGE_NPY

# Target dimensionality
C, H, W      = config.PCA_TARGET_SHAPE            # (4,16,16)
n_components = C * H * W                          # 1024

print(f"PCA components: {n_components}  →  target latent shape ({C}, {H}, {W})")

# Load data
print("Loading SNP vectors…")
X = np.load(input_path)
print("Loaded array shape:", X.shape)

# PCA reduction
print("Fitting PCA…")
pca = PCA(n_components=n_components, random_state=0)
X_reduced = pca.fit_transform(X)
print("Explained variance (first 5):", pca.explained_variance_ratio_[:5])

# Save flat result
np.save(output_path, X_reduced.astype(np.float32))
print(f"Saved flat PCA vectors to {output_path}  shape = {X_reduced.shape}")

