"""
Reshape SNP vectors into similarity-score vectors.

• Windows  = 4096 (config.SIMILARITY_WINDOW_COUNT)
• Output   = (N, 4096) array scaled to [0, 1]
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config


RAW_VECTOR_PATH    = config.SNP_VECTOR_NPY
SIM_VEC_OUT        = config.SIMILARITY_VECTOR_NPY
N_WINDOWS          = config.SIMILARITY_WINDOW_COUNT   # 4096

# Load SNP matrix
print("Loading raw SNP vectors…")
snp_matrix = np.load(RAW_VECTOR_PATH)        # (N, M)
N_SAMPLES, N_SNPS = snp_matrix.shape
print("Loaded SNP matrix:", snp_matrix.shape)

# Window statistics
snps_per_window = int(np.ceil(N_SNPS / N_WINDOWS))
print(f"{snps_per_window} SNPs per window × {N_WINDOWS} windows")

# Compute similarity vectors
sim_vectors = np.zeros((N_SAMPLES, N_WINDOWS), dtype=np.float32)

for i in range(N_WINDOWS):
    start = i * snps_per_window
    end   = min((i + 1) * snps_per_window, N_SNPS)
    window_sum = snp_matrix[:, start:end].sum(axis=1)
    sim_vectors[:, i] = window_sum / (2 * (end - start))   # normalize to [0, 1]

np.save(SIM_VEC_OUT, sim_vectors)
print(f"Saved similarity vectors to {SIM_VEC_OUT}   shape = {sim_vectors.shape}")
