# SNP Embedding Vector Preparation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config

# Paths
raw_file = config.FINAL_IMPUTED_DIR / "Final_Imputed_raw.raw"
embedding_output_dir = config.EMBEDDING_DIR
os.makedirs(embedding_output_dir, exist_ok=True)

print("Loading SNP genotype data…")
df = pd.read_csv(raw_file, delim_whitespace=True)

# Drop metadata
snp_df = df.drop(columns=["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"])

# Fill missing values (e.g., mean imputation)
print("Handling missing data…")
snp_df_imputed = snp_df.fillna(snp_df.mean())

# Normalize genotype matrix
print("Normalizing genotype values…")
scaler = StandardScaler()
snp_normalized = scaler.fit_transform(snp_df_imputed)

# Save individual SNP vectors
individual_ids = df["IID"].values
embeddings_path = embedding_output_dir / "snp_vectors.npy"
ids_path = embedding_output_dir / "sample_ids.txt"

np.save(embeddings_path, snp_normalized)
with open(ids_path, "w") as f:
    for iid in individual_ids:
        f.write(iid + "\n")

print(f"SNP vectors saved to: {embeddings_path}")
print(f"Corresponding IDs saved to: {ids_path}")
print(f"Total SNP features available: {snp_normalized.shape[1]}")
