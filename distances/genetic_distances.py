# compute distances between genotypes

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config

# Run PLINK to compute PI_HAT
plink_bin = config.PLINK1_BINARY
bfile = config.FINAL_IMPUTED_DIR / "Final_Imputed_LDpruned"
genome_out = config.GENOME_OUT_FILE

print("Running PLINK --genome to compute PI_HAT...")
subprocess.run([
    str(plink_bin),
    "--bfile", str(bfile),
    "--genome",
    "--out", str(genome_out).replace(".genome", "")
], check=True)

# Convert to distance matrix
print("Parsing PI_HAT into genetic distance matrix...")

df = pd.read_csv(genome_out, delim_whitespace=True)
samples = sorted(set(df["IID1"]).union(set(df["IID2"])))
idx_map = {iid: i for i, iid in enumerate(samples)}
N = len(samples)

dist = np.ones((N, N))  # default distance = 1
for _, row in df.iterrows():
    i, j = idx_map[row["IID1"]], idx_map[row["IID2"]]
    pi_hat = row["PI_HAT"]
    dist[i, j] = dist[j, i] = 1 - pi_hat

np.save(config.GENETIC_DISTANCE_NPY, dist)
np.save(config.SAMPLE_ORDER_NPY, np.array(samples))
print("Saved:")
print(f" - Genetic distance matrix: {config.GENETIC_DISTANCE_NPY}")
print(f" - Sample ID order: {config.SAMPLE_ORDER_NPY}")

# Filter for likely siblings (PI_HAT close to 0.5)
sibling_candidates = df[df["PI_HAT"] >= 0.45].copy()
sibling_candidates = sibling_candidates[["FID1", "IID1", "FID2", "IID2", "PI_HAT"]]

print("Likely sibling pairs (PI_HAT ≥ 0.45):")
print(sibling_candidates)
