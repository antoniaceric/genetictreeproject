# Preprocessing with PLINK2 – filter, QC, LD-prune, export .raw, and plot PCA

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config


# Paths
plink_binary = str(config.PLINK_BINARY)

data_dir      = config.FINAL_IMPUTED_DIR            # Path object
input_prefix  = data_dir / "Final_Imputed_QC"
maf_filtered  = data_dir / "Final_Imputed_filtered_maf"

# Step 1: MAF filter
subprocess.run([
    plink_binary,
    "--bfile", str(input_prefix),
    "--maf", str(config.MAF_THRESHOLD),
    "--make-bed",
    "--out", str(maf_filtered)
], check=True)
print("MAF filtering complete.")

# Step 2: Missingness filter
cleaned_output = data_dir / "Final_Imputed_cleaned"
subprocess.run([
    plink_binary,
    "--bfile", str(maf_filtered),
    "--geno", str(config.GENO_THRESHOLD),   # SNP missingness
    "--mind", str(config.MIND_THRESHOLD),   # sample missingness
    "--make-bed",
    "--out", str(cleaned_output)
], check=True)
print("Missingness filtering complete.")

# Step 3: LD pruning
prune_prefix = data_dir / "Final_Imputed_prune"
subprocess.run([
    plink_binary,
    "--bfile", str(cleaned_output),
    "--indep-pairwise",
    str(config.LD_WINDOW), str(config.LD_STEP), str(config.LD_R2),
    "--out", str(prune_prefix)
], check=True)

ld_pruned = data_dir / "Final_Imputed_LDpruned"
subprocess.run([
    plink_binary,
    "--bfile", str(cleaned_output),
    "--extract", f"{prune_prefix}.prune.in",
    "--make-bed",
    "--out", str(ld_pruned)
], check=True)
print("LD pruning complete.")

# Step 4: Export SNP matrix (.raw)
raw_out = data_dir / "Final_Imputed_raw"
subprocess.run([
    plink_binary,
    "--bfile", str(ld_pruned),
    "--recode", "A",
    "--out", str(raw_out)
], check=True)
print("SNP matrix exported to .raw")

# Step 5: PCA
print("Running PCA on SNP matrix…")
df = pd.read_csv(f"{raw_out}.raw", delim_whitespace=True)

# Drop non-genotype metadata
snp_data = df.drop(columns=["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(snp_data)

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of SNP Genotypes")
plt.tight_layout()
plt.savefig(data_dir / "snp_pca_plot.png")
plt.show()
print("PCA complete. Plot saved.")
