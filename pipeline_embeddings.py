# Pipeline Genetic Tree Project
"""
Adjust paths and parameters in `config.py` as needed.
Runs each processing stage unless its final output already exists,
except `prep_ldm.py` which is always executed.
Final output is a latent tensor ready for LDM, reshaped via selected method.
"""

import subprocess
import sys
from pathlib import Path
import config
from typing import Optional


PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent

# Ask user which reshaping method to use
print("Select reshaping method:")
print("  1 = PCA")
print("  2 = Similarity Scores")
method = input("Enter choice [1/2]: ").strip()

if method == "1":
    reshape_script = "reshape_snp_vectors_pca.py"
    sentinel = config.SNP_IMAGE_NPY
    config.USE_PCA = True
    config.LATENT_TENSOR_PT = config.LATENT_TENSOR_PCA_PT
    config.SNP_IMAGE_NPY = config.SNP_IMAGE_NPY
elif method == "2":
    reshape_script = "z_transformation.py"
    sentinel = config.ZSCORE_VECTOR_NPY
    config.USE_PCA = False
    config.LATENT_TENSOR_PT = config.LATENT_TENSOR_SIM_PT
    config.SNP_IMAGE_NPY = config.ZSCORE_VECTOR_NPY
else:
    raise ValueError("Invalid input. Please enter 1 or 2.")

# pipeline steps
STEPS = []
if not (config.FINAL_IMPUTED_DIR / "Final_Imputed_raw.raw").exists():
    STEPS.append(("embeddings/preprocessing.py", config.FINAL_IMPUTED_DIR / "Final_Imputed_raw.raw"))
if not config.SNP_VECTOR_NPY.exists():
    STEPS.append(("embeddings/prep_embeddings.py", config.SNP_VECTOR_NPY))

if method == "2":
    STEPS += [
        ("embeddings/reshape_snp_vectors_similarity.py", config.SIMILARITY_VECTOR_NPY),
        ("embeddings/z_transformation.py", config.ZSCORE_VECTOR_NPY),
    ]
else:
    STEPS += [
        ("embeddings/reshape_snp_vectors_pca.py", config.SNP_IMAGE_NPY),
    ]

# Always run prep_ldm.py (not skipped)
STEPS.append(("embeddings/prep_ldm.py", None))

def run_step(script: str, sentinel: Optional[Path]) -> None:
    if sentinel is not None and sentinel.exists():
        print(f"Skipping {script} — {sentinel.name} already exists.")
        return

    print(f"\n=== Running {script} ===")
    result = subprocess.run([PYTHON, str(SCRIPT_DIR / script)])
    if result.returncode != 0:
        raise RuntimeError(f"{script} exited with code {result.returncode}")

if __name__ == "__main__":
    for script, sentinel in STEPS:
        run_step(script, sentinel)

    print("\nPipeline finished successfully.")
