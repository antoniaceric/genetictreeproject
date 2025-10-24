"""
Runs the full genotype-to-image similarity correlation pipeline:
1. Computes PI_HAT-based genetic distances via PLINK (--genome)
2. Computes embedding distances (Euclidean, Correlation)
3. Computes image distances (SSIM, RGB-hist, chi-square hist, dense cosine)
4. Correlates all with genetic distances via Spearman and cluster agreements and saves results
"""

import subprocess
import sys
PYTHON = sys.executable
from pathlib import Path

SCRIPTS = {
    "genetic": "distances/genetic_distances.py",
    "embedding": "distances/embedding_distances.py",
    "image":   "distances/images_distances.py",
    "compare": "distances/compare_distances.py",
}

PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent

def run(script_name):
    path = SCRIPT_DIR / SCRIPTS[script_name]
    print(f"\n=== Running {script_name.upper()} ===")
    result = subprocess.run([PYTHON, str(path)])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} script failed with code {result.returncode}")

if __name__ == "__main__":
    print("Starting full genotype-to-image similarity analysis...")

    run("genetic")
    run("embedding")
    run("image")
    run("compare")

    print("\nAll steps completed successfully.")
