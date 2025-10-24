# CONFIG FILE GENETIC TREES

"""
Settings for Genetic Tree Project

Adapt paths and processing parameters as needed.
(`pipeline.py`, `preprocessing.py`, `prep_embedding.py`, `reshape_snp_vectors_[method].py`,
and `prep_ldm.py` import from here.)
"""
# CONFIG FILE GENETIC TREES
"""
Central settings for the Genetic Tree Project.
All pipeline stages import paths and hyper-parameters from here.
"""

from pathlib import Path

# Root & directory structure
ROOT_DIR              = Path("/files/data")

FINAL_IMPUTED_DIR     = ROOT_DIR / "Final_Imputed"
EMBEDDING_DIR         = FINAL_IMPUTED_DIR / "snp_embeddings"
LDM_DATASET_DIR       = ROOT_DIR / "ldm_dataset"
LDM_IMAGE_DIR         = LDM_DATASET_DIR / "snp_images_similarity"  # legacy PNGs (optional)

# External binaries
PLINK_BINARY          = Path("/opt/conda/envs/ldm/bin/plink2")

# Pre-processing thresholds (PLINK)
MAF_THRESHOLD         = 0.2
GENO_THRESHOLD        = 0.02
MIND_THRESHOLD        = 0.02
LD_WINDOW             = 50
LD_STEP               = 5
LD_R2                 = 0.2

# Embedding stage
SNP_VECTOR_NPY        = EMBEDDING_DIR / "snp_vectors.npy"
SAMPLE_IDS_TXT        = EMBEDDING_DIR / "sample_ids.txt"

# Similarity-score reshaping
# 4096 = 4 × 32 × 32  (matches LDM latent cube)
SIMILARITY_WINDOW_COUNT = 4096
SIMILARITY_VECTOR_NPY   = EMBEDDING_DIR / "snp_similarity_vectors.npy"
ZSCORE_VECTOR_NPY       = EMBEDDING_DIR / "snp_similarity_zscore_vectors.npy"

# Latent-tensor output (Similarity branch only)
LATENT_TENSOR_SIM_PT  = LDM_DATASET_DIR / "latents_batch_similarity.pt"
LATENT_TENSOR_PT      = LATENT_TENSOR_SIM_PT   # set once here

# Latent cube shape seen by diffusion model
PCA_TARGET_SHAPE      = (4, 32, 32)   # used *only* to tell prep_ldm.py the target cube

# Genetic similarity config
GENOME_OUT_FILE       = FINAL_IMPUTED_DIR / "genetic_relatedness.genome"
GENETIC_DISTANCE_NPY  = FINAL_IMPUTED_DIR / "genetic_distances.npy"
SAMPLE_ORDER_NPY      = FINAL_IMPUTED_DIR / "sample_order.npy"
PLINK1_BINARY = Path("/opt/conda/envs/ldm/bin/plink")

# Directory where generated images are saved
IMAGE_SAMPLE_DIR = Path("/files/latent-diffusion/outputs/txt2img-samples/samples")

# Output path for image distance matrix
IMAGE_DISTANCE_NPY = FINAL_IMPUTED_DIR / "image_ssim_distances.npy"
IMAGE_HIST_DISTANCE_NPY = ROOT_DIR / "Final_Imputed" / "image_hist_distances.npy"
IMAGE_CV_HIST_CHISQR_NPY = ROOT_DIR / "Final_Imputed" /"image_cv_hist_chisqr.npy"
IMAGE_DENSE_COSINE_NPY = ROOT_DIR / "Final_Imputed" /"image_dense_cosine.npy"

# Embedding distance matrices
EMBEDDING_EUCLIDEAN_DISTANCE_NPY  = EMBEDDING_DIR / "embedding_euclidean_distances.npy"
EMBEDDING_CORRELATION_DISTANCE_NPY = EMBEDDING_DIR / "embedding_correlation_distances.npy"

# Comparison output
COMPARISON_RESULT_TXT = FINAL_IMPUTED_DIR / "genotype_image_spearman.txt"
COMPARE_PLOT_FILE = ROOT_DIR / "Final_Imputed" / "genotype_vs_image_correlation.png"

# Case/Control labeling for xgb classifier
CASE_CONTROL_RAW = FINAL_IMPUTED_DIR /"match_ids" / "APC_IMPUTED.csv" #file containing partno, centre, group to get ID_LifeAndBrain
ID_EXCEL = FINAL_IMPUTED_DIR / "match_ids" / "femNATIDMATCH.xlsx" #file with ID_LifeAndBrain and ID_FemNAT
CASE_CONTROL_CSV = FINAL_IMPUTED_DIR / "match_ids" / "ID_LifeAndBrain_to_Group.csv"
CASE_CONTROL_LABELS = FINAL_IMPUTED_DIR / "snp_embeddings" / "case_control_labels.csv"