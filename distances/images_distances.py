# compute distances between images

import os
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config
import cv2
from sklearn.metrics.pairwise import cosine_distances

# Paths
image_dir = Path(config.IMAGE_SAMPLE_DIR)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
N = len(image_files)
print(f"Found {N} PNG images in {image_dir}")

# Load Images
def load_gray(path):
    return np.array(Image.open(path).convert("L"))

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

gray_images = [load_gray(image_dir / f) for f in image_files]
rgb_images = [load_rgb(image_dir / f) for f in image_files]

# SSIM (grayscale structural)
print("Computing SSIM (grayscale structure)...")
ssim_matrix = np.zeros((N, N))
for i in tqdm(range(N), desc="SSIM"):
    for j in range(i, N):
        score, _ = ssim(gray_images[i], gray_images[j], full=True)
        ssim_matrix[i, j] = ssim_matrix[j, i] = score

ssim_dist = 1 - ssim_matrix
np.save(config.IMAGE_DISTANCE_NPY, ssim_dist)

# RGB Histogram with numpy
def rgb_hist(im, bins=32):
    return np.concatenate([
        np.histogram(im[..., c], bins=bins, range=(0, 255), density=True)[0]
        for c in range(3)
    ])

print("Computing RGB histogram correlation distance...")
hist_matrix = np.zeros((N, N))
hists = [rgb_hist(im) for im in rgb_images]

for i in tqdm(range(N), desc="RGB hist (corr)"):
    for j in range(i, N):
        corr = np.corrcoef(hists[i], hists[j])[0, 1]
        dist = 1 - corr
        hist_matrix[i, j] = hist_matrix[j, i] = dist

np.save(config.IMAGE_HIST_DISTANCE_NPY, hist_matrix)

# cv2.compareHist
def to_cv_hist(im):
    # 3-channel histogram (32 bins per channel, normalized)
    chans = cv2.split(im)
    hists = []
    for ch in chans:
        hist = cv2.calcHist([ch], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)
    return np.vstack(hists)

cv_hists = [to_cv_hist(im.astype(np.uint8)) for im in rgb_images]

def compare_cv_metric(h1, h2, method):
    return cv2.compareHist(h1, h2, method)

# Example: Chi-square distance
print("Computing histogram distance via cv2 (Chi-Square)...")
chi_matrix = np.zeros((N, N))
for i in tqdm(range(N), desc="cv2 ChiSq"):
    for j in range(i, N):
        d = compare_cv_metric(cv_hists[i], cv_hists[j], cv2.HISTCMP_CHISQR)
        chi_matrix[i, j] = chi_matrix[j, i] = d

np.save(config.IMAGE_CV_HIST_CHISQR_NPY, chi_matrix)

# Dense RGB vector similarity (cosine)
print("Computing dense RGB vector cosine distances...")
flat_matrix = np.array([im.flatten() / 255.0 for im in rgb_images])
cosine_matrix = cosine_distances(flat_matrix)

np.save(config.IMAGE_DENSE_COSINE_NPY, cosine_matrix)

print("All distance matrices saved:")
print(f" - SSIM: {config.IMAGE_DISTANCE_NPY}")
print(f" - RGB histogram: {config.IMAGE_HIST_DISTANCE_NPY}")
print(f" - cv2 Chi-Square hist: {config.IMAGE_CV_HIST_CHISQR_NPY}")
print(f" - Dense RGB cosine: {config.IMAGE_DENSE_COSINE_NPY}")
