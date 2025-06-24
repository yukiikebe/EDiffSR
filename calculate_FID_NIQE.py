import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from pytorch_fid import fid_score
import scipy
import scipy.special

# Paths
gt_root = "AID_split_matched/HR/test"
sr_root = "results/yuki/Baseline-AID/AID"
device = 'cuda:0'

categories = sorted(os.listdir(gt_root))
fid_values = []
fid_by_category = {}
niqe_scores = []

print(f"{'Category':<20} | {'FID':>8}")
print("-" * 32)

# Function to preprocess images for FID
def preprocess_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.BICUBIC)
    return img

# --- FID per category ---
for category in categories:
    gt_dir = os.path.join(gt_root, category)
    sr_dir = os.path.join(sr_root, category)

    if not os.path.isdir(gt_dir) or not os.path.isdir(sr_dir):
        print(f"{category:<20} | Missing directory")
        continue

    # Preprocess images and save to temporary directories
    temp_gt_dir = os.path.join("temp_gt", category)
    temp_sr_dir = os.path.join("temp_sr", category)
    os.makedirs(temp_gt_dir, exist_ok=True)
    os.makedirs(temp_sr_dir, exist_ok=True)

    # Preprocess GT images
    for fname in os.listdir(gt_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = preprocess_image(os.path.join(gt_dir, fname))
            img.save(os.path.join(temp_gt_dir, fname))

    # Preprocess SR images
    for fname in os.listdir(sr_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = preprocess_image(os.path.join(sr_dir, fname))
            img.save(os.path.join(temp_sr_dir, fname))

    try:
        fid = fid_score.calculate_fid_given_paths(
            [temp_gt_dir, temp_sr_dir],
            batch_size=10,
            device=device,
            dims=2048
        )
        fid_values.append(fid)
        fid_by_category[category] = fid
        print(f"{category:<20} | {fid:.4f}")
    except Exception as e:
        print(f"{category:<20} | FID Error: {e}")

# Print best and average FID
if fid_values:
    avg_fid = np.mean(fid_values)
    best_category = min(fid_by_category, key=fid_by_category.get)
    best_fid = fid_by_category[best_category]

    print("\n--- FID Summary ---")
    print(f"Average FID across categories: {avg_fid:.4f}")
    print(f"Best FID: {best_fid:.4f} ({best_category})")
else:
    print("\nNo valid FID scores calculated.")

# --- NIQE across all SR images ---
print("\nCalculating NIQE for all SR images...")

# NIQE Functions
def compute_niqe_features(image, block_size=96):
    image = image.astype(np.float32)
    image = (image - np.mean(image)) / (np.std(image) + 1e-7)
    mu = cv2.blur(image, (7, 7))
    sigma = np.sqrt(cv2.blur(image**2, (7, 7)) - mu**2)
    structdis = (image - mu) / (sigma + 1)
    h, w = structdis.shape
    features = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = structdis[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            alpha, l_std, r_std = aggd_fit(block)
            features.append([alpha, l_std, r_std])
    return np.array(features).reshape(-1, 3)

def aggd_fit(structdis):
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (scipy.special.gamma(2.0 / gam) ** 2) / (
        scipy.special.gamma(1.0 / gam) * scipy.special.gamma(3.0 / gam)
    )
    structdis = structdis.flatten()
    left_std = np.sqrt((structdis[structdis < 0] ** 2).mean())
    right_std = np.sqrt((structdis[structdis > 0] ** 2).mean())
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(structdis))) ** 2 / np.mean(structdis ** 2)
    rhatnorm = rhat * (gammahat ** 3 + 1) * (gammahat + 1) / (
        (gammahat ** 2 + 1) ** 2
    )
    diff = np.abs(r_gam - rhatnorm)
    alpha = gam[np.argmin(diff)]
    return alpha, left_std, right_std

mu_pris_param = np.array([4.3589, 1.7492, 1.4739])
cov_pris_param = np.array([
    [0.4010, 0.0054, -0.0142],
    [0.0054, 0.2054, 0.0115],
    [-0.0142, 0.0115, 0.2701]
])

def calculate_niqe_score(image, mu_pris_param, cov_pris_param):
    features = compute_niqe_features(image)
    if features.size == 0:
        return np.nan
    features_mean = np.mean(features, axis=0)
    features_cov = np.cov(features, rowvar=False)
    X = features_mean - mu_pris_param
    covmat = (cov_pris_param + features_cov) / 2.0
    invcov = np.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, invcov), X.T))
    return niqe_score

for category in sorted(os.listdir(sr_root)):
    category_path = os.path.join(sr_root, category)
    if not os.path.isdir(category_path):
        continue

    for fname in os.listdir(category_path):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(category_path, fname)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            score = calculate_niqe_score(img, mu_pris_param, cov_pris_param)
            if not np.isnan(score):
                niqe_scores.append(score)
        except Exception as e:
            print(f"NIQE Error on {img_path}: {e}")

if len(niqe_scores) > 0:
    avg_niqe = np.mean(niqe_scores)
    print(f"\nAverage NIQE score across all SR images: {avg_niqe:.4f}")
else:
    print("\nNo valid NIQE scores calculated.")
