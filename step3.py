import os
import numpy as np
import cv2
import pywt
from scipy.ndimage import median_filter

# 📌 Input and output paths
INPUT_PATH = "C:/dip/reconstructed_archive"
OUTPUT_PATH = "C:/dip/final_denoised_archive"

CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# Ensure output directories exist
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# 📌 Wavelet-Based Shrinkage (WBS)
def wavelet_denoising(img, wavelet='db1', threshold=10):
    coeffs = pywt.wavedec2(img, wavelet, level=2)
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else tuple(pywt.threshold(sub, threshold, mode='soft') for sub in c) for c in coeffs]
    return pywt.waverec2(coeffs_thresh, wavelet)

# 📌 Hybrid Median Filter (HMF) implementation
def hybrid_median_filter(img):
    med_cross = median_filter(img, size=(3, 3))
    med_diag = median_filter(img, size=(5, 5))
    med_center = img
    return np.median(np.array([med_cross, med_diag, med_center]), axis=0)

# 📌 Process all images
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        input_dir = os.path.join(INPUT_PATH, folder, category)
        output_dir = os.path.join(OUTPUT_PATH, folder, category)

        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (256, 256))
            wavelet_denoised = wavelet_denoising(img, threshold=15)
            hmf_result = hybrid_median_filter(wavelet_denoised)

            hmf_result = np.clip(hmf_result, 0, 255).astype(np.uint8)
            cv2.imwrite(output_path, hmf_result)

        print(f"Step 3 completed for {folder}/{category}")

print("✅ Step 3: Multi-scale denoising & edge preservation done. Output saved in 'final_denoised_archive'.")