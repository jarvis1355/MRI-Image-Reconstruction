import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pyplot as plt

# 📌 Paths
INPUT_PATH = "C:/dip/slice_generated_archive"
GROUND_TRUTH_PATH = "C:/dip/archive (1)"
OUTPUT_PATH = "C:/dip/final_refined_archive"
CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# 📂 Ensure output directories exist
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# 📌 Apply Adaptive Histogram Equalization
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

# 📌 Frequency Analysis: Calculate energy from FFT
def frequency_energy(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return np.sum(magnitude)

# 📌 Processing loop
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        input_dir = os.path.join(INPUT_PATH, folder, category)
        gt_dir = os.path.join(GROUND_TRUTH_PATH, folder, category)
        output_dir = os.path.join(OUTPUT_PATH, folder, category)

        filenames = sorted(os.listdir(input_dir))
        psnr_scores = []
        ssim_scores = []
        freq_diffs = []

        for filename in filenames:
            input_path = os.path.join(input_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            output_path = os.path.join(output_dir, filename)

            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            enhanced_img = apply_clahe(img)

            # Save refined image
            cv2.imwrite(output_path, enhanced_img)

            # Ground truth comparison
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.resize(gt_img, (256, 256))

                psnr_val = psnr(gt_img, enhanced_img)
                ssim_val = ssim(gt_img, enhanced_img)
                freq_orig = frequency_energy(gt_img)
                freq_recon = frequency_energy(enhanced_img)

                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
                freq_diffs.append(abs(freq_orig - freq_recon))

        # 📊 Report
        if psnr_scores:
            print(f"\n📁 {category} ({folder}):")
            print(f"  - Avg PSNR:  {np.mean(psnr_scores):.2f}")
            print(f"  - Avg SSIM:  {np.mean(ssim_scores):.4f}")
            print(f"  - Avg Freq Δ: {np.mean(freq_diffs):.2f}")

print("\n✅ Step 7 Complete: Final refinement and evaluation saved in 'final_refined_archive'")
