import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_tv_chambolle

# 📌 Input and output paths
INPUT_PATH = "C:/dip/processed_archive"
OUTPUT_PATH = "C:/dip/reconstructed_archive"

CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# Ensure output directories exist
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# 📌 K-space undersampling mask generator (Variable Density)
def generate_k_space_mask(shape, undersample_ratio=0.5):
    center_fraction = 0.1
    mask = np.zeros(shape, dtype=np.float32)
    center_x, center_y = shape[0] // 2, shape[1] // 2
    delta = int(min(shape) * center_fraction)
    mask[center_x - delta:center_x + delta, center_y - delta:center_y + delta] = 1

    total_pixels = shape[0] * shape[1]
    num_samples = int(total_pixels * undersample_ratio) - int(np.sum(mask))

    indices = np.array(np.where(mask == 0)).T
    sampled_indices = indices[np.random.choice(len(indices), int(num_samples), replace=False)]
    for x, y in sampled_indices:
        mask[x, y] = 1

    return mask

# 📌 Fourier Transform + Undersampling
def undersample_fft(img, mask):
    fft = np.fft.fftshift(np.fft.fft2(img))
    return fft * mask

# 📌 Inverse FFT to reconstruct image
def reconstruct_ifft(k_space_data):
    img_recon = np.fft.ifft2(np.fft.ifftshift(k_space_data))
    return np.abs(img_recon)

# 📌 Apply Adaptive Total Variation (ATV) denoising
def apply_total_variation(img, weight=0.1, iterations=100):
    return denoise_tv_chambolle(img, weight=weight, max_num_iter=iterations)

# 📌 Apply Wavelet Thresholding to restore detail
def apply_wavelet_denoising(img, wavelet='db1', threshold=10):
    coeffs = pywt.wavedec2(img, wavelet, level=2)
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else tuple(pywt.threshold(sub, threshold, mode='soft') for sub in c) for c in coeffs]
    return pywt.waverec2(coeffs_thresh, wavelet)

# 📌 Process all images in the dataset
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
            mask = generate_k_space_mask(img.shape, undersample_ratio=0.4)
            k_space = undersample_fft(img, mask)
            recon = reconstruct_ifft(k_space)
            tv_denoised = apply_total_variation(recon, weight=0.1)
            wavelet_refined = apply_wavelet_denoising(tv_denoised)

            wavelet_refined = np.clip(wavelet_refined, 0, 255).astype(np.uint8)
            cv2.imwrite(output_path, wavelet_refined)

        print(f"Processed and saved images for {folder}/{category}")

print("✅ All MRI images reconstructed and saved in 'reconstructed_archive' folder.")