import os
import numpy as np
import cv2
from skimage.restoration import inpaint_biharmonic
from skimage.util import random_noise
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from cv2 import inpaint

# 📌 Input and output paths
INPUT_PATH = "C:/dip/final_denoised_archive"
OUTPUT_PATH = "C:/dip/inpainted_archive"

CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# Ensure output directories exist
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# 📌 Create artificial mask for testing (simulate missing regions)
def create_missing_mask(img_shape, missing_ratio=0.2):
    mask = np.zeros(img_shape, dtype=np.uint8)
    num_pixels = int(img_shape[0] * img_shape[1] * missing_ratio)
    xs = np.random.randint(0, img_shape[0], num_pixels)
    ys = np.random.randint(0, img_shape[1], num_pixels)
    mask[xs, ys] = 1
    return mask

# 📌 Patch-based exemplar inpainting (using OpenCV's Telea method)
def exemplar_inpaint(img, mask):
    return inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# 📌 Biharmonic inpainting for smooth interpolation
def biharmonic_inpaint(img, mask):
    return inpaint_biharmonic(img, mask.astype(bool), channel_axis=None)

# 📌 Fourier-Based Frequency Compensation (restore frequency domain details)
def fourier_frequency_compensation(original_img, inpainted_img):
    original_fft = np.fft.fft2(original_img)
    inpainted_fft = np.fft.fft2(inpainted_img)

    # Blend amplitudes: keep phase of inpainted, magnitude of original
    result_fft = np.abs(original_fft) * np.exp(1j * np.angle(inpainted_fft))
    compensated = np.fft.ifft2(result_fft)
    return np.abs(compensated)

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
            mask = create_missing_mask(img.shape, missing_ratio=0.15)

            exemplar = exemplar_inpaint(img, mask)
            biharmonic = biharmonic_inpaint(exemplar, mask)
            final_img = fourier_frequency_compensation(img, biharmonic)

            final_img = np.clip(final_img, 0, 255).astype(np.uint8)
            cv2.imwrite(output_path, final_img)

        print(f"Step 4 completed for {folder}/{category}")

print("✅ Step 4: Inpainting with exemplar, biharmonic, and Fourier compensation complete. Output saved in 'inpainted_archive'.")
