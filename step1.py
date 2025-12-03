import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 📌 Path to the dataset (modify this as needed)
DATASET_PATH = "C:/dip/archive (1)"
OUTPUT_PATH = "C:/dip/processed_archive"

# 📌 Categories (Brain Tumor Classes)
CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)
for subfolder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, subfolder, category), exist_ok=True)

# 📌 Function to load images from a given category
def load_images(folder, category):
    path = os.path.join(DATASET_PATH, folder, category)
    images = []
    filenames = os.listdir(path)
    for filename in filenames:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append((img, filename))
    return images

# 📌 Function to simulate undersampling
def undersample_image(img, mask_ratio=0.3):
    mask = np.random.rand(*img.shape) > mask_ratio
    return img * mask

# 📌 Function to add Gaussian noise
def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)

# 📌 Function to add Motion Blur
def apply_motion_blur(img, kernel_size=10):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

# 📌 Process all images and save results
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        print(f"Processing {folder}/{category}...")
        images = load_images(folder, category)

        if not images:
            print(f"No images found in {folder}/{category}!")
            continue

        for img, filename in images:
            undersampled_img = undersample_image(img, mask_ratio=0.4)
            noisy_img = add_gaussian_noise(undersampled_img, sigma=20)
            motion_blur_img = apply_motion_blur(noisy_img, kernel_size=10)

            save_path = os.path.join(OUTPUT_PATH, folder, category, filename)
            cv2.imwrite(save_path, motion_blur_img)

        print(f"Saved processed images in {OUTPUT_PATH}/{folder}/{category}/")

print("Processing complete. All images saved.")
