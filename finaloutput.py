import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
INPUT_PATH = "C:/dip/final_refined_archive"
OUTPUT_PATH = "C:/dip/final_ultra_enhanced_archive"
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

# Create output folders
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# ✅ Helper: Validate only brain tissue contours
def is_valid_contour(contour, image_shape):
    area = cv2.contourArea(contour)
    if area < 50:
        return False
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    if aspect_ratio > 6.0:
        return False
    
    img_center = np.array([image_shape[1] / 2, image_shape[0] / 2])
    contour_center = np.array([x + w / 2, y + h / 2])
    distance = np.linalg.norm(contour_center - img_center)
    
    if distance > 250 and area < 300:
        return False
    
    return True

# ✅ Updated Create mask with only valid brain contours
def create_brain_mask(img):
    # Threshold to binary
    _, binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Morph open to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter valid contours (only large, centered structures)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # ⛔ Skip small noisy patches
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if aspect_ratio > 5.0:
            continue

        img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
        contour_center = np.array([x + w / 2, y + h / 2])
        distance = np.linalg.norm(contour_center - img_center)
        if distance > 250 and area < 1000:
            continue

        valid_contours.append(cnt)

    # Final mask
    mask = np.zeros_like(img)
    cv2.drawContours(mask, valid_contours, -1, (255), thickness=cv2.FILLED)

    return mask

# ✅ Enhancement pipeline
def enhance_image_mri(img):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    
    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # Step 2: Denoising
    img_denoised = cv2.fastNlMeansDenoising(img_clahe, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 3: Unsharp Masking
    blur = cv2.GaussianBlur(img_denoised, (5, 5), 1.0)
    sharpened = cv2.addWeighted(img_denoised, 1.5, blur, -0.5, 0)
    
    # Step 4: Edge Detection (light)
    edges = cv2.Canny(sharpened, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    light_edges = (edges_dilated * 0.4).astype(np.uint8)
    
    # Step 5: Otsu + blend
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blended = cv2.addWeighted(sharpened, 0.8, otsu, 0.2, 0)
    
    # Step 6: Add light contours
    enhanced = cv2.add(blended, light_edges)
    
    # Step 7: Gamma correction
    gamma = 0.9
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    final = cv2.LUT(enhanced, look_up)
    
    # Step 8: Keep only valid brain region, black background
    brain_mask = create_brain_mask(final)
    final_masked = cv2.bitwise_and(final, final, mask=brain_mask)
    
    return final_masked

# 🔄 Enhance all images in batch
print("🔬 Enhancing all MRI images for ultra-realism and clarity...")
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        input_dir = os.path.join(INPUT_PATH, folder, category)
        output_dir = os.path.join(OUTPUT_PATH, folder, category)
        
        for filename in tqdm(os.listdir(input_dir), desc=f"{folder}/{category}"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                enhanced = enhance_image_mri(img)
                cv2.imwrite(output_path, enhanced)

print("✅ All MRI scans enhanced and saved to:", OUTPUT_PATH)
