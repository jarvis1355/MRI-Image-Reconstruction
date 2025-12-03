import os
import cv2
import numpy as np

# 📌 Input and output paths
INPUT_PATH = "C:/dip/gan_restored_archive"
OUTPUT_PATH = "C:/dip/slice_generated_archive"
CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# 📂 Ensure output directories exist
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# 📌 Function to sort filenames numerically if needed
def sorted_numerically(file_list):
    try:
        return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    except:
        return sorted(file_list)  # fallback to alphabetical

# 📌 Slice Generation: Insert interpolated slices between each pair
def generate_slices(image_stack):
    new_stack = []
    for i in range(len(image_stack) - 1):
        new_stack.append(image_stack[i])
        interp = cv2.addWeighted(image_stack[i], 0.5, image_stack[i + 1], 0.5, 0)
        new_stack.append(interp)  # add interpolated slice
    new_stack.append(image_stack[-1])
    return new_stack

# 📌 Main processing loop
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        input_dir = os.path.join(INPUT_PATH, folder, category)
        output_dir = os.path.join(OUTPUT_PATH, folder, category)

        # Collect and sort images
        filenames = sorted_numerically(os.listdir(input_dir))
        image_stack = []

        for fname in filenames:
            img_path = os.path.join(input_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                image_stack.append(img)

        # Generate slices
        generated_stack = generate_slices(image_stack)

        # Save all slices
        for idx, img in enumerate(generated_stack):
            filename = f"slice_{idx:03}.png"
            cv2.imwrite(os.path.join(output_dir, filename), img)

        print(f"✅ Processed {category} in {folder} — {len(generated_stack)} slices saved.")

print("\n✅ Step 6 Complete: Slice generation finished and saved in 'slice_generated_archive'")
