import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 📌 Paths
INPUT_PATH = "C:/dip/inpainted_archive"
OUTPUT_PATH = "C:/dip/gan_restored_archive"
CATEGORIES = ["pituitary", "notumor", "meningioma", "glioma"]

# 📂 Ensure output directories exist
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_PATH, folder, category), exist_ok=True)

# 📌 Patch extraction
PATCH_SIZE = 64
STRIDE = 32

def extract_patches(img, patch_size=PATCH_SIZE, stride=STRIDE):
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

# 📌 Lightweight PatchGAN Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.main(x)

# 📌 Dataset for extracted patches
class PatchDataset(Dataset):
    def __init__(self, image_paths):
        self.patches = []
        transform = transforms.ToTensor()

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            for patch in extract_patches(img):
                patch_tensor = transform(patch).float()
                self.patches.append(patch_tensor)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

# 📌 Collect image paths
image_paths = []
for folder in ["Training", "Testing"]:
    for category in CATEGORIES:
        input_dir = os.path.join(INPUT_PATH, folder, category)
        for filename in os.listdir(input_dir):
            image_paths.append(os.path.join(input_dir, filename))

# 📌 Setup GAN components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)
criterion = nn.L1Loss()
dataset = PatchDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 📌 Lightweight training
epochs = 10
print("🚀 Training Lightweight PatchGAN...")
gen.train()
for epoch in range(epochs):
    for patch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        patch = patch.to(device)
        output = gen(patch)
        loss = criterion(output, patch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
print("✅ GAN Training Complete")

# 📌 Enhance full images using trained generator
print("🧠 Enhancing full images with trained generator...")
gen.eval()
transform = transforms.ToTensor()
with torch.no_grad():
    for folder in ["Training", "Testing"]:
        for category in CATEGORIES:
            input_dir = os.path.join(INPUT_PATH, folder, category)
            output_dir = os.path.join(OUTPUT_PATH, folder, category)
            for filename in os.listdir(input_dir):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256))
                input_tensor = transform(img).unsqueeze(0).to(device)
                output_tensor = gen(input_tensor).squeeze(0).cpu()

                restored_img = output_tensor.squeeze().numpy() * 255
                restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)
                cv2.imwrite(output_path, restored_img)

print("✅ Step 5 Complete: GAN-enhanced images saved to 'gan_restored_archive'.")
