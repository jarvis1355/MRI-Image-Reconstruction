# 🧠 Brain MRI Reconstruction & Enhancement Pipeline

*A Multi-Stage Framework for Restoring Corrupted MRI Scans*

---

## 📌 Overview

This project implements a complete **MRI image reconstruction pipeline** designed to restore MRI brain scans affected by:

* Undersampling
* Gaussian noise
* Motion blur
* Missing/corrupted regions
* Low contrast
* Loss of structure or texture

The pipeline combines **classical image processing**, **frequency-domain reconstruction**, **wavelet-based denoising**, **inpainting**, **patch-based deep learning**, and **contrast enhancement** to generate **clear, radiologically reliable MRI images**.

Supported MRI categories:

* Glioma
* Meningioma
* Pituitary
* No Tumor

---

## 🚀 Key Features

* Multi-stage MRI Reconstruction
* k-space undersampling & frequency compensation
* Adaptive Total Variation (ATV) denoising
* Multi-scale wavelet shrinkage
* Motion blur removal
* Missing region restoration (Telea + Biharmonic inpainting)
* Patch-based Deep Learning texture enhancement
* CLAHE contrast improvement
* Unsharp masking & Canny edge sharpening
* Brain masking to remove background
* Slice interpolation for 3D continuity
* Evaluation using **PSNR, SSIM, Frequency Energy**

---

## 🏗 Pipeline Architecture

### **1. Preprocessing**

* Simulates real MRI defects:

  * undersampling
  * Gaussian noise
  * motion blur

### **2. Frequency Domain Reconstruction**

* FFT → k-space
* Undersampling mask applied
* IFFT to get corrupted image
* Adaptive Total Variation denoising
* Wavelet thresholding to remove high-frequency noise

### **3. Multi-Scale Denoising**

* Multi-level wavelet shrinkage
* Hybrid median filter

### **4. Missing Region Restoration**

* Telea inpainting
* Biharmonic PDE inpainting
* Fourier magnitude-phase recombination

### **5. Patch-Based Deep Learning Enhancement**

* Extract overlapping 64×64 patches
* Patch-based L1 reconstruction
* Aggregated to restore fine textures

### **6. Slice Interpolation**

Ensures smoother transitions between slices:

```
Interpolated_Slice = (Slice1 + Slice2) / 2
```

### **7. Contrast & Structural Enhancement**

* CLAHE
* Gamma correction
* Unsharp masking
* Canny edge enhancement
* Otsu thresholding

### **8. Brain Masking**

Retains only true brain region using:

* Area > 500
* Aspect ratio < 5
* Must be near image center

---

## 🧪 Evaluation Metrics

| Metric               | Description                            | Higher is Better |
| -------------------- | -------------------------------------- | ---------------- |
| **PSNR**             | Measures reconstruction clarity        | ✔                |
| **SSIM**             | Measures structural similarity         | ✔                |
| **Frequency Energy** | Measures edge and texture preservation | ✔                |

---

## 📁 Dataset

This project uses publicly available Brain MRI datasets containing:

* Glioma
* Meningioma
* Pituitary
* No-tumor

All images are resized to **256×256**.

---

## 🧩 Technologies Used

* Python 3
* NumPy
* OpenCV
* scikit-image
* PyWavelets
* TensorFlow / PyTorch
* Matplotlib

---

## 📂 Project Structure

```
📦 Brain-MRI-Reconstruction
│
├── preprocessing/
├── reconstruction/
├── denoising/
├── inpainting/
├── deep_learning/
├── slice_interpolation/
├── enhancement/
├── evaluation/
│
├── models/
├── results/
└── README.md
```

---

## 🏁 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Brain-MRI-Reconstruction.git
cd Brain-MRI-Reconstruction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

```bash
python main.py
```

Outputs appear in the **results/** folder.

---

## 🔬 Simple Summary (Layman-Friendly)

Think of a corrupted MRI as a damaged old photograph—
blurred, torn, noisy, and low-quality.

This pipeline:

1. Removes noise
2. Fixes blur
3. Fills missing pieces
4. Recovers lost texture
5. Enhances contrast
6. Sharpens edges
7. Keeps only actual brain region
8. Produces a clean, diagnostic-quality MRI

---

## 📌 Future Work

* GAN-based MRI reconstruction
* Diffusion model enhancement
* Real-time hospital-compatible pipelines
* Radiologist-driven feedback improvements

---

## 🤝 Contributors

* **Your Name** (Lead Developer)
* Team Members

---

## 📜 License

**MIT License**
Feel free to use, modify, and distribute.

---

## ⭐ If you find this useful, don't forget to star the repository!

---
