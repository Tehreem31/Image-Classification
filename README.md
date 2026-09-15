# Image Classification Model - CIFAR-10

## 📌 Project Overview
This repository contains a PyTorch-based Convolutional Neural Network (CNN) designed to classify 32x32 color images into 10 distinct categories using the CIFAR-10 dataset[cite: 1]. Developed as part of **Task 1 for the EncoderX Remote Internship (Batch 02)**[cite: 1].

---

## 🛠️ Dataset & Data Pipeline
- **Dataset:** CIFAR-10 (60,000 images across 10 categories)[cite: 1]
- **Splits:** 80% Training, 20% Validation, Standard Test Set[cite: 1]
- **Preprocessing & Augmentation:**
  - Channel-wise normalization: `Mean=(0.4914, 0.4822, 0.4465)`, `Std=(0.2470, 0.2435, 0.2616)`
  - Random Horizontal Flips & Random Cropping (padding=4) to prevent overfitting[cite: 1]

---

## 🏗️ Model Architecture
A custom modular CNN built using PyTorch:
1. **Conv Block 1:** Conv2D (3→32) → BatchNorm → ReLU → Conv2D (32→64) → BatchNorm → ReLU → MaxPool (2x2) → Dropout (0.25)
2. **Conv Block 2:** Conv2D (64→128) → BatchNorm → ReLU → MaxPool (2x2) → Dropout (0.30)
3. **Classifier Head:** Flatten → Dense (128*8*8 → 256) → BatchNorm → ReLU → Dropout (0.50) → Dense (256 → 10)

- **Optimizer:** `AdamW` (lr=1e-3, weight_decay=1e-4)[cite: 1]
- **Loss Function:** `CrossEntropyLoss`[cite: 1]

---

## 📊 Results & Performance Evaluation
- **Overall Test Accuracy:** ~80.2%[cite: 1]
- **Strengths:** High classification precision (>88%) on rigid mechanical classes (`automobile`, `truck`, `ship`)[cite: 1].
- **Weaknesses:** Slight confusion between fine-grained animal features (`cat` vs. `dog`) at 32x32 resolution[cite: 1].

---

## 📁 Repository Structure
