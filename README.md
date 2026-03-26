# Lung Disease Detection using Deep Learning

## Binary Classification of Chest X-Rays using ResNet18

---

## Overview

This project applies **transfer learning** on a pretrained **ResNet18** model to classify chest X-ray images into:

- **NORMAL** — healthy lungs  
- **PNEUMONIA** — infected lungs  

A **balanced subset (25%)** of the dataset is used to reduce training time while maintaining strong performance.

---

## Dataset

**Source:** Kaggle — Chest X-Ray Images (Pneumonia)

### Structure
chest_xray/
├── train/
├── test/
└── val/


### Classes

- **NORMAL**
- **PNEUMONIA**

---

## Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **Matplotlib**
- **scikit-learn**
- **Kaggle GPU**

---

## Model Architecture

Pretrained **ResNet18** with:

- Frozen feature extraction layers  
- Trainable final layer: **Linear (512 → 2)**  

This reduces computation and speeds up training.

---

## How to Run

1. Open a **Kaggle Notebook**
2. Add dataset: `chest-xray-pneumonia`
3. Enable **GPU**
4. Run all cells sequentially

### Dataset Path

```python
base_path = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"