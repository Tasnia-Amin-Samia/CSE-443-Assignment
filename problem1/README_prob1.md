# Pneumonia Classification using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) for binary classification of chest X-ray images into two classes:

- `NORMAL`
- `PNEUMONIA`

The model is implemented using TensorFlow/Keras in Google Colab. The notebook uses Google Drive for the dataset and trains a CNN from scratch.

## Dataset

The dataset is organized into separate training, validation, and testing directories.

| Split | Images |
|---|---:|
| Training | 5,216 |
| Validation | 16 |
| Testing | 624 |
| Classes | 2 |

The classes detected by the notebook are `NORMAL` and `PNEUMONIA`.

Images are resized to **224 × 224** pixels and loaded with binary labels.

## Data Augmentation

The training pipeline applies:

- Random horizontal flip
- Random rotation
- Random zoom
- Random contrast

Pixel values are then rescaled from `[0, 255]` to `[0, 1]`.

## CNN Architecture

The model contains four convolutional blocks followed by global average pooling and fully connected layers.

### Architecture

```text
Input: 224 × 224 × 3
        ↓
Data Augmentation
        ↓
Rescaling (1/255)
        ↓
Conv2D 32, 3×3, ReLU
Conv2D 32, 3×3, ReLU
MaxPooling 2×2
        ↓
Conv2D 64, 3×3, ReLU
Conv2D 64, 3×3, ReLU
MaxPooling 2×2
        ↓
Conv2D 128, 3×3, ReLU
Conv2D 128, 3×3, ReLU
MaxPooling 2×2
        ↓
Conv2D 256, 3×3, ReLU
MaxPooling 2×2
        ↓
GlobalAveragePooling2D
        ↓
Dense 128, ReLU
        ↓
Dropout 0.5
        ↓
Dense 1, Sigmoid
```

The model has **615,201 trainable parameters**.

## Training

The model is compiled with:

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss: Binary Crossentropy
- Metrics:
  - Accuracy
  - Precision
  - Recall

Training is configured for **10 epochs**.

The notebook uses:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

The best model is saved as:

```text
best_cnn_model.keras
```

## Results

The final evaluation on the test set produced:

| Metric | Score |
|---|---:|
| Accuracy | 81.25% |
| Precision | 90.27% |
| Recall | 78.46% |
| F1 Score | 83.95% |

Classification report:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| NORMAL | 0.71 | 0.86 | 0.77 |
| PNEUMONIA | 0.90 | 0.78 | 0.84 |

The test set contains 624 images.

## Visualization

The notebook includes:

- Training vs. validation accuracy graph
- Training vs. validation loss graph
- Classification report
- Confusion matrix

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## How to Run

1. Open `prob1.ipynb` in Google Colab.
2. Mount Google Drive.
3. Make sure the dataset is available at:

```text
/content/drive/MyDrive/problemds/
├── train/
├── val/
└── test/
```

4. Run the notebook cells sequentially.
5. The trained model will be saved as `best_cnn_model.keras`.

## Project Objective

The objective is to build a CNN-based image classification model that can distinguish between normal and pneumonia chest X-ray images.
