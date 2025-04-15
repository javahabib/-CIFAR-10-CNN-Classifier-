# -CIFAR-10-CNN-Classifier-

This project demonstrates two different Convolutional Neural Network (CNN) models trained on the CIFAR-10 dataset using TensorFlow/Keras.

---

## Models

### 1. `cnn.py` (Standard CNN)
- Basic CNN model trained on raw CIFAR-10 images.
- No data augmentation applied.
- Splits the dataset into training, validation, and testing.
- Plots training and validation accuracy/loss.
- Displays a confusion matrix and random prediction samples.

### 2. `cnn_augmented.py` (CNN with Data Augmentation)
- Same CNN architecture, but trained using **ImageDataGenerator** for:
  - Rotation
  - Width/height shifting
  - Horizontal flipping
- Helps model generalize better and reduce overfitting.
- Optionally includes early stopping.


## Requirements

Install required libraries using:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

## How to Run

Install required libraries using:

```bash
python cnn.py
```
```bash
python cnn_augmented.py
```

## Dataset Info
Dataset: CIFAR-10

60,000 32x32 color images in 10 classes

Automatically downloaded using cifar10.load_data()
