# 🧠 Part A: CNN Trained From Scratch on iNaturalist 12K (10-Class Subset)

## 📁 Dataset

- **Source**: iNaturalist 12K (10 class subset)
- **Structure**:
  - `train/`: Training images organized into 10 class folders.
  - `val/`: Validation images with the same class structure.(USED AS TEST DATA).

## 🧾 Code Overview

The code trains a custom Convolutional Neural Network (CNN) from scratch using PyTorch. Key components:

- **Model**: Defined a 5-layer CNN using `nn.Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d`, followed by `nn.Linear` dense layers.
- **Training Loop**: Handles forward pass, loss computation, backpropagation, gradient clipping, and optimizer stepping.
- **Evaluation Function**: Computes validation accuracy and loss.
- **Logging**: Uses Weights & Biases (wandb) for logging training metrics and running hyperparameter sweeps.
- **AMP**: Implements Automatic Mixed Precision (`torch.amp`) for faster training.
- **Model Saving**: Saves the best model as `best_model.pth`.

## 🛠️ Techniques Used

- Custom CNN architecture (5 conv layers)
- Batch Normalization
- MaxPooling
- Dense Layers
- Dropout for regularization
- Label Smoothing
- **Data Augmentation**:
  - Random Crop
  - Horizontal Flip
  - Color Jitter
  - Rotation
- Gradient Clipping
- Automatic Mixed Precision (AMP)
- StepLR Learning Rate Scheduler
- [Weights & Biases (wandb)](https://wandb.ai) for tracking

## 🔧 Hyperparameters Swept

- Dropout rate
- Data augmentation probability
- Gradient clipping threshold
- Learning rate
- Dense layer size

## ✅ Final Performance

- **Validation Accuracy**: ~36%
- **Test Accuracy**: ~37%
- **Model Checkpoint**: `best_model.pth`

## 🧹 Sweep Insights

- Higher dropout improved generalization.
- Data augmentation significantly reduced overfitting.
- Gradient clipping between 1.0–2.0 stabilized training.
- ReLU and SiLU activations were both effective.
- Best learning rates were around `1e-3` to `5e-4`.
