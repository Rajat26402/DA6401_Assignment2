# 🧠 Part B: Fine-Tuning Pretrained Models on iNaturalist 12K

## 📁 Dataset

- **Source**: iNaturalist 12K (10 class subset)
- **Structure**:
  - Same as Part A (`train/` and `val/` folders)

## 🧾 Code Overview

The code fine-tunes a **pretrained ResNet50** from `torchvision.models` using PyTorch.

- **Model**: Loads ImageNet-pretrained ResNet50.
- **Strategy**: Freezes all layers except `layer4` and `fc` to fine-tune only the deeper, task-specific layers.
- **Classifier Replacement**: Replaces the original `fc` layer with a new one for 10 output classes.
- **Data Augmentation**: Applied the same augmentations as Part A.
- **Training Loop**: Includes AMP and scheduler (StepLR).
- **Logging**: Training/validation/test performance logged via wandb.
- **Model Saving**: Saves fine-tuned weights to `resnet50_finetuned.pth`.

## 🛠️ Techniques Used

- Fine-tuning pretrained ResNet50
- Partial unfreezing (`layer4` + `fc`)
- Label smoothing
- Data augmentation
- Gradient clipping
- Automatic Mixed Precision (AMP)
- StepLR learning rate scheduler
- [wandb](https://wandb.ai) for logging

## ⚙️ Fine-Tuning Strategy

- **Strategy Used**: *Freezing Layers Up to a Certain Layer (e.g., up to `layer4`)*
- **Why?**
  - Keeps general low-level features from pretrained backbone.
  - Fine-tunes higher layers to adapt to the iNaturalist domain.
  - Efficient in terms of time and avoids catastrophic forgetting.
  - Great balance of accuracy and generalization.

## ✅ Final Performance

- **Validation Accuracy**: ~76%
- **Test Accuracy**: ~78%
- **Checkpoint Saved**: `resnet50_finetuned.pth`

## 🔄 Comparison with Part A

| Aspect                  | Part A (Scratch) | Part B (Fine-Tune) |
|------------------------|------------------|---------------------|
| Validation Accuracy    | ~36%             | ~76%                |
| Test Accuracy          | ~37%             | ~78%                |
| Training Time          | Higher           | Lower               |
| Generalization         | Lower            | Better              |
| Uses Pretrained Model? | ❌               | ✅                  |


