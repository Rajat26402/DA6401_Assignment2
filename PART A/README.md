Dataset
Source: iNaturalist 12K (10 class subset)

Structure:

train/ contains training images organized into 10 class folders.

val/ contains validation images with the same class structure.

Techniques Used
Custom CNN architecture (5 conv layers + batchnorm + maxpool + dense layers)

Data Augmentation (random crop, flip, color jitter, rotation)

Dropout for regularization

Batch Normalization

Label Smoothing

Automatic Mixed Precision (AMP)

Gradient Clipping

Learning Rate Scheduling (StepLR)

WandB for experiment tracking and hyperparameter sweep

Hyperparameters Swept
Dropout rate

Data augmentation probability

Gradient clipping threshold

Learning rate

Dense layer size

Final CNN Performance
Validation Accuracy: ~ 36%

Test Accuracy: ~ 37%

Model checkpoint saved as best_model.pth

🧹 Sweep Insights
Key insights from the sweep:

Higher dropout improved generalization.

Using data augmentation significantly reduced overfitting.

Clipping gradients between 1.0–2.0 stabilized training.

ReLU and SiLU were competitive, but ReLU remained reliable.

Optimal learning rate found to be ~1e-3 or 5e-4 depending on batch size.