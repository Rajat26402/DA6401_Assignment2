Dataset
Same as Part A (iNaturalist 12K - 10 classes)

Preprocessing
Image resized to 224x224

Normalized using ImageNet mean & std

Models Used
Pretrained Backbones: ResNet50

Custom classifier head adapted for 10-class output

Fine-Tuning Strategies Explored
Freezing all layers except the final fully connected layer

Freezing layers up to layer4 and fine-tuning the rest

Fine-tuning the entire model with different learning rates

Final Strategy Used
Freezing up to layer4 and fine-tuning layer4 + fc layers in ResNet50, based on the balance of adaptability, training cost, and validation performance.

Results
Achieved significantly higher accuracy compared to training from scratch.

Improved generalization and convergence speed.

Test Accuracy: ~ 78%

Tools & Features
WandB logging for metrics, losses, and model configs

Automatic Mixed Precision

Label smoothing

StepLR scheduler

Optimizer: Adam