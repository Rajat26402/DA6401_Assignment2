# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# **Resnet** 

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import wandb

# WandB login
wandb.login(key="d6f8c99f1fd73267470842bbf00f03ae845f7308")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "/kaggle/input/nature/nature_12K/inaturalist_12K"
batch_size = 64
img_size = 224
num_classes = 10
epochs = 15
lr = 1e-4

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# %%
# Transforms
train_transform = transforms.Compose([
    transforms.Resize((img_size + 32, img_size + 32)),
    transforms.RandomCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# %%
# Dataset and loaders
full_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
test_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

# Load pre-trained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# %%
# Fine-tuning strategy: unfreeze layer4 and fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss, optimizer, scaler, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scaler = GradScaler()

# %%
wandb.init(project="DLA2-PartB", name="ResNet50-finetune-enhanced", config={
    "strategy": "unfreeze layer4 and fc",
    "model": "ResNet50",
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": lr,
    "augmentation": True,
    "label_smoothing": 0.1,
    "scheduler": "StepLR(step_size=5, gamma=0.5)"
})
wandb.watch(model)

# %%
def train_one_epoch(loader):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Evaluation loop
def evaluate(loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# %%
def test(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# %%
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step()
    print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": scheduler.get_last_lr()[0]
    })

# %%
# Final test accuracy
test_acc = test(test_loader)
print(f"✅ Test Accuracy: {test_acc:.4f}")
wandb.log({"test_acc": test_acc})

# Save model
torch.save(model.state_dict(), "resnet50_finetuned.pth")
wandb.save("resnet50_finetuned.pth")

# %%



