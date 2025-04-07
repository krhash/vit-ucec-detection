import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageFile
import timm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import random

# Fix PIL issues
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Dataset class
class HistoDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=10000):
        self.classes = ['normal', 'tumor']
        self.samples = []
        self.transform = transform

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            self.samples += [
                (os.path.join(cls_dir, f), self.classes.index(cls))
                for f in sorted(os.listdir(cls_dir))
                if f.endswith('.png') and os.path.getsize(os.path.join(cls_dir, f)) > 1024
            ]

        random.shuffle(self.samples)
        self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB').resize((224, 224))
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return torch.zeros(3, 224, 224), -1

# Safe weight loading function
def safe_load_weights(model, checkpoint_path, device):
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state_dict
    state_dict = checkpoint.get('model', checkpoint)

    # Strip "model." prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[len("model."):] if k.startswith("model.") else k
        new_state_dict[new_key] = v

    # Filter matching keys
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}

    missing_keys = model_keys - filtered_state_dict.keys()
    unexpected_keys = set(new_state_dict.keys()) - model_keys

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Ignored unexpected keys: {unexpected_keys}")

    model.load_state_dict(filtered_state_dict, strict=False)
    print("Model weights loaded successfully.\n")

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return preds, targets

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Main
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "processed_slides"
    BATCH_SIZE = 32

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = HistoDataset(DATA_DIR, transform=val_transform)
    labels = [s[1] for s in full_dataset.samples]
    _, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=labels)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE)

    # --- Evaluate ViT Model ---
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    safe_load_weights(vit_model, "out_model/uterine_cancer_vit.pth", DEVICE)
    vit_model.to(DEVICE)

    vit_preds, vit_targets = evaluate(vit_model, val_loader, DEVICE)
    print("\n[ViT Evaluation]")
    print(classification_report(vit_targets, vit_preds, target_names=["Normal", "Tumor"]))
    plot_confusion_matrix(vit_targets, vit_preds, ["Normal", "Tumor"], title="ViT Confusion Matrix")

    # --- Evaluate ResNet34 Model ---
    resnet_model = models.resnet34(pretrained=False)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
    safe_load_weights(resnet_model, "out_model/uterine_cancer_resnet34.pth", DEVICE)
    resnet_model.to(DEVICE)

    resnet_preds, resnet_targets = evaluate(resnet_model, val_loader, DEVICE)
    print("\n[ResNet34 Evaluation]")
    print(classification_report(resnet_targets, resnet_preds, target_names=["Normal", "Tumor"]))
    plot_confusion_matrix(resnet_targets, resnet_preds, ["Normal", "Tumor"], title="ResNet34 Confusion Matrix")
