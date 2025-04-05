import os
import cv2
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassAccuracy

# Fix PIL image truncation
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 1. Dataset
# ----------
class HistoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'tumor']
        self.samples = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for f in sorted(os.listdir(cls_dir)):
                if f.endswith('.png'):  # Only PNGs
                    path = os.path.join(cls_dir, f)
                    if os.path.getsize(path) > 1024:  # Skip empty/corrupted files
                        self.samples.append((path, self.classes.index(cls)))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img)

            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            print(f"[ERROR] Skipping file {path}: {e}")
            return torch.zeros(3, 224, 224), -1

# 2. ResNet34 Classifier
# ----------------------
class ResNetClassifier(pl.LightningModule):
    def __init__(self, class_weights):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.train_acc = MulticlassAccuracy(num_classes=2)
        self.val_acc = MulticlassAccuracy(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.train_acc(logits, y)
        # Log training loss and accuracy at both step and epoch levels
        self.log('train_loss', loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_acc(logits, y)
        # Log validation loss and accuracy at both step and epoch levels
        self.log('val_loss', loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# 3. Training Setup
# -----------------
def main():
    DATA_DIR = "processed_slides"
    BATCH_SIZE = 32
    EPOCHS = 10

    torch.manual_seed(42)

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])

    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    # Full dataset for filtering + indexing
    full_dataset = HistoDataset(DATA_DIR)
    labels = [label for _, label in full_dataset.samples]

    # Stratified split
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = HistoDataset(DATA_DIR, transform=train_tf)
    val_dataset = HistoDataset(DATA_DIR, transform=val_tf)

    # Class weights
    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    weights = torch.tensor([total / (2 * class_counts[i]) for i in range(2)], dtype=torch.float)
    print(f"Class Weights: {weights}")

    # Sampler for class balancing
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

    # DataLoaders
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    # Model checkpointing
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_resnet",
        filename="resnet34-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_cb],
        enable_progress_bar=False  # Disable progress bar
    )

    model = ResNetClassifier(class_weights=weights)
    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), "./out_model/uterine_cancer_resnet34.pth")
    print("Training complete. Model saved to 'out_model/uterine_cancer_resnet34.pth'.")

if __name__ == "__main__":
    main()
