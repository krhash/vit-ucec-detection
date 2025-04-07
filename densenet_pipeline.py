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
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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

    def __len__(self):
        return len(self.samples)

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


# 2. DenseNet Classifier
# ----------------------
class DenseNetClassifier(pl.LightningModule):
    def __init__(self, class_weights):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, 2)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Metric containers
        self.train_preds, self.train_targets = [], []
        self.val_preds, self.val_targets = [], []

        # Epoch-wise logs
        self.train_metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        self.val_metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.train_preds.extend(preds.cpu().numpy())
        self.train_targets.extend(y.cpu().numpy())
        loss = self.loss_fn(logits, y)
        return loss

    def on_train_epoch_end(self):
        preds = np.array(self.train_preds)
        targets = np.array(self.train_targets)

        acc = (preds == targets).mean()
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        roc_auc = roc_auc_score(targets, preds)

        self.train_metrics['acc'].append(acc)
        self.train_metrics['precision'].append(precision)
        self.train_metrics['recall'].append(recall)
        self.train_metrics['f1'].append(f1)
        self.train_metrics['roc_auc'].append(roc_auc)

        self.log_dict({
            'train_acc': acc,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1,
            'train_roc_auc': roc_auc
        }, prog_bar=True)

        self.train_preds.clear()
        self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())
        loss = self.loss_fn(logits, y)
        return loss

    def on_validation_epoch_end(self):
        preds = np.array(self.val_preds)
        targets = np.array(self.val_targets)

        acc = (preds == targets).mean()
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        roc_auc = roc_auc_score(targets, preds)

        self.val_metrics['acc'].append(acc)
        self.val_metrics['precision'].append(precision)
        self.val_metrics['recall'].append(recall)
        self.val_metrics['f1'].append(f1)
        self.val_metrics['roc_auc'].append(roc_auc)

        self.log_dict({
            'val_acc': acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_roc_auc': roc_auc
        }, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# 3. Training Setup
# -----------------
def main():
    DATA_DIR = "processed_slides"
    BATCH_SIZE = 32
    EPOCHS = 7

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
        enable_progress_bar=True
    )

    model = DenseNetClassifier(class_weights=weights)
    trainer.fit(model, train_loader, val_loader)

    out_model_dir = "out_model"
    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
        print(f"Directory '{out_model_dir}' created.")
    else:
        print(f"Directory '{out_model_dir}' already exists.")

    # Save the model
    torch.save(model.state_dict(), os.path.join(out_model_dir, "uterine_cancer_densenet.pth"))
    print("\nTraining Complete! Model saved to out_model/uterine_cancer_densenet.pth")

    # Plot metrics
    def plot_metrics(metric_dict, title):
        plt.figure(figsize=(10, 6))
        for key, values in metric_dict.items():
            plt.plot(values, label=key.capitalize())
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.show()

    plot_metrics(model.train_metrics, "Training Metrics")
    plot_metrics(model.val_metrics, "Validation Metrics")


if __name__ == "__main__":
    main()