# Histopathology Vision Transformer Pipeline
# ------------------------------------------
# Features:
# - 224x224 input validation
# - Enhanced class balancing (35k normal vs 90k tumor)
# - GPU-optimized checkpointing
# - 10-epoch training with progress tracking

import os
import cv2
import json
import openslide
import numpy as np
import torch
import timm
import albumentations as A
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.morphology import remove_small_objects, remove_small_holes
from torchmetrics.classification import MulticlassAccuracy

# Configure image handling
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 1. Slide Processing Pipeline
# ----------------------------
# Step 1: Enhanced Slide Processing with Caching
# ----------------------------------------------
class SlideProcessor:
    def __init__(self, label_path, output_dir, target_size=224,
                 white_threshold=0.3, dark_threshold=0.3):
        self.label_path = label_path
        self.output_dir = output_dir
        self.target_size = target_size
        self.white_threshold = white_threshold  # Max allowed white pixels ratio
        self.dark_threshold = dark_threshold  # Max allowed dark pixels ratio
        self.cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_labels(self):
        with open(self.label_path) as f:
            return {item['slide_name']: item['description'] for item in json.load(f)}

    def _find_slide_path(self, slide_name, root_dir):
        for root, _, files in os.walk(root_dir):
            if slide_name in files:
                return os.path.join(root, slide_name)
        return None

    def _is_valid_patch(self, patch):
        """Check if patch contains sufficient tissue content"""
        """ Ignore dark, white patches"""
        # Convert to different color spaces for improved analysis
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

        # 1. Detect white regions (high brightness)
        white_mask = gray > 220
        white_percentage = np.mean(white_mask)

        # 2. Detect grey regions (medium brightness, low saturation)
        grey_mask = cv2.inRange(hsv, (0, 0, 100), (180, 30, 220))
        grey_percentage = np.mean(grey_mask > 0)

        # 3. Measure tissue presence through texture and color variation
        texture_score = np.std(gray) / 255.0  # Normalized texture variation
        color_score = np.mean(hsv[:, :, 1]) / 255.0  # Saturation channel

        # 4. Calculate overall tissue content score
        tissue_content = color_score * texture_score * 100

        # 5. Apply stricter thresholds based on the sample images
        return (white_percentage < 0.30 and
                grey_percentage < 0.40 and
                tissue_content > 2.0)

    def process_slides(self, input_dir):
        labels = self._load_labels()

        # Create process pool for parallelization
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = []
            for slide_name, desc in labels.items():
                slide_path = self._find_slide_path(slide_name, input_dir)
                if slide_path:
                    futures.append(executor.submit(
                        self._process_single_slide,
                        slide_path,
                        desc,
                        self.target_size
                    ))

            for future in futures:
                future.result()  # Wait for completion

    def _process_single_slide(self, slide_path, description, target_size):
        slide_id = os.path.basename(slide_path).split('.')[0]
        output_subdir = os.path.join(self.output_dir,
                                     'tumor' if 'tumor' in description.lower() else 'normal')
        os.makedirs(output_subdir, exist_ok=True)

        with openslide.OpenSlide(slide_path) as slide:
            best_level = slide.get_best_level_for_downsample(8)
            w, h = slide.level_dimensions[best_level]

            valid_patches = 0
            total_patches = 0

            for y in range(0, h, target_size):
                for x in range(0, w, target_size):
                    total_patches += 1
                    patch = slide.read_region(
                        (x * (2 ** best_level), y * (2 ** best_level)),
                        best_level,
                        (target_size, target_size)
                    ).convert('RGB')

                    patch_np = np.array(patch)

                    # Apply improved quality filter
                    if self._is_valid_patch(patch_np):
                        valid_patches += 1
                        patch_path = os.path.join(output_subdir,
                                                  f"{slide_id}_X{x}_Y{y}.png")
                        patch.save(patch_path)

            print(
                f"Slide {slide_id}: Kept {valid_patches}/{total_patches} patches ({(valid_patches / total_patches) * 100:.1f}%)")


# 2. Data Pipeline with Size Validation
# -------------------------------------
class HistoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = ['normal', 'tumor']
        self.samples = []
        # Sort files for consistent ordering
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            self.samples += [
                (os.path.join(cls_dir, f), self.classes.index(cls))
                for f in sorted(os.listdir(cls_dir))
                if f.endswith('.png')
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx][0]).convert('RGB')
            img = np.array(img)

            # Ensure 224x224 input
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))

            if self.transform:
                img = self.transform(image=img)['image']

            return torch.tensor(img).permute(2, 0, 1).float(), self.samples[idx][1]
        except Exception as e:
            print(f"Error loading {self.samples[idx][0]}: {str(e)}")
            return torch.zeros(3, 224, 224), -1

# VIT Classifier with weight balancing
class ViTClassifier(pl.LightningModule):
    def __init__(self, class_weights):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224',
                                       pretrained=True,
                                       num_classes=2)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.train_loss.append(loss.detach())
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = (torch.argmax(outputs, 1) == y).float().mean()
        self.val_loss.append(loss)
        self.val_acc.append(acc)
        self.log('val_loss', loss, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_loss).mean()
        print(f"\nEpoch {self.current_epoch+1}/7")
        print(f"Train Loss: {avg_loss:.4f}")
        self.train_loss = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_acc = torch.stack(self.val_acc).mean()
        print(f"Val Loss: {avg_loss:.4f} | Val Acc: {avg_acc:.2%}")
        print("-----------------------------------")
        self.val_loss = []
        self.val_acc = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-5, weight_decay=0.05)


# 4. Execution Workflow
# ---------------------
def main():
    # Configuration
    DATA_DIR = "processed_slides"
    LABEL_PATH = "labels.json"
    MAX_EPOCHS = 7

    # Process slides (uncomment to regenerate)
    SlideProcessor(LABEL_PATH, DATA_DIR).process_slides("UCEC")

    # Data transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(
            num_holes_range=(6, 8),
            hole_height_range=(24, 32),
            hole_width_range=(24, 32),
            p=0.5
        )
    ])

    val_transform = A.Compose([A.Resize(224, 224)])

    # Create base dataset for splitting
    full_dataset = HistoDataset(DATA_DIR)
    full_dataset.samples = [s for s in full_dataset.samples if os.path.getsize(s[0]) > 1024]

    # Split indices
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=[s[1] for s in full_dataset.samples]
    )

    # Create transformed datasets
    train_dataset = HistoDataset(DATA_DIR, transform=train_transform)
    val_dataset = HistoDataset(DATA_DIR, transform=val_transform)

    # Calculate class weights from training indices
    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = (class_counts.sum() / (2 * class_counts)).float()
    print(f"\nClass Weights - Normal: {class_weights[0]:.2f}, Tumor: {class_weights[1]:.2f}")

    # Create data loaders with transforms
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=32,
        sampler=WeightedRandomSampler(
            weights=class_weights[train_labels],
            num_samples=len(train_idx),
            replacement=True
        ),
        num_workers=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=32,
        num_workers=4,
        persistent_workers=True
    )

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train model
    print("\nStarting Training...")
    model = ViTClassifier(class_weights=class_weights)
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    best_model = ViTClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        class_weights=class_weights
    )

    out_model_dir = "out_model"
    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
        print(f"Directory '{out_model_dir}' created.")
    else:
        print(f"Directory '{out_model_dir}' already exists.")

    # Save the model state_dict
    torch.save(best_model.state_dict(), os.path.join(out_model_dir, "uterine_cancer_vit.pth"))
    print("\nTraining Complete! Model saved to out_model/uterine_cancer_vit.pth")

if __name__ == '__main__':
    main()