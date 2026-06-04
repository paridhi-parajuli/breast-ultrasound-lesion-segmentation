# Colab: install deps first (run once)
# !pip install segmentation-models-pytorch

import os
import random

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# ---------------- DATASET ----------------
# BreastCancerDataset()[i] = (image, mask, label), correctly pairing each
# image with its mask and cancer type.
class BreastCancerDataset(Dataset):

    def __init__(self, root_dir,
                 image_transform=None,
                 mask_transform=None):

        self.root = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.samples = []

        cancer_type_to_label = {
            "normal": 0,
            "benign": 1,
            "malignant": 2
        }

        for cancer_type in ["normal", "benign", "malignant"]:
            class_dir = os.path.join(root_dir, cancer_type)

            for f in os.listdir(class_dir):
                if "_mask" in f:
                    continue

                image_path = os.path.join(class_dir, f)
                mask_path = os.path.join(
                    class_dir,
                    f.replace(".png", "_mask.png")
                )

                if os.path.exists(mask_path):
                    self.samples.append(
                        (image_path,
                         mask_path,
                         cancer_type_to_label[cancer_type])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path, label = self.samples[i]

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0).float()

        return image, mask, label


# ---------------- DATA AUGMENTATION (train only) ----------------
# Applies the SAME random geometric transform to both image and mask so they
# stay aligned. This is the biggest lever for a small dataset like BUSI.
class AugmentedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        image, mask, label = self.subset[i]

        # random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # random vertical flip
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # random rotation (+/- 15 deg): bilinear for image, nearest for mask
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # keep mask strictly binary after interpolation
        mask = (mask > 0.5).float()

        return image, mask, label


# ---------------- MODEL ----------------
# ResNet34 U-Net for lesion segmentation. A classification head was tried on
# the shared encoder but hurt segmentation on this small dataset, so the
# model is segmentation-only.
class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=1,
            depth=5,
            weights="imagenet"
        )

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5
        )

        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        seg_out = self.seg_head(decoder_output)
        return seg_out


# ---------------- LOSS ----------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # convert logits -> probabilities
        probs = torch.sigmoid(logits)

        # flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


# ---------------- DATA ----------------
# Normalize the grayscale input to ~zero-mean/unit-var so it matches the
# distribution the imagenet-pretrained encoder expects (helps convergence).
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

mask_transform = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=transforms.InterpolationMode.NEAREST
    ),
    transforms.ToTensor(),
])

dataset = BreastCancerDataset(
    root_dir="/content/drive/MyDrive/Dataset_BUSI_with_GT",
    image_transform=image_transform,
    mask_transform=mask_transform,
)

# 70% train, 15% val, 15% test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Augment only the training split; val/test stay clean
train_dataset = AugmentedDataset(train_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# ---------------- TRAINING SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResUNet().to(device)

# Hyperparameters
LR = 1e-4
WEIGHT_DECAY = 1e-4
BCE_WEIGHT = 0.3
DICE_WEIGHT = 0.7
num_epochs = 50
patience = 10

# AdamW adds weight decay -> a bit of regularization on the small dataset
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Drop LR when val Dice plateaus so it can fine-tune instead of bouncing
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=4
)

seg_loss_fn = nn.BCEWithLogitsLoss()
dice_loss_fn = DiceLoss()


# ---------------- TRAINING LOOP ----------------
best_val_dice = 0
counter = 0

for epoch in range(num_epochs):

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0

    for images, masks, labels in train_dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        seg_out = model(images)

        seg_bce = seg_loss_fn(seg_out, masks)
        seg_dice = dice_loss_fn(seg_out, masks)
        loss = BCE_WEIGHT * seg_bce + DICE_WEIGHT * seg_dice

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0
    val_dice_total = 0

    with torch.no_grad():
        for images, masks, labels in val_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            seg_out = model(images)

            seg_bce = seg_loss_fn(seg_out, masks)
            seg_dice_loss = dice_loss_fn(seg_out, masks)
            loss = BCE_WEIGHT * seg_bce + DICE_WEIGHT * seg_dice_loss
            val_loss += loss.item()

            # Dice metric on thresholded predictions
            probs = torch.sigmoid(seg_out)
            preds = (probs > 0.5).float()

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            val_dice_total += dice.mean().item()

    val_loss /= len(val_dataloader)
    val_dice = val_dice_total / len(val_dataloader)

    # step the scheduler on the metric we care about
    scheduler.step(val_dice)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Dice: {val_dice:.4f} | "
          f"LR: {current_lr:.2e}")

    # ---------------- EARLY STOPPING + CHECKPOINT ----------------
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        counter = 0

        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved best model")

    else:
        counter += 1
        print(f"No improvement. Patience: {counter}/{patience}")

        if counter >= patience:
            print("🛑 Early stopping triggered")
            break


# ---------------- FINAL TEST ----------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_dice_total = 0

with torch.no_grad():
    for images, masks, labels in test_dataloader:
        images = images.to(device)
        masks = masks.to(device)

        seg_out = model(images)

        probs = torch.sigmoid(seg_out)
        preds = (probs > 0.5).float()

        intersection = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        test_dice_total += dice.mean().item()

test_dice = test_dice_total / len(test_dataloader)

print("🏁 FINAL TEST DICE:", test_dice)
