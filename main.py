# confirm the format of dataset, using 224*224 pixels dataset, 3 classes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

base_dir = "/mnt/storage/kellen/deep_learning/BCSS"

for subdir in ["train", "train_mask", "val", "val_mask", "test"]:
    path = os.path.join(base_dir, subdir)
    print(f"{subdir}: {len(os.listdir(path))} files")

# print("\nSample file names:")
# print("train:", os.listdir(os.path.join(base_dir, "train"))[:3])
# print("train_mask:", os.listdir(os.path.join(base_dir, "train_mask"))[:3])

class BCSSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))  # make the order same

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # read image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # to tensor (when not use transform)
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

train_dataset = BCSSDataset(
    img_dir="/mnt/storage/kellen/deep_learning/BCSS/train",
    mask_dir="/mnt/storage/kellen/deep_learning/BCSS/train_mask"
)

val_dataset = BCSSDataset(
    img_dir="/mnt/storage/kellen/deep_learning/BCSS/val",
    mask_dir="/mnt/storage/kellen/deep_learning/BCSS/val_mask"
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

# --- Basic building block ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# --- Full UNet ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        # Decoder
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return self.out_conv(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet(in_channels=3, out_channels=3)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50

def pixel_accuracy(pred, mask):
    return (pred == mask).float().mean().item()

def dice_coefficient(pred, mask, num_classes=3, eps=1e-6):
    dice = 0.0
    for c in range(num_classes):
        pred_c = (pred == c).float()
        mask_c = (mask == c).float()
        intersection = (pred_c * mask_c).sum()
        union = pred_c.sum() + mask_c.sum()
        dice += (2 * intersection + eps) / (union + eps)
    return (dice / num_classes).item()

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    pixel_acc_list = []
    dice_list = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            val_loss += criterion(outputs, masks).item()
            pixel_acc_list.append(pixel_accuracy(preds, masks))
            dice_list.append(dice_coefficient(preds, masks))
    
    avg_val_loss = val_loss / len(val_loader)
    avg_pixel_acc = sum(pixel_acc_list) / len(pixel_acc_list)
    avg_dice = sum(dice_list) / len(dice_list)

    print(f"Validation Loss: {avg_val_loss:.4f}, Pixel Acc: {avg_pixel_acc:.4f}, Dice: {avg_dice:.4f}")
    # torch.save(model.state_dict(), f"unet_epoch{epoch+1}.pt")