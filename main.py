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

import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random

base_dir = "/mnt/storage/kellen/deep_learning/BCSS"

for subdir in ["train", "train_mask", "val", "val_mask", "test"]:
    path = os.path.join(base_dir, subdir)
    print(f"{subdir}: {len(os.listdir(path))} files")

# print("\nSample file names:")
# print("train:", os.listdir(os.path.join(base_dir, "train"))[:3])
# print("train_mask:", os.listdir(os.path.join(base_dir, "train_mask"))[:3])

class BCSSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 資料增強（訓練時才用）
        if self.augment:
            # 隨機水平翻轉
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 隨機垂直翻轉
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # 隨機旋轉（90, 180, 270度）
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            
            # 隨機色彩調整（只對影像）
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        # 轉成 tensor
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

train_dataset = BCSSDataset(
    img_dir="/mnt/storage/kellen/deep_learning/BCSS/train",
    mask_dir="/mnt/storage/kellen/deep_learning/BCSS/train_mask",
    augment=True
)

val_dataset = BCSSDataset(
    img_dir="/mnt/storage/kellen/deep_learning/BCSS/val",
    mask_dir="/mnt/storage/kellen/deep_learning/BCSS/val_mask",
    augment=False
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

num_epochs = 30

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

history = {
    'train_loss': [],
    'val_loss': [],
    'train_dice': [],
    'val_dice': [],
    'pixel_acc': [],
    'learning_rate': []
}

best_dice = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    train_dice_list = []  # 新增：記錄訓練 dice
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # 計算訓練時的 dice
        preds = torch.argmax(outputs, dim=1)
        train_dice_list.append(dice_coefficient(preds, masks))
    
    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = sum(train_dice_list) / len(train_dice_list)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    pixel_acc_list = []
    dice_list = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            images, masks = images.to(device), masks.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            val_loss += criterion(outputs, masks).item()
            pixel_acc_list.append(pixel_accuracy(preds, masks))
            dice_list.append(dice_coefficient(preds, masks))
    
    avg_val_loss = val_loss / len(val_loader)
    avg_pixel_acc = sum(pixel_acc_list) / len(pixel_acc_list)
    avg_dice = sum(dice_list) / len(dice_list)

    print(f"  Val Loss: {avg_val_loss:.4f}, Pixel Acc: {avg_pixel_acc:.4f}, Val Dice: {avg_dice:.4f}")
    
    # ===== 記錄歷史 =====
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_dice'].append(avg_train_dice)
    history['val_dice'].append(avg_dice)
    history['pixel_acc'].append(avg_pixel_acc)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    # ===== 儲存最佳模型 =====
    if avg_dice > best_dice:
        best_dice = avg_dice
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': avg_dice,
        }, 'best_model.pth')
        print(f"  ✓ Saved best model! Best Dice: {best_dice:.4f}")
    
    print("-" * 60)

# ===== 訓練結束後畫圖 =====
print("\n" + "="*60)
print("Training Completed!")
print(f"Best Validation Dice: {best_dice:.4f} at Epoch {best_epoch}")
print("="*60)

# ===== 繪製訓練曲線 =====
import matplotlib
matplotlib.use('Agg')  # 新增：使用非互動式後端
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_range = range(1, num_epochs + 1)

# 1. Loss Curve
axes[0, 0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Dice Coefficient Curve
axes[0, 1].plot(epochs_range, history['train_dice'], 'b-o', label='Train Dice', linewidth=2, markersize=4)
axes[0, 1].plot(epochs_range, history['val_dice'], 'r-s', label='Val Dice', linewidth=2, markersize=4)
axes[0, 1].axhline(y=best_dice, color='g', linestyle='--', linewidth=2, label=f'Best: {best_dice:.4f}')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
axes[0, 1].set_title('Dice Coefficient Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. Pixel Accuracy Curve
axes[1, 0].plot(epochs_range, history['pixel_acc'], 'g-^', label='Pixel Accuracy', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Pixel Accuracy', fontsize=12)
axes[1, 0].set_title('Pixel Accuracy Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 4. Overfitting 分析 (Train vs Val Dice)
axes[1, 1].plot(epochs_range, history['train_dice'], 'b-o', label='Train Dice', linewidth=2, markersize=4)
axes[1, 1].plot(epochs_range, history['val_dice'], 'r-s', label='Val Dice', linewidth=2, markersize=4)
gap = [train - val for train, val in zip(history['train_dice'], history['val_dice'])]
axes[1, 1].fill_between(epochs_range, history['train_dice'], history['val_dice'], 
                         alpha=0.3, color='yellow', label='Overfitting Gap')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Dice Coefficient', fontsize=12)
axes[1, 1].set_title('Train vs Val Dice (Overfitting Analysis)', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./training_curves_combine_loss_1.png', dpi=300, bbox_inches='tight')
print("\n✓ Training curves saved to './training_curves_combine_loss_1.png'")
plt.close()  # 新增：關閉圖形釋放記憶體

# ===== 額外：繪製詳細的 Dice 趨勢 =====
fig2, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs_range, history['train_dice'], 'b-o', label='Train Dice', linewidth=2, markersize=5)
ax.plot(epochs_range, history['val_dice'], 'r-s', label='Val Dice', linewidth=2, markersize=5)
ax.axhline(y=best_dice, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax.scatter([best_epoch], [best_dice], color='gold', s=200, zorder=5, 
           edgecolors='black', linewidth=2, label=f'Best: {best_dice:.4f} @ Epoch {best_epoch}')

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Dice Coefficient', fontsize=14)
ax.set_title('Dice Coefficient Over Time', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# 加上數值標註（每5個epoch）
for i in range(0, num_epochs, 5):
    ax.annotate(f'{history["val_dice"][i]:.3f}', 
                xy=(i+1, history["val_dice"][i]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('./dice_trend_combine_loss_1.png', dpi=300, bbox_inches='tight')
print("✓ Dice trend plot saved to './dice_trend_combine_loss_1.png'")
plt.close()  # 新增：關閉圖形釋放記憶體

# ===== 印出最終統計 =====
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Total Epochs: {num_epochs}")
print(f"Best Validation Dice: {best_dice:.4f} (Epoch {best_epoch})")
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
print(f"Final Train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final Val Dice: {history['val_dice'][-1]:.4f}")
print(f"Final Pixel Accuracy: {history['pixel_acc'][-1]:.4f}")

# 分析 overfitting
final_gap = history['train_dice'][-1] - history['val_dice'][-1]
if final_gap > 0.05:
    print(f"\n⚠️  Warning: Potential overfitting detected! (Gap: {final_gap:.4f})")
    print("    Consider: data augmentation, dropout, or early stopping")
elif final_gap < 0:
    print(f"\n⚠️  Warning: Val Dice > Train Dice (Gap: {final_gap:.4f})")
    print("    This might indicate: insufficient training or data leakage")
else:
    print(f"\n✓ Good fit! Train-Val gap: {final_gap:.4f}")

print("="*60)

# ===== 儲存訓練歷史到 CSV =====
import pandas as pd

history_df = pd.DataFrame({
    'epoch': list(range(1, num_epochs + 1)),
    'train_loss': history['train_loss'],
    'val_loss': history['val_loss'],
    'train_dice': history['train_dice'],
    'val_dice': history['val_dice'],
    'pixel_acc': history['pixel_acc'],
    'learning_rate': history['learning_rate']
})

history_df.to_csv('./training_history.csv', index=False)
print("\n✓ Training history saved to './training_history.csv'")