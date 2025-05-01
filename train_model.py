import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from skimage.transform import resize
from XBNet import XBNet

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

out_dir = os.path.join("out")
data_dir = os.path.join("dataset", "train")
img_size = (256, 256)
os.makedirs(out_dir, exist_ok=True)

# Load data
class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size = img_size):
        """
        image_dir: path to folder with files like image_00001.npy
        mask_dir:  path to folder with files like mask_00001.npy
        """
        self.img_size = img_size
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "image_*.npy")))
        self.mask_paths  = sorted(glob.glob(os.path.join(mask_dir,  "mask_*.npy")))

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"
        
        # Optionally check that indexes match
        for img_path, msk_path in zip(self.image_paths, self.mask_paths):
            idx_img = os.path.splitext(os.path.basename(img_path))[0].split("_")[-1]
            idx_msk = os.path.splitext(os.path.basename(msk_path))[0].split("_")[-1]
            assert idx_img == idx_msk, f"Index mismatch: {img_path} vs {msk_path}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load
        img = np.load(self.image_paths[idx], allow_pickle=True).astype(np.float32) # shape: (1, H, W)
        msk = np.load(self.mask_paths[idx], allow_pickle=True).astype(np.float32)  # shape: (H, W)
        h, w = img.shape
        aug = random.randrange(6)

        if random.random() < 0.4:
            assert h >= self.img_size[0] and w >= self.img_size[1], "crop_size larger than image"
            r1 = random.randint(0, h - self.img_size[0])
            c1 = random.randint(0, w - self.img_size[1])
            img = img[r1:r1 + self.img_size[0], c1:c1 + self.img_size[1]]
            msk = msk[r1:r1 + self.img_size[0], c1:c1 + self.img_size[1]]
        else:
            img = resize(
                    img,
                    output_shape=self.img_size,
                    order=1,            # 0=nearest, 1=bilinear, 3=bicubic; choose what you like
                    anti_aliasing=True  
                )

            msk = resize(
                msk,
                output_shape=self.img_size,
                order=1,            # 0=nearest, 1=bilinear, 3=bicubic; choose what you like
                anti_aliasing=True  
                )
            
        match aug:
            case 0:
                img = np.fliplr(img); msk = np.fliplr(msk)
            case 1:
                img = np.flipud(img); msk = np.flipud(msk)
            case 2:
                img = np.rot90(img, k=1); msk = np.rot90(msk, k=1)
            case 3:
                img = np.rot90(img, k=2); msk = np.rot90(msk, k=2)
            case 4:
                img = np.rot90(img, k=3); msk = np.rot90(msk, k=3)
            case _:
                img, msk = img, msk
                

        return tuple([img.copy()[None, :, :], msk.copy()]) 
dataset = CellSegmentationDataset(
    image_dir = os.path.join(data_dir, "image"),
    mask_dir  = os.path.join(data_dir, "mask"),
)
      
n_test  = int(0.3 * len(dataset))
n_train = len(dataset) - n_test
print(f"Using {n_train} train samples")
print(f"Using {n_test} test samples")

train_ds, test_ds = random_split(dataset, [n_train, n_test])

train_loader = DataLoader(
    train_ds,
     batch_size=8,
    shuffle=True,      
    pin_memory=True,
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)

test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)

# Visualize Data
image_batch, mask_batch = next(iter(train_loader))
n = min(8, image_batch.size(0))

fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
for i in range(n):
    # take channel 0 → shape (H, W)
    img = image_batch[i, 0].cpu().numpy()
    msk = mask_batch[i].cpu().numpy()

    # Left: raw image
    ax = axes[i, 0]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Image  #{i}")
    ax.axis("off")

    # Right: mask
    ax = axes[i, 1]
    ax.imshow(msk, cmap="jet", vmin=0, vmax=msk.max())
    ax.set_title(f"Mask   #{i}")
    ax.axis("off")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(n, 1, figsize=(5, 3 * n))
for i in range(n):
    img = image_batch[i, 0].cpu().numpy()
    msk = mask_batch[i].cpu().numpy()

    ax = axes[i]
    ax.imshow(img, cmap="gray")
    ax.imshow(msk, cmap="jet", alpha=0.4, vmin=0, vmax=msk.max())
    ax.set_title(f"Overlay #{i}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Optimizer and Model
model = XBNet(base_chns = 32, n_group = 8).to(device)

weights_arg = [1,10,5]
weights = torch.tensor(weights_arg).to(device).float()
criterion = F.cross_entropy

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)

ckpt = {
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_history": [],
    "test_history": [],
}

torch.save(ckpt, os.path.join(out_dir, "model.pt"))

# Train
epochs = 1000
start_epoch = 0
train_history = []
test_history  = []

resume = True
if resume and os.path.exists(os.path.join(out_dir, "model.pt")):
    ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch    = ckpt["epoch"]
    train_history  = ckpt["train_history"]
    test_history   = ckpt["test_history"]

for epoch in range(epochs):
    model.train()
    train_loss_accum = 0.0
    train_loader_tq = tqdm(
        train_loader, 
        desc=f"Epoch {epoch+1}/{epochs} [Train]",
        leave=False
    )
    
    for image_batch, mask_batch in train_loader_tq:
        image_batch = image_batch.to(device, non_blocking=True).float()
        mask_batch  = mask_batch.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        pred = model(image_batch)
        loss = criterion(pred, mask_batch, weight=weights)
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()
        train_loader_tq.set_postfix(train_loss=train_loss_accum/(train_loader_tq.n+1))

    # — evaluate on test set —
    model.eval()
    test_loss_accum = 0.0
    test_loader_tq = tqdm(
        test_loader,
        desc=f"Epoch {epoch+1}/{epochs} [Test ]",
        leave=False
    )
    with torch.no_grad():
        for image_batch, mask_batch in test_loader_tq:
            image_batch = image_batch.to(device, non_blocking=True).float()
            mask_batch  = mask_batch.to(device, non_blocking=True).long()

            pred = model(image_batch)
            loss = criterion(pred, mask_batch, weight=weights)
            test_loss_accum += loss.item()
            test_loader_tq.set_postfix(test_loss=test_loss_accum/(test_loader_tq.n+1))

    train_loss = train_loss_accum / len(train_loader)
    test_loss  = test_loss_accum  / len(test_loader)

    train_history.append(train_loss)
    test_history.append(test_loss)

    print(
        f"Epoch {start_epoch + epoch + 1:02d}/{start_epoch + epochs:02d} — "
        f"Train loss: {train_loss:.6f} | "
        f"Test loss:  {test_loss:.6f} ({test_loss * 100:.2f}%)"
    )

    ckpt = {
        "epoch": start_epoch + epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_history": train_history,
        "test_history": test_history,
    }

    torch.save(ckpt, os.path.join(out_dir, "model.pt"))

# Loss History
plt.plot(train_history, label="Train Loss")
plt.plot(test_history,  label="Test  Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# Model eval
model.eval()
image_batch, gt_batch = next(iter(test_loader))

with torch.no_grad():
    image_batch = image_batch.to(device, non_blocking=True).float()
    gt_batch    = gt_batch.to(device,    non_blocking=True).long()    # <-- use gt_batch, not mask_batch
    pred_batch  = model(image_batch)

n = min(8, image_batch.size(0))
fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

for i in range(n):
    img       = image_batch[i, 0].cpu().numpy()            # (H, W)
    pred_mask = pred_batch[i, 2].cpu().numpy()             # (H, W)
    # adjust this if your gt_batch is (B,1,H,W) vs (B,H,W):
    if gt_batch.dim() == 4:
        gt_mask = gt_batch[i, 0].cpu().numpy()
    else:
        gt_mask = gt_batch[i].cpu().numpy()

    # Predicted overlay
    ax = axes[i, 0]
    ax.imshow(img,       cmap="gray")
    ax.imshow(pred_mask, cmap="jet", alpha=0.4, vmin=0, vmax=gt_mask.max())
    ax.set_title(f"Predicted #{i}")
    ax.axis("off")

    # GT overlay
    ax = axes[i, 1]
    ax.imshow(img,    cmap="gray")
    ax.imshow(gt_mask, cmap="jet", alpha=0.4, vmin=0, vmax=gt_mask.max())
    ax.set_title(f"Ground-Truth #{i}")
    ax.axis("off")

plt.tight_layout()
plt.show()