import os, glob, random
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.filters import gaussian, sobel, threshold_otsu, prewitt
from skimage.measure import label, regionprops
from XBNet import XBNet

# ── CONFIG ─────────────────────────────────────────────
VISUALIZE   = True           # set False to skip all plotting
MIN_CELL_PIX = 200           # minimum area in pixels
CONNECTIVITY = 2             # 1=4‐conn, 2=8‐conn for connected components
TH_FACTOR = 1.0             # 1.0 = pure Otsu, >1.0 = more stringent
# ───────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

case = "PhC-C2DH-U373"
out_dir = os.path.join("out")
data_dir = os.path.join("data", "train", case)
img_size = (256, 256)

model = XBNet(base_chns = 32, n_group = 8, input_size = img_size, output_size = img_size).to(device)
ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# first pass: find global min/max for normalization
Min, Max = np.inf, -np.inf
for subdir in ["01","02"]:
    for name in sorted(os.listdir(os.path.join(data_dir, subdir))):
        img = np.asarray(
            imageio.volread(os.path.join(data_dir, subdir, name)),
            dtype=np.double
        )
        Min = min(Min, img.min())
        Max = max(Max, img.max())

# container for all metrics
all_metrics = []

for subdir in ["01","02"]:
    for name in sorted(os.listdir(os.path.join(data_dir, subdir))):
        # ── 1) load & normalize ─────────────────────
        img = np.asarray(
            imageio.volread(os.path.join(data_dir, subdir, name)),
            dtype=np.double
        )

        img = ((img - Min) / (Max - Min)).astype(np.single)

        # ── 2) resize & batchify ────────────────────
        img_resized = resize(img, output_shape=img_size, order=1, anti_aliasing=True)
        batch = torch.from_numpy(img_resized[None,None,:,:]).to(device).float()

        # ── 3) predict & score map ──────────────────
        with torch.no_grad():
            pred = model(batch)
        score = pred[0, 2].cpu().numpy()   # choose your class‐index

        # ── 4) Otsu threshold → binary mask ───────
        th0      = threshold_otsu(score)
        th       = min(th0 * TH_FACTOR, score.max())
        mask_bin = (score > th).astype(np.uint8)

        # ── 5) optionally visualize raw, score, binary ──
        if VISUALIZE:
            fig, (a0,a1,a2) = plt.subplots(1,3,figsize=(12,4))
            a0.imshow(img_resized, cmap="gray");    a0.set_title("Input")
            a1.imshow(score,      cmap="viridis"); a1.set_title(f"Prediction")
            a2.imshow(mask_bin,   cmap="gray")    ; a2.set_title(f"Binarization")
            for ax in (a0,a1,a2): ax.axis("off")
            plt.tight_layout(); plt.show()

        # ── 6) extract & filter connected components ──
        labels = label(mask_bin, connectivity=CONNECTIVITY)
        props  = regionprops(labels)

        # prepare per‐cell metrics for this frame
        frame_metrics = []

        for p in props:
            if p.area <= MIN_CELL_PIX:
                continue

            cy, cx = p.centroid    # (row, col)
            A      = p.area        # pixel count
            P      = p.perimeter   # approximate contour length

            frame_metrics.append({
                "frame":     name,
                "subdir":    subdir,
                "label_id":  p.label,
                "centroid":  (float(cx), float(cy)), 
                "area":      int(A),
                "perimeter": float(P),
                "bbox":      p.bbox,            # optional
            })

        all_metrics.extend(frame_metrics)

        # ── 7) optionally visualize a few cell masks ──
        if VISUALIZE and frame_metrics:
            n = len(frame_metrics)
            fig, axes = plt.subplots(1, n, figsize=(4*n,4))
            for i, mets in enumerate(frame_metrics[:n]):
                lbl = mets["label_id"]
                cell_mask = (labels == lbl)
                ax = axes[i]
                ax.imshow(cell_mask, cmap="gray")
                ax.set_title(
                    f"Cell {lbl}\nA={mets['area']} P={mets['perimeter']:.1f}"
                )
                ax.axis("off")
            plt.tight_layout(); plt.show()

# ── after loop, you have all_metrics as a list of dicts ──
# Example: convert to a pandas DataFrame and save
import pandas as pd
df = pd.DataFrame(all_metrics)
df.to_csv(os.path.join(out_dir, "cell_metrics.csv"), index=False)