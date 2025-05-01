import os, glob, random
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import pandas as pd

from skimage.transform import resize
from skimage.filters import gaussian, sobel, threshold_otsu, prewitt
from skimage.measure import label, regionprops
from XBNet import XBNet

# ── CONFIG ─────────────────────────────────────────────
VISUALIZE   = False           # set False to skip all plotting
MIN_CELL_PIX = 200           # minimum area in pixels
CONNECTIVITY = 2             # 1=4‐conn, 2=8‐conn for connected components
TH_FACTOR = 1.0             # 1.0 = pure Otsu, >1.0 = more stringent

# PHYSICAL UNITS
img_size        = (256, 256)    # Size after resize
PIXEL_SIZE_UM   = 0.65          # microns per pixel
TIME_STEP_MIN   = 15            # minutes per frame
# ───────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

case = "PhC-C2DH-U373"
out_dir = os.path.join("out")
data_dir = "data"
os.makedirs(out_dir, exist_ok=True)
original_size = (0, 0)

model = XBNet(base_chns = 32, n_group = 8, input_size = img_size, output_size = img_size).to(device)
ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# first pass: find global min/max for normalization
Min, Max = np.inf, -np.inf

for data in ["test", "train"]:
    for subdir in ["01","02"]:
        for name in sorted(os.listdir(os.path.join(data_dir, data, case, subdir))):
            img = np.asarray(
                imageio.volread(os.path.join(data_dir, data, case, subdir, name)),
                dtype=np.double
            )
            Min = min(Min, img.min())
            Max = max(Max, img.max())
            original_size = img.shape

# container for all metrics
all_metrics = []

for data in ["test", "train"]:
    for subdir in ["01","02"]:
        for nframe, name in enumerate(sorted(os.listdir(os.path.join(data_dir, data, case, subdir)))):
            # ── 1) load & normalize ─────────────────────
            img = np.asarray(
                imageio.volread(os.path.join(data_dir, data, case, subdir, name)),
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
            th       = th0 * TH_FACTOR
            mask_bin = (score > th).astype(np.uint8)

            # ── 5) optionally visualize raw, score, binary ──
            if VISUALIZE or nframe == 0:
                fig, (a0,a1,a2) = plt.subplots(1,3,figsize=(12,4))
                a0.imshow(img_resized, cmap="gray");    a0.set_title("Input")
                a1.imshow(score,      cmap="viridis"); a1.set_title(f"Prediction")
                a2.imshow(mask_bin,   cmap="gray")    ; a2.set_title(f"Binarization")
                for ax in (a0,a1,a2): ax.axis("off")

                if nframe == 0:
                    plt.savefig(os.path.join(out_dir, "prediction.png"), dpi=200)

                plt.tight_layout()
                plt.show()

            # ── 6) extract & filter connected components ──
            labels = label(mask_bin, connectivity=CONNECTIVITY)
            props  = regionprops(labels)

            # prepare per‐cell metrics for this frame
            frame_metrics = []

            for p in props:
                if p.area <= MIN_CELL_PIX:
                    continue

                cy, cx = (PIXEL_SIZE_UM * p.centroid[0] * original_size[0] /  img_size[0], PIXEL_SIZE_UM * p.centroid[1] * original_size[1] /  img_size[1])
                A      = PIXEL_SIZE_UM**2 * p.area        # pixel count
                P      = PIXEL_SIZE_UM * p.perimeter   # approximate contour length

                frame_metrics.append({
                    "frame":     name,
                    "time":      nframe * TIME_STEP_MIN,
                    "subdir":    subdir,
                    "label_id":  p.label,
                    "centroid":  (float(cx), float(cy)), 
                    "area":      float(A),
                    "perimeter": float(P),
                    "bbox":      p.bbox,            # optional
                })

            all_metrics.extend(frame_metrics)

            # ── 7) optionally visualize a few cell masks ──
            if VISUALIZE and frame_metrics or nframe == 0:
                n = len(frame_metrics)
                fig, axes = plt.subplots(1, n, figsize=(4*n,4))
                for i, mets in enumerate(frame_metrics[:n]):
                    lbl = mets["label_id"]
                    cell_mask = (labels == lbl)
                    ax = axes[i]
                    ax.imshow(cell_mask, cmap="gray")
                    ax.set_title(
                        f"Cell {lbl}\nA={mets['area']:.1f} μm^2 P={mets['perimeter']:.1f} μm"
                    )
                    ax.axis("off")

                if nframe == 0:
                    plt.savefig(os.path.join(out_dir, "cell_mask.png"), dpi=200)

                plt.tight_layout()
                plt.show()


df = pd.DataFrame(all_metrics)
df.to_csv(os.path.join(out_dir, "cell_metrics.csv"), index=False)