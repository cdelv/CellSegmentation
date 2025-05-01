#!/usr/bin/env python3
import os
import glob
import random

import numpy as np
import imageio
import matplotlib.pyplot as plt
out_dir        = "out"

def find_case_dirs(root):
    # Return only real directories (no .zip)
    return [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

def pick_random_tiff(case_path):
    sub01 = os.path.join(case_path, "01")
    # look for both .tif and .tiff
    patterns = [os.path.join(sub01, "*.tif"), os.path.join(sub01, "*.tiff")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No .tif files in {sub01}")
    return random.choice(files)

def load_image(path):
    vol = np.asarray(imageio.volread(path), dtype=np.float32)
    # if it's a stack, take first slice
    if vol.ndim == 3:
        return vol[0]
    return vol

def plot_collage(imgs, titles, ncols=5, figsize=(15, 6)):
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # turn off any unused axes
    for ax in axes[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collage.png"), dpi=200)
    plt.show()

def main():
    root = os.path.join("data", "train")
    cases = sorted(find_case_dirs(root))
    # limit to 10 for a 2×5 grid
    cases = cases[:10]

    imgs, titles = [], []
    for case in cases:
        path = os.path.join(root, case)
        try:
            tiff_path = pick_random_tiff(path)
            img = load_image(tiff_path)
            imgs.append(img)
            titles.append(case)
        except FileNotFoundError as e:
            print(f"Skipping {case}: {e}")

    if imgs:
        plot_collage(imgs, titles, ncols=5)

if __name__ == "__main__":
    main()
