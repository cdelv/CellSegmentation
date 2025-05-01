import os
import shutil
import glob
import random

import numpy as np
import imageio
from scipy import ndimage
import skimage.color as clr
import matplotlib.pyplot as plt

def main():
    data_dir = os.path.join("data", "train")
    test_dir = os.path.join("data", "test")
    dataset_dir = "dataset"
    itr_dilation = 2
    itr_erosion = 2

    # Create dataset directory
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "train", "image"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "train", "mask"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "test"),  exist_ok=True)

    # Get all cases in the data
    cases = []
    for path in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, path)):
            cases.append(path)

    # Find the size
    Minx = np.inf
    Maxx = -np.inf
    Miny = np.inf
    Maxy = -np.inf

    """
    for case in cases:
        path = os.path.join(data_dir, case)
        for subdir in ["01", "02"]:
            for name in sorted(os.listdir(os.path.join(path, subdir))):
                image = np.asarray(imageio.volread(os.path.join(path, subdir, name)))

                Minx = min(Minx, image.shape[0])
                Maxx = max(Maxx, image.shape[0])
                Miny = min(Miny, image.shape[1])
                Maxy = max(Maxy, image.shape[1])

    print(Minx, Maxx, Miny, Maxy)

    for case in cases:
        path = os.path.join(test_dir, case)
        for subdir in ["01", "02"]:
            for name in sorted(os.listdir(os.path.join(path, subdir))):
                image = np.asarray(imageio.volread(os.path.join(path, subdir, name)))

                Minx = min(Minx, image.shape[0])
                Maxx = max(Maxx, image.shape[0])
                Miny = min(Miny, image.shape[1])
                Maxy = max(Maxy, image.shape[1])

    print(Minx, Maxx, Miny, Maxy)
    """

    # Process all the training data
    image_ID = 0
    for case in cases:
        path = os.path.join(data_dir, case)

        # find min and max values for the case
        Min = np.inf
        Max = -np.inf
        for subdir in ["01", "02"]:
            for name in sorted(os.listdir(os.path.join(path, subdir))):
                image = np.asarray(imageio.volread(os.path.join(path, subdir, name)), dtype=np.double)
                Min = np.minimum(Min, image.min())
                Max = np.maximum(Max, image.max())

        for subdir in ["01", "02"]:
            for name in os.listdir(os.path.join(path, subdir)):

                # Load image
                image = np.asarray(imageio.volread(os.path.join(path, subdir, name)), dtype=np.single)
                if Max > Min:
                    image = (image - Min) / (Max - Min)
                else:
                    image = np.zeros_like(image)

                # Process mask
                prefix = name.split(".")[0][1:]

                if os.path.exists(os.path.join(path, subdir+"_GT")):
                    if os.path.exists(os.path.join(path, subdir+"_GT", "SEG", "man_seg"+prefix+".tif")):
                        mask = np.asarray(imageio.volread(os.path.join(path, subdir+"_GT", "SEG", "man_seg"+prefix+".tif")), dtype=np.uint16)
                        mask = seg_to_lbl(mask, itr_dilation, itr_erosion)

                        # Save image and mask
                        np.save(os.path.join(dataset_dir, "train", "image", f"image_{image_ID:05d}"), image)
                        np.save(os.path.join(dataset_dir, "train", "mask", f"mask_{image_ID:05d}"), mask)
                        image_ID += 1

                elif os.path.exists(os.path.join(path, subdir+"_ST")):
                    if os.path.exists(os.path.join(path, subdir+"_ST", "SEG", "man_seg"+prefix+".tif")):
                        mask = np.asarray(imageio.volread(os.path.join(path, subdir+"_ST", "SEG", "man_seg"+prefix+".tif")), dtype=np.uint16)
                        mask = seg_to_lbl(mask, itr_dilation, itr_erosion)

                        # Save image and mask
                        np.save(os.path.join(dataset_dir, "train", "image", f"image_{image_ID:05d}"), image)
                        np.save(os.path.join(dataset_dir, "train", "mask", f"mask_{image_ID:05d}"), mask)
                        image_ID += 1

    # Get all cases in the data
    cases = []
    for path in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, path)):
            cases.append(path)

    # Process all the training data
    image_ID = 0
    for case in cases:
        path = os.path.join(test_dir, case)

        # find min and max values for the case
        Min = np.inf
        Max = -np.inf
        for subdir in ["01", "02"]:
            for name in sorted(os.listdir(os.path.join(path, subdir))):
                image = np.asarray(imageio.volread(os.path.join(path, subdir, name)), dtype=np.double)
                Min = np.minimum(Min, image.min())
                Max = np.maximum(Max, image.max())

        for subdir in ["01", "02"]:
            for name in os.listdir(os.path.join(path, subdir)):
                image = np.asarray(imageio.volread(os.path.join(path, subdir, name)), dtype=np.single)
                if Max > Min:
                    image = (image - Min) / (Max - Min)
                else:
                    image = np.zeros_like(image)

                np.save(os.path.join(dataset_dir, "test", f"image_{image_ID:05d}"), image)
                image_ID += 1


def seg_to_lbl(seg, itr_dilation, itr_erosion):
    s1, s2 = seg.shape
    cell_set = np.delete(np.unique(seg),0) 
    background = (seg==0)
    lbl = np.zeros_like(seg, dtype=np.uint8)
    for cell in cell_set:    
        mask_original = (seg == cell)
        mask_dilated = ndimage.binary_dilation(mask_original, iterations=itr_dilation)
        mask_dilated = np.logical_and(mask_dilated, background)
        lbl[mask_dilated] = 1
        lbl[mask_original] = 2 
        mask_eroded = ndimage.binary_erosion(mask_original, iterations=itr_erosion)
        mask_boundary = np.logical_and(mask_original,np.logical_not(mask_eroded))
        ys, xs = np.nonzero(mask_boundary)
        for y, x in zip(ys, xs):
            for dy in range(-itr_erosion,itr_erosion+1):
                for dx in range(-itr_erosion,itr_erosion+1):
                    yy = y + dy
                    xx = x + dx                
                    if 0<=yy<s1 and 0<=xx<s2:
                        if seg[yy][xx]>0 and seg[yy][xx]!=cell:
                            lbl[y][x] = 1
                            break
    return lbl


def visualize():
    data_dir = os.path.join("data", "train")
    test_dir = os.path.join("data", "test")
    dataset_dir = "dataset"

    N_training = len(glob.glob(os.path.join(dataset_dir, "train", "image", "*.npy")))
    idx = random.randint(0, N_training-1)
    img_training = np.load(os.path.join(dataset_dir, "train", "image", f"image_{idx:05d}.npy"), allow_pickle=True)

    lbl_training = np.load(os.path.join(dataset_dir, "train", "mask", f"mask_{idx:05d}.npy"), allow_pickle=True)
    #seg[0,0] = 90
    seg_clr = clr.label2rgb(lbl_training, image=None, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0))

    print("Training set:")
    print("# of training pairs: ", N_training, "range: ", img_training.min(), img_training.max())
    print("# of SEGs:", len(glob.glob(os.path.join(dataset_dir, "train", "mask", "*.npy"))))

    N_validation = len(glob.glob(os.path.join(dataset_dir, "test", "*.npy")))
    idx = random.randint(0, N_validation-1)
    img_validation = np.load(os.path.join(dataset_dir, "test", f"image_{idx:05d}.npy"), allow_pickle=True)
    print("Validation set:")
    print("# of validation pairs:", N_validation, "range:", img_validation.min(), img_validation.max())

    plt.figure(figsize=[8,6])

    plt.subplot(1,2,1)
    plt.imshow(img_training, cmap="gray")
    plt.title('Training image')

    plt.subplot(1,2,2)
    plt.imshow(lbl_training)
    plt.title('Training label')

    plt.tight_layout()
    plt.savefig('out/sample_pairs.png', dpi=200, transparent=False)
    plt.show()


if __name__ == '__main__':
    main()
    visualize()
