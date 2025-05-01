import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the metrics CSV
csv_path = os.path.join("out", "cell_metrics.csv")
df = pd.read_csv(csv_path)

# Compute shape parameter = 4π·A / P² and its mean
df["shape"] = 4 * np.pi * df["area"] / df["perimeter"]**2
mean_shape = df["shape"].mean()

# Define what to plot
columns = ["area", "perimeter", "shape"]
labels = ["area (μm^2)", "perimeter (μm)", "shape"]
colors  = ["C0",      "C1",        "C2"]

for col, label, color in zip(columns, labels, colors):
    data = df[col].values
    nbins = int(np.sqrt(len(data))) - 1 
    bins  = np.histogram_bin_edges(data, bins=nbins)
    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(6,4))
    plt.plot(centers, hist,
             marker='o', linestyle='None',
             markerfacecolor='none', markeredgecolor=color)
    plt.xlabel(label.capitalize())
    plt.ylabel("PDF")

    if col == "shape":
        plt.axvline(mean_shape, color=color, linestyle='--', linewidth=1.5,
                    label=f"Mean = {mean_shape:.2f}")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join("out", col+".png"), dpi=200)
    plt.show()