import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────
case           = "PhC-C2DH-U373"
out_dir        = "out"
file1          = os.path.join(out_dir, f"trajectories_{case}_01.csv")
file2          = os.path.join(out_dir, f"trajectories_{case}_02.csv")
file3          = os.path.join(out_dir, f"trajectories_{case}_03.csv")
file4          = os.path.join(out_dir, f"trajectories_{case}_04.csv")

PIXEL_SIZE_UM  = 0.65    # µm per pixel (already applied to x_um,y_um)
TIME_STEP_MIN  = 15      # minutes per frame
MAX_LAG_MIN    = 1100    # maximum lag time in minutes
FIT_LAG_MAX    = 300     # use τ ≤ 300 min for fitting
# ───────────────────────────────────────────────────────

os.makedirs(out_dir, exist_ok=True)

# 1) Load trajectories
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

# compute frame indices from time_min
for df in (df1, df2, df3, df4):
    df['frame_id'] = (df['time_min'] / TIME_STEP_MIN).astype(int)

def compute_msd(df, max_lag_min, time_step_min):
    # build per‐particle trajectories in frame-space
    trajs = {}
    for pid, grp in df.groupby("particle"):
        frames = grp['frame_id'].values
        xs     = grp['x_um'].values
        ys     = grp['y_um'].values
        trajs[pid] = (frames, xs, ys)
    # compute lag steps
    max_steps = int(max_lag_min // time_step_min)
    tau_steps = np.arange(1, max_steps + 1)
    taus_min  = tau_steps * time_step_min
    msd       = np.empty_like(taus_min, dtype=float)
    for i, dt in enumerate(tau_steps):
        disp2 = []
        for frames, xs, ys in trajs.values():
            idx_map = {f: j for j, f in enumerate(frames)}
            for j, f in enumerate(frames):
                f2 = f + dt
                if f2 in idx_map:
                    k = idx_map[f2]
                    dx = xs[k] - xs[j]
                    dy = ys[k] - ys[j]
                    disp2.append(dx*dx + dy*dy)
        msd[i] = np.nanmean(disp2) if disp2 else np.nan
    return taus_min, msd

# 2) Compute MSD up to MAX_LAG_MIN
taus1, msd1 = compute_msd(df1, MAX_LAG_MIN, TIME_STEP_MIN)
taus2, msd2 = compute_msd(df2, MAX_LAG_MIN, TIME_STEP_MIN)
taus3, msd3 = compute_msd(df3, MAX_LAG_MIN, TIME_STEP_MIN)
taus4, msd4 = compute_msd(df4, MAX_LAG_MIN, TIME_STEP_MIN)
msd_avg     = np.nanmean(np.vstack([msd1, msd2, msd3, msd4]), axis=0)

# 3) Fit over τ ≤ FIT_LAG_MAX
fit_mask = taus1 <= FIT_LAG_MAX
log_t = np.log10(taus1[fit_mask])
log_m = np.log10(msd_avg[fit_mask])
if len(log_t) >= 2 and not np.isnan(log_m).all():
    slope, intercept = np.polyfit(log_t, log_m, 1)
else:
    slope, intercept = np.nan, np.nan

# 4) Plot MSD curves and average with increased font, line & marker sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

fig, ax = plt.subplots(figsize=(9,6))

# Average MSD
ax.loglog(taus1, msd_avg,
          marker='s', linestyle='none',
          mfc='none', mec='k',
          markersize=10, label="Average MSD")

# Fit line
if not np.isnan(slope):
    fit_vals = 10**intercept * taus1**slope
    ax.loglog(taus1, fit_vals,
              '-', color='red',
              linewidth=3,
              label=f"Fit MSD (slope={slope:.2f})")

ax.set_xlabel("Lag τ (min)")
ax.set_ylabel("MSD(τ) (µm²)")

# increase axis spine width
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "MSD.png"), dpi=200)
plt.show()
