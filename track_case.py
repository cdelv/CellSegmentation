import os
import ast
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── CONFIG ─────────────────────────────────────────────
case           = "PhC-C2DH-U373"
data_root      = os.path.join("data")
out_dir        = "out"
csv_metrics    = os.path.join(out_dir, "cell_metrics.csv")
fps            = 5               # GIF frames per second
max_disp_pix   = 20              # max linking distance in pixels
VISUALIZE      = False           # True to pop up each frame
PIXEL_SIZE_UM  = 0.65            # microns per pixel
# ───────────────────────────────────────────────────────

os.makedirs(out_dir, exist_ok=True)

# 1) Load metrics CSV
df = pd.read_csv(csv_metrics, dtype={"subdir": str})
df["subdir"] = df["subdir"].str.zfill(2)
# parse bbox if string
if df["bbox"].dtype == object:
    df["bbox"] = df["bbox"].apply(ast.literal_eval)

# 2) Get x_px, y_px
if "centroid" in df.columns:
    # centroid as tuple
    coords = df["centroid"].apply(lambda c: c if isinstance(c, (list,tuple)) else ast.literal_eval(c))
    df[["x_px","y_px"]] = pd.DataFrame(coords.tolist(), index=df.index)
elif "x" in df.columns and "y" in df.columns:
    df = df.rename(columns={"x":"x_px", "y":"y_px"})
else:
    raise ValueError("No centroid or x/y columns found in metrics CSV")

# ensure time_min exists
if "time" in df.columns:
    df["time_min"] = df["time"].astype(float)
elif "time_min" not in df.columns:
    raise ValueError("No time or time_min column in CSV")

# 3) Link per subdir
for subdir in sorted(df["subdir"].unique()):
    if subdir in ["01", "02"]:
        s_dir = subdir
        d_dir = "train"
    else:
        s_dir = f"{int(subdir) - 2:02d}"
        d_dir = "test"

    subdf = df[df.subdir==subdir].copy()
    tracks=[]; next_pid=0; prev_pts=None; prev_pids=None

    for t in sorted(subdf["time_min"].unique()):
        grp = subdf[subdf.time_min==t]
        pts = grp[["x_px","y_px"]].to_numpy()
        pids = np.full(len(pts), -1, dtype=int)

        if prev_pts is None:
            for i in range(len(pts)):
                pids[i]=next_pid; next_pid+=1
        else:
            dmat = np.linalg.norm(pts[:,None,:]-prev_pts[None,:,:],axis=2)
            idx = np.argmin(dmat,axis=1)
            dmin = dmat[np.arange(len(pts)), idx]
            order = np.argsort(dmin)
            used=set()
            for i in order:
                j=idx[i]
                if dmin[i]<max_disp_pix and j not in used:
                    pids[i]=prev_pids[j]; used.add(j)
                else:
                    pids[i]=next_pid; next_pid+=1

        for i,pid in enumerate(pids):
            row=grp.iloc[i]
            tracks.append({
                "subdir": subdir,
                "frame": row["frame"],
                "time_min": t,
                "particle": int(pid),
                "x_px": float(row["x_px"]),
                "y_px": float(row["y_px"])
            })

        prev_pts, prev_pids = pts, pids

    tracks_df=pd.DataFrame(tracks)

    # convert px -> µm
    tracks_df["x_um"]=tracks_df["x_px"]*PIXEL_SIZE_UM
    tracks_df["y_um"]=tracks_df["y_px"]*PIXEL_SIZE_UM

    # save CSV
    out_csv=os.path.join(out_dir,f"trajectories_{case}_{subdir}.csv")
    tracks_df.to_csv(out_csv,index=False)
    print("Saved trajectories to", out_csv)

    # build GIF
    images=[]
    for t,grp in tracks_df.groupby("time_min"):
        fname=grp["frame"].iloc[0]
        img=imageio.volread(os.path.join(data_root,d_dir,case,s_dir,fname))
        fig,ax=plt.subplots(figsize=(6,6))
        ax.imshow(img,cmap="gray")

        # draw bboxes
        for _,r in subdf[subdf.time_min==t].iterrows():
            minr,minc,maxr,maxc=r["bbox"]
            ax.add_patch(Rectangle((minc,minr),maxc-minc,maxr-minr,
                                   edgecolor="r",facecolor="none"))
        # draw tracks
        sofar=tracks_df[tracks_df.time_min<=t]
        for pid,trk in sofar.groupby("particle"):
            ax.plot(trk["y_px"],trk["x_px"],"-",color="yellow",lw=1)
            last=trk[trk.time_min==t]
            ax.plot(last["y_px"],last["x_px"],"o",color="yellow",ms=4)
        ax.axis("off")
        if VISUALIZE: plt.show()
        fig.canvas.draw()
        buf,(w,h)=fig.canvas.print_to_buffer()
        frame=np.frombuffer(buf,dtype=np.uint8).reshape(h,w,4)[...,:3]
        images.append(frame); plt.close(fig)

    gif=os.path.join(out_dir,f"traj_boxes_{case}_{subdir}.gif")
    imageio.mimsave(gif,images,fps=fps)
    print("Saved GIF to",gif)
