"""
Interactive 3D viewer for CoMotion results.

Controls:
  Mouse drag      — rotate 3D view
  Scroll wheel    — zoom
  Frame slider    — scrub through frames
  Arrow keys ← →  — step one frame at a time
  Track buttons   — toggle individual tracks on/off

Usage:
    python view_results.py results/golfswing.pt
    python view_results.py results/golfswing.pt --video ~/Downloads/golfswing.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("macosx")   # native macOS backend — smooth rotation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import numpy as np
import torch

# ── SMPL skeleton ──────────────────────────────────────────────────────────────
SKELETON = [
    (0, 1), (0, 2),          # pelvis → hips
    (0, 3),                  # pelvis → spine1
    (1, 4), (2, 5),          # hips → knees
    (3, 6),                  # spine1 → spine2
    (4, 7), (5, 8),          # knees → ankles
    (6, 9),                  # spine2 → spine3
    (7, 10), (8, 11),        # ankles → feet
    (9, 12),                 # spine3 → neck
    (9, 13), (9, 14),        # spine3 → collars
    (12, 15),                # neck → head
    (13, 16), (14, 17),      # collars → shoulders
    (16, 18), (17, 19),      # shoulders → elbows
    (18, 20), (19, 21),      # elbows → wrists
]

COLORS = [
    "#FF6B35",   # orange
    "#2EC4B6",   # teal
    "#E63946",   # red
    "#A8DADC",   # light blue
    "#9B5DE5",   # purple
    "#F15BB5",   # pink
]


def load_results(pt_path: Path):
    preds = torch.load(pt_path, weights_only=False, map_location="cpu")
    # pred_3d is in camera space: X right, Y down, Z depth
    pred_3d = preds["pred_3d"].numpy()     # (N, 27, 3)
    frame_idxs = preds["frame_idx"].numpy()
    ids = preds["id"].numpy()

    unique_frames = np.unique(frame_idxs)
    unique_ids = np.unique(ids)

    # Organise as {frame_idx: {"pred_3d": ..., "ids": ...}}
    frame_data = {}
    for f in unique_frames:
        mask = frame_idxs == f
        frame_data[int(f)] = {
            "pred_3d": pred_3d[mask],    # (n, 27, 3)
            "ids": ids[mask],
        }

    # Precompute global axis limits (flip Y so up is up)
    all_pts = pred_3d.copy()
    all_pts[:, :, 1] *= -1                # flip Y

    center = all_pts.mean(axis=(0, 1))
    half_range = np.abs(all_pts - center).max() * 1.1

    limits = {
        "x": (center[0] - half_range, center[0] + half_range),
        "y": (center[1] - half_range, center[1] + half_range),
        "z": (center[2] - half_range, center[2] + half_range),
    }

    return frame_data, unique_frames, unique_ids, limits


def load_video_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def view(pt_path: Path, video_path: Path | None):
    frame_data, unique_frames, unique_ids, limits = load_results(pt_path)

    id_color = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(unique_ids)}
    active_tracks = {tid: True for tid in unique_ids}

    cap = None
    if video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))

    # ── figure layout ──────────────────────────────────────────────────────────
    has_video = cap is not None
    fig = plt.figure(figsize=(14 if has_video else 9, 8))
    fig.patch.set_facecolor("#1a1a2e")

    if has_video:
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[10, 1],
            width_ratios=[1, 1],
            hspace=0.05, wspace=0.1,
            left=0.05, right=0.95, top=0.93, bottom=0.12,
        )
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d = fig.add_subplot(gs[0, 1])
        ax2d.axis("off")
        ax_slider = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[10, 1],
            hspace=0.05,
            left=0.05, right=0.95, top=0.93, bottom=0.12,
        )
        ax3d = fig.add_subplot(gs[0], projection="3d")
        ax_slider = fig.add_subplot(gs[1])

    for ax in [ax3d]:
        ax.set_facecolor("#16213e")

    # ── slider ─────────────────────────────────────────────────────────────────
    slider = Slider(
        ax_slider, "Frame",
        unique_frames.min(), unique_frames.max(),
        valinit=unique_frames[0], valstep=1,
        color="#4cc9f0",
    )
    ax_slider.set_facecolor("#0f3460")

    # Track toggle buttons
    btn_axes = []
    btn_objs = []
    n_tracks = len(unique_ids)
    btn_w, btn_h = 0.07, 0.04
    btn_start_x = 0.05
    btn_y = 0.02

    for i, tid in enumerate(unique_ids):
        bax = fig.add_axes([btn_start_x + i * (btn_w + 0.01), btn_y, btn_w, btn_h])
        btn = Button(bax, f"id {int(tid)}", color=id_color[tid], hovercolor="#888")
        btn_axes.append(bax)
        btn_objs.append(btn)

    # ── draw function ──────────────────────────────────────────────────────────
    im2d_handle = [None]
    current_frame = [int(unique_frames[0])]

    def draw(frame_idx: int):
        current_frame[0] = frame_idx

        # Find nearest available frame
        nearest = unique_frames[np.argmin(np.abs(unique_frames - frame_idx))]
        data = frame_data[int(nearest)]

        # Preserve current 3D view angle between redraws
        elev = ax3d.elev
        azim = ax3d.azim
        ax3d.cla()
        ax3d.set_facecolor("#16213e")

        for kpts_cam, tid in zip(data["pred_3d"], data["ids"]):
            if not active_tracks.get(tid, True):
                continue

            color = id_color[tid]

            # Flip Y for display (camera Y-down → plot Y-up)
            kpts = kpts_cam.copy()
            kpts[:, 1] *= -1

            for j0, j1 in SKELETON:
                ax3d.plot(
                    [kpts[j0, 0], kpts[j1, 0]],
                    [kpts[j0, 2], kpts[j1, 2]],   # Z as depth axis (forward)
                    [kpts[j0, 1], kpts[j1, 1]],   # Y as vertical
                    color=color, linewidth=2.5, alpha=0.9,
                )

            ax3d.scatter(
                kpts[:22, 0], kpts[:22, 2], kpts[:22, 1],
                c=color, s=35, zorder=5, depthshade=False,
            )
            # Label at pelvis
            ax3d.text(
                kpts[0, 0], kpts[0, 2], kpts[0, 1],
                f" id={int(tid)}", color=color, fontsize=8,
            )

        ax3d.set_xlim(*limits["x"])
        ax3d.set_ylim(*limits["z"])     # depth on Y axis of plot
        ax3d.set_zlim(*limits["y"])     # height on Z axis of plot
        ax3d.set_xlabel("X", color="white", labelpad=2)
        ax3d.set_ylabel("Depth", color="white", labelpad=2)
        ax3d.set_zlabel("Height", color="white", labelpad=2)
        ax3d.tick_params(colors="gray", labelsize=7)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor("#333")
        ax3d.yaxis.pane.set_edgecolor("#333")
        ax3d.zaxis.pane.set_edgecolor("#333")
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_title(
            f"Frame {nearest}   |   drag: rotate   |   scroll: zoom",
            color="white", fontsize=10, pad=8,
        )

        # 2D video frame
        if cap is not None:
            frame_img = load_video_frame(cap, int(nearest))
            if frame_img is not None:
                if im2d_handle[0] is None:
                    im2d_handle[0] = ax2d.imshow(frame_img)
                else:
                    im2d_handle[0].set_data(frame_img)
                ax2d.set_title(f"Camera view  (frame {nearest})", color="white", fontsize=10)

        fig.canvas.draw_idle()

    def on_slider(val):
        draw(int(slider.val))

    slider.on_changed(on_slider)

    def make_toggle(tid):
        def toggle(_):
            active_tracks[tid] = not active_tracks[tid]
            draw(current_frame[0])
        return toggle

    for btn, tid in zip(btn_objs, unique_ids):
        btn.on_clicked(make_toggle(tid))

    # Arrow key frame stepping
    def on_key(event):
        cur = current_frame[0]
        if event.key == "right":
            nxt = unique_frames[unique_frames > cur]
            new = int(nxt[0]) if len(nxt) else cur
        elif event.key == "left":
            prv = unique_frames[unique_frames < cur]
            new = int(prv[-1]) if len(prv) else cur
        else:
            return
        slider.set_val(new)

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.suptitle(
        f"CoMotion 3D Viewer — {pt_path.stem}",
        color="white", fontsize=13, fontweight="bold",
    )

    draw(int(unique_frames[0]))
    plt.show()

    if cap:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_file", type=Path, nargs="?",
                        default=Path("results/golfswing.pt"))
    parser.add_argument("--video", type=Path, default=None,
                        help="Original video for side-by-side camera view")
    args = parser.parse_args()

    if not args.pt_file.exists():
        print(f"ERROR: results file not found: {args.pt_file}")
        print("Run run_golf.py first to generate results.")
        sys.exit(1)

    # Auto-detect video if not provided
    video_path = args.video
    if video_path is None:
        candidate = Path("/Users/seungjuhan/Downloads") / (args.pt_file.stem + ".mp4")
        if candidate.exists():
            video_path = candidate
            print(f"Auto-detected video: {video_path}")

    view(args.pt_file, video_path)


if __name__ == "__main__":
    main()
