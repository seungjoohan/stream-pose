"""
CoMotion inference on a pose video.
Outputs: results/<stem>.pt  — per-frame SMPL params + 3D/2D keypoints per track
         results/<stem>_viz.mp4 — 2D skeleton overlay video (no aitviewer needed)

Usage:
    python run_pose.py --video /path/to/video.mp4
    python run_pose.py --video /path/to/video.mp4 --no-viz
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# CoMotion imports are deferred to run() — smpl_kinematics asserts at import time
# that SMPL_NEUTRAL.pkl exists, so we check for it first in main().
sys.path.insert(0, str(Path(__file__).parent / "ml-comotion" / "src"))

# ── SMPL skeleton connections (indices into the 27-keypoint joints_face format) ──
# First 22 joints are standard SMPL body joints, 22-26 are face vertices
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

# Distinct colors per track (BGR)
TRACK_COLORS = [
    (0, 200, 255),   # yellow-orange
    (0, 255, 128),   # green
    (255, 100, 0),   # blue
    (255, 0, 200),   # magenta
    (0, 128, 255),   # orange
    (128, 0, 255),   # purple
]


def get_color(track_id: int):
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def draw_frame(frame: np.ndarray, track_state, orig_h: int, orig_w: int):
    """Draw 2D skeleton overlay for all active tracks on a frame."""
    out = frame.copy()

    # track_state shape: (1, 48, 27, 2) at network input resolution (512x512 cropped)
    # CoMotion returns pred_2d at the original input resolution already
    pred_2d = track_state.pred_2d[0].cpu().numpy()  # (48, 27, 2)
    ids = track_state.id[0].cpu().numpy().flatten()  # (48,)

    for i, (kpts, tid) in enumerate(zip(pred_2d, ids)):
        if tid == 0:
            continue  # padded / inactive track

        color = get_color(tid)
        kpts = kpts.astype(int)  # (27, 2) — x, y in pixel space

        # Draw skeleton bones
        for j0, j1 in SKELETON:
            p0, p1 = tuple(kpts[j0]), tuple(kpts[j1])
            if all(0 <= p0[0] < orig_w and 0 <= p0[1] < orig_h for _ in [1]) and \
               all(0 <= p1[0] < orig_w and 0 <= p1[1] < orig_h for _ in [1]):
                cv2.line(out, p0, p1, color, 2, cv2.LINE_AA)

        # Draw joint circles
        for j, (x, y) in enumerate(kpts[:22]):  # body joints only
            if 0 <= x < orig_w and 0 <= y < orig_h:
                cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)

        # Label track id near root (joint 0 = pelvis)
        rx, ry = kpts[0]
        if 0 <= rx < orig_w and 0 <= ry < orig_h:
            cv2.putText(out, f"id={int(tid)}", (rx + 5, ry - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return out


def run(video_path: Path, output_dir: Path, visualize: bool, start_frame: int, num_frames: int):
    # Deferred import — requires SMPL_NEUTRAL.pkl to exist
    from comotion_demo.models import comotion as comotion_module
    from comotion_demo.utils import dataloading, track as track_utils

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # ── device ──────────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    use_mps = torch.backends.mps.is_available()
    print(f"Device: {'MPS (Apple Silicon)' if use_mps else 'CPU'}")

    # ── load model ──────────────────────────────────────────────────────────────
    print("Loading CoMotion model...")
    model = comotion_module.CoMotion(use_coreml=False, pretrained=True)
    model.to(device).eval()

    # ── video writer setup ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0  # GIFs often report 0 fps
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # GIFs may report 0 for frame count — count manually if needed
    if total_frames <= 0:
        total_frames = 0
        while cap.read()[0]:
            total_frames += 1
    cap.release()

    print(f"Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")

    writer = None
    if visualize:
        viz_path = output_dir / f"{stem}_viz.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(viz_path), fourcc, fps, (orig_w, orig_h))

    # ── inference loop ───────────────────────────────────────────────────────────
    detections_log = []
    tracks_log = []
    initialized = False

    frame_gen = dataloading.yield_image_and_K(video_path, start_frame, num_frames)
    n = min(num_frames, total_frames - start_frame)

    # Also open with cv2 for visualization frames (dataloading resizes the tensor)
    cap_viz = cv2.VideoCapture(str(video_path))
    if start_frame > 0:
        cap_viz.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx, (image_tensor, K) in enumerate(tqdm(frame_gen, total=n, desc="Running CoMotion")):
        # Raw frame for visualization
        if visualize:
            ret, raw_frame = cap_viz.read()
            if not ret:
                break

        if not initialized:
            model.init_tracks(image_tensor.shape[-2:])
            initialized = True

        detection, track = model(image_tensor, K, use_mps=use_mps)

        detections_log.append({k: v.cpu() for k, v in detection.items()})
        tracks_log.append(track.cpu())

        if visualize:
            annotated = draw_frame(raw_frame, track, orig_h, orig_w)
            # Overlay frame index
            cv2.putText(annotated, f"frame {frame_idx + start_frame}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)

    if visualize:
        cap_viz.release()
        writer.release()
        print(f"Visualization saved → {viz_path}")

    # ── save SMPL results ────────────────────────────────────────────────────────
    print("Post-processing tracks...")
    stacked = torch.stack(tracks_log, dim=1)          # (1, T, 48, ...)
    tracks_dict = {k: getattr(stacked, k) for k in ["id", "pose", "trans", "betas"]}
    dets_dict = {k: [d[k] for d in detections_log] for k in detections_log[0].keys()}

    track_ref = track_utils.cleanup_tracks(
        {"detections": dets_dict, "tracks": tracks_dict},
        K,
        model.smpl_decoder.cpu(),
        min_matched_frames=1,
    )

    results_path = output_dir / f"{stem}.pt"
    if track_ref:
        frame_idxs, track_idxs = track_utils.convert_to_idxs(
            track_ref, tracks_dict["id"][0].squeeze(-1).long()
        )
        preds = {k: tracks_dict[k][0, frame_idxs, track_idxs] for k in tracks_dict}
        preds["id"] = preds["id"].squeeze(-1).long()
        preds["frame_idx"] = frame_idxs

        # Also compute 3D + 2D keypoints for all saved tracks
        pred_3d = model.smpl_decoder(
            preds["betas"], preds["pose"], preds["trans"], output_format="joints_face"
        )
        preds["pred_3d"] = pred_3d          # (N, 27, 3) — camera-space 3D
        preds["K"] = K.cpu()

        torch.save(preds, results_path)
        print(f"\nResults saved → {results_path}")
        print(f"  Tracks: {preds['id'].unique().numel()}")
        print(f"  Frames with detections: {len(frame_idxs)}")
        print(f"  SMPL pose shape: {preds['pose'].shape}   (N x 72)")
        print(f"  SMPL shape shape: {preds['betas'].shape}  (N x 10)")
        print(f"  3D keypoints shape: {preds['pred_3d'].shape}  (N x 27 x 3)")

        # Quick stats
        unique_ids = preds["id"].unique()
        for tid in unique_ids:
            mask = preds["id"] == tid
            print(f"  Track {tid.item():3d}: {mask.sum().item()} frames")
    else:
        print("No tracks found.")
        torch.save({"detections": dets_dict}, results_path)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="CoMotion inference on a pose video")
    parser.add_argument("--video", type=Path,
                        default=Path("/Users/seungjuhan/Downloads/golfswing.mp4"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "results")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip 2D visualization output")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1_000_000_000)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    smpl_path = Path(__file__).parent / "ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl"
    if not smpl_path.exists():
        print("ERROR: SMPL model not found.")
        print(f"  Expected: {smpl_path}")
        print("  1. Register at https://smpl.is.tue.mpg.de/")
        print("  2. Download v1.1.0: basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
        print(f"  3. Copy/rename to: {smpl_path}")
        sys.exit(1)

    run(
        video_path=args.video,
        output_dir=args.output_dir,
        visualize=not args.no_viz,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
