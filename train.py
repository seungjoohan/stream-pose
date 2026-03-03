"""
Training script for stream-pose 3D pose estimation.

Streams video through CoMotion labeler into a ring buffer, then trains
a DINOv2-ViT-S backbone + MLP head to regress pelvis-relative 3D keypoints.

Usage:
    # Train on a local video
    python train.py --video /path/to/video.mp4

    # Train on Pexels-fetched videos
    python train.py --fetch --queries "sports athlete" "yoga pose" --n 3

    # Custom training parameters
    python train.py --video clip.mp4 --lr 5e-4 --batch-size 32 --steps 20000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ml-comotion" / "src"))


def get_device(arg: str) -> str:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return arg


def mpjpe_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Mean per-joint position error on visible joints only.

    Args:
        pred:       (B, 27, 3) predicted keypoints
        target:     (B, 27, 3) ground-truth keypoints
        visibility: (B, 27) float32, 1=visible
    """
    err = torch.norm(pred - target, dim=-1)  # (B, 27)
    mask = visibility > 0.5
    if mask.sum() == 0:
        return err.mean()
    return err[mask].mean()


def pckh_05(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """PCKh@0.5: % of visible joints within 0.5 × head-segment length of GT.

    Uses neck (12) to head (15) distance as reference. Standard pose metric.

    Args:
        pred:       (B, 27, 3) predicted keypoints
        target:     (B, 27, 3) ground-truth keypoints
        visibility: (B, 27) float32, 1=visible

    Returns:
        Scalar in [0, 1]; 1.0 = all visible joints correct.
    """
    # ref = ||head - neck|| per sample, (B,)
    ref = torch.norm(target[:, 15] - target[:, 12], dim=-1).clamp(min=1e-4)
    dist = torch.norm(pred - target, dim=-1)  # (B, 27)
    threshold = 0.5 * ref.unsqueeze(-1)  # (B, 1)
    correct = (dist <= threshold) & (visibility > 0.5)
    mask = visibility > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return correct[mask].float().mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D pose model with streaming CoMotion labels",
    )
    parser.add_argument(
        "--queries", nargs="+",
        default=["sports athlete", "yoga pose", "fitness workout"],
        help="Pexels search queries (used with --fetch)",
    )
    parser.add_argument("--n", type=int, default=3,
                        help="Videos per query to download")
    parser.add_argument("--buffer-size", type=int, default=2048,
                        help="Ring buffer capacity")
    parser.add_argument("--min-samples", type=int, default=512,
                        help="Min samples before training starts")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for head (backbone stays frozen)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total training steps (0 = run forever)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print loss every N steps")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/"),
                        help="Where to save checkpoints")
    parser.add_argument("--video", type=Path, default=None,
                        help="Local video path (skip Pexels, use this instead)")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch videos from Pexels")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Device for training")
    parser.add_argument("--min-half-body-joints", type=int, default=14,
                        help="Skip videos with no person having this many visible joints (0=disable)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve video source ──────────────────────────────────────────────────
    if args.video is not None:
        video_paths = [args.video]
    elif args.fetch:
        from video_sources.fetch_videos import fetch_videos
        video_paths = fetch_videos(
            queries=args.queries,
            n_per_query=args.n,
        )
        if not video_paths:
            print("[train] No videos fetched. Exiting.")
            sys.exit(1)
    else:
        print("[train] Error: specify --video <path> or --fetch to provide video source.")
        sys.exit(1)

    print(f"[train] Video source ({len(video_paths)} files):")
    for vp in video_paths:
        print(f"  {vp}")

    # ── Build labeler + dataset + producer ────────────────────────────────────
    from labeler.comotion_labeler import CoMotionLabeler
    from data.streaming_dataset import StreamingPoseDataset, PoseProducer

    labeler = CoMotionLabeler()
    dataset = StreamingPoseDataset(
        capacity=args.buffer_size,
        augment=True,
        min_samples=args.min_samples,
    )
    producer = PoseProducer(
        video_paths,
        dataset,
        labeler=labeler,
        loop=True,
        min_half_body_joints=args.min_half_body_joints,
    )
    producer.start()
    print("[train] Producer started. Waiting for buffer to fill...")

    # ── Wait for buffer ───────────────────────────────────────────────────────
    while not dataset.is_ready:
        print(
            f"[train] Buffer: {dataset.buffer_size}/{args.min_samples} samples",
            flush=True,
        )
        time.sleep(2)
    print(f"[train] Buffer ready ({dataset.buffer_size} samples). Starting training.")

    # ── Build model + optimizer ───────────────────────────────────────────────
    from model.pose_model import DinoV2PoseModel

    device = get_device(args.device)
    print(f"[train] Device: {device}")

    model = DinoV2PoseModel(freeze_backbone=True).to(device)
    optim = torch.optim.Adam(model.head.parameters(), lr=args.lr)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    step = 0
    t0 = time.time()

    def save_checkpoint(step: int, tag: str = ""):
        name = f"pose_step{step}{tag}.pt"
        path = args.checkpoint_dir / name
        torch.save(
            {"step": step, "model": model.state_dict(), "optim": optim.state_dict()},
            path,
        )
        print(f"[train] Saved {path}")

    try:
        while True:
            for batch in loader:
                crop = batch["crop_rgb"].to(device)     # (B, 3, 224, 224)
                target = batch["kpts3d"].to(device)     # (B, 27, 3)
                vis = batch["visibility"].to(device)    # (B, 27)

                pred = model(crop)                       # (B, 27, 3)
                loss = mpjpe_loss(pred, target, vis)

                optim.zero_grad()
                loss.backward()
                optim.step()

                step += 1

                if step % args.log_every == 0:
                    with torch.no_grad():
                        pckh = pckh_05(pred, target, vis)
                    elapsed = time.time() - t0
                    print(
                        f"[train] step {step:6d} | "
                        f"loss {loss.item():.4f} | "
                        f"PCKh@0.5 {pckh.item():.3f} | "
                        f"buffer {dataset.buffer_size} | "
                        f"elapsed {elapsed:.0f}s"
                    )

                if step % args.save_every == 0:
                    save_checkpoint(step)

                if args.steps > 0 and step >= args.steps:
                    raise StopIteration

    except (KeyboardInterrupt, StopIteration):
        pass

    # ── Cleanup ───────────────────────────────────────────────────────────────
    save_checkpoint(step, tag="_final")
    print(f"[train] Training finished at step {step}.")
    producer.stop()
    producer.join(timeout=10)
    print("[train] Producer stopped. Done.")


if __name__ == "__main__":
    main()
