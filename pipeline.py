"""
Data pipeline runner: fetch videos → label with CoMotion → stream into buffer.

This script demonstrates the full pipeline and can be used to verify that
each stage works before connecting to a training loop.

Usage:
    # Label local video(s) — no Pexels API needed
    python pipeline.py --video /path/to/video.mp4 --dry-run

    # Fetch from Pexels and label (requires PEXELS_API_KEY env var)
    python pipeline.py --fetch --queries "sports athlete" "yoga pose" --n 3

    # Run as persistent background feeder (blocks until Ctrl-C)
    python pipeline.py --fetch --loop
"""

import argparse
import sys
import time
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ml-comotion" / "src"))


def run_dry(video_paths: list[Path], max_frames: int = 50):
    """
    Run the pipeline on local videos and print label stats.
    Does NOT start a training loop — just validates the stack.
    """
    from labeler.comotion_labeler import CoMotionLabeler
    from data.streaming_dataset import StreamingPoseDataset

    dataset = StreamingPoseDataset(capacity=4096, augment=False)
    labeler = CoMotionLabeler()

    total_frames = 0
    total_persons = 0

    for vp in video_paths:
        print(f"\nLabeling: {vp.name}")
        for frame_label in labeler.label_video(vp, num_frames=max_frames):
            dataset.push_frame(frame_label)
            n = len(frame_label["persons"])
            total_frames += 1
            total_persons += n
            if total_frames % 10 == 0:
                print(
                    f"  frame {frame_label['frame_idx']:4d} | "
                    f"persons: {n} | buffer: {dataset.buffer_size}"
                )

    print(f"\n--- Summary ---")
    print(f"Frames labeled:     {total_frames}")
    print(f"Person-frames:      {total_persons}")
    print(f"Buffer size:        {dataset.buffer_size}")

    if dataset.buffer_size > 0:
        # Sample one item to verify shapes
        item = dataset[0]
        print(f"kpts3d shape:       {tuple(item['kpts3d'].shape)}")
        print(f"visibility shape:   {tuple(item['visibility'].shape)}")
        print(f"kpts3d[0] (pelvis): {item['kpts3d'][0].tolist()}")  # should be [0,0,0]
        print(f"visible joints:     {int(item['visibility'].sum())}/{item['visibility'].numel()}")


def run_pipeline(
    fetch: bool,
    queries: list[str],
    n_per_query: int,
    video_dir: Path,
    loop: bool,
    frameskip: int,
):
    """
    Run the full persistent pipeline:
      1. (Optional) Fetch videos from Pexels
      2. Start PoseProducer thread to label videos and fill dataset
      3. Block printing buffer stats until Ctrl-C
    """
    from video_sources.fetch_videos import fetch_videos
    from labeler.comotion_labeler import CoMotionLabeler
    from data.streaming_dataset import StreamingPoseDataset, PoseProducer

    # ── 1. video list ──────────────────────────────────────────────────────────
    if fetch:
        video_paths = fetch_videos(
            queries=queries,
            n_per_query=n_per_query,
            output_dir=video_dir,
        )
    else:
        video_paths = sorted(video_dir.glob("*.mp4"))

    if not video_paths:
        print(f"No videos found in {video_dir}. Pass --fetch to download, "
              "or put .mp4 files in that directory.")
        sys.exit(1)

    print(f"Video list ({len(video_paths)} files):")
    for vp in video_paths:
        print(f"  {vp}")

    # ── 2. producer ───────────────────────────────────────────────────────────
    dataset = StreamingPoseDataset(capacity=16384, augment=True)
    labeler = CoMotionLabeler()
    producer = PoseProducer(
        video_paths, dataset, labeler=labeler, loop=loop, frameskip=frameskip
    )
    producer.start()
    print("\n[pipeline] Producer started. Press Ctrl-C to stop.\n")

    # ── 3. monitor ────────────────────────────────────────────────────────────
    try:
        while producer.is_alive():
            print(
                f"[pipeline] buffer: {dataset.buffer_size:6d} samples  "
                f"(capacity: {dataset._capacity})"
            )
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[pipeline] Stopping...")
        producer.stop()
        producer.join(timeout=10)
        print(f"[pipeline] Final buffer size: {dataset.buffer_size}")


def main():
    parser = argparse.ArgumentParser(description="Streaming pose data pipeline")
    parser.add_argument("--video", type=Path, nargs="+", default=[],
                        help="Local video file(s) for dry-run mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Label local video(s) and print stats without training")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch videos from Pexels (requires PEXELS_API_KEY)")
    parser.add_argument("--queries", nargs="+",
                        default=["sports athlete", "fitness workout", "yoga pose"],
                        help="Pexels search queries (used with --fetch)")
    parser.add_argument("--n", type=int, default=3,
                        help="Videos per query to download (used with --fetch)")
    parser.add_argument("--video-dir", type=Path,
                        default=Path("/tmp/stream_pose_videos"),
                        help="Directory for downloaded videos")
    parser.add_argument("--loop", action="store_true",
                        help="Cycle through video list indefinitely")
    parser.add_argument("--frameskip", type=int, default=2,
                        help="Process every Nth frame (default: 2)")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Max frames per video in dry-run mode")
    args = parser.parse_args()

    if args.dry_run or args.video:
        videos = args.video
        if not videos:
            parser.error("--dry-run requires at least one --video path")
        run_dry(videos, max_frames=args.max_frames)
    else:
        run_pipeline(
            fetch=args.fetch,
            queries=args.queries,
            n_per_query=args.n,
            video_dir=args.video_dir,
            loop=args.loop,
            frameskip=args.frameskip,
        )


if __name__ == "__main__":
    main()
