"""
Thread-safe ring buffer dataset for streaming pose labels.

Producer (labeler thread) pushes labeled samples into the buffer.
Consumer (training loop) reads samples as a standard torch Dataset.

Each sample in the buffer is one person-frame pair:
    kpts3d:     (27, 3) float32  pelvis-relative 3D keypoints
    visibility: (27,)   float32  per-joint visibility weight (0 or 1)
    track_id:   int
    frame_idx:  int

Usage (producer side):
    dataset = StreamingPoseDataset(capacity=4096)
    producer = PoseProducer(video_paths, dataset)
    producer.start()

Usage (consumer/training side):
    while not dataset.is_ready:
        time.sleep(1)
    loader = DataLoader(dataset, batch_size=32, ...)
    for batch in loader:
        kpts3d = batch["kpts3d"]        # (B, 27, 3)
        visibility = batch["visibility"] # (B, 27)
"""

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_JOINTS = 27  # SMPL joints_face

# Left/right flip pairs for data augmentation (joint index pairs)
# When flipping horizontally: swap left↔right joints
FLIP_PAIRS = [
    (1, 2),    # left hip ↔ right hip
    (4, 5),    # left knee ↔ right knee
    (7, 8),    # left ankle ↔ right ankle
    (10, 11),  # left foot ↔ right foot
    (13, 14),  # left collar ↔ right collar
    (16, 17),  # left shoulder ↔ right shoulder
    (18, 19),  # left elbow ↔ right elbow
    (20, 21),  # left wrist ↔ right wrist
    (23, 24),  # left eye ↔ right eye  (face vertices)
    (25, 26),  # left ear ↔ right ear  (face vertices)
]


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_flip(kpts3d: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Horizontal flip augmentation in pelvis-relative space.

    Negate the X axis and swap left/right joint pairs.

    Args:
        kpts3d:     (27, 3) pelvis-relative keypoints
        visibility: (27,)   bool or float visibility mask

    Returns:
        Tuple of (flipped kpts3d, flipped visibility).
    """
    kpts = kpts3d.copy()
    vis = visibility.copy()

    kpts[:, 0] *= -1  # flip X axis

    for l, r in FLIP_PAIRS:
        kpts[[l, r]] = kpts[[r, l]]
        vis[[l, r]] = vis[[r, l]]

    return kpts, vis


def augment_scale(kpts3d: np.ndarray, scale_range: tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """
    Random uniform scale augmentation.

    Multiplies all joint offsets by a random scale factor.
    Pelvis remains at origin (joint 0 is already [0,0,0]).

    Args:
        kpts3d:      (27, 3) pelvis-relative keypoints
        scale_range: (min, max) scale factor range

    Returns:
        Scaled keypoints (27, 3).
    """
    s = np.random.uniform(*scale_range)
    return kpts3d * s


def augment_jitter(
    kpts3d: np.ndarray,
    visibility: np.ndarray,
    sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to visible joints only.

    Args:
        kpts3d:     (27, 3) pelvis-relative keypoints
        visibility: (27,)   bool mask
        sigma:      noise standard deviation (in same units as kpts3d)

    Returns:
        Tuple of (noisy kpts3d, unchanged visibility).
    """
    noise = np.random.randn(*kpts3d.shape).astype(np.float32) * sigma
    noise[~visibility] = 0  # don't jitter invisible joints
    return kpts3d + noise, visibility


# ── Ring buffer dataset ────────────────────────────────────────────────────────

class StreamingPoseDataset(Dataset):
    """
    Thread-safe ring buffer holding the most recent `capacity` pose samples.

    Samples are pushed by a producer thread (the labeler) and read by the
    DataLoader in the training loop. Once the buffer is full, the oldest
    samples are silently dropped.

    IMPORTANT: DataLoader must use num_workers=0. This dataset uses threading.Lock
    for thread safety, which does NOT cross process boundaries. Using num_workers>0
    will cause each worker to see a stale copy of the buffer that is never updated
    by the producer thread.

    The training loop should call ``while not dataset.is_ready: time.sleep(1)``
    before creating the DataLoader to ensure enough samples are buffered.

    Args:
        capacity:  Maximum number of samples to keep in memory.
        augment:   If True, apply random flip + scale + jitter during __getitem__.
        min_samples: Minimum samples required before ``is_ready`` returns True.
    """

    def __init__(
        self,
        capacity: int = 8192,
        augment: bool = True,
        min_samples: int = 128,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._augment = augment
        self._min_samples = min_samples
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ── Producer API ──────────────────────────────────────────────────────────

    def push(self, sample: dict) -> None:
        """
        Add one labeled sample to the buffer.

        Args:
            sample: dict with keys:
                kpts3d:     (27, 3) np.float32
                visibility: (27,)   np.bool_ or np.float32
                track_id:   int     (optional)
                frame_idx:  int     (optional)
        """
        with self._lock:
            self._buffer.append(sample)

    def push_frame(self, frame_label: dict) -> None:
        """
        Push all persons from a frame_label dict (as returned by CoMotionLabeler).

        Args:
            frame_label: {"frame_idx": int, "persons": [per-person dicts]}
        """
        frame_idx = frame_label.get("frame_idx", -1)
        for person in frame_label.get("persons", []):
            self.push({
                "kpts3d": person["kpts3d"],
                "visibility": person["visibility"].astype(np.float32),
                "track_id": person.get("track_id", -1),
                "frame_idx": frame_idx,
            })

    # ── Consumer API (torch.utils.data.Dataset) ───────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __getitem__(self, idx: int) -> dict:
        # num_workers must be 0; see class docstring.
        with self._lock:
            try:
                sample = dict(self._buffer[idx])
            except IndexError:
                # Buffer rotated between __len__ and __getitem__; wrap around
                sample = dict(self._buffer[-1])

        kpts3d = sample["kpts3d"].copy()           # (27, 3) float32
        visibility = sample["visibility"].copy()   # (27,)  float32 (0 or 1)

        if self._augment:
            # Horizontal flip (50% probability)
            if np.random.rand() < 0.5:
                kpts3d, visibility = augment_flip(kpts3d, visibility.astype(bool))
                visibility = visibility.astype(np.float32)

            # Random scale (85%–115%)
            kpts3d = augment_scale(kpts3d)

            # Gaussian jitter on visible joints
            kpts3d, _ = augment_jitter(kpts3d, visibility.astype(bool))

        return {
            "kpts3d": torch.from_numpy(kpts3d),
            "visibility": torch.from_numpy(visibility),
            "track_id": sample.get("track_id", -1),
            "frame_idx": sample.get("frame_idx", -1),
        }

    @property
    def buffer_size(self) -> int:
        """Current number of samples in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True once the buffer has enough samples to begin training."""
        return self.buffer_size >= self._min_samples


# ── Background producer ────────────────────────────────────────────────────────

class PoseProducer(threading.Thread):
    """
    Background thread that fetches videos, runs CoMotion, and fills a dataset.

    Runs indefinitely, cycling through video_paths in order.
    Stops cleanly when stop() is called or the thread is set as daemon.

    Args:
        video_paths: List of video file paths to label.
        dataset:     StreamingPoseDataset to push samples into.
        labeler:     Optional pre-instantiated CoMotionLabeler.
                     If None, one is created lazily on first use.
        loop:        If True, cycle through video_paths indefinitely.
        frameskip:   Process every Nth frame (reduce labeling load).
    """

    def __init__(
        self,
        video_paths: list[Path | str],
        dataset: StreamingPoseDataset,
        labeler=None,
        loop: bool = True,
        frameskip: int = 2,
    ):
        super().__init__(daemon=True, name="PoseProducer")
        self._video_paths = [Path(p) for p in video_paths]
        self._dataset = dataset
        self._labeler = labeler
        self._loop = loop
        self._frameskip = frameskip
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the producer thread to stop after the current video."""
        self._stop_event.set()

    def run(self) -> None:
        # Import here so it only happens in the producer thread
        if self._labeler is None:
            import sys
            from pathlib import Path as _Path
            _root = _Path(__file__).parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from labeler.comotion_labeler import CoMotionLabeler
            self._labeler = CoMotionLabeler()

        fail_counts: dict[str, int] = {}
        skip_set: set[str] = set()

        while not self._stop_event.is_set():
            for vp in self._video_paths:
                if self._stop_event.is_set():
                    break
                key = str(vp)
                if key in skip_set:
                    continue
                try:
                    for frame_label in self._labeler.label_video(
                        vp, frameskip=self._frameskip
                    ):
                        self._dataset.push_frame(frame_label)
                        if self._stop_event.is_set():
                            break
                    fail_counts.pop(key, None)
                except Exception as e:
                    fail_counts[key] = fail_counts.get(key, 0) + 1
                    if fail_counts[key] >= 3:
                        print(f"[PoseProducer] Skipping {Path(vp).name} after 3 failures.")
                        skip_set.add(key)
                    else:
                        time.sleep(2 ** fail_counts[key])

            if not self._loop:
                break

        print("[PoseProducer] Stopped.")
