"""
CoMotion labeler: stream video frames → 3D pose labels.

For each active tracked person per frame, yields a label dict containing:
  - pelvis-relative 3D keypoints (27, 3)
  - per-joint visibility mask (27,)
  - CoMotion track ID and frame index

The SMPL model is only used internally by CoMotion; training targets contain
only raw 3D coordinates — no SMPL params are stored or passed downstream.

Usage:
    labeler = CoMotionLabeler()
    for frame_label in labeler.label_video("video.mp4"):
        # frame_label["persons"]: list of per-person dicts
        #   track_id:   int
        #   kpts3d:     (27, 3) float32  pelvis-relative
        #   visibility: (27,)   bool
        #   confidence: float
        buffer.push(frame_label)
"""

import sys
import threading
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import torch

# Add CoMotion to path before any comotion_demo imports
_COMOTION_SRC = Path(__file__).parent.parent / "ml-comotion" / "src"
_LABELER_DIR = Path(__file__).parent
_path_lock = threading.Lock()

with _path_lock:
    if str(_COMOTION_SRC) not in sys.path:
        sys.path.insert(0, str(_COMOTION_SRC))
    if str(_LABELER_DIR) not in sys.path:
        sys.path.insert(0, str(_LABELER_DIR))

from label_converter import convert_frame_labels  # noqa: E402


def _get_inference_config() -> dict:
    """
    Determine the best inference configuration for this environment.

    Returns a dict with:
        use_coreml: bool   — use CoreML for detection (Apple Neural Engine)
        use_mps:    bool   — offload refinement step to MPS
        device:     str    — device to load PyTorch model parameters on

    Two modes:
      CoreML (preferred on Apple Silicon with Python 3.12):
        Detection  → CoreML, runs on CPU/ANE automatically
        Refinement → MPS (passed as use_mps=True to forward)
        Model loaded on CPU (CoreML wrapper has no nn.Module parameters)

      Full-MPS (fallback, e.g. Python 3.14 where coremltools is broken):
        Detection  → PyTorch on MPS
        Refinement → MPS
        Model loaded on MPS; use_mps=False (no split needed, all on MPS)
    """
    mps_available = torch.backends.mps.is_available()

    # Check CoreML availability: coremltools importable + mlpackage on disk
    coreml_available = False
    if mps_available:
        try:
            import coremltools  # noqa: F401
            _mlpackage = (
                Path(__file__).parent.parent
                / "ml-comotion/src/comotion_demo/data/comotion_detection.mlpackage"
            )
            coreml_available = _mlpackage.exists()
        except ImportError:
            pass

    if coreml_available:
        return {"use_coreml": True, "use_mps": True, "device": "cpu"}
    elif mps_available:
        return {"use_coreml": False, "use_mps": False, "device": "mps"}
    else:
        return {"use_coreml": False, "use_mps": False, "device": "cpu"}


def _load_model(use_coreml: bool, device: str):
    """
    Load CoMotion model with the given configuration.
    Must be called after SMPL path is confirmed to exist.
    """
    from comotion_demo.models import comotion as comotion_module

    model = comotion_module.CoMotion(use_coreml=use_coreml, pretrained=True)
    model.to(device).eval()
    return model


def _check_smpl():
    """Raise a clear error if SMPL_NEUTRAL.pkl is missing."""
    smpl_path = (
        Path(__file__).parent.parent
        / "ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl"
    )
    if not smpl_path.exists():
        raise FileNotFoundError(
            "SMPL model not found.\n"
            f"  Expected: {smpl_path}\n"
            "  1. Register at https://smpl.is.tue.mpg.de/\n"
            "  2. Download v1.1.0: basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl\n"
            f"  3. Copy and rename to: {smpl_path}"
        )


def _bbox_from_kpts2d(kpts2d: np.ndarray, img_hw: tuple, pad: float = 0.2) -> tuple:
    """Return (x0, y0, x1, y1) int bbox from (N,2) pixel keypoints, with padding, clipped to image."""
    h, w = img_hw
    x0, y0 = kpts2d[:, 0].min(), kpts2d[:, 1].min()
    x1, y1 = kpts2d[:, 0].max(), kpts2d[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw = max(x1 - x0, y1 - y0) * (1 + pad) / 2
    x0, y0, x1, y1 = cx - hw, cy - hw, cx + hw, cy + hw
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1) + 1), min(h, int(y1) + 1)
    return x0, y0, x1, y1


def _extract_crop(image_hwc: np.ndarray, bbox: tuple, size: int = 128) -> np.ndarray:
    """Crop and resize to (size, size, 3) uint8."""
    x0, y0, x1, y1 = bbox
    crop = image_hwc[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


class CoMotionLabeler:
    """
    Runs CoMotion on video files and yields per-frame pose labels.

    The model is loaded lazily on the first call to label_video().
    After initialization the same model instance is reused across all videos,
    avoiding repeated load overhead.

    Example:
        labeler = CoMotionLabeler()
        for label in labeler.label_video("clip.mp4"):
            process(label["persons"])  # list of per-person dicts
    """

    def __init__(self):
        self._model = None
        self._use_mps: bool = False

    def _ensure_model(self):
        if self._model is not None:
            return
        _check_smpl()
        cfg = _get_inference_config()
        self._use_mps = cfg["use_mps"]

        mode = (
            "CoreML detection + MPS refinement" if cfg["use_coreml"]
            else f"PyTorch on {cfg['device']}"
        )
        print(f"[CoMotionLabeler] Loading model ({mode})...")
        self._model = _load_model(cfg["use_coreml"], cfg["device"])
        print("[CoMotionLabeler] Model ready.")

    def video_has_half_body(
        self,
        video_path: Path | str,
        sample_frames: int = 20,
        sample_frameskip: int = 30,
        min_visible_joints: int = 14,
    ) -> bool:
        """
        Quick check: does the video contain at least one person with ≥half body visible?

        Samples frames (every sample_frameskip) and returns True if any person has
        at least min_visible_joints visible (out of 27). Use to filter out videos
        with only legs, only arms, etc. before full labeling.

        Args:
            video_path:         Path to video.
            sample_frames:      Max frames to check before giving up.
            sample_frameskip:   Sample every Nth frame (sparse = faster).
            min_visible_joints: Threshold for "half body" (14 ≈ half of 27).

        Returns:
            True if video passes filter, False otherwise.
        """
        count = 0
        for frame_label in self.label_video(
            video_path,
            frameskip=sample_frameskip,
            min_visible_joints=1,  # accept all for this check
        ):
            for p in frame_label["persons"]:
                if int(p["visibility"].sum()) >= min_visible_joints:
                    return True
            count += 1
            if count >= sample_frames:
                break
        return False

    def label_video(
        self,
        video_path: Path | str,
        start_frame: int = 0,
        num_frames: int = 1_000_000_000,
        frameskip: int = 1,
        min_visible_joints: int = 8,
    ) -> Generator[dict, None, None]:
        """
        Stream-label a single video with CoMotion.

        Yields one dict per frame that has at least one active track:
            frame_idx:  int
            persons:    list of per-person dicts, each with:
                track_id:   int
                kpts3d:     (27, 3) float32  pelvis-relative 3D keypoints
                visibility: (27,)   bool     per-joint inbounds mask
                confidence: float            1.0 for active tracks (CoMotion considers
                                             the track valid); not a probability

        Args:
            video_path:         Path to .mp4 (or other OpenCV-readable format).
            start_frame:        First frame index to process.
            num_frames:         Maximum number of frames to process.
            frameskip:          Process every Nth frame (1 = every frame).
            min_visible_joints: Discard detections with fewer visible joints than this.
        """
        self._ensure_model()

        # Deferred imports — comotion_demo asserts SMPL at module load
        from comotion_demo.utils import dataloading
        from comotion_demo.utils.helper import check_inbounds

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get original image dimensions for visibility check
        cap = cv2.VideoCapture(str(video_path))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        img_hw = (orig_h, orig_w)

        initialized = False

        frame_gen = dataloading.yield_image_and_K(
            video_path, start_frame, num_frames, frameskip
        )

        for frame_offset, (image_tensor, K) in enumerate(frame_gen):
            frame_idx = int(start_frame + frame_offset * frameskip)

            if not initialized:
                self._model.init_tracks(image_tensor.shape[-2:])
                initialized = True

            with torch.no_grad():
                _, track_state = self._model(image_tensor, K, use_mps=self._use_mps)

            # TrackTensorState fields (batch dim 0, track dim 1):
            #   pred_3d: (1, 48, 27, 3)  camera-space 3D
            #   pred_2d: (1, 48, 27, 2)  pixel-space 2D
            #   id:      (1, 48, 1)      track ID (0 = inactive slot)
            pred_3d = track_state.pred_3d[0].cpu()   # (48, 27, 3)
            pred_2d = track_state.pred_2d[0].cpu()   # (48, 27, 2)
            ids = track_state.id[0].cpu().flatten()  # (48,)

            active_mask = ids != 0
            if not active_mask.any():
                continue

            pred_3d_active = pred_3d[active_mask].numpy()   # (n, 27, 3)
            pred_2d_active = pred_2d[active_mask].numpy()   # (n, 27, 2)
            ids_active = ids[active_mask].numpy()           # (n,)

            K_np = K.numpy()

            # Extract per-person image crops from current frame
            image_rgb_hwc = (
                image_tensor.permute(1, 2, 0).clamp(0, 1) * 255
            ).byte().cpu().numpy()

            n_active = len(ids_active)
            crops_rgb = []
            for i in range(n_active):
                bbox = _bbox_from_kpts2d(pred_2d_active[i], img_hw)
                crops_rgb.append(_extract_crop(image_rgb_hwc, bbox))

            # Build per-person labels — confidence=1.0 for all active tracks
            active_confidences = np.ones(n_active, dtype=np.float32)
            persons = convert_frame_labels(
                pred_3d_active, K_np, img_hw, ids_active, active_confidences,
                pred_2d_all=pred_2d_active,
                crops_rgb=crops_rgb,
            )

            # Filter out detections with too few visible joints
            persons = [
                p for p in persons
                if int(p["visibility"].sum()) >= min_visible_joints
            ]

            if persons:
                yield {
                    "frame_idx": frame_idx,
                    "persons": persons,
                }

    def label_video_list(
        self,
        video_paths: list[Path | str],
        **kwargs,
    ) -> Generator[dict, None, None]:
        """
        Label multiple videos sequentially with the same model instance.

        Args:
            video_paths: Sequence of video paths.
            **kwargs:    Forwarded to label_video().
        """
        for vp in video_paths:
            print(f"[CoMotionLabeler] Labeling {Path(vp).name}...")
            try:
                yield from self.label_video(vp, **kwargs)
            except Exception as e:
                print(f"[CoMotionLabeler] Error on {Path(vp).name}: {e}")
                continue
