"""
Convert CoMotion camera-space 3D keypoints to training-ready format.

Conversion steps:
  1. Project camera-space 3D → 2D pixel coords using K intrinsics
  2. Check each joint is within image bounds → per-joint visibility mask (27,)
  3. Subtract pelvis (joint 0) from all joints → pelvis-relative 3D coords
  4. Normalize by pelvis-to-head distance → scale-invariant (resolution, camera, body size)

Result: scale-invariant, camera-distance-independent, body-size-invariant representation.
Training generalizes across videos shot from different distances and people of different heights.

Joint convention — SMPL joints_face (27 joints):
  Index  Name
  0      pelvis (root)
  1      left hip        2  right hip
  3      spine1          4  left knee       5  right knee
  6      spine2          7  left ankle      8  right ankle
  9      spine3          10 left foot       11 right foot
  12     neck            13 left collar     14 right collar
  15     head
  16     left shoulder   17 right shoulder
  18     left elbow      19 right elbow
  20     left wrist      21 right wrist
  22-26  face vertices (nose tip, left/right eye, left/right ear)
"""

import numpy as np

NUM_JOINTS = 27  # SMPL joints_face


def project_to_2d(kpts3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project camera-space 3D keypoints to 2D pixel coordinates.

    Args:
        kpts3d: (..., N, 3) array in camera space (X right, Y down, Z forward)
        K:      (2, 3) intrinsics [[fx, 0, cx], [0, fy, cy]]

    Returns:
        kpts2d: (..., N, 2) pixel coordinates
    """
    z = kpts3d[..., 2:3].clip(0.01, None)  # 1cm minimum depth
    x_norm = kpts3d[..., 0:1] / z
    y_norm = kpts3d[..., 1:2] / z

    px = K[0, 0] * x_norm + K[0, 2]
    py = K[1, 1] * y_norm + K[1, 2]
    return np.concatenate([px, py], axis=-1)


def compute_visibility(kpts2d: np.ndarray, img_hw: tuple[int, int]) -> np.ndarray:
    """
    Boolean mask: True where joint projects within image bounds.

    Args:
        kpts2d: (..., N, 2) pixel coordinates
        img_hw: (height, width)

    Returns:
        visible: (..., N) bool array
    """
    h, w = img_hw
    in_x = (kpts2d[..., 0] >= 0) & (kpts2d[..., 0] <= w - 1)
    in_y = (kpts2d[..., 1] >= 0) & (kpts2d[..., 1] <= h - 1)
    return in_x & in_y


def to_pelvis_relative(kpts3d: np.ndarray) -> np.ndarray:
    """
    Express all joints relative to the pelvis (joint 0).

    After this transform:
      - kpts3d[..., 0, :] == [0, 0, 0]   (pelvis at origin)
      - All other joints are pelvis-relative offsets

    Args:
        kpts3d: (..., 27, 3)

    Returns:
        kpts3d_rel: (..., 27, 3)
    """
    pelvis = kpts3d[..., 0:1, :]   # (..., 1, 3)
    return kpts3d - pelvis


# Reference joints for scale normalization (pelvis=0, neck=12, head=15)
REF_PELVIS, REF_HEAD, REF_NECK = 0, 15, 12


def normalize_by_scale(
    kpts3d: np.ndarray,
    ref_a: int = REF_PELVIS,
    ref_b: int = REF_HEAD,
    fallback_ref_b: int = REF_NECK,
) -> np.ndarray:
    """
    Scale-normalize keypoints so ref distance = 1.0.

    Uses pelvis-to-head by default; falls back to pelvis-to-neck if head is
    degenerate (same as pelvis). Makes representation invariant to resolution,
    camera distance, and body size.

    Args:
        kpts3d: (..., 27, 3) pelvis-relative keypoints
        ref_a:  first reference joint (default pelvis)
        ref_b:  second reference joint (default head)
        fallback_ref_b: fallback if ref_b is degenerate (default neck)

    Returns:
        kpts3d_scaled: (..., 27, 3) with ||kpts[ref_b] - kpts[ref_a]|| = 1
    """
    ref_dist = np.linalg.norm(
        kpts3d[..., ref_b : ref_b + 1, :] - kpts3d[..., ref_a : ref_a + 1, :],
        axis=-1,
        keepdims=True,
    )
    if fallback_ref_b is not None:
        fallback_dist = np.linalg.norm(
            kpts3d[..., fallback_ref_b : fallback_ref_b + 1, :]
            - kpts3d[..., ref_a : ref_a + 1, :],
            axis=-1,
            keepdims=True,
        )
        use_fallback = (ref_dist < 1e-6) & (fallback_dist >= 1e-6)
        ref_dist = np.where(use_fallback, fallback_dist, ref_dist)
    ref_dist = np.maximum(ref_dist, 1e-6)
    return kpts3d / ref_dist


def convert_label(
    pred_3d_cam: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int],
    det_confidence: float = None,
    pred_2d: np.ndarray | None = None,
    crop_rgb: np.ndarray | None = None,
) -> dict:
    """
    Convert one person's camera-space output to a training-ready label.

    Args:
        pred_3d_cam:    (27, 3) float  camera-space keypoints
        K:              (2, 3)  float  camera intrinsics
        img_hw:         (H, W)         original image size
        det_confidence: float | None   CoMotion detection confidence score

    Returns dict:
        kpts3d:     (27, 3) float32  pelvis-relative, scale-normalized (ref dist=1)
        visibility: (27,)   bool     per-joint inbounds mask
        confidence: float            detection-level score (0.0 if unavailable)
    """
    kpts2d = project_to_2d(pred_3d_cam, K)           # (27, 2)
    visibility = compute_visibility(kpts2d, img_hw)   # (27,) bool
    kpts3d_rel = to_pelvis_relative(pred_3d_cam)      # (27, 3)
    kpts3d_norm = normalize_by_scale(kpts3d_rel)      # (27, 3) ref dist = 1

    label = {
        "kpts3d": kpts3d_norm.astype(np.float32),
        "visibility": visibility,
        "confidence": float(det_confidence) if det_confidence is not None else 0.0,
    }
    if pred_2d is not None:
        label["pred_2d"] = pred_2d.astype(np.float32)
    if crop_rgb is not None:
        label["crop_rgb"] = crop_rgb
    return label


def convert_frame_labels(
    pred_3d_cam: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int],
    track_ids: np.ndarray,
    confidences: np.ndarray = None,
    pred_2d_all: np.ndarray | None = None,
    crops_rgb: list[np.ndarray] | None = None,
) -> list[dict]:
    """
    Convert all detections in a frame to per-person training labels.

    Args:
        pred_3d_cam:  (n, 27, 3)  camera-space keypoints for n persons
        K:            (2, 3)      camera intrinsics
        img_hw:       (H, W)      original image dimensions
        track_ids:    (n,)        CoMotion track IDs
        confidences:  (n,) | None per-detection sigmoid confidence
        pred_2d_all:  (n, 27, 2) | None  pixel-space 2D keypoints
        crops_rgb:    list of (128,128,3) uint8 | None  per-person image crops

    Returns:
        List of n dicts, each containing:
            track_id:   int
            kpts3d:     (27, 3) float32  pelvis-relative, scale-normalized (ref=1)
            visibility: (27,)   bool
            confidence: float
            pred_2d:    (27, 2) float32  (if provided)
            crop_rgb:   (128, 128, 3) uint8  (if provided)
    """
    n = len(pred_3d_cam)
    confs = confidences if confidences is not None else [None] * n

    labels = []
    for i in range(n):
        p2d = pred_2d_all[i] if pred_2d_all is not None else None
        crop = crops_rgb[i] if crops_rgb is not None else None
        label = convert_label(pred_3d_cam[i], K, img_hw, confs[i], pred_2d=p2d, crop_rgb=crop)
        label["track_id"] = int(track_ids[i])
        labels.append(label)

    return labels
