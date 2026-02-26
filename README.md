# stream-pose

Experiment in training a 3D pose estimation model **without saving any raw data to disk**.

## Motivation

Vision models typically require large, pre-collected datasets stored locally. This project explores an alternative: a streaming pipeline that continuously fetches video, generates 3D pose labels on-the-fly using an auto-labeling model, and feeds labeled samples directly into a training buffer — all in memory.

The goal is to validate whether a student model can learn 3D human pose estimation purely from a teacher model's outputs, with no persistent dataset.

## How it works

```
Pexels API ──► fetch videos ──► CoMotion labeler ──► ring buffer ──► training loop
                                  (auto-label)        (in-memory)     (coming soon)
```

1. **Video source** (`video_sources/fetch_videos.py`): Streams sports and fitness videos from the [Pexels API](https://www.pexels.com/api/) (free, CC0 licensed).

2. **Auto-labeler** (`labeler/comotion_labeler.py`): Runs [CoMotion](https://github.com/apple/ml-comotion) on each video frame to produce per-person 3D pose labels — 27 keypoints in pelvis-relative coordinates with per-joint visibility masks. SMPL parameters are used internally but kept out of training targets.

3. **Streaming dataset** (`data/streaming_dataset.py`): A thread-safe ring buffer that holds the most recent N labeled samples. A background producer thread fills it continuously while the training loop consumes from it.

## Current status

- [x] Video fetching pipeline (Pexels API)
- [x] CoMotion auto-labeling with CoreML acceleration (Apple Silicon)
- [x] Streaming ring buffer dataset with augmentations (flip, scale, jitter)
- [ ] Training script (next)
- [ ] Student model architecture

## Setup

**Requirements:** Python 3.12, Apple Silicon (MPS + CoreML) recommended.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# chumpy requires special install (patched for NumPy 2.x)
pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"

# install CoMotion package (deps already covered by requirements.txt)
pip install --no-deps -e ml-comotion/
```

**Model weights** (required, not included):

1. **CoMotion checkpoints** — run the download script:
   ```bash
   cd ml-comotion && bash get_pretrained_models.sh && cd ..
   ```

2. **SMPL neutral body model** — register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/), download `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, and place it at:
   ```
   ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl
   ```

**API key** — create `.env` in the repo root:
```
PEXELS_API_KEY=your_key_here
```

## Usage

```bash
# Label a local video and inspect output
python pipeline.py --dry-run --video /path/to/video.mp4

# Fetch from Pexels and run persistent labeling pipeline
python pipeline.py --fetch --queries "sports athlete" "yoga pose" --n 5 --loop

# Visualise CoMotion results interactively
python view_results.py results/video.pt --video /path/to/video.mp4
```

## Acknowledgements

- **[CoMotion](https://github.com/apple/ml-comotion)** (Apple Inc.) — used as the auto-labeling teacher model. CoMotion code is licensed under the [Apple Sample Code License](ml-comotion/LICENSE.md); model weights are licensed for research use only under the [Apple ML Research Model License](ml-comotion/LICENSE_MODEL.md).

- **[SMPL](https://smpl.is.tue.mpg.de/)** (Max Planck Institute) — used internally by CoMotion for 3D body shape and pose. SMPL is for non-commercial research use only.

The student model trained by this pipeline is independent of SMPL and CoMotion's model weights — it learns only from their 3D coordinate outputs.
