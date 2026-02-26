"""
Pexels video fetcher for sports/fitness content.

Fetches and downloads videos from the Pexels API (free, CC0 licensed).
Downloads are stored locally; subsequent runs skip already-downloaded files.

Usage:
    python video_sources/fetch_videos.py --n 5 --out /tmp/stream_pose_videos

Requirements:
    PEXELS_API_KEY env var  (free key at https://www.pexels.com/api/)
    pip install requests
"""

import argparse
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent / ".env")

PEXELS_API_BASE = "https://api.pexels.com/videos"

# Sports/fitness queries that yield clear single-person pose footage
DEFAULT_QUERIES = [
    "sports athlete",
    "fitness workout",
    "yoga pose",
    "dancing",
    "tennis player",
    "golf swing",
    "basketball player",
    "running athlete",
    "gymnastics",
    "martial arts",
]

# Preferred video qualities (first match wins)
QUALITY_PRIORITY = ["hd", "sd", "uhd"]


def _request_with_retry(url, *, max_retries=4, base_sleep=5, **kwargs):
    """GET with exponential backoff on 429 Too Many Requests."""
    for attempt in range(max_retries):
        resp = requests.get(url, **kwargs)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", base_sleep * (2 ** attempt)))
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def get_api_key() -> str:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise RuntimeError(
            "PEXELS_API_KEY not found.\n"
            "Add it to .env in the repo root:\n"
            "  echo 'PEXELS_API_KEY=your_key_here' >> .env\n"
            "Get a free key at https://www.pexels.com/api/"
        )
    return key


def search_videos(
    query: str,
    per_page: int = 15,
    page: int = 1,
    api_key: str = None,
) -> list[dict]:
    """Search Pexels for videos matching query. Returns raw video metadata list."""
    headers = {"Authorization": api_key or get_api_key()}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": "landscape",
    }
    resp = _request_with_retry(
        f"{PEXELS_API_BASE}/search",
        headers=headers,
        params=params,
        timeout=15,
    )
    return resp.json().get("videos", [])


def pick_video_file(video: dict, min_width: int = 640, max_width: int = 1920) -> dict | None:
    """
    Pick best video file within resolution bounds.
    Prefers HD, then SD, then UHD. Falls back to any file if none fit bounds.
    """
    files = video.get("video_files", [])
    candidates = [f for f in files if min_width <= f.get("width", 0) <= max_width]
    if not candidates:
        candidates = files  # fall back to all files

    def quality_key(f):
        q = f.get("quality", "")
        try:
            return QUALITY_PRIORITY.index(q)
        except ValueError:
            return len(QUALITY_PRIORITY)

    candidates = sorted(candidates, key=quality_key)
    return candidates[0] if candidates else None


def download_video(url: str, dest: Path, timeout: int = 120) -> Path:
    """Stream-download video from URL to dest path. Returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with _request_with_retry(url, stream=True, timeout=timeout) as resp:
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB chunks
                    f.write(chunk)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def fetch_videos(
    queries: list[str] = None,
    n_per_query: int = 5,
    output_dir: Path = Path("/tmp/stream_pose_videos"),
    api_key: str = None,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Fetch and download videos from Pexels for each search query.

    Args:
        queries:       List of search strings. Defaults to DEFAULT_QUERIES.
        n_per_query:   Max videos to download per query.
        output_dir:    Directory to save .mp4 files into.
        api_key:       Pexels API key. Reads PEXELS_API_KEY env var if not provided.
        skip_existing: Skip download if file already exists.

    Returns:
        List of downloaded (or skipped) video paths.
    """
    api_key = api_key or get_api_key()
    queries = queries or DEFAULT_QUERIES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []

    for query in queries:
        print(f"[Pexels] Searching: '{query}' (up to {n_per_query} videos)...")
        try:
            videos = search_videos(query, per_page=n_per_query, api_key=api_key)
        except requests.RequestException as e:
            print(f"  Search failed: {e}")
            continue
        time.sleep(1.0)  # pace search requests

        for video in videos[:n_per_query]:
            vid_id = video["id"]
            dest = output_dir / f"pexels_{vid_id}.mp4"

            if skip_existing and dest.exists():
                print(f"  [skip] {dest.name}")
                downloaded.append(dest)
                continue

            vfile = pick_video_file(video)
            if not vfile:
                print(f"  [skip] id={vid_id}: no suitable video file found")
                continue

            url = vfile["link"]
            w = vfile.get("width", "?")
            h = vfile.get("height", "?")
            quality = vfile.get("quality", "?")
            print(f"  Downloading {dest.name} ({w}x{h}, {quality})...", end=" ", flush=True)

            try:
                download_video(url, dest)
                downloaded.append(dest)
                print("done")
                time.sleep(1.0)  # be polite to the API
            except Exception as e:
                print(f"FAILED: {e}")

    print(f"\n[Pexels] Total: {len(downloaded)} videos in {output_dir}")
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sports videos from Pexels")
    parser.add_argument("--query", nargs="+", default=DEFAULT_QUERIES,
                        help="Search queries (default: built-in sports queries)")
    parser.add_argument("--n", type=int, default=5, metavar="N",
                        help="Videos per query (default: 5)")
    parser.add_argument("--out", type=Path, default=Path("/tmp/stream_pose_videos"),
                        help="Output directory")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Pexels API key (or set PEXELS_API_KEY env var)")
    args = parser.parse_args()

    paths = fetch_videos(args.query, args.n, args.out, args.api_key)
    for p in paths:
        print(p)
