"""Frame sources: a video file, a folder of images, or a single image.

handcount.py and run.py both use ``iter_frames`` with the same ``every``
setting, so frame indices line up between the hand count and the run.
"""

import os

import cv2

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def source_info(path):
    """Return ('video'|'images', fps or None, frame_count or None)."""
    if os.path.isdir(path):
        files = list_images(path)
        return "images", None, len(files)
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "images", None, 1
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    return "video", fps, n


def list_images(folder):
    files = [f for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in IMAGE_EXT]
    return [os.path.join(folder, f) for f in files]


def iter_frames(path, every=1, max_frames=None):
    """Yield (frame_index, name, BGR image). Video frames keep their original index."""
    every = max(1, int(every))
    produced = 0
    if os.path.isdir(path) or os.path.splitext(path)[1].lower() in IMAGE_EXT:
        files = list_images(path) if os.path.isdir(path) else [path]
        for i, f in enumerate(files):
            if i % every:
                continue
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read image {f!r}")
            yield i, os.path.basename(f), img
            produced += 1
            if max_frames and produced >= max_frames:
                return
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path!r}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield i, f"frame_{i:05d}", frame
                produced += 1
                if max_frames and produced >= max_frames:
                    break
            i += 1
    finally:
        cap.release()
