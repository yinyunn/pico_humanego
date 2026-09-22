"""读取 PICO MP4 的解码 PTS，并提供左右目 crop。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

VALID_EYES = ("left", "right")


def crop_side_by_side(frame, eye: str, eye_width: int = 1080):
    """从 PICO 左右拼接帧中裁出指定目图像。"""
    if eye not in VALID_EYES:
        raise ValueError("eye must be left or right")
    eye_width = int(eye_width)
    if eye_width <= 0:
        raise ValueError("eye_width must be positive")
    if frame is None or getattr(frame, "ndim", 0) < 2:
        raise ValueError("frame must be an image array")
    height, width = frame.shape[:2]
    del height
    expected_width = 2 * eye_width
    if width != expected_width:
        raise ValueError(f"expected side-by-side width {expected_width}, got {width}")
    start = 0 if eye == "left" else eye_width
    return frame[:, start:start + eye_width]


def load_video_pts(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(int(a["frame_index"]) >= int(b["frame_index"]) for a, b in zip(rows, rows[1:])):
        raise ValueError("video PTS frame indices must be strictly increasing")
    if any(float(a["pts_seconds"]) > float(b["pts_seconds"]) for a, b in zip(rows, rows[1:])):
        raise ValueError("video PTS must be monotonically increasing")
    return rows


class PicoVideoReader:
    def __init__(self, video_path: str | Path, eye_width: int = 1080, eye: str = "left"):
        if eye not in VALID_EYES:
            raise ValueError("eye must be left or right")
        self.path = Path(video_path)
        self.eye_width = int(eye_width)
        if self.eye_width <= 0:
            raise ValueError("eye_width must be positive")
        self.eye = eye
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"cannot open PICO video: {self.path}")

    def read(self, frame_index: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode frame {frame_index} from {self.path}")
        return crop_side_by_side(frame, self.eye, self.eye_width)

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
