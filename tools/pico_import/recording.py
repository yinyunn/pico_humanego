"""PICO 原始视频/tracking 的同步中间层。

输出的 sync_manifest.jsonl 是 ``preprocess/pico`` 的唯一输入契约之一；它不写
HumanEgo 的 ``preprocess/all_data``，也不做任何手部坐标变换。
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path

from .calibration import PicoCalibration
from .tracking_reader import load_tracking
from .video_reader import load_video_pts


@dataclass
class PicoImportSession:
    video_path: Path
    tracking_path: Path
    pts_path: Path
    header: dict
    tracking: list[dict]
    video_pts: list[dict]
    calibration: PicoCalibration

    @classmethod
    def open(cls, video_path, tracking_path, pts_path, eye: str = "left"):
        header, tracking = load_tracking(tracking_path)
        video_pts = load_video_pts(pts_path)
        calibration = PicoCalibration.from_header(header, eye=eye)
        return cls(Path(video_path), Path(tracking_path), Path(pts_path), header, tracking, video_pts, calibration)

    def synchronize(self, max_delta_ms: float = 50.0) -> list[dict]:
        if not self.tracking:
            raise ValueError("tracking stream is empty")
        base_ns = int(self.header["timeStampNs"])
        timestamps = [row["timestamp_ns"] for row in self.tracking]
        max_delta_ns = int(float(max_delta_ms) * 1e6)
        result = []
        for video in self.video_pts:
            video_ts = base_ns + round(float(video["pts_seconds"]) * 1e9)
            right = bisect.bisect_left(timestamps, video_ts)
            candidates = [index for index in (right - 1, right) if 0 <= index < len(timestamps)]
            index = min(candidates, key=lambda item: abs(timestamps[item] - video_ts))
            delta_ns = timestamps[index] - video_ts
            result.append({
                "frame_index": int(video["frame_index"]),
                "pts_seconds": float(video["pts_seconds"]),
                "video_timestamp_ns": int(video_ts),
                "tracking_index": int(index),
                "tracking_timestamp_ns": int(timestamps[index]),
                "delta_ns": int(delta_ns),
                "delta_ms": float(delta_ns / 1e6),
                "within_tolerance": abs(delta_ns) <= max_delta_ns,
            })
        return result

    def write_intermediate(self, output_dir: str | Path, max_delta_ms: float = 50.0) -> dict:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        matches = self.synchronize(max_delta_ms)
        with (output / "sync_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for match in matches:
                handle.write(json.dumps(match, ensure_ascii=False) + "\n")
        (output / "calibration.json").write_text(json.dumps(self.calibration.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        deltas = [abs(row["delta_ms"]) for row in matches]
        summary = {
            "video": str(self.video_path),
            "tracking": str(self.tracking_path),
            "pts": str(self.pts_path),
            "video_frame_count": len(matches),
            "tracking_record_count": len(self.tracking),
            "sync_max_abs_ms": max(deltas) if deltas else None,
            "sync_mean_abs_ms": sum(deltas) / len(deltas) if deltas else None,
            "sync_within_tolerance_ratio": sum(row["within_tolerance"] for row in matches) / len(matches) if matches else 0.0,
            "status": "PASS" if matches and all(row["within_tolerance"] for row in matches) else "CHECK",
            "next_consumer": "preprocess/pico",
        }
        (output / "sync_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary


def synchronize_recording(video_path, tracking_path, pts_path, output_dir, max_delta_ms=50.0, eye="left"):
    return PicoImportSession.open(video_path, tracking_path, pts_path, eye=eye).write_intermediate(output_dir, max_delta_ms)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synchronize PICO MP4 PTS with trackingData before HumanEgo preprocessing")
    parser.add_argument("video", type=Path)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("pts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-ms", type=float, default=50.0)
    parser.add_argument("--eye", choices=("left", "right"), default="left", help="Select the matching PICO extrinsic (left=E1, right=E0)")
    args = parser.parse_args(argv)
    summary = synchronize_recording(args.video, args.tracking, args.pts, args.output, args.max_delta_ms, args.eye)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
