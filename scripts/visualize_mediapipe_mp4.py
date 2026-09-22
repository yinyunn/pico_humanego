"""在 PICO 左右目视频上分别检测双手，并输出双目匹配结果。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess.MediaPipeHands import create_hand_landmarker
from tools.pico_import.video_reader import crop_side_by_side, load_video_pts


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_SIDES = ("Left", "Right")
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "weights" / "mediapipe" / "hand_landmarker.task"


def draw_hand(frame, landmarks, label: str, score: float, hand_id: int):
    """在单目裁切图上绘制一只手，并返回局部像素点。"""
    h, w = frame.shape[:2]
    points = []
    for index, lm in enumerate(landmarks):
        u = int(lm.x * w)
        v = int(lm.y * h)
        points.append((u, v))
        cv2.circle(frame, (u, v), 5, (0, 255, 0), -1)
        cv2.putText(
            frame, str(index), (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

    if points:
        x, y = points[0]
        cv2.putText(
            frame, f"{label} {score:.2f} id={hand_id}",
            (x, max(30, y - 25)), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return points


def _hand_label(handedness: Any) -> tuple[str, float]:
    if not handedness:
        return "Unknown", 0.0
    category = handedness[0]
    label = getattr(category, "category_name", None) or "Unknown"
    score = getattr(category, "score", None)
    return str(label), float(score or 0.0)


def _swap_label(label: str) -> str:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def _detect_eye(
    eye_frame,
    landmarker,
    timestamp_ms: int,
    eye: str,
    x_offset: int,
    swap_handedness: bool,
) -> list[dict]:
    """在一个目图像上检测最多两只手，返回全幅坐标。"""
    rgb = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    detections = []

    for hand_id, landmarks in enumerate(result.hand_landmarks or []):
        handedness = []
        if result.handedness and hand_id < len(result.handedness):
            handedness = result.handedness[hand_id]
        raw_label, score = _hand_label(handedness)
        hand_side = _swap_label(raw_label) if swap_handedness else raw_label
        draw_hand(eye_frame, landmarks, hand_side, score, hand_id)

        joints = []
        for joint_id, lm in enumerate(landmarks):
            joints.append({
                "id": joint_id,
                "u": float(lm.x * eye_frame.shape[1] + x_offset),
                "v": float(lm.y * eye_frame.shape[0]),
                "u_eye": float(lm.x * eye_frame.shape[1]),
                "v_eye": float(lm.y * eye_frame.shape[0]),
                "x_norm": float(lm.x),
                "y_norm": float(lm.y),
            })
        detections.append({
            "eye": eye,
            "eye_hand_id": hand_id,
            "label": raw_label,
            "hand_side": hand_side,
            "score": score,
            "score_type": "handedness_classification_not_per_joint_confidence",
            "joints": joints,
        })
    return detections


def _best_hand(hands: list[dict], side: str) -> dict | None:
    candidates = [hand for hand in hands if hand["hand_side"] == side]
    if not candidates:
        return None
    return max(candidates, key=lambda hand: hand["score"])


def _match_stereo_hands(left_hands: list[dict], right_hands: list[dict]) -> list[dict]:
    """仅按手性产生候选；未经几何验证，不可直接用于三角化。"""
    matches = []
    for side in HAND_SIDES:
        left = _best_hand(left_hands, side)
        right = _best_hand(right_hands, side)
        matches.append({
            "hand_side": side,
            "matched": left is not None and right is not None,
            "match_method": "handedness",
            "geometry_verified": False,
            "ambiguous": any(sum(h["hand_side"] == side for h in hands) > 1
                             for hands in (left_hands, right_hands)),
            "left_eye_hand_id": None if left is None else left["eye_hand_id"],
            "right_eye_hand_id": None if right is None else right["eye_hand_id"],
            "left_score": None if left is None else left["score"],
            "right_score": None if right is None else right["score"],
        })
    return matches


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Separate PICO stereo views, detect up to two hands in each eye, and match them by handedness."
    )
    parser.add_argument("--input", required=True, help="Input side-by-side PICO MP4")
    parser.add_argument("--output", default="mediapipe_vis.mp4", help="Output visualization MP4")
    parser.add_argument("--json", default="mediapipe_2d.jsonl", help="Output stereo hand JSONL")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="MediaPipe HandLandmarker model")
    parser.add_argument("--eye-width", type=int, default=1080, help="Width of each eye crop")
    parser.add_argument("--pts", type=Path, help="Decoded video PTS JSONL (required by writer smoke)")
    parser.add_argument("--sync-manifest", type=Path, help="Existing recording synchronization manifest")
    parser.add_argument("--max-frames", type=int, help="Limit smoke test length")
    parser.add_argument(
        "--swap-handedness", action="store_true",
        help="Swap MediaPipe Left/Right labels before stereo matching for mirrored-input conventions",
    )
    parser.add_argument("--show", action="store_true", help="Show visualization window while processing")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    model_path = Path(args.model)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
    if args.eye_width <= 0:
        raise ValueError("--eye-width must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    pts = {int(row["frame_index"]): row for row in load_video_pts(args.pts)} if args.pts else {}
    sync = {}
    if args.sync_manifest:
        sync = {int(row["frame_index"]): row for row in
                (json.loads(line) for line in args.sync_manifest.read_text().splitlines() if line.strip())}
    outputs = [Path(args.output).resolve(), Path(args.json).resolve(),
               Path(args.json).with_suffix(".summary.json").resolve(),
               Path(args.json).with_suffix(".contact_sheet.jpg").resolve()]
    inputs = {input_path.resolve(), model_path.resolve()}
    inputs.update(p.resolve() for p in (args.pts, args.sync_manifest) if p)
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError("Input and output paths must be distinct")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_width = 2 * args.eye_width
    if width != expected_width:
        cap.release()
        raise ValueError(f"expected side-by-side width {expected_width}, got {width}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video: {output_path}")

    print("Video:")
    print("  resolution:", width, "x", height)
    print("  eye resolution:", args.eye_width, "x", height)
    print("  fps:", fps)
    print("  frames:", frame_count)
    print("  detector mode: left eye + right eye, up to 2 hands per eye")

    frame_idx = 0
    left_detection_count = 0
    right_detection_count = 0
    matched_count = 0
    missing = {"left": 0, "right": 0}
    samples = []
    limit = min(frame_count, args.max_frames or frame_count)
    sample_indices = set(np.linspace(0, max(0, limit - 1), 6, dtype=int).tolist())
    previous_timestamp_ms = -1
    timestamp_adjustments = 0
    try:
        with (
            create_hand_landmarker(model_path=model_path, video_mode=True, allow_download=False) as left_landmarker,
            create_hand_landmarker(model_path=model_path, video_mode=True, allow_download=False) as right_landmarker,
            open(json_path, "w", encoding="utf-8") as json_file,
        ):
            while True:
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                left_view = crop_side_by_side(frame, "left", args.eye_width)
                right_view = crop_side_by_side(frame, "right", args.eye_width)
                if args.pts and frame_idx not in pts:
                    raise ValueError(f"Missing decoded PTS for frame {frame_idx}")
                pts_seconds = float(pts[frame_idx]["pts_seconds"]) if args.pts else frame_idx / fps
                nominal_ms = round(pts_seconds * 1000)
                timestamp_ms = max(previous_timestamp_ms + 1, nominal_ms)
                timestamp_adjustments += timestamp_ms != nominal_ms
                previous_timestamp_ms = timestamp_ms
                left_hands = _detect_eye(
                    left_view, left_landmarker, timestamp_ms, "left", 0, args.swap_handedness
                )
                right_hands = _detect_eye(
                    right_view, right_landmarker, timestamp_ms, "right", args.eye_width, args.swap_handedness
                )
                matches = _match_stereo_hands(left_hands, right_hands)
                flat_hands = left_hands + right_hands
                left_detection_count += len(left_hands)
                right_detection_count += len(right_hands)
                matched_count += sum(match["matched"] for match in matches)
                missing["left"] += not bool(left_hands)
                missing["right"] += not bool(right_hands)

                cv2.putText(
                    frame, f"frame={frame_idx} L={len(left_hands)} R={len(right_hands)} matched={sum(m['matched'] for m in matches)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.line(frame, (args.eye_width, 0), (args.eye_width, height), (0, 255, 255), 2)

                json_file.write(json.dumps({
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "pts_seconds": pts_seconds,
                    "timestamp_source": "decoded_pts" if args.pts else "nominal_fps",
                    "sync": sync.get(frame_idx),
                    "hand_source": "mediapipe_stereo_2d_debug",
                    "is_final_hand_label": False,
                    "eyes": {
                        "left": {"crop_x": 0, "width": args.eye_width, "hands": left_hands},
                        "right": {"crop_x": args.eye_width, "width": args.eye_width, "hands": right_hands},
                    },
                    "stereo_matches": matches,
                    "hands": flat_hands,
                }, ensure_ascii=False) + "\n")
                writer.write(frame)
                if frame_idx in sample_indices:
                    samples.append(cv2.resize(frame, (1080, round(height * 1080 / width))))
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx}/{limit} frames", flush=True)

                if args.show:
                    cv2.imshow("MediaPipe Stereo Hands", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_idx == 0:
        raise ValueError("No frames decoded")
    if frame_idx < limit and not args.show:
        raise ValueError(f"Video ended early: decoded {frame_idx}, expected {limit}")
    summary = {
        "status": "NEEDS_2D_VISUAL_REVIEW", "input": str(input_path.resolve()),
        "model": str(model_path.resolve()), "frames": frame_idx,
        "left_detections": left_detection_count, "right_detections": right_detection_count,
        "handedness_candidate_pairs": matched_count, "geometry_verified": False,
        "no_hand_detection_frame_ratio": {eye: n / frame_idx for eye, n in missing.items()},
        "timestamp_source": "decoded_pts" if args.pts else "nominal_fps",
        "timestamp_ms_adjustments": timestamp_adjustments,
        "score_note": "MediaPipe handedness score is not per-joint detection confidence",
        "depth_source": None, "is_final_hand_label": False,
    }
    json_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if samples and not cv2.imwrite(str(json_path.with_suffix(".contact_sheet.jpg")), np.vstack(samples)):
        raise IOError("Cannot write contact sheet")

    print("\nDone.")
    print("  processed frames:", frame_idx)
    print("  left-eye detections:", left_detection_count)
    print("  right-eye detections:", right_detection_count)
    print("  stereo matched hand observations:", matched_count)
    print("Visualization:", output_path)
    print("Stereo 2D keypoints:", json_path)
    return summary


if __name__ == "__main__":
    main()
