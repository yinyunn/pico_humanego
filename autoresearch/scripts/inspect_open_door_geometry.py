#!/usr/bin/env python3
"""Audit raw open_door ICT, anchor, and coordinate-frame consistency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rotation_jump_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    relative = r0.T @ r1
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def read_session(session_dir: Path) -> dict:
    files = sorted(session_dir.glob("preprocess/all_data/*/training_data.json"))
    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data["metadata"]
            hands = data.get("entities", {}).get("hands") or {}
            hand = hands.get("left")
            if hand is None:
                continue
            objects = data.get("entities", {}).get("objects") or {}
            anchor_key = metadata.get("anchor_key", "obj1")
            anchor = objects.get(anchor_key)
            if anchor is None:
                continue
            hand_world = np.asarray(hand["T_hand_to_world"], dtype=np.float64)
            anchor_world = np.asarray(anchor["T_obj_to_world"], dtype=np.float64)
            cam_ref_to_world = np.asarray(
                metadata["world_transforms"]["cam0"], dtype=np.float64
            )
            world_to_cam = np.linalg.inv(cam_ref_to_world)
            hand_cam = world_to_cam @ hand_world
            anchor_cam = world_to_cam @ anchor_world
            rows.append({
                "frame": int(metadata.get("idx", int(path.parent.name))),
                "hand_world": hand_world,
                "hand_cam": hand_cam,
                "anchor_world": anchor_world,
                "anchor_cam": anchor_cam,
                "grasp": float(hand.get("grasp", 0.0)) > 0.5,
                "finished": float(metadata.get("is_finished", 0.0)) > 0.5,
                "anchor_key": anchor_key,
            })
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue

    rows.sort(key=lambda row: row["frame"])
    if not rows:
        return {"session": session_dir.name, "frames": 0}

    hand_world = np.stack([row["hand_world"] for row in rows])
    hand_cam = np.stack([row["hand_cam"] for row in rows])
    anchor_world = np.stack([row["anchor_world"] for row in rows])
    anchor_cam = np.stack([row["anchor_cam"] for row in rows])

    hand_world_pos_step = np.linalg.norm(np.diff(hand_world[:, :3, 3], axis=0), axis=1)
    hand_cam_pos_step = np.linalg.norm(np.diff(hand_cam[:, :3, 3], axis=0), axis=1)
    hand_cam_rot_step = [
        rotation_jump_deg(hand_cam[i - 1, :3, :3], hand_cam[i, :3, :3])
        for i in range(1, len(hand_cam))
    ]
    anchor_world_pos_step = np.linalg.norm(
        np.diff(anchor_world[:, :3, 3], axis=0), axis=1
    )
    anchor_cam_pos_step = np.linalg.norm(
        np.diff(anchor_cam[:, :3, 3], axis=0), axis=1
    )

    return {
        "session": session_dir.name,
        "frames": len(rows),
        "anchor_key": rows[0]["anchor_key"],
        "grasp_fraction": float(np.mean([row["grasp"] for row in rows])),
        "finished_fraction": float(np.mean([row["finished"] for row in rows])),
        "first_finished_frame": next(
            (row["frame"] for row in rows if row["finished"]), None
        ),
        "hand_world_path_m": float(hand_world_pos_step.sum()),
        "hand_cam_path_m": float(hand_cam_pos_step.sum()),
        "hand_world_step_p95_mm": percentile((hand_world_pos_step * 1000).tolist(), 95),
        "hand_world_step_max_mm": float(hand_world_pos_step.max() * 1000)
        if len(hand_world_pos_step) else 0.0,
        "hand_cam_step_p95_mm": percentile((hand_cam_pos_step * 1000).tolist(), 95),
        "hand_cam_step_max_mm": float(hand_cam_pos_step.max() * 1000)
        if len(hand_cam_pos_step) else 0.0,
        "hand_cam_rot_step_p95_deg": percentile(hand_cam_rot_step, 95),
        "hand_cam_rot_step_max_deg": float(max(hand_cam_rot_step))
        if hand_cam_rot_step else 0.0,
        "hand_cam_pos_span_mm": float(
            np.linalg.norm(hand_cam[:, :3, 3].max(axis=0) - hand_cam[:, :3, 3].min(axis=0))
            * 1000
        ),
        "anchor_world_pos_span_mm": float(
            np.linalg.norm(
                anchor_world[:, :3, 3].max(axis=0)
                - anchor_world[:, :3, 3].min(axis=0)
            )
            * 1000
        ),
        "anchor_cam_pos_span_mm": float(
            np.linalg.norm(
                anchor_cam[:, :3, 3].max(axis=0)
                - anchor_cam[:, :3, 3].min(axis=0)
            )
            * 1000
        ),
        "anchor_world_step_max_mm": float(anchor_world_pos_step.max() * 1000)
        if len(anchor_world_pos_step) else 0.0,
        "anchor_cam_step_max_mm": float(anchor_cam_pos_step.max() * 1000)
        if len(anchor_cam_pos_step) else 0.0,
        "_hand_cam_positions": hand_cam[:, :3, 3].tolist(),
    }


def plot_trajectories(rows: list[dict], output: Path) -> None:
    columns = 5
    plot_rows = max(1, math.ceil(len(rows) / columns))
    figure = plt.figure(figsize=(columns * 3.0, plot_rows * 2.8))
    for index, row in enumerate(rows):
        axis = figure.add_subplot(plot_rows, columns, index + 1, projection="3d")
        points = np.asarray(row["_hand_cam_positions"], dtype=np.float64)
        if len(points):
            axis.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=0.8)
            axis.scatter(points[0, 0], points[0, 1], points[0, 2], s=8, label="start")
            axis.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=8, label="end")
        axis.set_title(row["session"][-6:], fontsize=8)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.tick_params(labelsize=6)
    figure.suptitle("open_door ICT left-hand trajectories in per-frame camera reference")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sessions = sorted(
        path for path in args.data_root.iterdir()
        if path.is_dir() and (path / "preprocess" / "all_data").is_dir()
    )
    rows = [read_session(path) for path in sessions]
    for row in rows:
        row.pop("_hand_cam_positions", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "geometry_diagnostics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_rows = [read_session(path) for path in sessions]
    plot_trajectories(plot_rows, args.output_dir / "ict_trajectory_camera_frame.png")
    print(json.dumps({
        "sessions": len(rows),
        "output_json": str(args.output_dir / "geometry_diagnostics.json"),
        "trajectory_plot": str(args.output_dir / "ict_trajectory_camera_frame.png"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
