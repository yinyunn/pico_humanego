"""Generate Aria-compatible task phases from PICO kinematics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preprocess.AriaPhasesOps import AriaPhasesOps
from preprocess.AriaPhasesTypes import AriaPhases, AriaPhasesFrame
from utils.utils_io import load_cfg


MODE_NAMES = {
    0: "MANIPULATION",
    1: "NAVIGATION",
    2: "NAVIGATION",
    3: "TRANSITION",
    4: "FINISHED",
}


def _frame_dirs(root: Path) -> list[Path]:
    all_data = root / "preprocess" / "all_data"
    dirs = [p for p in all_data.iterdir() if p.is_dir() and p.name.isdigit()]
    dirs.sort(key=lambda p: int(p.name))
    if not dirs:
        raise FileNotFoundError(f"no PICO frames under {all_data}")
    return dirs


def _hand_velocity_entry(hand: dict | None) -> SimpleNamespace:
    if not hand or hand.get("midpoint_lin_vel_opt_world") is None:
        return SimpleNamespace(midpoint_lin_vel_opt_world=None)
    return SimpleNamespace(
        midpoint_lin_vel_opt_world=np.asarray(hand["midpoint_lin_vel_opt_world"], dtype=np.float64)
    )


def _cfg_dict(cfg) -> dict:
    try:
        return {str(k): v for k, v in dict(cfg).items()}
    except (TypeError, AttributeError):
        return {}


def generate_pico_phases(
    root: str | Path,
    cfg_path: str | Path | None = None,
    finished_frames: int | None = None,
) -> dict:
    """Generate per-frame and sequence-level phase JSON for a PICO recording."""

    root = Path(root)
    if cfg_path is None:
        cfg_path = Path(__file__).resolve().parents[2] / "cfg" / "preprocess" / "base" / "AriaPhases.yaml"
    cfg = load_cfg(str(cfg_path))
    frame_dirs = _frame_dirs(root)

    slam_rows = []
    hand_rows = []
    for frame_dir in frame_dirs:
        with (frame_dir / "aria_slam.json").open(encoding="utf-8") as handle:
            slam = json.load(handle)
        with (frame_dir / "aria_hands.json").open(encoding="utf-8") as handle:
            hands = json.load(handle)
        slam_rows.append(slam)
        hand_rows.append(SimpleNamespace(
            hand_l=_hand_velocity_entry(hands.get("hand_l")),
            hand_r=_hand_velocity_entry(hands.get("hand_r")),
        ))

    v = np.asarray([row.get("linear_speed_mps", 0.0) for row in slam_rows], dtype=np.float64)
    w = np.asarray([row.get("angular_speed_rps", 0.0) for row in slam_rows], dtype=np.float64)
    yaw = np.asarray([row.get("yaw_unwrapped_deg", 0.0) for row in slam_rows], dtype=np.float64)
    ts = [int(row["ts"]) for row in slam_rows]

    # Reuse the exact AriaPhases heuristic stages over PICO-derived kinematics.
    stop = AriaPhasesOps.compute_stop_mask_from_vw(
        v, w, cfg.v_stop_thresh, cfg.w_stop_thresh,
        cfg.stop_hold_frames, cfg.stop_debounce_frames,
    )
    stop = AriaPhasesOps.close_binary_mask(stop, kernel_size=cfg.stop_debounce_frames)
    stop = AriaPhasesOps.postprocess_stop_mask(
        stop, yaw, cfg.stop_min_on_frames, cfg.stop_yaw_veto_deg,
    )
    stop = AriaPhasesOps.apply_stop_offset(stop, cfg.stop_offset_frames)
    mode, _, mode_stats = AriaPhasesOps.compute_mode_from_stop_vw(
        stop, v, w, yaw, cfg.w_rot_thresh, cfg.v_rot_max, cfg,
    )
    has_hand_velocity = any(
        hand.midpoint_lin_vel_opt_world is not None
        for row in hand_rows
        for hand in (row.hand_l, row.hand_r)
    )
    if has_hand_velocity:
        mode, hand_stats = AriaPhasesOps.refine_manip_phases_with_hand_vel(
            mode,
            SimpleNamespace(hands=hand_rows),
            getattr(cfg, "manip_clean_vel_thresh", 0.15),
            getattr(cfg, "manip_clean_wait_frames", 5),
            getattr(cfg, "manip_clean_manual_offset", 15),
        )
    else:
        hand_stats = {
            "stop_frames": int(np.sum(mode == 0)),
            "forward_frames": int(np.sum(mode == 1)),
            "rotate_frames": int(np.sum(mode == 2)),
            "transition_frames": int(np.sum(mode == 3)),
            "skipped": "no valid PICO hand velocity",
        }
    mode = AriaPhasesOps.inject_finished_phase(
        mode,
        int(finished_frames if finished_frames is not None else getattr(cfg, "finished_frames", 30)),
    )

    fps = 30.0
    if len(ts) > 1:
        fps = float(1e9 / np.median(np.diff(np.asarray(ts, dtype=np.int64))))

    phases = AriaPhases(mps_path=str(root))
    for frame_dir, slam, stop_value, mode_value in zip(frame_dirs, slam_rows, stop, mode):
        mode_int = int(mode_value)
        phases.frames.append(AriaPhasesFrame(
            idx=int(frame_dir.name),
            ts=int(slam["ts"]),
            mode=mode_int,
            stop=int(stop_value),
            mode_str=MODE_NAMES.get(mode_int, "UNKNOWN"),
            v=float(slam.get("linear_speed_mps", 0.0)),
            w=float(slam.get("angular_speed_rps", 0.0)),
            yaw_u_deg=float(slam.get("yaw_unwrapped_deg", 0.0)),
        ))

    duration_s = (ts[-1] - ts[0]) * 1e-9 if len(ts) > 1 else 0.0
    mode_counts = {
        "manipulation_frames": int(np.sum(mode == 0)),
        "navigation_frames": int(np.sum((mode == 1) | (mode == 2))),
        "transition_frames": int(np.sum(mode == 3)),
        "finished_frames": int(np.sum(mode == 4)),
    }
    phases.summary = {
        "status": "PICO_ARIA_HEURISTIC",
        "sensor_source": "pico",
        "total_frames": len(frame_dirs),
        "fps_median": fps,
        "duration_s": duration_s,
        "algorithm": "AriaPhasesOps over PICO Head.pose and hand midpoint kinematics",
        "mode_encoding": MODE_NAMES,
        "stage_window_check": AriaPhasesOps.stage_window_check(mode, [4, 3, 2, 1, 0], True, False),
        "mode_stats_before_hand_refinement": mode_stats,
        "mode_stats_after_hand_refinement": hand_stats,
        "mode_counts": mode_counts,
        "hyperparams": _cfg_dict(cfg),
    }
    phases.save_aria_phases_json()
    return phases.summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Aria-compatible phases from PICO HumanEgo frames")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, default=None)
    parser.add_argument("--finished-frames", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(generate_pico_phases(args.input, args.cfg, args.finished_frames), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
