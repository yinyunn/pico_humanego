"""将已同步的 PICO 原始录制导出为 HumanEgo 标准逐帧目录。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from .recording import PicoImportSession
from .pico_phases import generate_pico_phases
from .video_reader import PicoVideoReader
from utils.utils_io import load_cfg


# PICO 26 点 -> HumanEgo/Aria 兼容的 21 点顺序。
# HumanEgo 的旧手部代码约定 0/1 为拇指/食指尖，5 为腕部，6/8 为拇指/食指 MCP。
PICO26_TO_HUMANEGO21 = (5, 10, 15, 20, 25, 1, 2, 3, 6, 7, 9, 11, 12, 14, 16, 17, 19, 21, 22, 24, 0)


def _quat_to_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((x, y, z, w))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = np.asarray((x, y, z, w), dtype=np.float64) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pose_matrix(pose):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_matrix(pose["quaternion_xyzw"])
    transform[:3, 3] = np.asarray(pose["position"], dtype=np.float64)
    return transform


def _inverse(transform):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _rpy_zyx(rotation):
    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    if sy > 1e-6:
        return [
            math.atan2(float(rotation[2, 1]), float(rotation[2, 2])),
            math.atan2(float(-rotation[2, 0]), sy),
            math.atan2(float(rotation[1, 0]), float(rotation[0, 0])),
        ]
    return [
        math.atan2(float(-rotation[1, 2]), float(rotation[1, 1])),
        math.atan2(float(-rotation[2, 0]), sy),
        0.0,
    ]


def _safe_normalize(value, fallback):
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else fallback


def _hand_pose(joints, index):
    return _pose_matrix(joints[index])


def _midpoint_pose(world_points, indices=(1, 2, 6, 5, 10)):
    # Parameterize only point indices; preserve the existing PICO frame math.
    wrist_i, thumb_base_i, index_base_i, thumb_tip_i, index_tip_i = indices
    wrist = world_points[wrist_i]
    thumb_base = world_points[thumb_base_i]
    index_base = world_points[index_base_i]
    x = _safe_normalize(index_base - thumb_base, np.array([1.0, 0.0, 0.0]))
    y_raw = (thumb_base + index_base) / 2.0 - wrist
    y_raw = y_raw - np.dot(y_raw, x) * x
    y = _safe_normalize(y_raw, np.array([0.0, 1.0, 0.0]))
    z = _safe_normalize(np.cross(x, y), np.array([0.0, 0.0, 1.0]))
    y = _safe_normalize(np.cross(z, x), np.array([0.0, 1.0, 0.0]))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x, y, z))
    transform[:3, 3] = (world_points[thumb_tip_i] + world_points[index_tip_i]) / 2.0
    return transform


def _project(points_camera, K):
    points = np.asarray(points_camera, dtype=np.float64)
    homogeneous = (np.asarray(K, dtype=np.float64) @ points.T).T
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return pixels, valid


def _pack_hand(raw_hand, T_c2w, K, width, height, midpoint_velocity=None, wrist_velocity=None):
    if raw_hand is None or not raw_hand.get("is_active"):
        return None
    joints = raw_hand["joints"]
    points_world = np.asarray([joint["position"] for joint in joints], dtype=np.float64)
    if points_world.shape != (26, 3):
        return None
    T_w2c = _inverse(T_c2w)
    points_camera = (T_w2c[:3, :3] @ points_world.T).T + T_w2c[:3, 3]
    points21_world = points_world[list(PICO26_TO_HUMANEGO21)]
    points21_camera = points_camera[list(PICO26_TO_HUMANEGO21)]
    pixels, valid = _project(points21_camera, K)

    wrist_world = _hand_pose(joints, 1)
    palm_world = _hand_pose(joints, 0)
    midpoint_world = _midpoint_pose(points_world)
    thumb_index_distance = float(np.linalg.norm(points_world[5] - points_world[10]))
    grasp = 1 if thumb_index_distance < 0.035 else 0

    def camera_pose(world_pose):
        return T_w2c @ world_pose

    confidence = 1.0 if all((int(joint.get("status", 0)) & 8) != 0 for joint in joints) else 0.0
    midpoint_velocity = np.zeros(3, dtype=np.float64) if midpoint_velocity is None else np.asarray(midpoint_velocity, dtype=np.float64)
    wrist_velocity = np.zeros(3, dtype=np.float64) if wrist_velocity is None else np.asarray(wrist_velocity, dtype=np.float64)
    return {
        "d2c": T_w2c.tolist(),
        "c2w": T_c2w.tolist(),
        "confidence": confidence,
        "grasp_state": grasp,
        "wrist_pose": camera_pose(wrist_world).tolist(),
        "palm_pose": camera_pose(palm_world).tolist(),
        "kpts_3d": points21_camera.tolist(),
        "kpts_2d": pixels.tolist(),
        "kpts_2d_valid": valid.tolist(),
        "joint_angles": {},
        "wrist_pose_raw_world": wrist_world.tolist(),
        "wrist_pose_opt_world": wrist_world.tolist(),
        "wrist_lin_vel_raw_world": wrist_velocity.tolist(),
        "wrist_ang_vel_raw_world": [0.0, 0.0, 0.0],
        "wrist_lin_vel_opt_world": wrist_velocity.tolist(),
        "wrist_ang_vel_opt_world": [0.0, 0.0, 0.0],
        "index_translation_raw_world": points_world[10].tolist(),
        "index_translation_opt_world": points_world[10].tolist(),
        "thumb_translation_raw_world": points_world[5].tolist(),
        "thumb_translation_opt_world": points_world[5].tolist(),
        "midpoint_pose_raw_world": midpoint_world.tolist(),
        "midpoint_pose_opt_world": midpoint_world.tolist(),
        "midpoint_translation_raw_world": midpoint_world[:3, 3].tolist(),
        "midpoint_orientation_raw_world": midpoint_world[:3, :3].tolist(),
        "midpoint_translation_opt_world": midpoint_world[:3, 3].tolist(),
        "midpoint_orientation_opt_world": midpoint_world[:3, :3].tolist(),
        "midpoint_lin_vel_raw_world": midpoint_velocity.tolist(),
        "midpoint_ang_vel_raw_world": [0.0, 0.0, 0.0],
        "midpoint_lin_vel_opt_world": midpoint_velocity.tolist(),
        "midpoint_ang_vel_opt_world": [0.0, 0.0, 0.0],
        "distance_midpoint2wrist_raw_world": float(np.linalg.norm(midpoint_world[:3, 3] - wrist_world[:3, 3])),
        "distance_midpoint2wrist_opt_world": float(np.linalg.norm(midpoint_world[:3, 3] - wrist_world[:3, 3])),
        "pico_joints_26": joints,
        "hand_frame": "app_tracking_world",
        "image_width": int(width),
        "image_height": int(height),
    }


def _nullable_points(points):
    return [[float(value) for value in point] if np.isfinite(point).all() else [None] * len(point)
            for point in np.asarray(points)]


def _nullable_vector(vector):
    vector = np.asarray(vector)
    return vector.tolist() if np.isfinite(vector).all() else None


def _smooth_rotations(rotations, valid, alpha):
    """Symmetric matrix EMA followed by nearest-SO(3) projection."""
    rotations = np.asarray(rotations, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    output = np.full_like(rotations, np.nan)
    index = 0
    while index < len(rotations):
        while index < len(rotations) and not valid[index]:
            index += 1
        start = index
        while index < len(rotations) and valid[index]:
            index += 1
        end = index
        if end <= start:
            continue
        segment = rotations[start:end]
        forward, backward = segment.copy(), segment.copy()
        for i in range(1, len(segment)):
            forward[i] = alpha * segment[i] + (1 - alpha) * forward[i - 1]
        for i in range(len(segment) - 2, -1, -1):
            backward[i] = alpha * segment[i] + (1 - alpha) * backward[i + 1]
        for i, matrix in enumerate(0.5 * (forward + backward), start):
            u, _, vt = np.linalg.svd(matrix)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            output[i] = rotation
    return output


def _grasp_hysteresis(ratios, close_ratio, open_ratio, close_hold, open_hold, missing_hold):
    if close_ratio >= open_ratio:
        raise ValueError("grasp_close_ratio must be smaller than grasp_open_ratio")
    states = np.zeros(len(ratios), dtype=np.uint8)
    state = 0
    close_count = open_count = missing_count = 0
    for i, ratio in enumerate(ratios):
        if not np.isfinite(ratio):
            missing_count += 1
            if missing_count > missing_hold:
                close_count = open_count = 0
            states[i] = state
            continue
        missing_count = 0
        if state == 0:
            close_count = close_count + 1 if ratio <= close_ratio else 0
            if close_count >= close_hold:
                state, close_count = 1, 0
        else:
            open_count = open_count + 1 if ratio >= open_ratio else 0
            if open_count >= open_hold:
                state, open_count = 0, 0
        states[i] = state
    return states


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class HumanEgoWriter:
    """把一个 ``PicoImportSession`` materialize 成 HumanEgo 基础阶段目录。"""

    def __init__(self, session: PicoImportSession, output_root: str | Path):
        self.session = session
        self.root = Path(output_root)
        self.all_data = self.root / "preprocess" / "all_data"

    def export_stereo_2d_smoke(self, max_delta_ms=50.0, max_frames=None):
        """Review both-eye MediaPipe pixels before replacing native hand labels.

        This intentionally writes no aria_hands/phase/training files. Geometry
        and hand-source switching are a later milestone after visual review.
        """
        from scripts.visualize_mediapipe_mp4 import main as visualize_stereo

        output = self.root / "stereo_2d"
        output.mkdir(parents=True, exist_ok=False)
        self.session.write_intermediate(output, max_delta_ms=max_delta_ms)
        argv = ["--input", str(self.session.video_path),
                "--pts", str(self.session.pts_path),
                "--sync-manifest", str(output / "sync_manifest.jsonl"),
                "--output", str(output / "mediapipe_stereo_vis.mp4"),
                "--json", str(output / "mediapipe_stereo_2d.jsonl")]
        if max_frames is not None:
            argv.extend(["--max-frames", str(max_frames)])
        return visualize_stereo(argv)

    def export_stereo_3d_diagnostic(self, stereo_json, max_delta_ms=50.0):
        """Audit real stereo geometry before changing the final hand source.

        Use only the confirmed E(raw camera -> head) and D(raw -> image) meanings.
        A mismatch is reported instead of fitting or inverting the calibration.
        """
        from .stereo_geometry import camera_to_head_pair, match_hands, epipolar_errors, stereo_matrices
        from .visuals import render_stereo_3d_diagnostic

        cache = Path(stereo_json)
        provenance = json.loads(cache.with_suffix(".summary.json").read_text())
        if Path(provenance["input"]).resolve() != self.session.video_path.resolve():
            raise ValueError("Stereo 2D cache belongs to another recording")
        if provenance.get("timestamp_source") != "decoded_pts":
            raise ValueError("Stereo 3D requires decoded PTS, not nominal FPS")
        rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
        matches = self.session.synchronize(max_delta_ms=max_delta_ms)
        if len(rows) != len(matches):
            raise ValueError("Stereo 2D cache must cover the entire recording")
        for row, match in zip(rows, matches):
            if row["frame_idx"] != match["frame_index"] or abs(row["pts_seconds"] - match["pts_seconds"]) > 1e-8:
                raise ValueError("Stereo cache and source video PTS mismatch")
            if any(row["eyes"][eye]["width"] != 1080 for eye in ("left", "right")):
                raise ValueError("Stereo cache uses an incompatible crop")

        output = self.root / "stereo_3d_diagnostic"
        output.mkdir(parents=True, exist_ok=False)
        cfg = {"reprojection_px": 4.0, "epipolar_px": 6.0,
               "min_depth_m": 0.10, "max_depth_m": 2.0, "min_ray_angle_deg": 0.5}
        calibration = self.session.calibration
        left, right = camera_to_head_pair(calibration)
        confirmed_relative = np.linalg.inv(right) @ left
        rectified_relative = np.eye(4)
        rectified_relative[0, 3] = -float(np.linalg.norm(confirmed_relative[:3, 3]))
        variants = {
            "confirmed_K_D_invE": {
                "left": left, "right": right, "relative": confirmed_relative,
                "triangulation": "explicit_K_D_invE",
            },
            "render_mode_3d_rectified_candidate": {
                "left": left,
                "right": left @ np.linalg.inv(rectified_relative),
                "relative": rectified_relative,
                "triangulation": "rectified_image_camera_candidate",
            },
        }

        def serial(value):
            if isinstance(value, np.ndarray):
                return serial(value.tolist())
            if isinstance(value, dict):
                return {k: serial(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [serial(v) for v in value]
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, (np.bool_,)):
                return bool(value)
            return value

        results = []
        for row, sync in zip(rows, matches):
            tracking = self.session.tracking[sync["tracking_index"]]
            H = _pose_matrix(tracking["head"])
            frame = {"frame_idx": row["frame_idx"], "pts_seconds": row["pts_seconds"], "sync": sync,
                     "is_final_hand_label": False, "variants": {}}
            for name, matrices in variants.items():
                geometry = ({
                    "E_left_raw_to_head": calibration.E_left,
                    "E_right_raw_to_head": calibration.E_right,
                    "raw_to_image": calibration.D,
                } if matrices["triangulation"] == "explicit_K_D_invE" else None)
                accepted, candidates = match_hands(
                    row["eyes"]["left"]["hands"], row["eyes"]["right"]["hands"],
                    calibration.K, matrices["relative"], camera_geometry=geometry, **cfg,
                )
                for c in candidates:
                    c["accepted"] = bool(c["accepted"] and sync["within_tolerance"])
                    if not sync["within_tolerance"]:
                        c["rejection"] = "tracking_sync_outlier"
                    if "xyz_head" in c["triangulation"]:
                        xyz_head = c["triangulation"]["xyz_head"]
                    else:
                        xyz_left = c["triangulation"]["xyz_left_image_camera_relative"]
                        xyz_head = xyz_left @ matrices["left"][:3, :3].T + matrices["left"][:3, 3]
                    world = xyz_head @ H[:3, :3].T + H[:3, 3]
                    c["xyz_world_diagnostic"] = world
                    c["T_hand_to_world_diagnostic"] = (_midpoint_pose(world, (0, 2, 5, 4, 8))
                                                         if c["accepted"] else None)
                    c["opening_ratio_diagnostic"] = (float(np.linalg.norm(world[4] - world[8]) / np.linalg.norm(world[5] - world[17]))
                                                       if c["accepted"] else None)
                frame["variants"][name] = candidates
            results.append(frame)

        # Static scene features test the supplied geometry independently of hand
        # tracking. Do not fit F/R/t to the hand data or replace the calibration.
        orb = cv2.ORB_create(nfeatures=2500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        scene = []
        from .video_reader import crop_side_by_side
        cap = cv2.VideoCapture(str(self.session.video_path))
        try:
            for index in np.linspace(0, len(rows) - 1, 8, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(rows[index]["frame_idx"]))
                ok, image = cap.read()
                if not ok:
                    raise ValueError("Cannot read stereo diagnostic sample")
                eye_images = [crop_side_by_side(image, eye) for eye in ("left", "right")]
                (kl, dl), (kr, dr) = [orb.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in eye_images]
                if dl is None or dr is None:
                    continue
                forward = [m[0] for m in matcher.knnMatch(dl, dr, k=2) if len(m) == 2 and m[0].distance < .7 * m[1].distance]
                backward = {m[0].queryIdx: m[0].trainIdx for m in matcher.knnMatch(dr, dl, k=2)
                            if len(m) == 2 and m[0].distance < .7 * m[1].distance}
                good = [m for m in forward if backward.get(m.trainIdx) == m.queryIdx]
                if not good:
                    continue
                a = np.array([kl[m.queryIdx].pt for m in good]); b = np.array([kr[m.trainIdx].pt for m in good])
                item = {"frame_idx": rows[index]["frame_idx"], "matches": len(good),
                        "median_right_minus_left_uv": np.median(b - a, axis=0)}
                for name, matrices in variants.items():
                    _, _, F = stereo_matrices(calibration.K, calibration.K, matrices["relative"])
                    errors = epipolar_errors(a, b, F)
                    item[name] = {"median_epipolar_px": np.median(errors), "p90_epipolar_px": np.percentile(errors, 90)}
                scene.append(item)
        finally:
            cap.release()

        summary = {"status": "BLOCKED_CALIBRATION_REVIEW", "source_video": str(self.session.video_path.resolve()),
                   "frames": len(rows), "is_final_hand_label": False,
                   "geometry_source": calibration.as_dict(), "thresholds": cfg,
                   "variants": {}, "scene_feature_check": scene,
                   "distortion_state": "not established by recording metadata",
                   "capture_render_mode": "PXRCapture_RenderMode_3D (from XRoboToolkit UICameraCtrl.cs)",
                   "reason": "Confirmed raw-camera geometry does not explain the RenderMode_3D pixel correspondences; final hand labels remain unchanged"}
        for name, matrices in variants.items():
            candidates = [c for frame in results for c in frame["variants"][name]]
            errors = np.array([c["triangulation"]["epipolar_error_px"] for c in candidates])
            valid = np.array([c["triangulation"]["valid"] for c in candidates])
            summary["variants"][name] = {"left_to_right": matrices["relative"],
                "triangulation": matrices["triangulation"],
                "baseline_m": np.linalg.norm(matrices["relative"][:3, 3]),
                "candidate_count": len(candidates), "accepted_hand_observations": sum(c["accepted"] for c in candidates),
                "joint_valid_ratio_of_candidates": float(np.mean(valid)) if valid.size else None,
                "median_epipolar_px": float(np.median(errors)) if errors.size else None}
            disparities = np.array([c["right_uv"] - c["left_uv"] for c in candidates])
            summary["variants"][name]["median_observed_right_minus_left_uv_px"] = (
                np.median(disparities.reshape(-1, 2), axis=0) if disparities.size else None
            )
            for eye in ("left", "right"):
                e = np.array([c["triangulation"][eye + "_reprojection_error_px"] for c in candidates])
                finite = e[np.isfinite(e)]
                summary["variants"][name][eye + "_reprojection"] = {
                    "mean_px": np.mean(finite) if finite.size else None,
                    "median_px": np.median(finite) if finite.size else None,
                    "per_joint_mean_px": [float(np.mean(col[np.isfinite(col)])) if np.isfinite(col).any() else None
                                          for col in e.T] if e.size else []}
        with (output / "stereo_3d.jsonl").open("w") as handle:
            for frame in results:
                handle.write(json.dumps(serial(frame), allow_nan=False) + "\n")
        (output / "summary.json").write_text(json.dumps(serial(summary), indent=2, allow_nan=False) + "\n")
        render_stereo_3d_diagnostic(self.session, rows, results, variants, output)
        return serial(summary)

    def write_stereo_hand_labels(self, stereo_json, cfg_path):
        """Replace this fresh smoke export's hand JSONs with filtered stereo hands."""
        from preprocess.MediaPipeHands import MP_TO_ARIA, remap_mp_to_aria
        from .stereo_geometry import (
            interpolate_short_gaps, match_hands, smooth_valid_points,
        )
        from .visuals import render_stereo_hand_comparison

        cfg = load_cfg(str(cfg_path))
        cache = Path(stereo_json)
        provenance = json.loads(cache.with_suffix(".summary.json").read_text())
        if Path(provenance["input"]).resolve() != self.session.video_path.resolve():
            raise ValueError("Stereo 2D cache belongs to another recording")
        rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
        by_frame = {int(row["frame_idx"]): row for row in rows}
        matches = [row for row in self.session.synchronize() if row["within_tolerance"]]
        if any(row["frame_index"] not in by_frame for row in matches):
            raise ValueError("Stereo cache does not cover all exported frames")

        calibration = self.session.calibration
        image_to_raw = np.eye(4)
        image_to_raw[:3, :3] = calibration.D.T
        C_left = calibration.E_left @ image_to_raw
        baseline = float(np.linalg.norm(calibration.E_left[:3, 3] - calibration.E_right[:3, 3]))
        T_left_to_right = np.eye(4)
        T_left_to_right[0, 3] = -baseline
        C_right = C_left @ np.linalg.inv(T_left_to_right)
        camera_to_head = {"left": C_left, "right": C_right}
        thresholds = {
            "reprojection_px": float(cfg.reprojection_error_px),
            "epipolar_px": float(cfg.epipolar_error_px),
            "min_depth_m": float(cfg.min_depth_m),
            "max_depth_m": float(cfg.max_depth_m),
            "min_ray_angle_deg": float(cfg.min_ray_angle_deg),
        }
        count = len(matches)
        observations = {side: [None] * count for side in ("left", "right")}
        raw_world = {side: np.full((count, 21, 3), np.nan) for side in observations}
        raw_valid = {side: np.zeros((count, 21), dtype=bool) for side in observations}

        for output_index, match in enumerate(matches):
            detection = by_frame[match["frame_index"]]
            accepted, _ = match_hands(
                detection["eyes"]["left"]["hands"], detection["eyes"]["right"]["hands"],
                calibration.K, T_left_to_right,
                confidence_min=float(cfg.handedness_confidence_min),
                ambiguity_margin_px=float(cfg.correspondence_ambiguity_margin_px),
                require_full_hand=False,
                min_valid_ratio=float(cfg.min_valid_joint_ratio),
                **thresholds,
            )
            tracking = self.session.tracking[match["tracking_index"]]
            H = _pose_matrix(tracking["head"])
            for candidate in accepted:
                side = candidate["side"].lower()
                tri = candidate["triangulation"]
                xyz_left = tri["xyz_left_image_camera_relative"]
                xyz_head = xyz_left @ C_left[:3, :3].T + C_left[:3, 3]
                xyz_world = xyz_head @ H[:3, :3].T + H[:3, 3]
                valid = np.asarray(tri["valid"], dtype=bool)
                raw_world[side][output_index, valid] = xyz_world[valid]
                raw_valid[side][output_index] = valid
                observations[side][output_index] = {
                    "handedness_score": float(candidate["handedness_score"]),
                    "left_uv": candidate["left_uv"], "right_uv": candidate["right_uv"],
                    "left_reprojection_error_px": tri["left_reprojection_error_px"],
                    "right_reprojection_error_px": tri["right_reprojection_error_px"],
                    "epipolar_error_px": tri["epipolar_error_px"],
                    "ray_angle_deg": tri["ray_angle_deg"],
                }

        packed = [{"hand_l": None, "hand_r": None} for _ in range(count)]
        side_stats = {}
        timestamps = np.asarray([row["video_timestamp_ns"] for row in matches], dtype=np.int64)
        for side in ("left", "right"):
            filled, filled_valid = interpolate_short_gaps(
                raw_world[side], raw_valid[side], int(cfg.interpolation_max_gap_frames)
            )
            filtered = smooth_valid_points(filled, filled_valid, float(cfg.position_ema_alpha))
            core_valid = filled_valid[:, [0, 2, 4, 5, 8, 17]].all(axis=1)
            raw_core_valid = raw_valid[side][:, [0, 2, 4, 5, 8, 17]].all(axis=1)
            raw_poses = np.full((count, 4, 4), np.nan)
            filtered_poses = np.full((count, 4, 4), np.nan)
            rotations = np.full((count, 3, 3), np.nan)
            for index in np.flatnonzero(core_valid):
                pose = _midpoint_pose(filtered[index], (0, 2, 5, 4, 8))
                filtered_poses[index] = pose
                rotations[index] = pose[:3, :3]
            for index in np.flatnonzero(raw_core_valid):
                raw_poses[index] = _midpoint_pose(raw_world[side][index], (0, 2, 5, 4, 8))
            stable_rotations = _smooth_rotations(
                rotations, core_valid, float(cfg.orientation_ema_alpha)
            )
            filtered_poses[core_valid, :3, :3] = stable_rotations[core_valid]
            ratios = np.full(count, np.nan)
            for index in np.flatnonzero(core_valid):
                ratios[index] = (np.linalg.norm(filtered[index, 4] - filtered[index, 8]) /
                                 np.linalg.norm(filtered[index, 5] - filtered[index, 17]))
            grasps = _grasp_hysteresis(
                ratios, float(cfg.grasp_close_ratio), float(cfg.grasp_open_ratio),
                int(cfg.grasp_hold_frames), int(cfg.release_hold_frames),
                int(cfg.missing_state_hold_frames),
            )
            midpoint_velocity = np.zeros((count, 3))
            wrist_velocity = np.zeros((count, 3))
            for index in range(1, count):
                if core_valid[index] and core_valid[index - 1]:
                    dt = max((timestamps[index] - timestamps[index - 1]) / 1e9, 1e-6)
                    midpoint_velocity[index] = ((filtered_poses[index, :3, 3] - filtered_poses[index - 1, :3, 3]) / dt)
                    wrist_velocity[index] = ((filtered[index, 0] - filtered[index - 1, 0]) / dt)

            for index in np.flatnonzero(core_valid):
                match = matches[index]
                tracking = self.session.tracking[match["tracking_index"]]
                H = _pose_matrix(tracking["head"])
                c2w = H @ C_left
                w2c = _inverse(c2w)
                pose = filtered_poses[index]
                wrist_world = pose.copy(); wrist_world[:3, 3] = filtered[index, 0]
                raw_wrist_world = None
                if raw_pose is not None:
                    raw_wrist_world = raw_pose.copy()
                    raw_wrist_world[:3, 3] = raw_world[side][index, 0]
                palm_world = pose.copy()
                palm_ids = [joint for joint in (0, 5, 9) if filled_valid[side][index, joint]]
                palm_world[:3, 3] = np.mean(filtered[index, palm_ids], axis=0)
                camera_points = filtered[index] @ w2c[:3, :3].T + w2c[:3, 3]
                kpts_camera_aria = remap_mp_to_aria(camera_points)
                reprojections = {}
                for eye, C_eye in camera_to_head.items():
                    eye_c2w = H @ C_eye
                    eye_w2c = _inverse(eye_c2w)
                    eye_points = filtered[index] @ eye_w2c[:3, :3].T + eye_w2c[:3, 3]
                    reprojections[eye], _ = _project(eye_points, calibration.K)
                observation = observations[side][index]
                score = 0.5 if observation is None else observation["handedness_score"]
                valid_ratio = float(np.mean(filled_valid[side][index]))
                raw_pose = raw_poses[index] if raw_core_valid[index] else None
                aria_valid = [bool(filled_valid[side][index, MP_TO_ARIA[i]]) for i in range(20)]
                aria_valid.append(bool(filled_valid[side][index, [0, 5, 9]].all()))
                kpts_2d_aria = remap_mp_to_aria(
                    np.column_stack([reprojections["left"], np.zeros(21)])
                )[:, :2]
                hand = {
                    "d2c": w2c.tolist(), "c2w": c2w.tolist(),
                    "confidence": float(max(0.5, score)),
                    "grasp_state": int(grasps[index]),
                    "gripper_opening_ratio": float(ratios[index]),
                    "wrist_pose": (w2c @ wrist_world).tolist(),
                    "palm_pose": (w2c @ palm_world).tolist(),
                    "kpts_3d": kpts_camera_aria.tolist() if filled_valid[side][index].all() else None,
                    "kpts_2d": _nullable_points(kpts_2d_aria),
                    "kpts_2d_valid": aria_valid,
                    "joint_angles": {},
                    "wrist_pose_raw_world": None if raw_wrist_world is None else raw_wrist_world.tolist(),
                    "wrist_pose_opt_world": wrist_world.tolist(),
                    "wrist_lin_vel_raw_world": wrist_velocity[index].tolist(),
                    "wrist_ang_vel_raw_world": [0.0, 0.0, 0.0],
                    "wrist_lin_vel_opt_world": wrist_velocity[index].tolist(),
                    "wrist_ang_vel_opt_world": [0.0, 0.0, 0.0],
                    "index_translation_raw_world": _nullable_vector(raw_world[side][index, 8]),
                    "index_translation_opt_world": filtered[index, 8].tolist(),
                    "thumb_translation_raw_world": _nullable_vector(raw_world[side][index, 4]),
                    "thumb_translation_opt_world": filtered[index, 4].tolist(),
                    "midpoint_pose_raw_world": None if raw_pose is None else raw_pose.tolist(),
                    "midpoint_pose_opt_world": pose.tolist(),
                    "midpoint_translation_raw_world": None if raw_pose is None else raw_pose[:3, 3].tolist(),
                    "midpoint_orientation_raw_world": None if raw_pose is None else raw_pose[:3, :3].tolist(),
                    "midpoint_translation_opt_world": pose[:3, 3].tolist(),
                    "midpoint_orientation_opt_world": pose[:3, :3].tolist(),
                    "midpoint_lin_vel_raw_world": midpoint_velocity[index].tolist(),
                    "midpoint_ang_vel_raw_world": [0.0, 0.0, 0.0],
                    "midpoint_lin_vel_opt_world": midpoint_velocity[index].tolist(),
                    "midpoint_ang_vel_opt_world": [0.0, 0.0, 0.0],
                    "distance_midpoint2wrist_raw_world": None if raw_pose is None else float(np.linalg.norm(raw_pose[:3, 3] - raw_world[side][index, 0])),
                    "distance_midpoint2wrist_opt_world": float(np.linalg.norm(pose[:3, 3] - filtered[index, 0])),
                    "hand_source": "mediapipe_stereo",
                    "stereo_geometry": "render_mode_3d_rectified_candidate",
                    "landmarks_world_raw_mp_order": _nullable_points(raw_world[side][index]),
                    "landmarks_world_filtered_mp_order": _nullable_points(filtered[index]),
                    "landmarks_valid_raw_mp_order": raw_valid[side][index].tolist(),
                    "landmarks_valid_final_mp_order": filled_valid[side][index].tolist(),
                    "valid_joint_ratio": valid_ratio,
                    "interpolated_frame": observation is None,
                    "filtered_reprojection_left": reprojections["left"].tolist(),
                    "filtered_reprojection_right": reprojections["right"].tolist(),
                    "stereo_observation": None if observation is None else {
                        key: (value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in observation.items()
                    },
                    "native_pico_debug_file": "aria_hands_pico_native.json",
                }
                packed[index]["hand_l" if side == "left" else "hand_r"] = hand
            side_stats[side] = {
                "raw_frame_count": int(sum(obs is not None for obs in observations[side])),
                "final_frame_count": int(np.sum(core_valid)),
                "interpolated_final_frame_count": int(sum(core_valid[i] and observations[side][i] is None for i in range(count))),
                "grasp_transition_count": int(np.sum(np.diff(grasps.astype(int)) != 0)),
                "opening_ratio_percentiles": np.nanpercentile(ratios, [5, 25, 50, 75, 95]).tolist() if np.isfinite(ratios).any() else None,
            }

        for index, (frame, match) in enumerate(zip(packed, matches)):
            frame_dir = self.all_data / f"{index:05d}"
            native_path = frame_dir / "aria_hands.json"
            shutil.copy2(native_path, frame_dir / "aria_hands_pico_native.json")
            native = json.loads(native_path.read_text())
            _write_json(native_path, {
                "idx": index, "source_frame_index": int(match["frame_index"]),
                "ts": int(match["video_timestamp_ns"]),
                "hand_r": frame["hand_r"], "hand_l": frame["hand_l"],
                "hand_source": "mediapipe_stereo",
                "stereo_geometry": "render_mode_3d_rectified_candidate",
                "native_pico_debug_file": "aria_hands_pico_native.json",
                "native_joint_mapping": native.get("joint_mapping"),
            })
        summary = {
            "status": "NEEDS_FILTERED_HAND_VISUAL_REVIEW",
            "hand_source": "mediapipe_stereo",
            "stereo_geometry": "render_mode_3d_rectified_candidate",
            "frame_count": count, "baseline_m": baseline,
            "config": str(Path(cfg_path).resolve()), "sides": side_stats,
            "phase_generated": False,
        }
        preprocess = self.root / "preprocess"
        _write_json(preprocess / "mediapipe_stereo_hands_summary.json", summary)
        render_stereo_hand_comparison(self.session, matches, self.root, camera_to_head,
                                      preprocess / "stereo_hand_validation")
        return summary

    def export(self, max_delta_ms: float = 50.0, keep_sync_outliers: bool = False,
               generate_phases: bool = True) -> dict:
        matches = self.session.synchronize(max_delta_ms=max_delta_ms)
        selected = [row for row in matches if keep_sync_outliers or row["within_tolerance"]]
        if not selected:
            raise ValueError("no video frames remain after synchronization filtering")
        self.all_data.mkdir(parents=True, exist_ok=True)
        fps = 1e9 / np.median(np.diff([row["video_timestamp_ns"] for row in selected])) if len(selected) > 1 else 30.0
        previous_head = None
        previous_ts = None
        first_head_translation = None
        slam_rows = []
        # Precompute PICO hand midpoint/wrist velocities for AriaPhases-compatible
        # manipulation-boundary refinement.  The old basic importer populated
        # these fields with zeros, which made hand-based phase cleaning inert.
        hand_velocities = {"left": [], "right": []}
        previous_hand_midpoint = {"left": None, "right": None}
        previous_hand_wrist = {"left": None, "right": None}
        previous_hand_ts = None
        for match in selected:
            tracking = self.session.tracking[match["tracking_index"]]
            current_ts = int(match["video_timestamp_ns"])
            dt = None if previous_hand_ts is None else max((current_ts - previous_hand_ts) / 1e9, 1e-6)
            for side in ("left", "right"):
                raw_hand = tracking["hands"].get(side)
                midpoint = None
                wrist = None
                if raw_hand is not None and raw_hand.get("is_active"):
                    points = np.asarray([joint["position"] for joint in raw_hand["joints"]], dtype=np.float64)
                    if points.shape == (26, 3):
                        midpoint = (points[5] + points[10]) / 2.0
                        wrist = points[1]
                if dt is not None and midpoint is not None and previous_hand_midpoint[side] is not None:
                    midpoint_velocity = (midpoint - previous_hand_midpoint[side]) / dt
                else:
                    midpoint_velocity = np.zeros(3, dtype=np.float64)
                if dt is not None and wrist is not None and previous_hand_wrist[side] is not None:
                    wrist_velocity = (wrist - previous_hand_wrist[side]) / dt
                else:
                    wrist_velocity = np.zeros(3, dtype=np.float64)
                hand_velocities[side].append((midpoint_velocity, wrist_velocity))
                previous_hand_midpoint[side] = midpoint
                previous_hand_wrist[side] = wrist
            previous_hand_ts = current_ts
        calibration = self.session.calibration
        eye = calibration.eye
        camera_profile = calibration.camera_profile
        with PicoVideoReader(self.session.video_path, eye_width=1080, eye=eye) as video:
            for output_index, match in enumerate(selected):
                tracking = self.session.tracking[match["tracking_index"]]
                head = tracking["head"]
                H = _pose_matrix(head)
                if first_head_translation is None:
                    first_head_translation = H[:3, 3].copy()
                raw_c2w = H @ calibration.E_eye
                image_to_raw = np.eye(4, dtype=np.float64)
                image_to_raw[:3, :3] = calibration.D.T
                T_c2w = raw_c2w @ image_to_raw
                frame_dir = self.all_data / f"{output_index:05d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                image = video.read(match["frame_index"])
                image_path = frame_dir / "rgb.png"
                if not cv2.imwrite(str(image_path), image):
                    raise IOError(f"failed to write {image_path}")
                ts = int(match["video_timestamp_ns"])
                camera = {
                    "idx": output_index,
                    "source_frame_index": int(match["frame_index"]),
                    "ts": ts,
                    "w": int(image.shape[1]),
                    "h": int(image.shape[0]),
                    "fps": float(fps),
                    "k": calibration.K.tolist(),
                    "d": [],
                    "c2w": T_c2w.tolist(),
                    "c2d": (calibration.E_eye @ image_to_raw).tolist(),
                    "d2w": H.tolist(),
                    "geometry_profile": camera_profile,
                    "camera_axis_to_image": calibration.D.tolist(),
                    "rgb_path": str(image_path.relative_to(self.root)),
                    "sync": match,
                }
                _write_json(frame_dir / "aria_cam_rgb.json", camera)

                hands = tracking["hands"]
                _write_json(frame_dir / "aria_hands.json", {
                    "idx": output_index,
                    "source_frame_index": int(match["frame_index"]),
                    "ts": ts,
                    "hand_r": _pack_hand(hands.get("right"), T_c2w, calibration.K, image.shape[1], image.shape[0], *hand_velocities["right"][output_index]),
                    "hand_l": _pack_hand(hands.get("left"), T_c2w, calibration.K, image.shape[1], image.shape[0], *hand_velocities["left"][output_index]),
                    "joint_mapping": list(PICO26_TO_HUMANEGO21),
                })

                linear_speed = 0.0
                angular_speed = 0.0
                if previous_head is not None and previous_ts is not None:
                    dt = max((ts - previous_ts) / 1e9, 1e-6)
                    linear_speed = float(np.linalg.norm(H[:3, 3] - previous_head[:3, 3]) / dt)
                    relative = previous_head[:3, :3].T @ H[:3, :3]
                    angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)))
                    angular_speed = float(angle / dt)
                slam_data = {
                    "idx": output_index,
                    "source_frame_index": int(match["frame_index"]),
                    "ts": ts,
                    "t_world": H[:3, 3].tolist(),
                    "rpy_deg": (np.asarray(_rpy_zyx(H[:3, :3])) * 180.0 / math.pi).tolist(),
                    "delta_t_world": (H[:3, 3] - first_head_translation).tolist(),
                    "delta_rpy_deg": [0.0, 0.0, 0.0],
                    "linear_speed_mps": linear_speed,
                    "angular_speed_rps": angular_speed,
                    "yaw_unwrapped_deg": float(_rpy_zyx(H[:3, :3])[2] * 180.0 / math.pi),
                }
                _write_json(frame_dir / "aria_slam.json", slam_data)
                slam_rows.append(slam_data)
                previous_head, previous_ts = H, ts

        _write_json(self.root / "preprocess" / "aria_cam_rgb_config.json", {
            "total_frames": len(selected),
            "fps": float(fps),
            "first_ts": int(selected[0]["video_timestamp_ns"]),
            "h": 810,
            "w": 1080,
            "k": calibration.K.tolist(),
            "d": [],
            "c2d": (calibration.E_eye @ np.block([
                [calibration.D.T, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ])).tolist(),
        })

        # Reuse HumanEgo's AriaPhases heuristics over the PICO-derived Head/Hand
        # kinematics instead of labeling the whole recording as MANIPULATION.
        phase_summary = (generate_pico_phases(self.root) if generate_phases else
                         {"status": "NOT_RUN_HAND_VALIDATION_GATE"})
        _write_json(self.root / "preprocess" / "pico_import_meta.json", {
            "sensor_source": "pico",
            "source_video": str(self.session.video_path),
            "source_tracking": str(self.session.tracking_path),
            "source_pts": str(self.session.pts_path),
            "hand_frame": "app_tracking_world",
            "camera_profile": camera_profile,
            "camera_formula": f"image_camera = D @ inv(H @ E_{eye}) @ hand_point",
        })

        sync_manifest = []
        selected_ids = {id(row): index for index, row in enumerate(selected)}
        for row in matches:
            sync_manifest.append({**row, "humanego_frame_index": selected_ids.get(id(row))})
        (self.root / "sync_manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sync_manifest), encoding="utf-8"
        )
        summary = {
            "status": "PASS" if len(selected) == len(matches) else "PASS_WITH_DROPPED_SYNC_OUTLIERS",
            "eye": eye,
            "humanego_root": str(self.root),
            "humanego_frame_count": len(selected),
            "source_video_frame_count": len(matches),
            "dropped_sync_outlier_count": len(matches) - len(selected),
            "geometry_profile": camera_profile,
            "phase_status": phase_summary.get("status"),
            "phase_results": "preprocess/aria_phases_results.json",
            "files_per_frame": (["rgb.png", "aria_cam_rgb.json", "aria_hands.json", "aria_slam.json"]
                                + (["aria_phases.json"] if generate_phases else [])),
        }
        _write_json(self.root / "humanego_import_summary.json", summary)
        return summary


def export_humanego(video_path, tracking_path, pts_path, output_root, max_delta_ms=50.0, keep_sync_outliers=False, eye="left"):
    session = PicoImportSession.open(video_path, tracking_path, pts_path, eye=eye)
    return HumanEgoWriter(session, output_root).export(max_delta_ms, keep_sync_outliers)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export synchronized PICO recording to HumanEgo standard frame directories")
    parser.add_argument("video", type=Path)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("pts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-ms", type=float, default=50.0)
    parser.add_argument("--keep-sync-outliers", action="store_true", help="Keep frames outside the synchronization tolerance")
    parser.add_argument("--stereo-2d-only", action="store_true", help="Only export reviewable stereo MediaPipe 2D; do not change final hand labels or phases")
    parser.add_argument("--stereo-3d-diagnostic", type=Path, metavar="STEREO_JSONL",
                        help="Audit same-frame stereo with existing 2D cache; never writes final hand labels")
    parser.add_argument("--stereo-hand-smoke", type=Path, metavar="STEREO_JSONL",
                        help="Write filtered stereo aria_hands in a fresh output and stop before phase")
    parser.add_argument(
        "--stereo-hand-config", type=Path,
        default=Path(__file__).resolve().parents[2] / "cfg/preprocess/base/PicoStereoHands.yaml",
    )
    parser.add_argument("--max-frames", type=int, help="Frame limit for --stereo-2d-only")
    parser.add_argument("--eye", choices=("left", "right"), default="left", help="Select the eye crop and matching PICO extrinsic (left=E1, right=E0)")
    args = parser.parse_args(argv)
    if args.stereo_hand_smoke:
        if args.stereo_2d_only or args.stereo_3d_diagnostic or args.max_frames is not None:
            parser.error("--stereo-hand-smoke cannot be combined with another smoke/diagnostic mode")
        session = PicoImportSession.open(args.video, args.tracking, args.pts, eye=args.eye)
        writer = HumanEgoWriter(session, args.output)
        writer.export(args.max_delta_ms, args.keep_sync_outliers, generate_phases=False)
        summary = writer.write_stereo_hand_labels(args.stereo_hand_smoke, args.stereo_hand_config)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.stereo_3d_diagnostic:
        if args.stereo_2d_only or args.max_frames is not None:
            parser.error("3D diagnostic requires a complete cache and cannot be combined with 2D-only/max-frames")
        session = PicoImportSession.open(args.video, args.tracking, args.pts, eye=args.eye)
        summary = HumanEgoWriter(session, args.output).export_stereo_3d_diagnostic(args.stereo_3d_diagnostic, args.max_delta_ms)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.max_frames is not None and (not args.stereo_2d_only or args.max_frames <= 0):
        parser.error("--max-frames requires --stereo-2d-only and a positive value")
    if args.stereo_2d_only:
        session = PicoImportSession.open(args.video, args.tracking, args.pts, eye=args.eye)
        summary = HumanEgoWriter(session, args.output).export_stereo_2d_smoke(args.max_delta_ms, args.max_frames)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    summary = export_humanego(args.video, args.tracking, args.pts, args.output, args.max_delta_ms, args.keep_sync_outliers, args.eye)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
