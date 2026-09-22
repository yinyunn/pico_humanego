# -*- coding: utf-8 -*-
# @FileName: HandTrackingComparison.py

"""
====================================================================================================
Hand Tracking Comparison (HandTrackingComparison.py)
====================================================================================================

Comprehensive comparison of hand tracking methods with opt (optimized/smoothed)
vs orig (raw/unsmoothed) variants.

Generates (per session, under {mps_path}/preprocess/hand_tracking/):

  {opt,orig}/
    {method}/
      object_centric.png         — Per-method 3D trajectory render
      object_centric.ply         — Per-method 3D mesh (if open3d available)
    comparison_3d_overlay.png    — All methods 3D overlay
    comparison_temporal.png      — XYZ + Grasp over time
    comparison_ate.png           — ATE vs reference over time
    comparison_rot_err.png       — Rotation error over time
    comparison_grasp.png         — Grasp state comparison
    comparison_velocity.png      — Velocity profiles
    comparison_jerk.png          — Jerk (smoothness) profiles
    comparison_reprojection.png  — Reprojection on sample frames
    metrics.json                 — Full quantitative metrics

  summary_opt_vs_orig.png        — Side-by-side opt vs orig comparison
  metrics_summary.json           — Combined metrics for both variants
====================================================================================================
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# ═════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════

HAND_ENTITY_KEYS = {
    "aria_mps":  "hands",
    "mediapipe": "hands_mediapipe",
    "wilor":     "hands_wilor",
    "hamer":     "hands_hamer",
}

# Maps method name to the per-frame JSON filename
METHOD_JSON_FILES = {
    "aria_mps":  "aria_hands.json",
    "mediapipe": "mediapipe_hands.json",
    "wilor":     "wilor_hands.json",
    "hamer":     "hamer_hands.json",
}

HAND_SIDE_KEY = {"right": "hand_r", "left": "hand_l"}

METHOD_STYLES = {
    "aria_mps":  {"color": "#E74C3C", "label": "Aria MPS",  "ls": "-",  "marker": "o", "lw": 2.0},
    "mediapipe": {"color": "#3498DB", "label": "MediaPipe", "ls": "-",  "marker": "s", "lw": 1.5},
    "wilor":     {"color": "#2ECC71", "label": "WiLoR",     "ls": "-",  "marker": "^", "lw": 1.5},
    "hamer":     {"color": "#9B59B6", "label": "HaMeR",     "ls": "-",  "marker": "D", "lw": 1.5},
}


# ═════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════

def _load_hand_from_per_frame_json(frame_dir: str, method: str, side: str, variant: str):
    """
    Load hand **midpoint** pose from per-frame {method}_hands.json.

    Uses midpoint_pose (gripper-like frame) for comparison, NOT wrist.
    No cross-method fallback: if the method's JSON says hand_r=None, return None.

    Args:
        variant: "opt" or "orig"

    Returns:
        dict with pos_w, rot_w, grasp, confidence, velocity, valid
        or None if not available
    """
    json_name = METHOD_JSON_FILES.get(method)
    if json_name is None:
        return None
    json_path = os.path.join(frame_dir, json_name)
    if not os.path.isfile(json_path):
        return None

    with open(json_path) as f:
        d = json.load(f)

    side_key = HAND_SIDE_KEY.get(side, "hand_r")
    hand = d.get(side_key)
    # Strict: if this method detected nothing, return None (no fallback)
    if hand is None or not isinstance(hand, dict):
        return None

    # Prefer midpoint pose (gripper-like), fall back to wrist if midpoint unavailable
    if variant == "opt":
        pose_key = "midpoint_pose_opt_world"
        vel_key = "midpoint_lin_vel_opt_world"
        fallback_pose_key = "wrist_pose_opt_world"
        fallback_vel_key = "wrist_lin_vel_opt_world"
    else:  # orig
        pose_key = "midpoint_pose_raw_world"
        vel_key = "midpoint_lin_vel_raw_world"
        fallback_pose_key = "wrist_pose_raw_world"
        fallback_vel_key = "wrist_lin_vel_raw_world"

    pose_data = hand.get(pose_key)
    vel_key_used = vel_key
    if pose_data is None:
        pose_data = hand.get(fallback_pose_key)
        vel_key_used = fallback_vel_key
    if pose_data is None:
        return None

    T = np.array(pose_data, dtype=np.float64).reshape(4, 4)
    pos_w = T[:3, 3].copy()
    rot_w = T[:3, :3].copy()

    confidence = float(hand.get("confidence", 0.0))
    grasp = float(hand.get("grasp_state", 0))

    vel_data = hand.get(vel_key_used)
    velocity = np.array(vel_data, dtype=np.float64) if vel_data is not None else np.zeros(3)

    return {
        "pos_w": pos_w,
        "rot_w": rot_w,
        "grasp": grasp,
        "confidence": confidence,
        "velocity": velocity,
        "valid": True,
    }


def load_manip_frames(mps_path: str, methods: List[str], side: str = "right",
                      variant: str = "opt"):
    """
    Load manip frames from per-frame JSON files.

    Args:
        variant: "opt" (optimized/smoothed) or "orig" (raw)

    Returns:
        results, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K
    """
    all_data_dir = os.path.join(mps_path, "preprocess", "all_data")
    if not os.path.isdir(all_data_dir):
        print(f"  [WARN] all_data dir not found: {all_data_dir}")
        return {}, [], None, None, {}, None

    frame_names = sorted([
        d for d in os.listdir(all_data_dir)
        if os.path.isfile(os.path.join(all_data_dir, d, "training_data.json"))
    ])
    if not frame_names:
        return {}, [], None, None, {}, None

    results = {m: [] for m in methods}
    T_w2cam0 = None
    T_w2anchor = None
    obj_meta = {}
    cam_K = None

    _null_entry = lambda: {"pos_w": np.zeros(3), "rot_w": np.eye(3), "grasp": 0.0,
                           "confidence": 0.0, "velocity": np.zeros(3), "valid": False}

    # ── Load shared transforms & object data (once) ──
    # 1) From first available training_data.json: cam0, anchor, intrinsics, T_ok2w
    first_td_path = os.path.join(all_data_dir, frame_names[0], "training_data.json")
    if os.path.isfile(first_td_path):
        with open(first_td_path) as f:
            td = json.load(f)
        meta = td.get("metadata", {})
        wt = meta.get("world_transforms", {})

        # cam0 (camera-to-world)
        cam0_c2w = wt.get("cam0")
        if cam0_c2w is not None:
            T_cam0_w = np.array(cam0_c2w, dtype=np.float64).reshape(4, 4)
            T_w2cam0 = np.linalg.inv(T_cam0_w)

        # Virtual static anchor (anchor-to-world) — preferred over cam0 for PLY
        anchor_a2w = wt.get("virtual_static_anchor")
        if anchor_a2w is not None:
            T_anchor2w = np.array(anchor_a2w, dtype=np.float64).reshape(4, 4)
            T_w2anchor = np.linalg.inv(T_anchor2w)
        elif T_w2cam0 is not None:
            T_w2anchor = T_w2cam0.copy()

        # Camera intrinsics
        k_data = meta.get("k")
        if k_data is not None:
            cam_K = np.array(k_data, dtype=np.float64).reshape(3, 3)

        # Object T_ok2w from training_data
        objects = td.get("entities", {}).get("objects", {})
        for obj_key, obj_val in objects.items():
            if isinstance(obj_val, dict):
                T_ok2w = obj_val.get("T_obj_to_world")
                if T_ok2w is not None:
                    obj_meta[obj_key] = {
                        "T_ok2w": np.array(T_ok2w, dtype=np.float64).reshape(4, 4),
                    }

    # 2) From camtriangulator_results.json: object keypoints in local frame (pts_ok)
    preprocess_dir = os.path.join(mps_path, "preprocess")
    ct_path = os.path.join(preprocess_dir, "camtriangulator_results.json")
    if os.path.isfile(ct_path):
        with open(ct_path) as f:
            ct = json.load(f)
        ct_objects = ct.get("objects", {})
        for obj_key, obj_data in ct_objects.items():
            pts_c0 = obj_data.get("points_3d_cam0")
            T_ok2c0_raw = obj_data.get("object_to_cam0_matrix")
            if pts_c0 is not None and T_ok2c0_raw is not None:
                pts_c0 = np.array(pts_c0, dtype=np.float64)
                T_ok2c0 = np.array(T_ok2c0_raw, dtype=np.float64).reshape(4, 4)
                # Transform cam0-frame points → object-local frame
                T_c02ok = np.linalg.inv(T_ok2c0)
                pts_ok = (T_c02ok[:3, :3] @ pts_c0.T + T_c02ok[:3, 3][:, None]).T
                if obj_key in obj_meta:
                    obj_meta[obj_key]["pts_ok"] = pts_ok
                elif pts_c0.shape[0] > 0:
                    # T_ok2w not in training_data but we can derive from cam0
                    cam0_c2w = ct.get("cam0_c2w")
                    if cam0_c2w is not None:
                        T_c02w = np.array(cam0_c2w, dtype=np.float64).reshape(4, 4)
                        T_ok2w = T_c02w @ T_ok2c0
                        obj_meta[obj_key] = {"T_ok2w": T_ok2w, "pts_ok": pts_ok}

    # ── Load per-frame hand data ──
    for fname in frame_names:
        frame_dir = os.path.join(all_data_dir, fname)
        for m in methods:
            entry = _load_hand_from_per_frame_json(frame_dir, m, side, variant)
            if entry is not None:
                results[m].append(entry)
            else:
                results[m].append(_null_entry())

    return results, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K


# ═════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════

def compute_metrics(results: dict, ref_method: str = "aria_mps") -> dict:
    """Compute comprehensive metrics comparing each method to reference."""
    ref_data = results.get(ref_method, [])
    n_frames = len(ref_data)

    metrics = {}
    for method, data in results.items():
        n_det = sum(1 for d in data if d["valid"])
        detection_rate = round(100.0 * n_det / max(n_frames, 1), 1)

        # Smoothness: jerk magnitude (3rd derivative)
        positions = np.array([d["pos_w"] for d in data if d["valid"]])
        if len(positions) >= 4:
            jerk = np.diff(positions, n=3, axis=0)
            jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=-1))) * 1000  # mm/frame^3
        else:
            jerk_mag = float("nan")

        # Velocity stats
        velocities = np.array([np.linalg.norm(d["velocity"]) for d in data if d["valid"]])
        vel_mean = float(np.mean(velocities)) if len(velocities) > 0 else float("nan")
        vel_std = float(np.std(velocities)) if len(velocities) > 0 else float("nan")

        # Confidence stats
        confidences = np.array([d["confidence"] for d in data if d["valid"]])
        conf_mean = float(np.mean(confidences)) if len(confidences) > 0 else float("nan")

        # ATE, rotation, grasp vs reference
        ate_errors, rot_errors, grasp_agree = [], [], []
        for i in range(min(len(data), len(ref_data))):
            if not data[i]["valid"] or not ref_data[i]["valid"]:
                continue
            ate_errors.append(np.linalg.norm(data[i]["pos_w"] - ref_data[i]["pos_w"]) * 100)
            R_diff = data[i]["rot_w"] @ ref_data[i]["rot_w"].T
            trace = np.clip(np.trace(R_diff), -1.0, 3.0)
            rot_errors.append(np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))))
            grasp_agree.append(
                (1 if data[i]["grasp"] >= 0.5 else 0) == (1 if ref_data[i]["grasp"] >= 0.5 else 0))

        ate_arr = np.array(ate_errors) if ate_errors else np.array([])
        rot_arr = np.array(rot_errors) if rot_errors else np.array([])

        metrics[method] = {
            "n_manip_frames": n_frames,
            "n_detected": n_det,
            "detection_rate": detection_rate,
            "confidence_mean": round(conf_mean, 3),
            "ate_mean_cm": round(float(np.mean(ate_arr)), 2) if len(ate_arr) > 0 else float("nan"),
            "ate_std_cm": round(float(np.std(ate_arr)), 2) if len(ate_arr) > 0 else float("nan"),
            "ate_median_cm": round(float(np.median(ate_arr)), 2) if len(ate_arr) > 0 else float("nan"),
            "ate_max_cm": round(float(np.max(ate_arr)), 2) if len(ate_arr) > 0 else float("nan"),
            "rot_mean_deg": round(float(np.mean(rot_arr)), 1) if len(rot_arr) > 0 else float("nan"),
            "rot_std_deg": round(float(np.std(rot_arr)), 1) if len(rot_arr) > 0 else float("nan"),
            "rot_median_deg": round(float(np.median(rot_arr)), 1) if len(rot_arr) > 0 else float("nan"),
            "grasp_acc": round(100.0 * sum(grasp_agree) / max(len(grasp_agree), 1), 1),
            "vel_mean_m_s": round(vel_mean, 4),
            "vel_std_m_s": round(vel_std, 4),
            "jerk_mm_frame3": round(jerk_mag, 2) if not np.isnan(jerk_mag) else float("nan"),
            "n_codetected": len(ate_arr),
        }

    return metrics


def print_metrics_table(metrics: dict, ref_method: str, variant: str = ""):
    """Print formatted metrics table + LaTeX version."""
    n_frames = next(iter(metrics.values())).get("n_manip_frames", 0)
    tag = f" [{variant.upper()}]" if variant else ""
    print()
    print("=" * 100)
    print(f"  Hand Tracking Comparison{tag} — {n_frames} manip frames (ref: {ref_method})")
    print("=" * 100)

    header = (f"  {'Method':<12} {'Det%':>6} {'Conf':>5} "
              f"{'ATE(cm)':>9} {'ATE_med':>8} {'Rot(d)':>7} {'Grasp%':>7} "
              f"{'Vel(m/s)':>9} {'Jerk':>7}")
    print(header)
    print("-" * 100)

    for method, m in metrics.items():
        style = METHOD_STYLES.get(method, {})
        label = style.get("label", method)
        _f = lambda v, fmt: f"{v:{fmt}}" if not np.isnan(v) else "  —"
        print(f"  {label:<12} {m['detection_rate']:>5.1f}% {m['confidence_mean']:>5.3f} "
              f"{_f(m['ate_mean_cm'], '>9.2f')} {_f(m['ate_median_cm'], '>8.2f')} "
              f"{_f(m['rot_mean_deg'], '>7.1f')} {m['grasp_acc']:>6.1f}% "
              f"{_f(m['vel_mean_m_s'], '>9.4f')} {_f(m['jerk_mm_frame3'], '>7.2f')}")

    print("=" * 100)

    # LaTeX
    print(f"\n  LaTeX{tag}:")
    for method, m in metrics.items():
        label = METHOD_STYLES.get(method, {}).get("label", method)
        ate = f"{m['ate_mean_cm']:.2f}" if not np.isnan(m['ate_mean_cm']) else "--"
        rot = f"{m['rot_mean_deg']:.1f}" if not np.isnan(m['rot_mean_deg']) else "--"
        jrk = f"{m['jerk_mm_frame3']:.2f}" if not np.isnan(m['jerk_mm_frame3']) else "--"
        print(f"  {label} & {m['detection_rate']:.1f}\\% & {ate} & {rot} "
              f"& {m['grasp_acc']:.1f}\\% & {jrk} \\\\")
    print()


# ═════════════════════════════════════════════
# Coordinate helpers
# ═════════════════════════════════════════════

def _to_anchor(pos_w, T_w2anchor):
    return T_w2anchor[:3, :3] @ pos_w + T_w2anchor[:3, 3]


# ═════════════════════════════════════════════
# 3D Visualizations
# ═════════════════════════════════════════════

def plot_multi_method_3d(results, T_w2anchor, obj_meta, out_path,
                         elev=25, azim=-60, title_suffix=""):
    """Plot all methods' 3D trajectories in one figure."""
    fig = plt.figure(dpi=300, figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Objects
    for obj_key, om in obj_meta.items():
        pts_ok = om.get("pts_ok")
        if pts_ok is not None and len(pts_ok) > 0:
            pts_a = np.array([_to_anchor(p, T_w2anchor) for p in pts_ok])
            ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                       c='#BDC3C7', s=4, alpha=0.25, label="Objects")

    # Method trajectories
    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "gray", "ls": "-", "lw": 1.5, "label": method})
        pts = np.array([_to_anchor(d["pos_w"], T_w2anchor) for d in data if d["valid"]])
        if len(pts) == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=style["color"], ls=style["ls"], lw=style.get("lw", 1.5),
                label=style.get("label", method), alpha=0.85)
        ax.scatter(*pts[0], color=style["color"], marker="o", s=20, zorder=5)
        ax.scatter(*pts[-1], color=style["color"], marker="*", s=40, zorder=5)

    # Axis limits
    all_pts = np.array([_to_anchor(d["pos_w"], T_w2anchor)
                        for data in results.values() for d in data if d["valid"]])
    if len(all_pts) > 0:
        center = all_pts.mean(axis=0)
        extent = max(np.abs(all_pts - center).max(), 0.05) * 1.5
        for setter, ci in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(center[ci] - extent, center[ci] + extent)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Hand Tracking 3D Comparison{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [+] {out_path}")


def export_per_method_png(method, data, T_w2anchor, obj_meta, out_path):
    """Per-method 3D trajectory render."""
    style = METHOD_STYLES.get(method, {"color": "#888888"})
    fig = plt.figure(dpi=220, figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    for om in obj_meta.values():
        pts_ok = om.get("pts_ok")
        if pts_ok is not None and len(pts_ok) > 0:
            pts_a = np.array([_to_anchor(p, T_w2anchor) for p in pts_ok])
            ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                       c='#BDC3C7', s=4, alpha=0.5, label="Objects")

    pts = np.array([_to_anchor(d["pos_w"], T_w2anchor) for d in data if d["valid"]])
    if len(pts) > 0:
        # Color by grasp state
        grasps = np.array([d["grasp"] for d in data if d["valid"]])
        for i in range(len(pts) - 1):
            c = '#E74C3C' if grasps[i] >= 0.5 else style["color"]
            ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], pts[i:i+2, 2], color=c, lw=1.5)
        ax.scatter(*pts[0], color='#2ECC71', s=30, marker="o", label="Start")
        ax.scatter(*pts[-1], color='#E74C3C', s=30, marker="*", label="End")

        center = pts.mean(axis=0)
        ext = max(np.abs(pts - center).max(), 0.05) * 1.8
        for setter, ci in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(center[ci] - ext, center[ci] + ext)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"{style.get('label', method)} — Object-Centric (red=grasp)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] {out_path}")


# ═════════════════════════════════════════════
# Trimesh PLY Export  (matches DatasetGen style)
# ═════════════════════════════════════════════
# Geometry primitives: arrows (cylinder + cone), spheres, connecting lines.
# Layout mirrors DatasetGen._export_object_and_traj_ply():
#   A) Object keypoints (coloured spheres) + coordinate axes at each object pose
#   B) Static anchor origin marker (yellow sphere + full RGB axes)
#   C) Trajectory points + connecting lines + mini-axes at stride

_OBJ_COLORS = [[0.2, 0.9, 0.2], [0.2, 0.8, 0.9], [0.9, 0.2, 0.8], [0.9, 0.2, 0.2]]
_OBJ_KPT_RADIUS = 0.004
_OBJ_AXES_LEN   = 0.06          # object coordinate axes length
_TRAJ_PT_RADIUS  = 0.003
_TRAJ_LINE_RADIUS = 0.0012
_TRAJ_AXES_LEN   = 0.025        # mini-axes along trajectory
_TRAJ_AXES_STRIDE = 10           # draw axes every N valid frames
_ANCHOR_RADIUS   = 0.007
_ANCHOR_AXES_LEN = 0.10


def _z_align_rotation(direction):
    """Return 3×3 rotation that maps +Z to `direction`."""
    d = np.asarray(direction, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = np.dot(z, d)
    if np.linalg.norm(v) < 1e-6:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _color_mesh(mesh, color):
    c8 = [int(c * 255) for c in color]
    mesh.visual.face_colors = np.tile(c8 + [255], (len(mesh.faces), 1)).astype(np.uint8)
    return mesh


def _make_sphere(center, radius, color, subdivisions=2):
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    s.apply_translation(np.asarray(center))
    return _color_mesh(s, color)


def _make_cylinder_line(start, end, radius, color):
    """Cylinder connecting two 3D points (for trajectory lines)."""
    diff = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(diff))
    if length < 1e-6:
        return _make_sphere(start, radius, color)
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    T = np.eye(4)
    T[:3, :3] = _z_align_rotation(diff)
    T[:3, 3] = (np.asarray(start) + np.asarray(end)) / 2
    cyl.apply_transform(T)
    return _color_mesh(cyl, color)


def _make_arrow(start, end, cyl_r, cone_r, color):
    """Arrow from start to end (cylinder shaft + cone tip), matching DatasetGen style."""
    diff = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(diff))
    if length < 1e-6:
        return _make_sphere(start, cyl_r, color)
    direction = diff / length
    R = _z_align_rotation(direction)

    # Shaft: 80% of length
    shaft_len = length * 0.8
    shaft = trimesh.creation.cylinder(radius=cyl_r, height=shaft_len, sections=8)
    T_shaft = np.eye(4)
    T_shaft[:3, :3] = R
    T_shaft[:3, 3] = np.asarray(start) + direction * (shaft_len / 2)
    shaft.apply_transform(T_shaft)
    _color_mesh(shaft, color)

    # Cone tip: 20% of length
    cone_len = length * 0.2
    cone = trimesh.creation.cone(radius=cone_r, height=cone_len, sections=8)
    T_cone = np.eye(4)
    T_cone[:3, :3] = R
    T_cone[:3, 3] = np.asarray(start) + direction * (shaft_len + cone_len / 2)
    cone.apply_transform(T_cone)
    _color_mesh(cone, color)

    return trimesh.util.concatenate([shaft, cone])


def _add_axes(geos, origin, R, axis_len, cyl_r=0.001, cone_r=0.003, alpha=1.0):
    """Append RGB XYZ arrows at pose (origin, R) to geometry list."""
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for i in range(3):
        end = np.asarray(origin) + R[:, i] * axis_len
        col = [axis_colors[i][j] * alpha for j in range(3)]
        geos.append(_make_arrow(np.asarray(origin), end, cyl_r, cone_r, col))


def _build_object_meshes(obj_meta, T_w2anchor):
    """Object keypoints (coloured spheres) + coordinate axes at each object pose.
    Matches DatasetGen._export_object_and_traj_ply() layout exactly:
      pts_ok are in **object-local** frame → T_ok2anchor maps them to anchor frame."""
    geos = []
    for idx, (obj_key, om) in enumerate(sorted(obj_meta.items())):
        T_ok2w = om.get("T_ok2w", om.get("T_ok2w_static", np.eye(4)))
        # T_ok_in_anchor = T_world_to_anchor @ T_ok_to_world
        T_ok2anchor = T_w2anchor @ T_ok2w
        R_rel, t_rel = T_ok2anchor[:3, :3], T_ok2anchor[:3, 3]
        color = _OBJ_COLORS[idx % len(_OBJ_COLORS)]

        # Object keypoints: pts_ok is in object-local frame, map to anchor
        pts_ok = om.get("pts_ok")
        if pts_ok is not None:
            pts_ok = np.asarray(pts_ok)
            if pts_ok.ndim == 2 and pts_ok.shape[0] > 0:
                pts_in_anchor = (R_rel @ pts_ok.T + t_rel[:, None]).T
                for pt in pts_in_anchor:
                    geos.append(_make_sphere(pt, _OBJ_KPT_RADIUS, color))

        # Object coordinate axes at object origin in anchor frame
        _add_axes(geos, t_rel, R_rel, _OBJ_AXES_LEN, cyl_r=0.001, cone_r=0.003, alpha=0.8)
    return geos


def _build_anchor_marker():
    """Static anchor origin marker: yellow sphere + full RGB axes."""
    geos = []
    origin = np.zeros(3)
    geos.append(_make_sphere(origin, _ANCHOR_RADIUS, [1.0, 0.8, 0.0]))
    _add_axes(geos, origin, np.eye(3), _ANCHOR_AXES_LEN, cyl_r=0.002, cone_r=0.004)
    return geos


def _build_trajectory_meshes(data, T_w2anchor, color_pt, color_line,
                              pt_r=_TRAJ_PT_RADIUS, line_r=_TRAJ_LINE_RADIUS,
                              axes_stride=_TRAJ_AXES_STRIDE, axes_len=_TRAJ_AXES_LEN):
    """Trajectory: points + connecting lines + mini-axes at stride.
    Grasp-closed frames shown in red."""
    geos = []
    valid = [(i, d) for i, d in enumerate(data) if d["valid"]]
    if not valid:
        return geos

    pts_a = [_to_anchor(d["pos_w"], T_w2anchor) for _, d in valid]
    rots_a = [T_w2anchor[:3, :3] @ d["rot_w"] for _, d in valid]
    grasps = [d["grasp"] for _, d in valid]

    # Trajectory point spheres
    for i, p in enumerate(pts_a):
        c = [0.9, 0.2, 0.2] if grasps[i] >= 0.5 else color_pt
        geos.append(_make_sphere(p, pt_r, c))

    # Connecting lines
    for i in range(len(pts_a) - 1):
        geos.append(_make_cylinder_line(pts_a[i], pts_a[i + 1], line_r, color_line))

    # Mini-axes at stride
    for i in range(0, len(pts_a), max(axes_stride, 1)):
        _add_axes(geos, pts_a[i], rots_a[i], axes_len,
                  cyl_r=0.0008, cone_r=0.0018, alpha=0.95)

    # Start marker (larger green sphere)
    geos.append(_make_sphere(pts_a[0], pt_r * 2.5, [0.2, 0.8, 0.2]))
    return geos


def export_per_method_ply(method, data, T_w2anchor, obj_meta, out_path):
    """Export per-method PLY matching DatasetGen format:
    objects (keypoints + axes) + anchor marker + trajectory (points + lines + mini-axes)."""
    if not HAS_TRIMESH:
        return
    style = METHOD_STYLES.get(method, {"color": "#888888"})
    rgb = [int(style["color"][i:i+2], 16) / 255 for i in (1, 3, 5)]
    rgb_line = [min(1.0, c * 0.7 + 0.3) for c in rgb]  # lighter for lines

    geos = _build_object_meshes(obj_meta, T_w2anchor)
    geos.extend(_build_anchor_marker())
    geos.extend(_build_trajectory_meshes(data, T_w2anchor, rgb, rgb_line))

    if geos:
        combined = trimesh.util.concatenate(geos)
        combined.export(out_path)
        print(f"  [+] {out_path}")


def export_combined_ply(active_results, T_w2anchor, obj_meta, out_path):
    """Export combined PLY: all methods + objects + anchor marker."""
    if not HAS_TRIMESH:
        return
    geos = _build_object_meshes(obj_meta, T_w2anchor)
    geos.extend(_build_anchor_marker())

    for method, data in active_results.items():
        style = METHOD_STYLES.get(method, {"color": "#888888"})
        rgb = [int(style["color"][i:i+2], 16) / 255 for i in (1, 3, 5)]
        rgb_line = [min(1.0, c * 0.7 + 0.3) for c in rgb]
        geos.extend(_build_trajectory_meshes(data, T_w2anchor, rgb, rgb_line,
                                              pt_r=0.0025, line_r=0.001,
                                              axes_stride=15, axes_len=0.02))

    if geos:
        combined = trimesh.util.concatenate(geos)
        combined.export(out_path)
        print(f"  [+] {out_path}")


# ═════════════════════════════════════════════
# Rich Temporal Plots
# ═════════════════════════════════════════════

def _rotmat_to_euler(R):
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def plot_temporal_comparison(results, frame_names, out_path, title_suffix=""):
    """XYZ position + Roll/Pitch/Yaw rotation + Grasp over time."""
    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)", "Roll (°)", "Pitch (°)", "Yaw (°)", "Grasp"]

    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "gray", "ls": "-", "lw": 1.2})
        idx = [i for i, d in enumerate(data) if d["valid"]]
        if not idx:
            continue
        pos = np.array([data[i]["pos_w"] for i in idx])
        rots = np.array([_rotmat_to_euler(data[i]["rot_w"]) for i in idx])  # (N, 3)
        grs = [data[i]["grasp"] for i in idx]

        lbl = style.get("label", method)
        for a in range(3):
            axes[a].plot(idx, pos[:, a], color=style["color"], ls=style["ls"],
                         lw=style.get("lw", 1.2), label=lbl, alpha=0.85)
        for a in range(3):
            axes[3 + a].plot(idx, rots[:, a], color=style["color"], ls=style["ls"],
                             lw=style.get("lw", 1.2), label=lbl, alpha=0.85)
        axes[6].plot(idx, grs, color=style["color"], ls=style["ls"],
                     lw=style.get("lw", 1.2), label=lbl, alpha=0.85)

    for i in range(7):
        axes[i].set_ylabel(labels[i]); axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)
    axes[6].set_xlabel("Manip Frame Index"); axes[6].set_ylim(-0.1, 1.1)
    fig.suptitle(f"Temporal Comparison{title_suffix}", fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


def plot_ate_over_time(results, ref_method, out_path, title_suffix=""):
    """ATE vs reference over time with distribution inset."""
    ref = results.get(ref_method, [])
    if not ref:
        return

    fig, (ax_main, ax_hist) = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={"width_ratios": [3, 1]})

    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "gray"})
        ates, idxs = [], []
        for i in range(min(len(data), len(ref))):
            if data[i]["valid"] and ref[i]["valid"]:
                ates.append(np.linalg.norm(data[i]["pos_w"] - ref[i]["pos_w"]) * 100)
                idxs.append(i)
        if not ates:
            continue
        # Smoothed line
        w = min(15, len(ates))
        if w > 1:
            sm = np.convolve(ates, np.ones(w) / w, mode="valid")
            ax_main.plot(idxs[w - 1:], sm, color=style["color"], lw=style.get("lw", 1.5),
                         label=f"{style['label']} ({np.mean(ates):.2f}cm)", alpha=0.85)
        else:
            ax_main.plot(idxs, ates, color=style["color"], lw=1.2, label=style["label"])
        # Histogram
        ax_hist.hist(ates, bins=30, color=style["color"], alpha=0.4, orientation="horizontal",
                     label=style["label"])

    ax_main.set_xlabel("Frame Index"); ax_main.set_ylabel("ATE (cm)")
    ax_main.legend(fontsize=8); ax_main.grid(True, alpha=0.3)
    ax_main.set_title(f"ATE Over Time{title_suffix}")
    ax_hist.set_xlabel("Count"); ax_hist.set_title("Distribution")
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


def plot_rot_err_over_time(results, ref_method, out_path, title_suffix=""):
    """Rotation error vs reference over time."""
    ref = results.get(ref_method, [])
    if not ref:
        return
    fig, (ax_main, ax_hist) = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={"width_ratios": [3, 1]})
    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "gray"})
        errs, idxs = [], []
        for i in range(min(len(data), len(ref))):
            if data[i]["valid"] and ref[i]["valid"]:
                R_diff = data[i]["rot_w"] @ ref[i]["rot_w"].T
                trace = np.clip(np.trace(R_diff), -1.0, 3.0)
                errs.append(np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))))
                idxs.append(i)
        if not errs:
            continue
        w = min(15, len(errs))
        if w > 1:
            sm = np.convolve(errs, np.ones(w) / w, mode="valid")
            ax_main.plot(idxs[w - 1:], sm, color=style["color"], lw=style.get("lw", 1.5),
                         label=f"{style['label']} ({np.mean(errs):.1f}deg)", alpha=0.85)
        ax_hist.hist(errs, bins=30, color=style["color"], alpha=0.4, orientation="horizontal",
                     label=style["label"])
    ax_main.set_xlabel("Frame Index"); ax_main.set_ylabel("Rotation Error (deg)")
    ax_main.legend(fontsize=8); ax_main.grid(True, alpha=0.3)
    ax_main.set_title(f"Rotation Error Over Time{title_suffix}")
    ax_hist.set_xlabel("Count"); ax_hist.set_title("Distribution")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


def plot_grasp_comparison(results, frame_names, out_path, title_suffix=""):
    """Grasp state over time for all methods (stacked subplots)."""
    methods = list(results.keys())
    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        style = METHOD_STYLES.get(method, {"color": "gray"})
        data = results[method]
        idx = [j for j, d in enumerate(data) if d["valid"]]
        grs = [data[j]["grasp"] for j in idx]

        axes[i].fill_between(idx, grs, alpha=0.3, color=style["color"])
        axes[i].plot(idx, grs, color=style["color"], lw=1.2)
        axes[i].set_ylabel(style.get("label", method), fontsize=10)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        axes[i].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Manip Frame Index")
    fig.suptitle(f"Grasp State Comparison{title_suffix}", fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


def plot_velocity_comparison(results, frame_names, out_path, title_suffix=""):
    """Velocity magnitude over time."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "gray"})
        idx = [i for i, d in enumerate(data) if d["valid"]]
        if not idx:
            continue
        # Compute velocity from finite differences if velocity field is zero
        pos = np.array([data[i]["pos_w"] for i in idx])
        vel_fd = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
        mean_v = float(np.mean(vel_fd))
        ax.plot(idx[1:], vel_fd, color=style["color"], lw=style.get("lw", 1.2),
                label=f"{style['label']} (mean={mean_v:.4f} m/f)", alpha=0.75)

    ax.set_xlabel("Frame Index"); ax.set_ylabel("Velocity (m/frame)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"Velocity Magnitude{title_suffix}")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


def plot_jerk_comparison(results, frame_names, out_path, title_suffix=""):
    """Jerk (3rd derivative) magnitude — smoothness metric."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "gray"})
        pos = np.array([d["pos_w"] for d in data if d["valid"]])
        if len(pos) < 4:
            continue
        jerk = np.diff(pos, n=3, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=-1) * 1000  # mm
        # Smooth for display
        w = min(15, len(jerk_mag))
        if w > 1:
            sm = np.convolve(jerk_mag, np.ones(w) / w, mode="valid")
        else:
            sm = jerk_mag
        ax.plot(range(len(sm)), sm, color=style["color"], lw=style.get("lw", 1.2),
                label=f"{style['label']} (mean={np.mean(jerk_mag):.2f})", alpha=0.75)

    ax.set_xlabel("Frame Index"); ax.set_ylabel("Jerk (mm/frame^3)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"Trajectory Jerk (Smoothness){title_suffix}")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


# ═════════════════════════════════════════════
# Reprojection Strip
# ═════════════════════════════════════════════

def create_reprojection_strip(results, mps_path, cam_K, T_w2cam0,
                              frame_names, frame_indices, out_path):
    """Strip of images with reprojected hand positions per method."""
    all_data_dir = os.path.join(mps_path, "preprocess", "all_data")
    n = len(frame_indices)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for fi, frame_idx in enumerate(frame_indices):
        ax = axes[fi]
        fname = frame_names[frame_idx]
        img_path = os.path.join(all_data_dir, fname, "rgb.png")
        if os.path.exists(img_path):
            ax.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        else:
            ax.set_facecolor("black")

        # Load per-frame camera pose instead of using fixed T_w2cam0
        cam_json_path = os.path.join(all_data_dir, fname, "aria_cam_rgb.json")
        if os.path.exists(cam_json_path):
            with open(cam_json_path) as f:
                cam_d = json.load(f)
            c2w_frame = np.array(cam_d["c2w"], dtype=np.float64).reshape(4, 4)
            w2c_frame = np.linalg.inv(c2w_frame)
        else:
            w2c_frame = T_w2cam0  # fallback to fixed pose if json missing

        for method, data in results.items():
            style = METHOD_STYLES.get(method, {"color": "gray"})
            d = data[frame_idx] if frame_idx < len(data) else None
            if d is None or not d["valid"]:
                continue
            pos_cam = w2c_frame[:3, :3] @ d["pos_w"] + w2c_frame[:3, 3]
            if pos_cam[2] > 1e-4:
                u = cam_K[0, 0] * pos_cam[0] / pos_cam[2] + cam_K[0, 2]
                v = cam_K[1, 1] * pos_cam[1] / pos_cam[2] + cam_K[1, 2]
                marker = 'o' if d["grasp"] < 0.5 else 'X'
                ax.scatter(u, v, color=style["color"], marker=marker, s=80, zorder=5,
                           edgecolors='white', linewidths=0.5)
        ax.set_title(f"Frame {fname}", fontsize=9)
        ax.axis("off")

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=METHOD_STYLES.get(m, {"color": "gray"})["color"],
                      markersize=8, label=METHOD_STYLES.get(m, {"label": m})["label"])
               for m in results.keys()]
    fig.legend(handles=handles, loc="lower center", ncol=len(results), fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


# ═════════════════════════════════════════════
# Opt vs Orig Summary
# ═════════════════════════════════════════════

def plot_opt_vs_orig_summary(metrics_opt, metrics_orig, out_path):
    """Side-by-side bar chart: opt vs orig for key metrics."""
    methods = list(metrics_opt.keys())
    x = np.arange(len(methods))
    w = 0.35
    labels = [METHOD_STYLES.get(m, {}).get("label", m) for m in methods]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    metric_keys = [
        ("ate_mean_cm", "ATE (cm)"),
        ("rot_mean_deg", "Rot Error (deg)"),
        ("grasp_acc", "Grasp Acc (%)"),
        ("jerk_mm_frame3", "Jerk (mm/f^3)"),
        ("detection_rate", "Detection Rate (%)"),
        ("confidence_mean", "Confidence"),
    ]

    for idx, (key, title) in enumerate(metric_keys):
        ax = axes[idx // 3][idx % 3]
        vals_opt = [metrics_opt[m].get(key, 0) for m in methods]
        vals_orig = [metrics_orig[m].get(key, 0) for m in methods]
        # Replace nan with 0 for bar chart
        vals_opt = [0 if np.isnan(v) else v for v in vals_opt]
        vals_orig = [0 if np.isnan(v) else v for v in vals_orig]

        bars1 = ax.bar(x - w / 2, vals_opt, w, label="Optimized", color="#2ECC71", alpha=0.8)
        bars2 = ax.bar(x + w / 2, vals_orig, w, label="Raw", color="#E74C3C", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.2, axis='y')

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}",
                        ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}",
                        ha='center', va='bottom', fontsize=7)

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Optimized vs Raw Hand Tracking Comparison", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [+] {out_path}")


# ═════════════════════════════════════════════
# Run One Variant
# ═════════════════════════════════════════════

def _run_variant(mps_path, methods, side, ref_method, variant, out_dir,
                 T_w2anchor_override=None, obj_meta_override=None, cam_K_override=None):
    """Run full comparison for one variant (opt or orig). Returns metrics + shared data."""
    variant_dir = os.path.join(out_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)
    tag = f" [{variant.upper()}]"

    print(f"\n  Loading {variant} data...")
    results, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K = \
        load_manip_frames(mps_path, methods, side, variant)

    if T_w2anchor_override is not None:
        T_w2anchor = T_w2anchor_override
    if obj_meta_override:
        obj_meta = obj_meta_override
    if cam_K_override is not None:
        cam_K = cam_K_override

    if not frame_names:
        print(f"  [!] No manip frames found for {variant}.")
        return {}, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K

    # Filter active methods
    active = []
    for m in methods:
        n_det = sum(1 for d in results.get(m, []) if d["valid"])
        n_total = len(frame_names)
        print(f"    {m}: {n_det}/{n_total} detected ({100*n_det/max(n_total,1):.1f}%)")
        if n_det > 0:
            active.append(m)
    if not active:
        print(f"  [!] No detections for {variant}.")
        return {}, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K

    active_results = {m: results[m] for m in active}

    # Metrics
    print(f"\n  Computing metrics{tag}...")
    metrics = compute_metrics(active_results, ref_method)
    print_metrics_table(metrics, ref_method, variant)

    with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    print(f"\n  Generating plots{tag}...")

    if T_w2anchor is not None:
        plot_multi_method_3d(active_results, T_w2anchor, obj_meta,
                             os.path.join(variant_dir, "comparison_3d_overlay.png"),
                             title_suffix=tag)
        for method in active:
            md = os.path.join(variant_dir, method)
            os.makedirs(md, exist_ok=True)
            export_per_method_png(method, active_results[method], T_w2anchor, obj_meta,
                                  os.path.join(md, "object_centric.png"))
            if HAS_TRIMESH:
                export_per_method_ply(method, active_results[method], T_w2anchor, obj_meta,
                                      os.path.join(md, "object_centric.ply"))
        # Combined all-methods PLY
        if HAS_TRIMESH:
            export_combined_ply(active_results, T_w2anchor, obj_meta,
                                os.path.join(variant_dir, "object_centric.ply"))

    plot_temporal_comparison(active_results, frame_names,
                            os.path.join(variant_dir, "comparison_temporal.png"), tag)
    plot_ate_over_time(active_results, ref_method,
                       os.path.join(variant_dir, "comparison_ate.png"), tag)
    plot_rot_err_over_time(active_results, ref_method,
                           os.path.join(variant_dir, "comparison_rot_err.png"), tag)
    plot_grasp_comparison(active_results, frame_names,
                          os.path.join(variant_dir, "comparison_grasp.png"), tag)
    plot_velocity_comparison(active_results, frame_names,
                             os.path.join(variant_dir, "comparison_velocity.png"), tag)
    plot_jerk_comparison(active_results, frame_names,
                         os.path.join(variant_dir, "comparison_jerk.png"), tag)

    if cam_K is not None and T_w2cam0 is not None:
        n_strip = min(7, len(frame_names))
        strip_idx = [int(i * len(frame_names) / n_strip) for i in range(n_strip)]
        create_reprojection_strip(active_results, mps_path, cam_K, T_w2cam0,
                                  frame_names, strip_idx,
                                  os.path.join(variant_dir, "comparison_reprojection.png"))

    return metrics, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K


# ═════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════

def run_hand_tracking_comparison(
    mps_path: str,
    methods: List[str] = None,
    side: str = "right",
    ref_method: str = "aria_mps",
    out_dir: str = None,
) -> dict:
    """
    Run full hand tracking comparison for both opt and orig variants.

    Returns: combined metrics dict {"opt": {...}, "orig": {...}}
    """
    if methods is None:
        methods = ["aria_mps", "mediapipe", "wilor", "hamer"]
    if out_dir is None:
        out_dir = os.path.join(mps_path, "preprocess", "hand_tracking")
    os.makedirs(out_dir, exist_ok=True)

    session_name = Path(mps_path).name
    print()
    print("=" * 70)
    print(f"  Hand Tracking Comparison: {session_name}")
    print(f"  Methods: {', '.join(methods)} | Side: {side} | Ref: {ref_method}")
    print(f"  Variants: opt (optimized) + orig (raw)")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # Run OPT variant first (to get shared transforms)
    metrics_opt, frame_names, T_w2cam0, T_w2anchor, obj_meta, cam_K = \
        _run_variant(mps_path, methods, side, ref_method, "opt", out_dir)

    # Run ORIG variant (reuse shared transforms)
    metrics_orig, *_ = \
        _run_variant(mps_path, methods, side, ref_method, "orig", out_dir,
                     T_w2anchor_override=T_w2anchor,
                     obj_meta_override=obj_meta,
                     cam_K_override=cam_K)

    # Opt vs Orig summary
    if metrics_opt and metrics_orig:
        print("\n  Generating opt vs orig summary...")
        plot_opt_vs_orig_summary(metrics_opt, metrics_orig,
                                 os.path.join(out_dir, "summary_opt_vs_orig.png"))

        combined = {"opt": metrics_opt, "orig": metrics_orig}
        with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
            json.dump(combined, f, indent=2)
        print(f"  [+] {os.path.join(out_dir, 'metrics_summary.json')}")

    print(f"\n  All done! Outputs in: {out_dir}")
    return {"opt": metrics_opt, "orig": metrics_orig}


# ═════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare hand tracking methods — opt vs orig, all metrics")
    parser.add_argument("--mps_path", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["aria_mps", "mediapipe", "wilor", "hamer"])
    parser.add_argument("--side", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--ref_method", type=str, default="aria_mps")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    run_hand_tracking_comparison(
        mps_path=args.mps_path,
        methods=args.methods,
        side=args.side,
        ref_method=args.ref_method,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
