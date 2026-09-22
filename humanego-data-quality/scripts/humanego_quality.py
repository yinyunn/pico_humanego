from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml


DEFAULT_IMAGE_CANDIDATES = [
    "rgb_WoArm_WArmObjKpts.png",
    "rgb_WoArm.png",
    "rgb.png",
]


@dataclass
class Issue:
    severity: str
    category: str
    episode: str
    frame: str
    metric: str
    value: Any
    threshold: Any
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    default_path = Path(__file__).resolve().parents[1] / "quality_config.yaml"
    target = Path(path) if path else default_path
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_sessions(data_root: Path, task: str, sources: Optional[Sequence[str]] = None) -> List[Path]:
    task_root = data_root / task
    if not task_root.exists():
        return []
    roots: List[Path] = []
    if sources:
        roots = [task_root / s for s in sources]
    else:
        roots = [p for p in task_root.iterdir() if p.is_dir()]
        if not roots:
            roots = [task_root]
    sessions: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            if (p / "preprocess" / "all_data").is_dir():
                sessions.append(p)
    # Also accept task root directly containing recordings
    for p in sorted(task_root.iterdir()):
        if p.is_dir() and (p / "preprocess" / "all_data").is_dir() and p not in sessions:
            sessions.append(p)
    return sorted(sessions)


def discover_frame_dirs(session: Path) -> List[Path]:
    all_data = session / "preprocess" / "all_data"
    if not all_data.is_dir():
        return []
    dirs = [p for p in all_data.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name))


def resolve_obs_path(frame_dir: Path, obs_value: Any) -> Optional[Path]:
    if not isinstance(obs_value, str) or not obs_value:
        return None
    p = Path(obs_value)
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([frame_dir / p, frame_dir / p.name])
    for c in candidates:
        if c.exists():
            return c
    return None


def choose_image(frame_dir: Path, training: Dict[str, Any], requested: Optional[str]) -> Optional[Path]:
    obs = training.get("obs", {}) if isinstance(training, dict) else {}
    keys = []
    if requested:
        # requested may be a filename or obs key
        keys.extend([requested, requested.replace(".png", "_path")])
    keys.extend(["rgb_WoArm_WArmObjKpts_path", "rgb_WoArm_path", "rgb_path"])
    for key in keys:
        if key in obs:
            p = resolve_obs_path(frame_dir, obs.get(key))
            if p:
                return p
    for name in ([requested] if requested else []) + DEFAULT_IMAGE_CANDIDATES:
        if not name:
            continue
        p = frame_dir / name
        if p.exists():
            return p
    return None


def image_metrics(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"image_read_ok": False}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    nz = probs > 0
    entropy = float(-(probs[nz] * np.log2(probs[nz])).sum())
    low_t = int(cfg["image"]["underexposed_pixel_threshold"])
    high_t = int(cfg["image"]["overexposed_pixel_threshold"])
    return {
        "image_read_ok": True,
        "image_w": int(img.shape[1]),
        "image_h": int(img.shape[0]),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur_laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": entropy,
        "underexposed_frac": float((gray <= low_t).mean()),
        "overexposed_frac": float((gray >= high_t).mean()),
        "dhash": dhash(gray),
        "appearance_feature": appearance_feature(img).tolist(),
    }


def dhash(gray: np.ndarray, hash_size: int = 8) -> str:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_hex(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        return int(int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None


def appearance_feature(img_bgr: np.ndarray) -> np.ndarray:
    """Cheap appearance vector for quick PCA; not a semantic embedding."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    low = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = [low]
    for ch, maxv in [(0, 180), (1, 256), (2, 256)]:
        h = cv2.calcHist([hsv], [ch], None, [12], [0, maxv]).ravel().astype(np.float32)
        h /= max(float(h.sum()), 1.0)
        feats.append(h)
    return np.concatenate(feats)


def mask_metrics(path: Path) -> Dict[str, Any]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"ok": False, "coverage": np.nan}
    return {"ok": True, "coverage": float((mask > 127).mean())}


def as_matrix(value: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    return a if a.shape == (4, 4) else None


def validate_se3(T: Any) -> Dict[str, Any]:
    a = as_matrix(T)
    if a is None:
        return {"valid_shape": False, "finite": False, "ortho_err": np.nan, "det": np.nan, "bottom_err": np.nan}
    finite = bool(np.isfinite(a).all())
    if not finite:
        return {"valid_shape": True, "finite": False, "ortho_err": np.nan, "det": np.nan, "bottom_err": np.nan}
    R = a[:3, :3]
    ortho = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    bottom = float(np.linalg.norm(a[3, :] - np.array([0.0, 0.0, 0.0, 1.0])))
    return {"valid_shape": True, "finite": True, "ortho_err": ortho, "det": det, "bottom_err": bottom}


def translation(T: Any) -> Optional[np.ndarray]:
    a = as_matrix(T)
    if a is None or not np.isfinite(a).all():
        return None
    return a[:3, 3].astype(float)


def rotation_angle_deg(T0: Any, T1: Any) -> Optional[float]:
    a, b = as_matrix(T0), as_matrix(T1)
    if a is None or b is None:
        return None
    R = a[:3, :3].T @ b[:3, :3]
    c = float((np.trace(R) - 1.0) / 2.0)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def safe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def timestamp_scale(values: Sequence[Any]) -> float:
    """Infer a scale that converts metadata timestamps to seconds.

    PICO exports ``metadata.ts`` as nanoseconds, while some HumanEgo sources
    store seconds or milliseconds.  Keeping the normalization here makes the
    temporal and motion diagnostics comparable across preprocessing sources.
    """
    try:
        ts = np.asarray(values, dtype=float)
        ts = ts[np.isfinite(ts)]
        if len(ts) < 2:
            return 1.0
        med_dt = float(np.median(np.abs(np.diff(ts))))
    except Exception:
        return 1.0
    if med_dt >= 1e7:       # nanoseconds, e.g. 40,000,000
        return 1e-9
    if med_dt >= 1e4:       # microseconds
        return 1e-6
    if med_dt >= 10.0:      # milliseconds
        return 1e-3
    return 1.0


def extract_frame_record(session: Path, frame_dir: Path, image_name: Optional[str], do_image: bool, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Issue]]:
    episode = session.name
    frame = frame_dir.name
    issues: List[Issue] = []
    rec: Dict[str, Any] = {"episode": episode, "frame": frame, "frame_dir": str(frame_dir)}
    tpath = frame_dir / "training_data.json"
    if not tpath.exists():
        issues.append(Issue("error", "schema", episode, frame, "training_data", "missing", "exists", "training_data.json is missing"))
        rec["training_json_ok"] = False
        return rec, issues
    try:
        data = read_json(tpath)
        rec["training_json_ok"] = True
    except Exception as e:
        issues.append(Issue("error", "schema", episode, frame, "training_data", type(e).__name__, "valid JSON", f"training_data.json could not be read: {e}"))
        rec["training_json_ok"] = False
        return rec, issues

    md = data.get("metadata", {}) or {}
    entities = data.get("entities", {}) or {}
    rec["idx"] = md.get("idx", frame)
    rec["ts"] = safe_float(md.get("ts"))
    rec["fps_meta"] = safe_float(md.get("fps"))
    rec["meta_w"] = safe_float(md.get("w"))
    rec["meta_h"] = safe_float(md.get("h"))
    rec["anchor_key"] = md.get("anchor_key")
    rec["is_finished"] = safe_float(md.get("is_finished"))

    K = np.asarray(md.get("k"), dtype=float) if md.get("k") is not None else np.array([])
    rec["intrinsics_ok"] = bool(K.shape == (3, 3) and np.isfinite(K).all())
    if not rec["intrinsics_ok"]:
        issues.append(Issue("warning", "schema", episode, frame, "metadata.k", str(K.shape), "3x3 finite", "Camera intrinsics missing or malformed"))

    c2w = md.get("c2w")
    rec["camera_T"] = c2w
    se = validate_se3(c2w)
    for k, v in se.items():
        rec[f"camera_{k}"] = v
    _se3_issues(issues, episode, frame, "camera", se, cfg)
    p = translation(c2w)
    if p is not None:
        rec.update({"camera_x": p[0], "camera_y": p[1], "camera_z": p[2]})

    hands = entities.get("hands", {}) or {}
    rec["hand_count"] = len(hands) if isinstance(hands, dict) else 0
    if not hands:
        issues.append(Issue("warning", "tracking", episode, frame, "hands", 0, ">=1 for manipulation", "No hand entity in training target"))
    if isinstance(hands, dict):
        for side, hdata in hands.items():
            if not isinstance(hdata, dict):
                continue
            T = hdata.get("T_hand_to_world")
            se_h = validate_se3(T)
            prefix = f"hand_{side}"
            rec[f"{prefix}_T"] = T
            rec[f"{prefix}_grasp"] = safe_float(hdata.get("grasp"))
            for k, v in se_h.items():
                rec[f"{prefix}_{k}"] = v
            _se3_issues(issues, episode, frame, prefix, se_h, cfg)
            hp = translation(T)
            if hp is not None:
                rec.update({f"{prefix}_x": hp[0], f"{prefix}_y": hp[1], f"{prefix}_z": hp[2]})

    objects = entities.get("objects", {}) or {}
    rec["object_count"] = len(objects) if isinstance(objects, dict) else 0
    if isinstance(objects, dict):
        for name, odata in objects.items():
            if not isinstance(odata, dict):
                continue
            T = odata.get("T_obj_to_world")
            se_o = validate_se3(T)
            prefix = f"object_{name}"
            rec[f"{prefix}_T"] = T
            rec[f"{prefix}_dynamic"] = bool(odata.get("is_dynamic", False))
            for k, v in se_o.items():
                rec[f"{prefix}_{k}"] = v
            _se3_issues(issues, episode, frame, prefix, se_o, cfg)
            op = translation(T)
            if op is not None:
                rec.update({f"{prefix}_x": op[0], f"{prefix}_y": op[1], f"{prefix}_z": op[2]})

    img_path = choose_image(frame_dir, data, image_name)
    rec["image_path"] = str(img_path) if img_path else ""
    if img_path is None:
        issues.append(Issue("error", "image", episode, frame, "image_path", "missing", image_name or "default image", "Training/diagnostic image not found"))
    elif do_image:
        im = image_metrics(img_path, cfg)
        rec.update(im)
        if not im.get("image_read_ok"):
            issues.append(Issue("error", "image", episode, frame, "image_read", "failed", "readable", f"Could not decode {img_path.name}"))
        else:
            _image_issues(issues, episode, frame, im, cfg)
            if rec.get("meta_w") and int(rec["meta_w"]) != im["image_w"]:
                issues.append(Issue("warning", "schema", episode, frame, "image_w", im["image_w"], rec["meta_w"], "Image width differs from metadata.w"))
            if rec.get("meta_h") and int(rec["meta_h"]) != im["image_h"]:
                issues.append(Issue("warning", "schema", episode, frame, "image_h", im["image_h"], rec["meta_h"], "Image height differs from metadata.h"))

    # Masks: direct filenames are stable in official preprocessing output.
    for key, filename in [("mask_arm", "mask_arm.png"), ("mask_combined", "mask_arm_and_obj.png")]:
        mp = frame_dir / filename
        if mp.exists() and do_image:
            mm = mask_metrics(mp)
            rec[f"{key}_coverage"] = mm["coverage"]
            if mm["ok"]:
                cov = mm["coverage"]
                low = float(cfg["mask"]["nearly_empty_fraction"])
                high = float(cfg["mask"]["nearly_full_fraction"])
                if cov <= low:
                    issues.append(Issue("warning", "mask", episode, frame, f"{key}_coverage", cov, f"> {low}", f"{filename} is nearly empty"))
                elif cov >= high:
                    issues.append(Issue("warning", "mask", episode, frame, f"{key}_coverage", cov, f"< {high}", f"{filename} covers most of the image"))
    obj_masks = sorted(frame_dir.glob("mask_obj*.png"))
    if obj_masks and do_image:
        covs = []
        for mp in obj_masks:
            mm = mask_metrics(mp)
            if mm["ok"]:
                covs.append(mm["coverage"])
                rec[f"{mp.stem}_coverage"] = mm["coverage"]
        if covs:
            rec["mask_objects_mean_coverage"] = float(np.mean(covs))
            rec["mask_objects_min_coverage"] = float(np.min(covs))

    return rec, issues


def _se3_issues(issues: List[Issue], episode: str, frame: str, name: str, se: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    c = cfg["se3"]
    if not se["valid_shape"] or not se["finite"]:
        issues.append(Issue("error", "geometry", episode, frame, f"{name}_T", "invalid", "finite 4x4", f"{name} transform is missing/malformed"))
        return
    if se["ortho_err"] > float(c["orthonormality_error_warn"]):
        issues.append(Issue("warning", "geometry", episode, frame, f"{name}_ortho_err", se["ortho_err"], c["orthonormality_error_warn"], f"{name} rotation is not sufficiently orthonormal"))
    if abs(se["det"] - 1.0) > float(c["determinant_error_warn"]):
        issues.append(Issue("warning", "geometry", episode, frame, f"{name}_det", se["det"], f"1±{c['determinant_error_warn']}", f"{name} rotation determinant deviates from +1"))
    if se["bottom_err"] > float(c["bottom_row_error_warn"]):
        issues.append(Issue("warning", "geometry", episode, frame, f"{name}_bottom_err", se["bottom_err"], c["bottom_row_error_warn"], f"{name} homogeneous bottom row is invalid"))


def _image_issues(issues: List[Issue], episode: str, frame: str, im: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    c = cfg["image"]
    checks = [
        ("brightness", im["brightness"], c["brightness_low"], c["brightness_high"], "Brightness is outside configured range"),
        ("contrast", im["contrast"], c["contrast_low"], None, "Image contrast is low"),
        ("blur_laplacian", im["blur_laplacian"], c["blur_laplacian_low"], None, "Image may be blurry"),
        ("entropy", im["entropy"], c["entropy_low"], None, "Image entropy/information content is low"),
    ]
    for metric, value, low, high, msg in checks:
        bad = value < float(low) if high is None else (value < float(low) or value > float(high))
        if bad:
            thr = f">={low}" if high is None else f"[{low}, {high}]"
            issues.append(Issue("warning", "image", episode, frame, metric, value, thr, msg))
    exp = float(c["exposure_fraction_warn"])
    if im["underexposed_frac"] > exp:
        issues.append(Issue("warning", "image", episode, frame, "underexposed_frac", im["underexposed_frac"], f"<= {exp}", "Large underexposed region"))
    if im["overexposed_frac"] > exp:
        issues.append(Issue("warning", "image", episode, frame, "overexposed_frac", im["overexposed_frac"], f"<= {exp}", "Large overexposed region"))


def add_temporal_and_motion_metrics(df: pd.DataFrame, issues: List[Issue], cfg: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for ep, idxs in out.groupby("episode", sort=False).groups.items():
        inds = list(idxs)
        sub = out.loc[inds]
        raw_ts = pd.to_numeric(sub.get("ts"), errors="coerce").to_numpy(dtype=float)
        ts = raw_ts * timestamp_scale(raw_ts)
        dts = np.r_[np.nan, np.diff(ts)] if len(ts) else np.array([])
        out.loc[inds, "dt"] = dts
        valid_dt = dts[np.isfinite(dts) & (dts > 0)]
        med_dt = float(np.median(valid_dt)) if valid_dt.size else np.nan
        out.loc[inds, "episode_median_dt"] = med_dt
        if np.isfinite(med_dt) and med_dt > 0:
            out.loc[inds, "effective_fps"] = 1.0 / med_dt
        for local_i in range(1, len(inds)):
            row_i = inds[local_i]
            frame = str(out.at[row_i, "frame"])
            dt = dts[local_i]
            if not np.isfinite(dt) or dt <= 0:
                issues.append(Issue("error", "temporal", ep, frame, "dt", dt, "> 0", "Timestamp is non-monotonic or invalid"))
            elif np.isfinite(med_dt) and dt > med_dt * float(cfg["time"]["gap_factor_warn"]):
                issues.append(Issue("warning", "temporal", ep, frame, "dt", dt, f"<= {cfg['time']['gap_factor_warn']}×median", "Large timestamp/frame gap"))

        # Consecutive duplicate check among rows where images were sampled.
        prev_hash = None
        for row_i in inds:
            h = out.at[row_i, "dhash"] if "dhash" in out.columns else None
            if isinstance(h, str) and h:
                hd = hamming_hex(prev_hash, h) if prev_hash else None
                out.at[row_i, "dhash_prev_distance"] = hd
                if hd is not None and hd <= int(cfg["image"]["near_duplicate_hamming"]):
                    issues.append(Issue("info", "redundancy", ep, str(out.at[row_i, "frame"]), "dhash_prev_distance", hd, f"> {cfg['image']['near_duplicate_hamming']}", "Near-consecutive duplicate appearance"))
                    out.at[row_i, "near_duplicate_prev"] = True
                prev_hash = h

        # Motion metrics for camera / each discovered hand / object.
        prefixes = ["camera"]
        prefixes += sorted({c[:-2] for c in out.columns if c.startswith("hand_") and c.endswith("_x")})
        prefixes += sorted({c[:-2] for c in out.columns if c.startswith("object_") and c.endswith("_x")})
        for prefix in prefixes:
            xcol, ycol, zcol = f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"
            if not all(c in out.columns for c in (xcol, ycol, zcol)):
                continue
            pts = sub[[xcol, ycol, zcol]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            dist = np.r_[np.nan, np.linalg.norm(np.diff(pts, axis=0), axis=1)]
            speed = dist / dts
            out.loc[inds, f"{prefix}_translation_step"] = dist
            out.loc[inds, f"{prefix}_speed"] = speed
            Tcol = f"{prefix}_T"
            rot = np.full(len(inds), np.nan)
            if Tcol in out.columns:
                vals = list(sub[Tcol])
                for i in range(1, len(vals)):
                    a = rotation_angle_deg(vals[i - 1], vals[i])
                    rot[i] = a if a is not None else np.nan
                out.loc[inds, f"{prefix}_rotation_step_deg"] = rot
            _motion_issues(out, inds, ep, prefix, dist, speed, rot, issues, cfg)
    return out


def _motion_issues(df: pd.DataFrame, inds: List[int], ep: str, prefix: str, dist: np.ndarray, speed: np.ndarray, rot: np.ndarray, issues: List[Issue], cfg: Dict[str, Any]) -> None:
    c = cfg["trajectory"]
    kind = "camera" if prefix == "camera" else ("hand" if prefix.startswith("hand_") else "object")
    jump_t = float(c[f"{kind}_translation_jump_warn_m"])
    speed_t = float(c[f"{kind}_translation_speed_warn_mps"])
    rot_t = float(c[f"{kind}_rotation_jump_warn_deg"])
    for j in range(1, len(inds)):
        row_i = inds[j]
        frame = str(df.at[row_i, "frame"])
        if np.isfinite(dist[j]) and dist[j] > jump_t:
            issues.append(Issue("warning", "trajectory", ep, frame, f"{prefix}_translation_step", float(dist[j]), jump_t, f"Large {prefix} translation jump"))
        if np.isfinite(speed[j]) and speed[j] > speed_t:
            issues.append(Issue("warning", "trajectory", ep, frame, f"{prefix}_speed", float(speed[j]), speed_t, f"High {prefix} translation speed"))
        if np.isfinite(rot[j]) and rot[j] > rot_t:
            issues.append(Issue("warning", "trajectory", ep, frame, f"{prefix}_rotation_step_deg", float(rot[j]), rot_t, f"Large {prefix} rotation jump"))


def add_mask_jump_issues(df: pd.DataFrame, issues: List[Issue], cfg: Dict[str, Any]) -> None:
    cols = [c for c in df.columns if c.endswith("_coverage") and c.startswith("mask")]
    t = float(cfg["mask"]["coverage_jump_warn"])
    for ep, sub in df.groupby("episode", sort=False):
        for col in cols:
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            dv = np.r_[np.nan, np.abs(np.diff(vals))]
            for j, val in enumerate(dv):
                if np.isfinite(val) and val > t:
                    frame = str(sub.iloc[j]["frame"])
                    issues.append(Issue("warning", "mask", ep, frame, f"{col}_jump", float(val), t, f"Abrupt {col} change"))


def add_episode_label_issues(df: pd.DataFrame, issues: List[Issue]) -> None:
    """HumanEgo-specific episode-level checks for anchor/grasp/finished labels."""
    if df.empty:
        return
    for ep, sub in df.groupby("episode", sort=False):
        if "anchor_key" in sub:
            anchors = sub["anchor_key"].dropna().astype(str).unique().tolist()
            if len(anchors) > 1:
                issues.append(Issue("warning", "labels", ep, "*", "anchor_key", ",".join(anchors), "constant within episode", "Anchor key changes inside one demonstration"))
        if "is_finished" in sub:
            fin = pd.to_numeric(sub["is_finished"], errors="coerce").dropna()
            if len(fin) and float(fin.max()) < 0.5:
                issues.append(Issue("warning", "labels", ep, "*", "is_finished", float(fin.max()), ">=0.5 at demonstration end", "No finished-positive frame found"))
        grasp_cols = [c for c in sub.columns if c.startswith("hand_") and c.endswith("_grasp")]
        for col in grasp_cols:
            g = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(g):
                continue
            ratio = float((g >= 0.5).mean())
            transitions = int(np.sum((g[1:] >= 0.5) != (g[:-1] >= 0.5))) if len(g) > 1 else 0
            if ratio in (0.0, 1.0):
                issues.append(Issue("info", "labels", ep, "*", col, ratio, "task-dependent", f"{col} is constant for the entire episode; verify grasp labeling if the task includes contact transitions"))
            elif transitions == 0:
                issues.append(Issue("info", "labels", ep, "*", f"{col}_transitions", transitions, ">=1 when task grasps/releases", "No grasp-state transition detected"))


def split_distribution_shift(df: pd.DataFrame, issues: List[Issue], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap train/eval appearance shift using standardized feature-mean distance."""
    result = {"available": False}
    if "appearance_feature" not in df or "split" not in df:
        return result
    rows = []
    splits = []
    for _, r in df.iterrows():
        f = r.get("appearance_feature")
        if isinstance(f, (list, tuple, np.ndarray)) and len(f) > 2:
            a = np.asarray(f, dtype=float)
            if np.isfinite(a).all():
                rows.append(a); splits.append(str(r.get("split", "train")))
    if len(rows) < 6 or "train" not in splits or "eval" not in splits:
        return result
    X = np.vstack(rows)
    splits = np.asarray(splits)
    tr, ev = X[splits == "train"], X[splits == "eval"]
    if len(tr) < 3 or len(ev) < 3:
        return result
    pooled = X.std(axis=0)
    pooled[pooled < 1e-6] = 1.0
    delta = (tr.mean(axis=0) - ev.mean(axis=0)) / pooled
    shift = float(np.sqrt(np.mean(delta ** 2)))
    result = {"available": True, "mean_shift_rms": shift, "train_samples": int(len(tr)), "eval_samples": int(len(ev))}
    dc = cfg.get("distribution", {})
    warn = float(dc.get("train_eval_mean_shift_warn", 0.8))
    err = float(dc.get("train_eval_mean_shift_error", 1.5))
    if shift >= err:
        issues.append(Issue("error", "distribution", "train_vs_eval", "*", "appearance_mean_shift_rms", shift, f"< {err}", "Strong train/eval appearance distribution separation in lightweight features"))
    elif shift >= warn:
        issues.append(Issue("warning", "distribution", "train_vs_eval", "*", "appearance_mean_shift_rms", shift, f"< {warn}", "Train/eval appearance distributions differ in lightweight features"))
    return result


def exact_duplicate_issues(df: pd.DataFrame, issues: List[Issue]) -> None:
    if "dhash" not in df.columns:
        return
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, r in df.iterrows():
        h = r.get("dhash")
        if isinstance(h, str) and h:
            groups[h].append((str(r["episode"]), str(r["frame"])))
    for h, members in groups.items():
        eps = {e for e, _ in members}
        if len(members) > 1 and len(eps) > 1:
            for ep, fr in members:
                issues.append(Issue("info", "redundancy", ep, fr, "cross_episode_exact_dhash", h, "unique across episodes", f"Exact dHash shared by {len(members)} sampled frames across {len(eps)} episodes"))


def pca_2d(features: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    valid_idx = []
    rows = []
    for i, f in enumerate(features):
        if isinstance(f, (list, tuple, np.ndarray)) and len(f) > 2:
            a = np.asarray(f, dtype=float)
            if np.isfinite(a).all():
                valid_idx.append(i)
                rows.append(a)
    coords = np.full((len(features), 2), np.nan, dtype=float)
    if len(rows) < 3:
        return coords, np.array(valid_idx, dtype=int)
    X = np.vstack(rows)
    X = X - X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    X /= scale
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        z = X @ vt[:2].T
        coords[np.array(valid_idx), :] = z
    except np.linalg.LinAlgError:
        pass
    return coords, np.array(valid_idx, dtype=int)


def episode_summary(df: pd.DataFrame, issues_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep, sub in df.groupby("episode", sort=False):
        raw_ts = pd.to_numeric(sub.get("ts"), errors="coerce")
        ts = raw_ts * timestamp_scale(raw_ts.to_numpy(dtype=float))
        duration = float(ts.max() - ts.min()) if ts.notna().sum() >= 2 else np.nan
        r: Dict[str, Any] = {
            "episode": ep,
            "frames": int(len(sub)),
            "duration_s": duration,
            "sampled_images": int(pd.to_numeric(sub.get("image_read_ok"), errors="coerce").fillna(0).sum()) if "image_read_ok" in sub else 0,
            "missing_training_json": int((sub.get("training_json_ok", True) == False).sum()) if "training_json_ok" in sub else 0,
            "anchor_count": int(sub["anchor_key"].dropna().nunique()) if "anchor_key" in sub else 0,
        }
        for col in ["brightness", "blur_laplacian", "entropy", "mask_arm_coverage", "mask_objects_mean_coverage", "effective_fps"]:
            if col in sub:
                r[f"mean_{col}"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        near = sub.get("near_duplicate_prev")
        r["near_duplicate_ratio"] = float(pd.Series(near).map(lambda x: bool(x) if pd.notna(x) else False).mean()) if near is not None else 0.0
        for c in [x for x in sub.columns if x.endswith("_speed")]:
            r[f"p95_{c}"] = float(pd.to_numeric(sub[c], errors="coerce").quantile(0.95))
        epi = issues_df[issues_df["episode"] == ep] if not issues_df.empty else pd.DataFrame()
        r["errors"] = int((epi["severity"] == "error").sum()) if not epi.empty else 0
        r["warnings"] = int((epi["severity"] == "warning").sum()) if not epi.empty else 0
        r["infos"] = int((epi["severity"] == "info").sum()) if not epi.empty else 0
        rows.append(r)
    return pd.DataFrame(rows)


def score_quality(df: pd.DataFrame, issues_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    weights = cfg.get("score_weights", {})
    categories = ["schema", "geometry", "temporal", "image", "mask", "redundancy", "labels", "distribution"]
    cat_scores: Dict[str, float] = {}
    n = max(len(df), 1)
    for cat in categories:
        w = float(weights.get(cat, 0))
        if issues_df.empty:
            penalty_rate = 0.0
        else:
            sub = issues_df[issues_df["category"] == cat]
            # Errors cost 3x warnings, infos 0.25x; cap at one weighted failure/frame.
            p = 3.0 * (sub["severity"] == "error").sum() + 1.0 * (sub["severity"] == "warning").sum() + 0.25 * (sub["severity"] == "info").sum()
            penalty_rate = min(1.0, float(p) / n)
        cat_scores[cat] = max(0.0, w * (1.0 - penalty_rate))
    total_weight = sum(float(weights.get(c, 0)) for c in categories) or 100.0
    score = 100.0 * sum(cat_scores.values()) / total_weight
    return float(score), cat_scores


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)
