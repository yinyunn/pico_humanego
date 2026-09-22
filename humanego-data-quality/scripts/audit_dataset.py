#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from humanego_quality import (
    Issue,
    add_episode_label_issues,
    add_mask_jump_issues,
    add_temporal_and_motion_metrics,
    discover_frame_dirs,
    discover_sessions,
    episode_summary,
    exact_duplicate_issues,
    extract_frame_record,
    load_config,
    pca_2d,
    save_json,
    score_quality,
    split_distribution_shift,
)
from reporting import create_plots, write_html_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit HumanEgo preprocessing/training data quality")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--session", type=Path, help="Audit one recording directory")
    group.add_argument("--task", type=str, help="Task name under --data-root")
    p.add_argument("--data-root", type=Path, default=Path("./data"))
    p.add_argument("--sources", nargs="*", help="Optional source folders, e.g. aria pico")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--image-name", type=str, default="rgb_WoArm_WArmObjKpts.png")
    p.add_argument("--image-stride", type=int, default=1, help="Compute image/mask metrics every N frames")
    p.add_argument("--max-frames-per-session", type=int, default=0, help="0 = no cap")
    p.add_argument("--eval-session-pattern", default="*_000_vrs", help="Mark matching episodes as eval in CSV")
    p.add_argument("--reprojection-adapter", type=Path, default=None, help="Optional adapter .py; see example")
    return p.parse_args()


def load_adapter(path: Optional[Path]):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("humanego_reprojection_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapter: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_world_points"):
        raise RuntimeError("Reprojection adapter must define load_world_points(frame_dir, training_json)")
    return mod


def apply_reprojection(frame_dir: Path, rec: Dict[str, Any], adapter, issues: List[Issue]) -> None:
    if adapter is None or not rec.get("training_json_ok"):
        return
    tpath = frame_dir / "training_data.json"
    try:
        data = json.loads(tpath.read_text(encoding="utf-8"))
        pts = adapter.load_world_points(frame_dir, data)
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        md = data.get("metadata", {})
        K = np.asarray(md.get("k"), dtype=float)
        c2w = np.asarray(md.get("c2w"), dtype=float)
        if K.shape != (3, 3) or c2w.shape != (4, 4):
            return
        w2c = np.linalg.inv(c2w)
        ph = np.c_[pts, np.ones(len(pts))]
        cam = (w2c @ ph.T).T[:, :3]
        z = cam[:, 2]
        valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
        rec["reprojection_points_total"] = int(len(pts))
        rec["reprojection_points_front"] = int(valid.sum())
        if not valid.any():
            issues.append(Issue("warning", "reprojection", rec["episode"], rec["frame"], "points_in_front", 0, ">0", "No adapter points project in front of camera; check coordinate convention"))
            return
        uvw = (K @ cam[valid].T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        rec["reprojection_u_mean"] = float(np.mean(uv[:, 0]))
        rec["reprojection_v_mean"] = float(np.mean(uv[:, 1]))
        # True pixel error needs 2D ground-truth correspondences; adapter may supply them.
        if hasattr(adapter, "load_image_points"):
            gt = adapter.load_image_points(frame_dir, data)
            if gt is not None:
                gt = np.asarray(gt, dtype=float).reshape(-1, 2)
                if len(gt) == len(pts):
                    err = np.linalg.norm(uv - gt[valid], axis=1)
                    rec["reprojection_error_mean_px"] = float(np.mean(err))
                    rec["reprojection_error_p95_px"] = float(np.quantile(err, 0.95))
    except Exception as e:
        issues.append(Issue("warning", "reprojection", rec["episode"], rec["frame"], "adapter", type(e).__name__, "successful", f"Reprojection adapter failed: {e}"))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    stride = max(1, args.image_stride)
    adapter = load_adapter(args.reprojection_adapter)

    if args.session:
        sessions = [args.session.resolve()]
        dataset_name = args.session.name
    else:
        if not args.task:
            print("ERROR: pass --task <name> or --session <recording>", file=sys.stderr)
            return 2
        sessions = discover_sessions(args.data_root.resolve(), args.task, args.sources)
        dataset_name = args.task
    sessions = [s for s in sessions if (s / "preprocess" / "all_data").is_dir()]
    if not sessions:
        print("ERROR: no sessions with preprocess/all_data found", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.output or Path("quality_reports") / dataset_name / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    issues: List[Issue] = []
    for session in sessions:
        frame_dirs = discover_frame_dirs(session)
        if args.max_frames_per_session > 0:
            frame_dirs = frame_dirs[: args.max_frames_per_session]
        for i, frame_dir in enumerate(frame_dirs):
            do_image = (i % stride == 0)
            rec, rec_issues = extract_frame_record(session, frame_dir, args.image_name, do_image, cfg)
            rec["split"] = "eval" if session.match(args.eval_session_pattern) else "train"
            apply_reprojection(frame_dir, rec, adapter, rec_issues)
            records.append(rec)
            issues.extend(rec_issues)

    df = pd.DataFrame(records)
    df = add_temporal_and_motion_metrics(df, issues, cfg)
    add_mask_jump_issues(df, issues, cfg)
    add_episode_label_issues(df, issues)
    exact_duplicate_issues(df, issues)

    if "appearance_feature" in df:
        coords, _ = pca_2d(df["appearance_feature"].tolist())
        df["pca_x"] = coords[:, 0]
        df["pca_y"] = coords[:, 1]

    distribution_summary = split_distribution_shift(df, issues, cfg)

    issues_df = pd.DataFrame([i.as_dict() for i in issues])
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["severity", "category", "episode", "frame", "metric", "value", "threshold", "message"])
    ep_df = episode_summary(df, issues_df)
    score, score_parts = score_quality(df, issues_df, cfg)

    # Avoid huge nested JSON/list columns in CSV; retain paths and scalar metrics.
    csv_df = df.copy()
    drop_cols = [c for c in csv_df.columns if c.endswith("_T") or c == "appearance_feature"]
    csv_df = csv_df.drop(columns=drop_cols, errors="ignore")
    csv_df.to_csv(out_dir / "frame_metrics.csv", index=False)
    issues_df.to_csv(out_dir / "issues.csv", index=False)
    ep_df.to_csv(out_dir / "episode_metrics.csv", index=False)

    sev = Counter(issues_df["severity"].tolist())
    cat = Counter(issues_df["category"].tolist())
    summary = {
        "dataset": dataset_name,
        "sessions": [str(s) for s in sessions],
        "episodes": len(sessions),
        "frames": int(len(df)),
        "sampled_images": int(pd.to_numeric(df.get("image_read_ok"), errors="coerce").fillna(0).sum()) if "image_read_ok" in df else 0,
        "image_stride": stride,
        "quality_score": score,
        "score_components": score_parts,
        "train_eval_distribution": distribution_summary,
        "issue_counts": {k: int(v) for k, v in sev.items()},
        "issue_categories": {k: int(v) for k, v in cat.items()},
        "config": cfg,
        "notes": [
            "Quality score is a heuristic triage score; compare runs only with the same thresholds.",
            "Lightweight PCA is not a semantic CLIP/DINO embedding.",
            "Temporal checks validate post-preprocessing metadata.ts, not raw sensor clock alignment.",
        ],
    }
    save_json(out_dir / "summary.json", summary)
    plots = create_plots(df, ep_df, out_dir)
    report = write_html_report(out_dir, summary, ep_df, issues_df, plots)

    print(f"HumanEgo data-quality audit complete")
    print(f"  dataset:       {dataset_name}")
    print(f"  episodes:      {len(sessions)}")
    print(f"  frames:        {len(df)}")
    print(f"  quality score: {score:.1f}/100")
    print(f"  report:        {report}")
    print(f"  issues:        {out_dir / 'issues.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
