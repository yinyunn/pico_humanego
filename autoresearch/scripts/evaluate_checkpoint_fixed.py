#!/usr/bin/env python3
"""Evaluate a checkpoint with a reproducible eval protocol.

This diagnostic intentionally reuses the existing HumanEgo trainer evaluator
without changing it.  It fixes eval ordering and augmentation behavior, sets
the RNG before every pass, and optionally compares Euler step counts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# When invoked as ``python autoresearch/scripts/...py``, Python puts only the
# script directory on sys.path. Add the repository root for HumanEgo imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions
from training.FlowMatchingModel import FlowMatchingModel
from training.FlowMatchingTrainer import TrainConfig, eval_ode_inference
from utils.utils_math import set_seed


ALIASES = {
    "model_h_weighting": "model_horizon_weighting",
    "model_h_beta": "model_horizon_beta",
    "enable_aug_jitter": "enable_aug_target_jittering",
    "enable_aug_stride": "enable_aug_temporal_stride",
}


def load_config(path: Path) -> TrainConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = TrainConfig()
    for key, value in values.items():
        key = ALIASES.get(key, key)
        if hasattr(cfg, key):
            if key == "image_size" and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def make_model(cfg: TrainConfig) -> FlowMatchingModel:
    return FlowMatchingModel(
        single_hand=cfg.single_hand,
        pred_horizon=cfg.pred_horizon,
        max_ict=cfg.max_ict,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        vision_embed_dim=cfg.vision_embed_dim,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        horizon_weighting=cfg.model_horizon_weighting,
        horizon_beta=cfg.model_horizon_beta,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn,
        use_pre_norm=cfg.use_pre_norm,
        use_ctx_norm=cfg.use_ctx_norm,
        use_done_in_flow=cfg.use_done_in_flow,
    ).to(cfg.device)


def score(metrics: dict) -> float:
    return (
        float(metrics["pos_err_w_m"])
        + float(metrics["rot_err_w_deg"]) / 100.0
        - float(metrics["grasp_f1_w"]) * 0.10
    )


def mean_std(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[20])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Reuse the exact same seed for every repeat to test determinism",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 1 or any(step < 1 for step in args.steps):
        parser.error("repeats and steps must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.stats.is_file():
        parser.error(f"stats file does not exist: {args.stats}")

    cfg = load_config(args.config)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.num_workers = 0
    cfg.enable_augmentation = False
    cfg.enable_aug_img = False
    cfg.enable_aug_rrc = False
    cfg.enable_aug_target_jittering = False
    cfg.enable_aug_cutout = False
    cfg.enable_aug_temporal_stride = False
    cfg.enable_aug_interpolation = False
    stats = json.loads(args.stats.read_text(encoding="utf-8"))

    eval_sessions = [MPSSessions(path) for path in cfg.MPS_PATHS_EVAL]
    ds_eval = FlowMatchingDataloader(
        sessions=eval_sessions,
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand,
        single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name,
        centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=False,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=cfg.use_legacy_image_loading,
        use_legacy_rng=cfg.use_legacy_rng,
        seed=cfg.seed,
        stats=stats,
    )
    loader = DataLoader(
        ds_eval,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = make_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    raw = []
    for steps in args.steps:
        cfg.num_inference_steps = steps
        for repeat in range(args.repeats):
            repeat_seed = args.seed if args.same_seed else args.seed + repeat
            random.seed(repeat_seed)
            np.random.seed(repeat_seed)
            set_seed(repeat_seed)
            metrics = eval_ode_inference(model, loader, cfg, max_batches=None)
            row = {
                "steps": int(steps),
                "repeat": int(repeat + 1),
                "seed": int(repeat_seed),
                **{key: float(value) if isinstance(value, (float, int)) else value
                   for key, value in metrics.items()},
            }
            row["score"] = float(score(row))
            if not all(math.isfinite(float(row[key])) for key in (
                "pos_err_w_m", "rot_err_w_deg", "grasp_f1_w", "score"
            )):
                raise RuntimeError(f"non-finite evaluation result: {row}")
            raw.append(row)
            print(json.dumps(row, ensure_ascii=False))

    summaries = {}
    for steps in args.steps:
        rows = [row for row in raw if row["steps"] == steps]
        summaries[str(steps)] = {
            key: mean_std(rows, key)
            for key in (
                "pos_err_k1_m", "pos_err_kK_m",
                "rot_err_k1_deg", "rot_err_kK_deg",
                "grasp_f1_k1", "grasp_f1_kK",
                "pos_err_w_m", "rot_err_w_deg", "grasp_f1_w", "score",
            )
        }

    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "stats": str(args.stats),
        "eval_sessions": [session.mps_path for session in eval_sessions],
        "protocol": {
            "augmentation": False,
            "shuffle": False,
            "num_workers": 0,
            "seed_schedule": [
                args.seed if args.same_seed else args.seed + i
                for i in range(args.repeats)
            ],
            "solver": "Euler",
            "steps": [int(step) for step in args.steps],
        },
        "raw": raw,
        "summary_by_steps": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
