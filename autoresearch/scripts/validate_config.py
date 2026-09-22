#!/usr/bin/env python3
"""Validate an AutoHumanEgo YAML without importing or modifying HumanEgo code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# Mirrored from the inspected TrainConfig dataclass. This tool is deliberately
# independent of the protected trainer module.
TRAIN_CONFIG_FIELDS = {
    "out_dir", "data_root", "MPS_PATHS_TRAIN", "MPS_PATHS_EVAL",
    "data_paths_explicit", "data_sources", "eval_source", "data_num", "task",
    "img_name", "centric_mode", "frame_mode", "action_mode", "use_pcd_features",
    "use_aux_obj_dynamics", "use_aux_visual_foresight",
    "use_aux_temporal_contrastive", "use_region_attn", "use_ot_cfm",
    "enable_augmentation", "enable_aug_img", "enable_aug_rrc",
    "enable_aug_target_jittering", "enable_aug_cutout", "enable_aug_temporal_stride",
    "enable_aug_interpolation", "use_amp", "use_lr_schedule", "warmup_steps",
    "min_lr_ratio", "image_size", "pred_horizon", "single_hand",
    "single_hand_side", "max_ict", "batch_size", "epochs", "lr", "weight_decay",
    "num_workers", "grad_clip", "use_ema", "ema_decay", "w_flow", "w_pos",
    "w_rot", "w_g", "w_done", "w_foresight", "w_contrastive", "use_pre_norm",
    "use_ctx_norm", "use_done_in_flow", "use_legacy_image_loading", "use_legacy_rng",
    "use_state_noise", "state_pos_noise_std", "state_rot_noise_deg",
    "state_grasp_noise_std", "hand_tracking_method", "num_inference_steps",
    "model_horizon_weighting", "model_horizon_beta", "patch_size",
    "vision_embed_dim", "num_decoder_layers", "num_heads", "mlp_ratio", "dropout",
    "eval_every", "vis_eval_every", "early_stopping_patience",
    "early_stopping_min_delta", "make_video", "seed", "device",
}

ALIASES = {
    "model_h_weighting": "model_horizon_weighting",
    "model_h_beta": "model_horizon_beta",
    "enable_aug_jitter": "enable_aug_target_jittering",
    "enable_aug_stride": "enable_aug_temporal_stride",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        value = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    if not isinstance(value, dict):
        print("INVALID: YAML root must be a mapping", file=sys.stderr)
        return 2
    unknown = []
    for key in value:
        normalized = ALIASES.get(key, key)
        if normalized not in TRAIN_CONFIG_FIELDS:
            unknown.append(key)
    if unknown:
        print("INVALID: unknown TrainConfig keys: " + ", ".join(sorted(unknown)), file=sys.stderr)
        return 2
    if "image_size" in value and (not isinstance(value["image_size"], (list, tuple)) or len(value["image_size"]) != 2):
        print("INVALID: image_size must contain [height, width]", file=sys.stderr)
        return 2
    if "epochs" in value and int(value["epochs"]) < 1:
        print("INVALID: epochs must be >= 1", file=sys.stderr)
        return 2
    print(f"VALID: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
