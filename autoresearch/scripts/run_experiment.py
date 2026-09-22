#!/usr/bin/env python3
"""Launch exactly one explicitly selected HumanEgo training job."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="open_door")
    parser.add_argument("--group", default="autoresearch")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-num", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ROOT / "cfg" / "training" / args.task / args.group / f"{args.experiment_id}.yaml"
    if not config.is_file():
        parser.error(f"config does not exist: {config}")
    try:
        yaml_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"cannot read config: {exc}")
    command = [
        sys.executable, "-m", "training.FlowMatchingTrainer",
        "--task", args.task, "--use_cfg", "--exp", args.group,
        "--job", args.experiment_id,
    ]
    # The inspected Trainer resets data_paths_explicit from CLI arguments after
    # loading YAML. Pass the configured explicit paths through the real CLI so
    # open_door's external PICO sessions are actually selected.
    train_paths = yaml_config.get("MPS_PATHS_TRAIN", [])
    eval_paths = yaml_config.get("MPS_PATHS_EVAL", [])
    if yaml_config.get("data_paths_explicit") and train_paths:
        command += ["--train_data", *map(str, train_paths)]
        if eval_paths:
            command += ["--eval_data", *map(str, eval_paths)]
    for flag, value in (("--epochs", args.epochs), ("--data_num", args.data_num),
                        ("--num_workers", args.num_workers), ("--device", args.device),
                        ("--seed", args.seed)):
        if value is not None:
            command += [flag, str(value)]
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
