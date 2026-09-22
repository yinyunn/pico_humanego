#!/usr/bin/env python3
"""Parse one HumanEgo evaluation snapshot and compute the fixed AutoResearch score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = ("pos_err_w_m", "rot_err_w_deg", "grasp_f1_w")


def result(experiment_id: str, status: str, **values) -> dict:
    return {"experiment_id": experiment_id, "status": status, **values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshots = sorted((args.run_dir / "eval_snapshots").glob("eval_ep_*.json"))
    if not snapshots:
        print(json.dumps(result(args.experiment_id, "EVAL_MISSING"), indent=2))
        return 1
    snapshot = snapshots[-1]
    try:
        data = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps(result(args.experiment_id, "EVAL_ERROR", error=str(exc)), indent=2))
        return 1
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        print(json.dumps(result(args.experiment_id, "EVAL_ERROR", missing=missing, snapshot=str(snapshot)), indent=2))
        return 1
    values = {key: float(data[key]) for key in REQUIRED}
    if not all(math.isfinite(value) for value in values.values()):
        print(json.dumps(result(args.experiment_id, "NAN", snapshot=str(snapshot)), indent=2))
        return 1
    score = values["pos_err_w_m"] + values["rot_err_w_deg"] / 100.0 - values["grasp_f1_w"] * 0.10
    payload = result(args.experiment_id, "SUCCESS", **values, score=float(score), snapshot=str(snapshot))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
