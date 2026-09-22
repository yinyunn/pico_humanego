#!/usr/bin/env python3
"""Print successful AutoHumanEgo results ordered by the fixed score."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).parents[1] / "results.tsv")
    args = parser.parse_args()
    with args.results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = [row for row in rows if row.get("status") == "SUCCESS" and row.get("score") not in (None, "")]
    rows.sort(key=lambda row: float(row["score"]))
    print("Rank\tExperiment\tScore\tPos(m)\tRot(deg)\tGraspF1\tDecision")
    for rank, row in enumerate(rows, 1):
        print("\t".join([
            str(rank), row["experiment_id"], f"{float(row['score']):.6f}",
            f"{float(row['pos_err_w_m']):.6f}", f"{float(row['rot_err_w_deg']):.4f}",
            f"{float(row['grasp_f1_w']):.6f}", row.get("decision", ""),
        ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
