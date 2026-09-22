#!/usr/bin/env python3
"""Load audited HumanEgo sampled frames into FiftyOne for interactive filtering.

FiftyOne is optional and intentionally not in requirements-quality.txt.
Install separately: pip install fiftyone
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame-csv", type=Path, required=True)
    p.add_argument("--dataset-name", default="humanego_quality")
    p.add_argument("--persistent", action="store_true")
    args = p.parse_args()

    try:
        import fiftyone as fo
    except ImportError as e:
        raise SystemExit("FiftyOne is not installed. Run: pip install fiftyone") from e

    df = pd.read_csv(args.frame_csv)
    if "image_path" not in df:
        raise SystemExit("frame_metrics.csv has no image_path column")
    df = df[df["image_path"].astype(str).str.len() > 0]
    df = df[df["image_path"].map(lambda p: Path(str(p)).exists())]

    if args.dataset_name in fo.list_datasets():
        fo.delete_dataset(args.dataset_name)
    ds = fo.Dataset(args.dataset_name, persistent=args.persistent)

    skip = {"image_path", "frame_dir"}
    samples = []
    for _, r in df.iterrows():
        s = fo.Sample(filepath=str(r["image_path"]))
        for col, val in r.items():
            if col in skip or pd.isna(val):
                continue
            if isinstance(val, (str, int, float, bool)):
                safe = col.replace(".", "_").replace("-", "_")
                try:
                    s[safe] = val.item() if hasattr(val, "item") else val
                except Exception:
                    pass
        samples.append(s)
    ds.add_samples(samples)
    print(f"Loaded {len(ds)} samples into FiftyOne dataset: {ds.name}")
    print("Tip: color/filter by episode, split, blur_laplacian, brightness, mask coverage, speed, or PCA coordinates.")
    session = fo.launch_app(ds)
    session.wait()


if __name__ == "__main__":
    main()
