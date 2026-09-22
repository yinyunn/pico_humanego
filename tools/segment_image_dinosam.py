#!/usr/bin/env python3
"""Segment an image using HumanEgo's Grounding DINO + SAM2 engine.

Example (in the HumanEgo environment)::

    python tools/segment_image_dinosam.py --image photo.jpg \
        --prompt "hand. arm." --output-dir outputs/segmentation

Repeat --prompt to save separate masks. Objects within one prompt are merged.
Model weights use the existing Hugging Face cache and download if missing.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--prompt", action="append", required=True,
                        help='English target description, e.g. "hand. arm."; repeatable')
    parser.add_argument("--output-dir", type=Path, help="Default: <image_stem>_dinosam beside input")
    parser.add_argument("--cfg", type=Path, default=ROOT / "cfg/preprocess/base/DINOSAM.yaml")
    parser.add_argument("--box-threshold", type=float, help="Detection threshold in [0, 1]")
    parser.add_argument("--max-boxes", type=int, help="Keep highest scoring N boxes per prompt")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.cfg.is_file():
        parser.error(f"Config does not exist: {args.cfg}")
    if args.box_threshold is not None and not 0 <= args.box_threshold <= 1:
        parser.error("--box-threshold must be in [0, 1]")
    if args.max_boxes is not None and args.max_boxes < 1:
        parser.error("--max-boxes must be positive")
    args.prompt = [p.strip() for p in args.prompt]
    if not all(args.prompt):
        parser.error("--prompt cannot be empty")
    return args


def main():
    args = parse_args()
    # Keep --help available even outside the model environment.
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np
    import torch
    from preprocess.DINOSAM import DINOSAMEngine
    from utils.utils_io import load_cfg

    image_path = args.image.resolve()
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    output = (args.output_dir or image_path.with_name(image_path.stem + "_dinosam")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(str(args.cfg.resolve()))
    if args.box_threshold is not None:
        cfg.box_threshold = args.box_threshold

    def save(name, array):
        path = output / name
        if path == image_path:
            raise ValueError(f"Output would overwrite input: {path}")
        if not cv2.imwrite(str(path), array):
            raise OSError(f"Cannot write image: {path}")

    engine = DINOSAMEngine(cfg)
    results = []
    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    try:
        engine.dino_model.eval()
        with torch.inference_mode():
            # SAM2 expects RGB; the existing DINO engine accepts OpenCV BGR.
            engine.predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for index, prompt in enumerate(args.prompt, 1):
                prompt = prompt if prompt.endswith(".") else prompt + "."
                mask, confidence, boxes, scores = engine.predict_frame_internal(
                    img, prompt, max_boxes=args.max_boxes)
                if mask is None:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                name = f"mask_{index:02d}.png"
                save(name, mask)
                combined = np.maximum(combined, mask)
                count = 0 if boxes is None else len(boxes)
                results.append({"prompt": prompt, "mask": name,
                                "box_count": count, "mean_score": float(confidence),
                                "boxes_xyxy": [] if boxes is None else boxes.tolist(),
                                "scores": [] if scores is None else scores.tolist()})
                print(f"[{index}] {prompt} -> {count} boxes, {int(np.count_nonzero(mask))} pixels")
                if count == 0:
                    print("  No detection; saved an empty mask. Try another prompt or lower threshold.")
    finally:
        engine.cleanup()

    save("mask.png", combined)
    overlay = img.copy()
    selected = combined > 0
    overlay[selected] = (0.55 * img[selected] + 0.45 * np.array([0, 220, 80])).astype(np.uint8)
    save("overlay.png", overlay)
    save("cutout.png", np.dstack([img, combined]))
    metadata = {"image": str(image_path), "width": img.shape[1], "height": img.shape[0],
                "cfg": str(args.cfg.resolve()), "box_threshold": float(cfg.box_threshold),
                "max_boxes": args.max_boxes, "results": results}
    (output / "results.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved segmentation to: {output}")


if __name__ == "__main__":
    main()
