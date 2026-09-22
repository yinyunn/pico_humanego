"""从 humanego_ready 运行 PICO 的对象追踪、三角化和 DatasetGen。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from preprocess.CamTriangulator import reset_camtriagulator, run_camtriagulator
from preprocess.CoTrackerOffline import print_cotracker_offline_stats, reset_cotracker_offline, run_cotracker_offline
from preprocess.DINOSAMOps import print_dinosam_stats, run_dinosam
from preprocess.DatasetGen import run_datasetgen
from preprocess.KptsSelector import run_kptsselector
from tools.pico_import.visuals import run_visual_preprocess
from utils.utils_io import load_cfg

OBJECT_LABELS = {"obj1": "soda can", "obj2": "white tray"}


def frame_paths(root: Path) -> list[str]:
    paths = []
    for directory in sorted((root / "preprocess" / "all_data").iterdir()):
        image = directory / "rgb.png"
        if directory.is_dir() and directory.name.isdigit() and image.exists():
            paths.append(str(image))
    if not paths:
        raise FileNotFoundError(f"no HumanEgo RGB frames under {root / 'preprocess' / 'all_data'}")
    return paths


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_pipeline(
    root: str | Path,
    reference_frame: int,
    dino_cfg: str | Path,
    max_frames: int | None = None,
    obj1_triangulation_end: int | None = None,
    export_visuals: bool = True,
    force_visuals: bool = False,
):
    root = Path(root)
    paths = frame_paths(root)
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    if not 0 <= reference_frame < len(paths):
        raise ValueError(f"reference frame {reference_frame} outside 0..{len(paths) - 1}")

    preprocess = root / "preprocess"
    reference_image = paths[reference_frame]
    dino_cfg = str(dino_cfg)
    kpts_cfg = str(Path(__file__).resolve().parents[2] / "cfg" / "preprocess" / "base" / "KptsSelector.yaml")
    cotracker_cfg = str(Path(__file__).resolve().parents[2] / "cfg" / "preprocess" / "base" / "CoTracker.yaml")
    project_root = Path(__file__).resolve().parents[2]
    cam_cfg_base = project_root / "cfg" / "preprocess" / "base" / "PicoCanMousepadCamTriangulator.yaml"
    cam_cfg = str(cam_cfg_base)
    if obj1_triangulation_end is not None:
        cam_cfg_runtime = preprocess / "pico_camtriangulator_runtime.yaml"
        cam_cfg_runtime.write_text(
            cam_cfg_base.read_text(encoding="utf-8")
            + f"\ntriangulation_ranges:\n  obj1:\n    start: 0\n    end: {int(obj1_triangulation_end)}\n",
            encoding="utf-8",
        )
        cam_cfg = str(cam_cfg_runtime)
    dataset_cfg = str(project_root / "cfg" / "preprocess" / "base" / "PicoCanMousepadDatasetGen.yaml")

    write_json(preprocess / "pico_downstream_plan.json", {
        "sensor_source": "pico",
        "reference_frame": reference_frame,
        "reference_image": reference_image,
        "image_count": len(paths),
        "objects": OBJECT_LABELS,
        "anchor_key": "obj2",
        "triangulation_ranges": {"obj1": {"start": 0, "end": obj1_triangulation_end}},
        "stages": ["dinosam_reference", "kptsselector", "cotracker", "pico_visuals", "camtriangulator", "datasetgen"],
    })

    print(f"[PICO] reference frame: {reference_frame} -> {reference_image}")
    print("[PICO] Stage 1/5: DINO-SAM2 on reference frame")
    run_dinosam(cfg_path=dino_cfg, image_path=reference_image)
    print_dinosam_stats()

    print("[PICO] Stage 2/5: selecting object keypoints")
    selected = {}
    for obj_key in ("obj1", "obj2"):
        mask = Path(reference_image).parent / f"mask_{obj_key}.png"
        if not mask.exists():
            raise FileNotFoundError(f"DINO-SAM2 did not produce {mask}")
        points = run_kptsselector(
            cfg_path=kpts_cfg,
            mask_path=str(mask),
            save_img_path=str(preprocess / f"kptsselector_vis_{obj_key}.png"),
            rgb_path=reference_image,
        )
        if not points or len(points) < 3:
            raise RuntimeError(f"insufficient keypoints for {obj_key}: {points}")
        selected[obj_key] = points
    write_json(preprocess / "kptsselector_results.json", {"method": "PICO_REFERENCE_DINOSAM", "objects": selected})

    print("[PICO] Stage 3/5: CoTracker over all frames")
    cotracker_config = load_cfg(cotracker_cfg)
    cotracker_config.ref_idx = reference_frame
    # CoTracker's loader expects a YAML path; create a session-local override so the
    # reference frame is explicit and the base config remains untouched.
    cotracker_override = preprocess / "pico_cotracker_runtime.yaml"
    cotracker_override.write_text(
        f"cotracker_res: {int(cotracker_config.cotracker_res)}\n"
        f"cotracker_chunk_size: {int(cotracker_config.cotracker_chunk_size)}\n"
        f"ref_idx: {reference_frame}\n"
        f"cotracker_viz_trail_len: {int(cotracker_config.cotracker_viz_trail_len)}\n",
        encoding="utf-8",
    )
    reset_cotracker_offline()
    run_cotracker_offline(paths[0], str(cotracker_override), 0, paths, str(root))
    print_cotracker_offline_stats()

    visuals_summary = None
    if export_visuals:
        print("[PICO] Stage 4/6: PICO visual preprocessing")
        visuals_summary = run_visual_preprocess(
            root=root,
            dino_cfg=dino_cfg,
            lama_cfg=project_root / "cfg" / "preprocess" / "base" / "Lama.yaml",
            visualkpts_cfg=project_root / "cfg" / "preprocess" / "base" / "VisualKpts.yaml",
            force=force_visuals,
        )

    print("[PICO] Stage 5/6: CamTriangulator")
    # CamTriangulator keeps sequence state in a module singleton.  Reset it
    # when this function is reused for another PICO recording in one process.
    reset_camtriagulator()
    run_camtriagulator(paths[0], cam_cfg, 0, paths, str(root))

    print("[PICO] Stage 6/6: DatasetGen")
    dataset_summary = run_datasetgen(paths, str(root), dataset_cfg)
    summary = {
        "status": "PASS",
        "reference_frame": reference_frame,
        "image_count": len(paths),
        "objects": OBJECT_LABELS,
        "anchor_key": "obj2",
        "visuals": visuals_summary,
        "camtriangulator_results": str(preprocess / "camtriangulator_results.json"),
        "dataset_summary": dataset_summary,
    }
    write_json(preprocess / "pico_downstream_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=750)
    parser.add_argument(
        "--dino-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "cfg" / "preprocess" / "base" / "PicoCanTrayDINOSAM.yaml",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="debug limit; omit for the full recording")
    parser.add_argument("--no-visuals", action="store_false", dest="export_visuals", help="skip PICO image/video preprocessing")
    parser.add_argument("--force-visuals", action="store_true", help="regenerate arm masks and LaMa visual outputs")
    parser.set_defaults(export_visuals=True)
    parser.add_argument(
        "--obj1-triangulation-end",
        type=int,
        default=None,
        help="inclusive end frame of the stable obj1 (soda-can) triangulation window",
    )
    args = parser.parse_args(argv)
    return 0 if run_pipeline(
        args.input,
        args.reference_frame,
        args.dino_config,
        args.max_frames,
        args.obj1_triangulation_end,
        args.export_visuals,
        args.force_visuals,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
