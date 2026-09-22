"""PICO -> HumanEgo visual preprocessing and diagnostic video export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from preprocess.DINOSAMOps import close_dinosam_worker, run_dinosam
from preprocess.Lama import close_lama_worker, run_lama
from preprocess.VisualKpts import reset_visualkpts, run_visualkpts


def render_stereo_3d_diagnostic(session, detections, reconstructions, variants, output):
    """Existing native projection vs same-time stereo; candidate is NOT a label."""
    from .video_reader import crop_side_by_side
    from .humanego_writer import _pose_matrix
    from .stereo_geometry import MP_BONES, stereo_matrices
    from preprocess.AriaHandsOps import AriaHandsOps

    output = Path(output)
    cap = cv2.VideoCapture(str(session.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    epipolar_writer = None
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot open diagnostic video input/output")
    samples = []
    epipolar_samples = []
    indices = set(np.linspace(0, len(detections) - 1, 6, dtype=int))

    def points(canvas, uv, color, radius=3, lines=False):
        uv = np.asarray(uv)
        visible = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < canvas.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < canvas.shape[0])
        for p in uv[visible]:
            cv2.circle(canvas, tuple(np.rint(p).astype(int)), radius, color, 1, cv2.LINE_AA)
        if lines:
            for a, b in MP_BONES:
                if visible[a] and visible[b]:
                    cv2.line(canvas, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 1, cv2.LINE_AA)

    def epiline(canvas, line, color):
        a, b, c = np.asarray(line, dtype=float)
        h, w = canvas.shape[:2]
        if abs(b) >= abs(a) and abs(b) > 1e-10:
            p1, p2 = (0, int(round(-c / b))), (w - 1, int(round(-(a * (w - 1) + c) / b)))
        elif abs(a) > 1e-10:
            p1, p2 = (int(round(-c / a)), 0), (int(round(-(b * (h - 1) + c) / a)), h - 1)
        else:
            return
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    try:
        for i, (det, frame) in enumerate(zip(detections, reconstructions)):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode diagnostic frame {i}")
            # PTS can use non-contiguous frame ids; seek only when necessary.
            if det["frame_idx"] != i:
                cap.set(cv2.CAP_PROP_POS_FRAMES, det["frame_idx"])
                ok, image = cap.read()
                if not ok:
                    raise ValueError("Cannot seek diagnostic source frame")
            tracking = session.tracking[frame["sync"]["tracking_index"]]
            H = _pose_matrix(tracking["head"])
            panels = []
            epipolar_panels = []
            for name, matrices in variants.items():
                views = []
                for eye in ("left", "right"):
                    canvas = crop_side_by_side(image, eye).copy()
                    if name == "confirmed_K_D_invE":
                        native = {"hand_l": {"pico_joints_26": tracking["hands"]["left"]["joints"]} if tracking["hands"]["left"] and tracking["hands"]["left"]["is_active"] else None,
                                  "hand_r": {"pico_joints_26": tracking["hands"]["right"]["joints"]} if tracking["hands"]["right"] and tracking["hands"]["right"]["is_active"] else None}
                        canvas = _draw_full_hand_skeleton(canvas, native, {"k": session.calibration.K, "c2w": H @ matrices[eye]})
                    for hand in det["eyes"][eye]["hands"]:
                        uv = [[j["u_eye"], j["v_eye"]] for j in hand["joints"]]
                        points(canvas, uv, (0, 255, 0), lines=True)
                    for c in frame["variants"][name]:
                        tri = c["triangulation"]
                        points(canvas, tri[eye + "_reprojected_uv"], (0, 0, 255), radius=5)
                        pose = c["T_hand_to_world_diagnostic"]
                        if pose is not None:
                            canvas = AriaHandsOps._draw_axis(canvas, np.linalg.inv(H @ matrices[eye]) @ pose,
                                                            session.calibration.K, np.zeros(5))
                    accepted = sum(c["accepted"] for c in frame["variants"][name])
                    cv2.rectangle(canvas, (0, 0), (1080, 65), (20, 20, 20), -1)
                    cv2.putText(canvas, f"{name} / {eye} / frame {det['frame_idx']} / quality pass {accepted}",
                                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(canvas, "GREEN: MediaPipe 2D  RED: 3D reprojection | DIAGNOSTIC ONLY", (12, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 1, cv2.LINE_AA)
                    views.append(canvas)
                panels.append(np.hstack(views))
                epipolar_views = [crop_side_by_side(image, eye).copy() for eye in ("left", "right")]
                _, _, F = stereo_matrices(session.calibration.K, session.calibration.K, matrices["relative"])
                colors = [(0, 255, 255), (255, 0, 255), (0, 180, 255),
                          (255, 180, 0), (80, 255, 80), (255, 80, 80)]
                for candidate in frame["variants"][name]:
                    for color, joint in zip(colors, (0, 4, 8, 12, 16, 20)):
                        left_uv = np.asarray(candidate["left_uv"][joint])
                        right_uv = np.asarray(candidate["right_uv"][joint])
                        cv2.circle(epipolar_views[0], tuple(np.rint(left_uv).astype(int)), 5, color, -1, cv2.LINE_AA)
                        cv2.circle(epipolar_views[1], tuple(np.rint(right_uv).astype(int)), 5, color, -1, cv2.LINE_AA)
                        epiline(epipolar_views[1], F @ np.r_[left_uv, 1.0], color)
                epipolar_panel = np.hstack(epipolar_views)
                cv2.rectangle(epipolar_panel, (0, 0), (2160, 65), (20, 20, 20), -1)
                cv2.putText(epipolar_panel, f"{name} | frame {det['frame_idx']}",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(epipolar_panel, "Same colors: left landmark, detected right landmark, expected right epipolar line",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 1, cv2.LINE_AA)
                epipolar_panels.append(epipolar_panel)
            epipolar_full = np.vstack(epipolar_panels)
            if epipolar_writer is None:
                epipolar_writer = cv2.VideoWriter(
                    str(output / "epipolar_geometry.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (epipolar_full.shape[1], epipolar_full.shape[0]),
                )
                if not epipolar_writer.isOpened():
                    raise ValueError("Cannot open epipolar diagnostic video output")
            epipolar_writer.write(epipolar_full)
            if i in indices:
                epipolar_samples.append(cv2.resize(epipolar_full, (1080, 810)))
            full = np.vstack(panels)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output / "stereo_reprojection.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (full.shape[1], full.shape[0]),
                )
                if not writer.isOpened():
                    raise ValueError("Cannot open diagnostic video output")
            writer.write(full)
            if i in indices:
                samples.append(cv2.resize(full, (1080, 810)))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if epipolar_writer is not None:
            epipolar_writer.release()
    if samples and not cv2.imwrite(str(output / "stereo_reprojection_contact_sheet.jpg"), np.vstack(samples)):
        raise ValueError("Cannot write stereo diagnostic sheet")
    if epipolar_samples and not cv2.imwrite(
        str(output / "epipolar_geometry_contact_sheet.jpg"), np.vstack(epipolar_samples)
    ):
        raise ValueError("Cannot write epipolar diagnostic sheet")


def render_stereo_hand_comparison(session, matches, root, camera_to_head, output):
    """Render native baseline above filtered stereo labels for manual review."""
    from .humanego_writer import _pose_matrix
    from .stereo_geometry import MP_BONES
    from .video_reader import crop_side_by_side
    from preprocess.AriaHandsOps import AriaHandsOps

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(session.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(output / "native_vs_filtered_stereo.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (2160, 1620),
    )
    if not cap.isOpened() or not writer.isOpened():
        cap.release(); writer.release()
        raise ValueError("Cannot open filtered-hand comparison video")
    sample_indices = set(np.linspace(0, len(matches) - 1, 6, dtype=int))
    samples = []
    colors = {"hand_l": (255, 80, 220), "hand_r": (0, 220, 255)}
    try:
        for index, match in enumerate(matches):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(match["frame_index"]))
            ok, full = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {match['frame_index']}")
            tracking = session.tracking[match["tracking_index"]]
            H = _pose_matrix(tracking["head"])
            frame_dir = root / "preprocess/all_data" / f"{index:05d}"
            final = json.loads((frame_dir / "aria_hands.json").read_text())
            native = json.loads((frame_dir / "aria_hands_pico_native.json").read_text())
            native_views, stereo_views = [], []
            for eye in ("left", "right"):
                source = crop_side_by_side(full, eye)
                c2w = H @ camera_to_head[eye]
                native_view = _draw_full_hand_skeleton(source, native, {
                    "k": session.calibration.K, "c2w": c2w,
                })
                stereo_view = source.copy()
                for hand_key in ("hand_l", "hand_r"):
                    hand = final.get(hand_key)
                    if not hand:
                        continue
                    points = np.asarray(hand[f"filtered_reprojection_{eye}"], dtype=float)
                    visible = np.isfinite(points).all(axis=1)
                    color = colors[hand_key]
                    for a, b in MP_BONES:
                        if visible[a] and visible[b]:
                            cv2.line(stereo_view, tuple(np.rint(points[a]).astype(int)),
                                     tuple(np.rint(points[b]).astype(int)), color, 2, cv2.LINE_AA)
                    for point in points[visible]:
                        cv2.circle(stereo_view, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
                    pose_world = np.asarray(hand["midpoint_pose_opt_world"], dtype=float)
                    stereo_view = AriaHandsOps._draw_axis(
                        stereo_view, np.linalg.inv(c2w) @ pose_world,
                        session.calibration.K, np.zeros(5),
                    )
                    label = ("L" if hand_key == "hand_l" else "R") + \
                            f" grasp={hand['grasp_state']} ratio={hand['gripper_opening_ratio']:.2f}"
                    cv2.putText(stereo_view, label,
                                (15, 92 if hand_key == "hand_l" else 120),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
                native_views.append(native_view)
                stereo_views.append(stereo_view)
            native_panel, stereo_panel = np.hstack(native_views), np.hstack(stereo_views)
            for panel, text in ((native_panel, "PICO native 26-joint baseline"),
                                (stereo_panel, "MediaPipe same-frame stereo | interpolated + zero-lag smoothing")):
                cv2.rectangle(panel, (0, 0), (2160, 65), (20, 20, 20), -1)
                cv2.putText(panel, f"{text} | source frame {match['frame_index']}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, "Left eye | Right eye", (12, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 1, cv2.LINE_AA)
            comparison = np.vstack([native_panel, stereo_panel])
            writer.write(comparison)
            if index in sample_indices:
                samples.append(cv2.resize(comparison, (1080, 810)))
    finally:
        cap.release(); writer.release()
    if samples and not cv2.imwrite(str(output / "native_vs_filtered_stereo_contact_sheet.jpg"), np.vstack(samples)):
        raise ValueError("Cannot write filtered-hand contact sheet")


class _VideoWriter:
    """Streaming MP4 writer; keeps a full recording out of RAM."""

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = float(fps or 30.0)
        self.writer = None

    def write(self, frame: np.ndarray | None):
        if frame is None:
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = np.zeros_like(image)
    red[:, :, 2] = mask
    return cv2.addWeighted(image, 0.75, red, 0.45, 0.0)


def _draw_full_hand_skeleton(image: np.ndarray, hands: dict, camera: dict) -> np.ndarray:
    """Draw the tracked PICO 26-joint skeleton in the camera image."""
    canvas = image.copy()
    K = np.asarray(camera["k"], dtype=np.float64)
    T_w2c = np.linalg.inv(np.asarray(camera["c2w"], dtype=np.float64))
    h, w = canvas.shape[:2]
    palette = {"hand_r": (0, 220, 255), "hand_l": (255, 80, 220)}

    for side in ("hand_r", "hand_l"):
        hand = hands.get(side)
        if not hand:
            continue
        joints = hand.get("pico_joints_26") or []
        if len(joints) != 26:
            continue
        world = np.asarray([j["position"] for j in joints], dtype=np.float64)
        cam = (T_w2c[:3, :3] @ world.T).T + T_w2c[:3, 3]
        uvh = (K @ cam.T).T
        valid = cam[:, 2] > 1e-4
        uv = np.zeros((26, 2), dtype=np.float64)
        uv[valid] = uvh[valid, :2] / uvh[valid, 2:3]
        color = palette[side]

        chains = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        chains += [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        chains += [(0, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
        chains += [(0, 16), (16, 17), (17, 18), (18, 19), (19, 20)]
        chains += [(0, 21), (21, 22), (22, 23), (23, 24), (24, 25)]
        for a, b in chains:
            if valid[a] and valid[b]:
                p1 = tuple(np.round(uv[a]).astype(int))
                p2 = tuple(np.round(uv[b]).astype(int))
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        for i, point in enumerate(uv):
            if valid[i] and 0 <= point[0] < w and 0 <= point[1] < h:
                p = tuple(np.round(point).astype(int))
                cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, p, 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, side.replace("hand_", "").upper(),
                    (20, 30 if side == "hand_r" else 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas


def _arm_dino_cfg(base_cfg: Path, output_path: Path) -> Path:
    text = base_cfg.read_text(encoding="utf-8")
    prefix = text.split("dinosam_prompt:", 1)[0]
    output_path.write_text(
        prefix + 'dinosam_prompt:\n  arm: "human arms . human hands ."\n',
        encoding="utf-8",
    )
    return output_path


def _protect_tracked_objects_from_arm_mask(
    root: Path, paths: list[Path], overwrite_raw: bool = False
) -> None:
    """Produce the Aria-style arm-only mask expected by the unchanged LaMa stage."""
    result_path = root / "preprocess" / "cotracker_results.json"
    if not result_path.exists():
        return
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    for frame_index, image_path in enumerate(paths):
        mask_path = image_path.parent / "mask_arm.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Keep the raw DINO result for diagnosing accidental over-segmentation.
        raw_path = image_path.parent / "mask_arm_dino.png"
        if overwrite_raw or not raw_path.exists():
            cv2.imwrite(str(raw_path), mask)

        protected = np.zeros_like(mask)
        for obj_key in ("obj1", "obj2"):
            obj = result.get(obj_key, {})
            tracks = np.asarray(obj.get("tracks", []), dtype=np.float32)
            visibility = np.asarray(obj.get("visibility", []), dtype=np.float32)
            if tracks.ndim != 3 or frame_index >= len(tracks):
                continue
            points = tracks[frame_index]
            if visibility.ndim == 2 and frame_index < len(visibility):
                points = points[visibility[frame_index] > 0.5]
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) < 3:
                continue
            hull = cv2.convexHull(np.round(points).astype(np.int32))
            cv2.fillConvexPoly(protected, hull, 255)

        # Match the Aria-side mask margin: 9x9 morphology before the
        # unchanged LaMa stage applies its own 9x9 dilation twice.
        protected = cv2.dilate(protected, np.ones((9, 9), np.uint8), iterations=1)
        mask[protected > 0] = 0
        cv2.imwrite(str(mask_path), mask)


def run_visual_preprocess(
    root: str | Path,
    dino_cfg: str | Path,
    lama_cfg: str | Path,
    visualkpts_cfg: str | Path,
    force: bool = False,
) -> dict:
    """Generate PICO image variants and HumanEgo-compatible visual clips."""
    root = Path(root)
    paths = sorted((root / "preprocess" / "all_data").glob("[0-9]*/rgb.png"))
    if not paths:
        raise FileNotFoundError(f"no RGB frames under {root / 'preprocess' / 'all_data'}")
    preprocess = root / "preprocess"
    vis_dir = preprocess / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((preprocess / "aria_cam_rgb_config.json").read_text(encoding="utf-8"))
    fps = float(meta.get("fps", 30.0))

    mask_paths = [p.parent / "mask_arm.png" for p in paths]
    mask_video = _VideoWriter(vis_dir / "mask_vis.mp4", fps)
    try:
        if force or not all(p.exists() for p in mask_paths):
            arm_cfg = _arm_dino_cfg(Path(dino_cfg), preprocess / "pico_arm_DINOSAM_runtime.yaml")
            print(f"[PICO visuals] DINO-SAM2 arm masks: {len(paths)} frames")
            for image_path in paths:
                mask_video.write(run_dinosam(cfg_path=str(arm_cfg), image_path=str(image_path)))
        else:
            print("[PICO visuals] all arm masks already exist; skipping DINO-SAM2")
            for image_path, mask_path in zip(paths, mask_paths):
                image = cv2.imread(str(image_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_video.write(_mask_overlay(image, mask))
    finally:
        close_dinosam_worker()
        mask_video.close()

    _protect_tracked_objects_from_arm_mask(root, paths, overwrite_raw=force)

    lama_video = _VideoWriter(vis_dir / "lama_vis.mp4", fps)
    try:
        print(f"[PICO visuals] LaMa arm removal: {len(paths)} frames")
        for index, image_path in enumerate(paths):
            output_path = image_path.parent / "rgb_WoArm.png"
            if force or not output_path.exists():
                diagnostic = run_lama(str(image_path), str(lama_cfg), index)
            else:
                diagnostic = cv2.hconcat([
                    cv2.imread(str(image_path)), cv2.imread(str(output_path))
                ])
            lama_video.write(diagnostic)
    finally:
        close_lama_worker()
        lama_video.close()

    reset_visualkpts()
    original_video = _VideoWriter(vis_dir / "aria_cam_rgb.mp4", fps)
    overlay_video = _VideoWriter(vis_dir / "aria_vis.mp4", fps)
    clean_overlay_video = _VideoWriter(vis_dir / "visualkpts_vis.mp4", fps)
    full_hand_video = _VideoWriter(vis_dir / "aria_vis_full_hands.mp4", fps)
    try:
        print(f"[PICO visuals] VisualKpts overlays: {len(paths)} frames")
        path_list = [str(p) for p in paths]
        for index, image_path in enumerate(paths):
            image = cv2.imread(str(image_path))
            frame_dir = image_path.parent
            camera = json.loads((frame_dir / "aria_cam_rgb.json").read_text(encoding="utf-8"))
            hands = json.loads((frame_dir / "aria_hands.json").read_text(encoding="utf-8"))
            run_visualkpts(str(image_path), str(visualkpts_cfg), index, path_list, str(root))
            original_video.write(image)
            overlay_video.write(cv2.imread(str(frame_dir / "rgb_WArmObjKpts.png")))
            clean_overlay_video.write(cv2.imread(str(frame_dir / "rgb_WoArm_WArmObjKpts.png")))
            full_hand_video.write(_draw_full_hand_skeleton(image, hands, camera))
    finally:
        original_video.close()
        overlay_video.close()
        clean_overlay_video.close()
        full_hand_video.close()

    return {
        "status": "PASS",
        "frame_count": len(paths),
        "fps": fps,
        "files_per_frame": [
            "rgb.png", "rgb_WoArm.png", "rgb_WArmObjKpts.png", "rgb_WoArm_WArmObjKpts.png",
            "mask_arm_dino.png",
        ],
        "videos": [
            "vis/aria_cam_rgb.mp4", "vis/mask_vis.mp4", "vis/lama_vis.mp4",
            "vis/aria_vis.mp4", "vis/aria_vis_full_hands.mp4", "vis/visualkpts_vis.mp4",
        ],
    }


def main(argv=None):
    import argparse

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate PICO HumanEgo visual assets and MP4 diagnostics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--dino-config", type=Path,
        default=project_root / "cfg" / "preprocess" / "base" / "PicoCanTrayDINOSAM.yaml",
    )
    parser.add_argument(
        "--lama-config", type=Path,
        default=project_root / "cfg" / "preprocess" / "base" / "Lama.yaml",
    )
    parser.add_argument(
        "--visualkpts-config", type=Path,
        default=project_root / "cfg" / "preprocess" / "base" / "VisualKpts.yaml",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = run_visual_preprocess(args.input, args.dino_config, args.lama_config, args.visualkpts_config, args.force)
    out = args.input / "preprocess" / "pico_visuals_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
