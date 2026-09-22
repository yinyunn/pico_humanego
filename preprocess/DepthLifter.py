# -*- coding: utf-8 -*-
# @FileName: DepthLifter.py

"""
====================================================================================================
Single-View Depth Lifting & Pose Estimation Pipeline (DepthLifter.py)
====================================================================================================

Description:
    This script replaces CamTriangulator for robot (RealSense) data. Instead of multi-view
    triangulation, it uses single-camera depth maps + camera intrinsics to lift 2D CoTracker
    keypoints into 3D, then estimates 6-DOF object poses using the same OrientAnything methods
    available in CamTriangulator.

    Because the robot setup uses a static RealSense camera, the camera-to-world transform is
    the identity matrix (c2w = I). Depth maps are stored as uint16 PNG images in millimetres.

Core Functionalities:
    1.  Depth-Based 3D Lifting: Back-projects 2D pixel tracks into 3D using per-pixel depth
        values from RealSense depth maps and known camera intrinsics.
    2.  Median Filtering: Optionally aggregates depth readings across multiple visible frames
        to suppress sensor noise and transient depth dropouts.
    3.  Pose Estimation: Computes 6-DOF object frames using three available methods:
        - pca1: Star-shaped Relational Pose Estimator (Y-axis = physical normal).
        - pca2: Object-Centric Relational Estimator (Handles vertical/horizontal geometries).
        - vlm:  Semantic Pose Estimator using Orient-Anything V2.
    4.  Relational Constraints: Forces the X-axis of context objects to point toward the Anchor.
    5.  3D Export & QA Rendering: Exports Open3D `.ply` files and renders HUD overlays.

Data Layout (Robot / RealSense):
    {session_dir}/aria/
        session_meta.json          <- camera intrinsics (k), image size (w, h), c2w (identity)
        cotracker_results.json     <- 2D tracks and visibility from CoTracker
        all_data/{idx:05d}/
            rgb.png                <- colour frame
            depth.png              <- uint16 depth in millimetres
            robot_state.json       <- joint states (not used here)

Generated Outputs:
    - {session_dir}/aria/depthlifter_results.json  : 3D points and 6-DOF matrices per object.
    - {session_dir}/aria/depthlifter_vis.ply       : 3D mesh containing points and coordinate axes.
    - {session_dir}/aria/depthlifter_vis.png       : 2D QA image overlay with axes and labels.

Config Keys (DepthLifter.yaml):
    - pose_method           : "vlm" | "pca1" | "pca2"  (default: "vlm")
    - depth_median_frames   : int   (number of visible frames to median-filter, default: 5)
    - min_valid_depth_m     : float (minimum acceptable depth in metres, default: 0.1)
    - max_valid_depth_m     : float (maximum acceptable depth in metres, default: 3.0)
    - pca_anisotropy_threshold : float (default: 0.08)
====================================================================================================
"""

import os
import cv2
import json
import logging
import open3d as o3d
import numpy as np
from typing import Optional, Tuple, List, Dict
from PIL import Image
import torch

from utils.utils_vis import draw_glass_rect
from utils.utils_io import load_cfg

from preprocess.OrientAnything import (
    estimate_frame_pca1,
    estimate_frame_pca2,
    estimate_frame_vlm,
    get_crop_from_2d_kpts,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# [Engine]
# ==============================================================================
class DepthLifterEngine:
    """Low-level depth lifting operations: pixel unproject and median filtering."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.depth_median_frames = getattr(cfg, "depth_median_frames", 5)
        self.min_valid_depth_m = getattr(cfg, "min_valid_depth_m", 0.1)
        self.max_valid_depth_m = getattr(cfg, "max_valid_depth_m", 3.0)

    def _load_depth(self, depth_path: str) -> Optional[np.ndarray]:
        """
        Load a uint16 depth image and convert to metres.

        Args:
            depth_path: Absolute path to the depth.png file (uint16, millimetres).

        Returns:
            (H, W) float64 array in metres, or None if the file is missing / corrupt.
        """
        if not os.path.exists(depth_path):
            logger.warning("Depth file not found: %s", depth_path)
            return None
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            logger.warning("Failed to decode depth file: %s", depth_path)
            return None
        depth_m = depth_mm.astype(np.float64) / 1000.0
        return depth_m

    def _is_valid_depth(self, z: float) -> bool:
        """Check whether a depth reading is within the acceptable range."""
        return self.min_valid_depth_m <= z <= self.max_valid_depth_m

    def unproject_pixel(self, u: float, v: float, depth_m: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
        """
        Lift a single (u, v) pixel to 3D using the corresponding depth map.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            depth_m: (H, W) depth map in metres.
            K: (3, 3) camera intrinsic matrix.

        Returns:
            (3,) numpy array [X, Y, Z] in camera frame, or None if depth is invalid.
        """
        H, W = depth_m.shape[:2]
        vi, ui = int(round(v)), int(round(u))

        # Bounds check
        if vi < 0 or vi >= H or ui < 0 or ui >= W:
            return None

        Z = depth_m[vi, ui]
        if Z <= 0 or not self._is_valid_depth(Z):
            return None

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=np.float64)

    def lift_keypoint_median(
        self,
        kpt_tracks: np.ndarray,
        kpt_vis: np.ndarray,
        all_depth_paths: List[str],
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Lift a single keypoint to 3D by median-filtering depth across multiple visible frames.

        Args:
            kpt_tracks: (T, 2) per-frame pixel positions for this keypoint.
            kpt_vis:    (T,)   per-frame visibility flags (0 or 1).
            all_depth_paths: List of depth.png paths, one per frame.
            K: (3, 3) camera intrinsic matrix.

        Returns:
            (3,) median-filtered 3D point in camera coordinates, or None if lifting failed.
        """
        candidates = []
        visible_indices = np.where(kpt_vis > 0)[0]

        for idx in visible_indices:
            if len(candidates) >= self.depth_median_frames:
                break

            if idx >= len(all_depth_paths):
                continue

            depth_m = self._load_depth(all_depth_paths[idx])
            if depth_m is None:
                continue

            u, v = kpt_tracks[idx]
            pt3d = self.unproject_pixel(u, v, depth_m, K)
            if pt3d is not None:
                candidates.append(pt3d)

        if len(candidates) == 0:
            return None

        return np.median(np.array(candidates), axis=0)

    def lift_keypoint_single(
        self,
        kpt_tracks: np.ndarray,
        kpt_vis: np.ndarray,
        all_depth_paths: List[str],
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Lift a single keypoint to 3D using the first visible frame with valid depth.

        Args:
            kpt_tracks: (T, 2) per-frame pixel positions for this keypoint.
            kpt_vis:    (T,)   per-frame visibility flags (0 or 1).
            all_depth_paths: List of depth.png paths, one per frame.
            K: (3, 3) camera intrinsic matrix.

        Returns:
            (3,) 3D point in camera coordinates, or None if lifting failed.
        """
        visible_indices = np.where(kpt_vis > 0)[0]

        for idx in visible_indices:
            if idx >= len(all_depth_paths):
                continue

            depth_m = self._load_depth(all_depth_paths[idx])
            if depth_m is None:
                continue

            u, v = kpt_tracks[idx]
            pt3d = self.unproject_pixel(u, v, depth_m, K)
            if pt3d is not None:
                return pt3d

        return None


# ==============================================================================
# [Manager]
# ==============================================================================
class DepthLifterManager:
    """Manages single-view depth lifting, pose estimation, and visualization for robot data."""

    def __init__(self):
        self.objects_3d = None
        self.T_o2c0 = None
        self.T_o2w_vis = None
        self.engine = None
        self.cfg = None
        self.T_limit = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self, image_path, cfg_path, frame_idx, all_image_paths, session_path):
        """
        Main entry point. On the first call (frame_idx == 0), performs full depth lifting
        and pose estimation for all objects. On subsequent calls, returns per-frame QA
        visualization overlays.

        Args:
            image_path:      Path to the current frame's rgb.png.
            cfg_path:        Path to DepthLifter.yaml config file.
            frame_idx:       Current frame index (0-based).
            all_image_paths: List of all rgb.png paths in the session.
            session_path:    Root session directory (contains aria/ subfolder).
        """
        # ==================================================================
        # 1) INIT — Executes only once at sequence start
        # ==================================================================
        if self.objects_3d is None:
            cfg = load_cfg(cfg_path)
            self.cfg = cfg
            self.engine = DepthLifterEngine(cfg)

            # ---[ Pose Method Switch ] ---
            pose_method_cfg = getattr(cfg, "pose_method", "vlm")
            if isinstance(pose_method_cfg, dict) or hasattr(pose_method_cfg, "keys"):
                print(f"║ [DepthLifter] Pose Estimation Methods: {dict(pose_method_cfg)}")
            else:
                print(f"║ [DepthLifter] Pose Estimation Method (Global): {str(pose_method_cfg).upper()}")

            # ---[ Load CoTracker Results ] ---
            tracks_path = os.path.join(session_path, "preprocess", "cotracker_results.json")
            if not os.path.exists(tracks_path):
                raise FileNotFoundError(f"Missing cotracker results at {tracks_path}")

            with open(tracks_path, "r") as f:
                cotracker_data = json.load(f)

            # ---[ Load Camera Intrinsics from session_meta.json ] ---
            meta_path = os.path.join(session_path, "preprocess", "session_meta.json")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing session_meta.json at {meta_path}")

            with open(meta_path, "r") as f:
                session_meta = json.load(f)

            # Support both flat layout {"k":..., "w":..., "h":...} and
            # nested layout {"camera": {"k_rgb":..., "w":..., "h":...}}
            if "k" in session_meta:
                K = np.array(session_meta["k"], dtype=np.float64)
                img_w = session_meta.get("w", None)
                img_h = session_meta.get("h", None)
            elif "camera" in session_meta:
                cam = session_meta["camera"]
                K = np.array(cam.get("k_rgb", cam.get("k")), dtype=np.float64)
                img_w = cam.get("w", None)
                img_h = cam.get("h", None)
            else:
                raise KeyError("session_meta.json has no 'k' or 'camera.k_rgb' field for intrinsics")

            # Camera frame = World frame (static RealSense, identity c2w)
            self.T_c02w = np.eye(4, dtype=np.float64)
            self.T_w2c0 = np.eye(4, dtype=np.float64)

            T_provided = len(all_image_paths)

            # ---[ Build Depth Paths ] ---
            all_depth_paths = []
            for rgb_p in all_image_paths:
                depth_p = rgb_p.replace("rgb.png", "depth.png")
                all_depth_paths.append(depth_p)

            # ---[ VLM image for cropping ] ---
            needs_vlm = False
            if isinstance(pose_method_cfg, dict) or hasattr(pose_method_cfg, "keys"):
                pm_dict = dict(pose_method_cfg) if hasattr(pose_method_cfg, "keys") else pose_method_cfg
                if "vlm" in pm_dict.values() or pm_dict.get("default", "vlm") == "vlm":
                    needs_vlm = True
            elif str(pose_method_cfg).lower() == "vlm":
                needs_vlm = True

            cam0_rgb_full = None
            if needs_vlm:
                cam0_raw = cv2.imread(all_image_paths[0])
                if cam0_raw is not None:
                    cam0_rgb_full = cv2.cvtColor(cam0_raw, cv2.COLOR_BGR2RGB)

            # ---[ Extract Object Keys ] ---
            obj_keys = sorted([k for k in cotracker_data.keys() if k.startswith("obj")])
            if not obj_keys:
                raise ValueError("No 'objx' keys found in cotracker_results.json!")

            self.objects_3d = {}
            global_T_limit = 0
            anchor_key = obj_keys[0]  # Primary Anchor Object
            anchor_center_cam = None

            # -----------------------------------------------------------------
            # Internal Helper: Crop object image (VLM mode only)
            # -----------------------------------------------------------------
            def get_crop_from_tracks(full_img, obj_key_name, pad=40):
                if full_img is None:
                    return None
                tracks_raw = np.array(cotracker_data[obj_key_name]["tracks"])
                vis_raw = np.array(cotracker_data[obj_key_name]["visibility"])

                # Use frame 0 keypoints
                if tracks_raw.ndim == 3:
                    d0, d1, _ = tracks_raw.shape
                    if d0 > d1:
                        # (N_frames, N_kpts, 2) → frame 0 = all kpts at t=0
                        tracks_0 = tracks_raw[0, :, :]
                        vis_0 = vis_raw[0, :]
                    else:
                        # (N_kpts, N_frames, 2) → frame 0 = kpt[:, 0]
                        tracks_0 = tracks_raw[:, 0, :]
                        vis_0 = vis_raw[:, 0]
                else:
                    tracks_0 = tracks_raw[0]
                    vis_0 = vis_raw[0]

                valid_pts = tracks_0[vis_0 > 0]
                if len(valid_pts) == 0:
                    return Image.fromarray(full_img)

                x1, y1 = np.min(valid_pts, axis=0) - pad
                x2, y2 = np.max(valid_pts, axis=0) + pad

                h, w = full_img.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                return Image.fromarray(full_img[y1:y2, x1:x2])

            # -----------------------------------------------------------------
            # Internal Helper: Depth Lifting for a Single Object
            # -----------------------------------------------------------------
            def process_single_object_tracks(obj_key_to_process):
                nonlocal global_T_limit
                print(f"║ [DepthLifter] Processing {obj_key_to_process} 3D points...")

                tracks_raw = np.array(cotracker_data[obj_key_to_process]["tracks"])
                vis_raw = np.array(cotracker_data[obj_key_to_process]["visibility"])

                if tracks_raw.ndim != 3:
                    raise ValueError(
                        f"Unexpected tracks shape {tracks_raw.shape} for {obj_key_to_process}. "
                        f"Expected 3D array."
                    )

                # CoTracker output: (N_frames, N_keypoints, 2)
                # We need:           (N_keypoints, N_frames, 2)
                d0, d1, _ = tracks_raw.shape
                if d0 > d1:
                    # (N_frames, N_kpts, 2) → transpose to (N_kpts, N_frames, 2)
                    tracks_all = tracks_raw.transpose(1, 0, 2)
                    vis_all = vis_raw.T
                else:
                    tracks_all = tracks_raw
                    vis_all = vis_raw

                N_kpts, N_frames, _ = tracks_all.shape

                if N_frames > global_T_limit:
                    global_T_limit = N_frames

                use_median = getattr(cfg, "depth_median_frames", 5) > 1

                pts_3d = []
                n_failed = 0
                for n in range(N_kpts):
                    kpt_tracks = tracks_all[n]   # (N_frames, 2)
                    kpt_vis = vis_all[n]          # (N_frames,)

                    if use_median:
                        pt3d = self.engine.lift_keypoint_median(
                            kpt_tracks, kpt_vis, all_depth_paths, K
                        )
                    else:
                        pt3d = self.engine.lift_keypoint_single(
                            kpt_tracks, kpt_vis, all_depth_paths, K
                        )

                    if pt3d is not None:
                        pts_3d.append(pt3d)
                    else:
                        n_failed += 1
                        logger.warning(
                            "Could not lift keypoint %d of %s (no valid depth in any visible frame).",
                            n, obj_key_to_process
                        )

                if n_failed > 0:
                    print(f"║ [DepthLifter] WARNING: {n_failed}/{N_kpts} keypoints of "
                          f"{obj_key_to_process} could not be lifted.")

                if len(pts_3d) < 3:
                    raise ValueError(
                        f"║ [DepthLifter Error] {obj_key_to_process} lifted points < 3! "
                        f"({len(pts_3d)} succeeded out of {N_kpts})"
                    )

                pts_3d = np.array(pts_3d, dtype=np.float64)

                # Since c2w = identity, world coords == camera coords
                pts_3d_world = pts_3d.copy()
                pts_3d_cam0 = pts_3d.copy()
                return pts_3d_world, pts_3d_cam0

            # -----------------------------------------------------------------
            # Core Loop: Compute poses for all objects
            # -----------------------------------------------------------------
            for i, obj_key in enumerate(obj_keys):
                is_anchor = (obj_key == anchor_key)
                pts_w, pts_c0 = process_single_object_tracks(obj_key)
                t_c0 = pts_c0.mean(axis=0)

                # Determine per-object pose method dynamically
                if isinstance(pose_method_cfg, dict) or hasattr(pose_method_cfg, "keys"):
                    pm_dict = dict(pose_method_cfg) if hasattr(pose_method_cfg, "keys") else pose_method_cfg
                    curr_pose_method = pm_dict.get(obj_key, pm_dict.get("default", "vlm")).lower()
                else:
                    curr_pose_method = str(pose_method_cfg).lower()

                if curr_pose_method == "vlm":
                    crop_img = get_crop_from_tracks(cam0_rgb_full, obj_key)
                    T_o2c0, info = estimate_frame_vlm(
                        image=crop_img,
                        t_cam=t_c0,
                        is_anchor=is_anchor,
                        anchor_center_cam=anchor_center_cam,
                        do_rm_bkg=True,
                    )
                elif curr_pose_method == "pca1":
                    T_o2c0, info = estimate_frame_pca1(
                        pts_cam=pts_c0,
                        is_anchor=is_anchor,
                        anchor_center_cam=anchor_center_cam,
                    )
                elif curr_pose_method == "pca2":
                    T_o2c0, info = estimate_frame_pca2(
                        pts_cam=pts_c0,
                        is_anchor=is_anchor,
                        anchor_center_cam=anchor_center_cam,
                    )
                else:
                    raise ValueError(f"Unknown pose_method '{curr_pose_method}' for object '{obj_key}'")

                # If this is the anchor, record its center for subsequent context objects
                if is_anchor:
                    anchor_center_cam = T_o2c0[:3, 3]

                self.objects_3d[obj_key] = {
                    "points_3d_world": pts_w,
                    "points_3d_cam0": pts_c0,
                    "T_o2c0": T_o2c0,
                    "T_o2w_vis": self.T_c02w @ T_o2c0,
                    "frame_info": info,
                }

            self.T_limit = global_T_limit

            # ------------- Save Results & Export 3D -------------
            res_dict = {
                "cam0_c2w": self.T_c02w.tolist(),
                "objects": {},
            }
            for k, v in self.objects_3d.items():
                res_dict["objects"][k] = {
                    "points_3d_world": v["points_3d_world"].tolist(),
                    "points_3d_cam0": v["points_3d_cam0"].tolist(),
                    "object_to_cam0_matrix": v["T_o2c0"].tolist(),
                    "info": v["frame_info"],
                }

            json_path = os.path.join(session_path, "preprocess", "depthlifter_results.json")
            with open(json_path, "w") as f:
                json.dump(res_dict, f, indent=4)
            print(f"║ [DepthLifter] Results saved to: {json_path}")

            self._export_depthlifter_vis_ply(session_path, cfg)

        # ==================================================================
        # 2) Render QA Visualization Overlay per frame
        # ==================================================================
        img = cv2.imread(image_path)
        if img is None or frame_idx >= self.T_limit:
            return img

        # For static RealSense: w2c = identity, K from session_meta
        meta_path = os.path.join(session_path, "preprocess", "session_meta.json")
        with open(meta_path, "r") as f:
            session_meta = json.load(f)
        if "k" in session_meta:
            K = np.array(session_meta["k"], dtype=np.float64)
        elif "camera" in session_meta:
            cam = session_meta["camera"]
            K = np.array(cam.get("k_rgb", cam.get("k")), dtype=np.float64)
        else:
            raise KeyError("session_meta.json missing intrinsics")
        w2c = np.eye(4, dtype=np.float64)

        vis_img = self._draw_qa(img, w2c, K, frame_idx)

        # Save the 2D preview image on the last frame
        if frame_idx == self.T_limit - 1:
            save_p = os.path.join(session_path, "preprocess", "depthlifter_vis.png")
            cv2.imwrite(save_p, vis_img)

        return vis_img

    # ------------------------------------------------------------------
    # 3D Export
    # ------------------------------------------------------------------
    def _export_depthlifter_vis_ply(self, session_path, cfg):
        """Export PLY containing all objects in camera0 coordinates."""
        geos = []
        obj_colors = [[0.2, 0.9, 0.2], [0.2, 0.8, 0.9], [0.9, 0.2, 0.8], [0.9, 0.2, 0.2]]

        point_radius = getattr(cfg, "point_radius_m", 0.003)
        axes_len = getattr(cfg, "axes_len_m", 0.05)

        for idx, (obj_key, data) in enumerate(self.objects_3d.items()):
            pts = data["points_3d_cam0"]
            T = data["T_o2c0"]
            origin = T[:3, 3]
            base_color = obj_colors[idx % len(obj_colors)]

            # 1) Keypoints
            for pt in pts:
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=point_radius)
                sph.translate(pt)
                sph.paint_uniform_color(base_color)
                geos.append(sph)

            # 2) Origin marker
            center_sph = o3d.geometry.TriangleMesh.create_sphere(radius=point_radius * 1.6)
            center_sph.translate(origin)
            center_sph.paint_uniform_color([1.0, 0.8, 0.0])  # Yellow center
            geos.append(center_sph)

            # 3) Axes arrows
            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for i in range(3):
                axis_vec = T[:3, i]
                arrow = self._create_arrow(origin, origin + axis_vec * axes_len, axis_colors[i])
                geos.append(arrow)

        combined = o3d.geometry.TriangleMesh()
        for g in geos:
            combined += g
        save_p = os.path.join(session_path, "preprocess", "depthlifter_vis.ply")
        o3d.io.write_triangle_mesh(save_p, combined)
        print(f"║ [DepthLifter] Multi-object PLY saved to: {save_p}")

    # ------------------------------------------------------------------
    # Arrow helper
    # ------------------------------------------------------------------
    def _create_arrow(self, start, end, color):
        """Create an Open3D arrow mesh between two 3D points."""
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return o3d.geometry.TriangleMesh()
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002,
            cone_radius=0.005,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2,
        )
        arrow.paint_uniform_color(color)
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        rotation_matrix = self._get_rotation_matrix(z_axis, vec / length)
        arrow.rotate(rotation_matrix, center=(0, 0, 0))
        arrow.translate(start)
        return arrow

    @staticmethod
    def _get_rotation_matrix(vec1, vec2):
        """Compute rotation matrix that rotates vec1 to vec2 (Rodrigues formula)."""
        a = (vec1 / (np.linalg.norm(vec1) + 1e-12)).reshape(3)
        b = (vec2 / (np.linalg.norm(vec2) + 1e-12)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-8:
            return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-12))

    # ------------------------------------------------------------------
    # QA Overlay
    # ------------------------------------------------------------------
    def _draw_qa(self, img, w2c, K, frame_idx):
        """Draw 3D point projections and coordinate axes onto the current frame."""

        def proj(p_w):
            pc = w2c[:3, :3] @ p_w + w2c[:3, 3]
            if pc[2] < 1e-4:
                return None
            uv = K @ pc
            return (int(uv[0] / uv[2]), int(uv[1] / uv[2]))

        # Draw each object
        for obj_key, data in self.objects_3d.items():
            pts_w = data["points_3d_world"]
            T_o2w = data["T_o2w_vis"]

            # Draw point cloud
            for p in pts_w:
                uv = proj(p)
                if uv:
                    cv2.circle(img, uv, 4, (0, 255, 0), -1, cv2.LINE_AA)

            # Draw coordinate axes
            origin_w = T_o2w[:3, 3]
            uv0 = proj(origin_w)
            if uv0:
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: R, G, B
                labels = ["X", "Y", "Z"]
                for i in range(3):
                    ax_w = origin_w + T_o2w[:3, i] * 0.12
                    uv1 = proj(ax_w)
                    if uv1:
                        cv2.arrowedLine(img, uv0, uv1, colors[i], 2, tipLength=0.2, line_type=cv2.LINE_AA)
                        cv2.putText(
                            img, labels[i], (uv1[0] + 5, uv1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA,
                        )

                # Yellow dot at origin
                cv2.circle(img, uv0, 5, (0, 215, 255), -1)
                # Label the object key
                cv2.putText(
                    img, obj_key, (uv0[0] + 10, uv0[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

        draw_glass_rect(img, (10, 10), (360, 70))
        cv2.putText(
            img, f"DEPTH LIFTER QA ({len(self.objects_3d)} OBJS)", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            img, f"Frame: {frame_idx:04d}", (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return img


# ==============================================================================
# [Singleton + Public API]
# ==============================================================================
_MANAGER_INSTANCE = DepthLifterManager()


def reset_depthlifter():
    """Reset the singleton so a new session gets a fresh inference run."""
    global _MANAGER_INSTANCE
    _MANAGER_INSTANCE = DepthLifterManager()


def run_depthlifter(image_path, cfg_path, frame_idx, all_image_paths=None, session_path=None):
    """
    Public entry point for the Depth Lifter pipeline.

    Mirrors the CamTriangulator API: processes all objects on the first frame (frame_idx == 0),
    then returns per-frame QA visualizations for subsequent frames.

    Args:
        image_path:      Path to the current frame's rgb.png.
        cfg_path:        Path to DepthLifter.yaml config file.
        frame_idx:       Current frame index (0-based).
        all_image_paths: List of all rgb.png paths in the session.
        session_path:    Root session directory (contains aria/ subfolder).

    Returns:
        numpy.ndarray: BGR image with QA overlay, or None if the frame could not be loaded.
    """
    return _MANAGER_INSTANCE.process(image_path, cfg_path, frame_idx, all_image_paths, session_path)
