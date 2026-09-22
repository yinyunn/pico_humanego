# -*- coding: utf-8 -*-
# @FileName: CamTriangulator.py

"""
====================================================================================================
Project Aria Camera Triangulation & Pose Estimation Pipeline (CamTriangulator.py)
====================================================================================================

Description:
    This script converts 2D tracked keypoints (e.g., from CoTracker) into 3D spatial coordinates 
    using Multi-View Direct Linear Transformation (DLT) and Bundle Adjustment (BA) refinement. 
    It then estimates the 6-DOF pose of multiple objects using either purely geometric (PCA-based) 
    or semantic (Vision-Language Model, VLM) methods, establishing relational spatial contexts 
    between a main "Anchor" object and surrounding "Context" objects.

Core Functionalities:
    1.  Multi-View Triangulation: Reconstructs 3D points from 2D pixel tracks across frames.
    2.  Bundle Adjustment: Refines 3D point coordinates using Huber-loss optimization to 
        minimize reprojection errors.
    3.  Pose Estimation: Computes 6-DOF object frames using three available methods:
        - pca1: Star-shaped Relational Pose Estimator (Y-axis = physical normal).
        - pca2: Object-Centric Relational Estimator (Handles vertical/horizontal geometries).
        - vlm: Semantic Pose Estimator using Orient-Anything V2.
    4.  Relational Constraints: Forces the X-axis of context objects to point toward the Anchor.
    5.  3D Export & QA Rendering: Exports Open3D `.ply` files and renders HUD overlays.

Generated Outputs:
    - [mps_path]/aria/camtriangulator_results.json: 3D points and 6-DOF matrices per object.
    - [mps_path]/aria/camtriangulator_vis.ply: 3D mesh containing points and coordinate axes.
    - [mps_path]/aria/camtriangulator_vis.png: 2D QA image overlay with axes and labels.
====================================================================================================
"""

import os
import cv2
import json
import open3d as o3d
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import torch

from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from utils.utils_vis import draw_glass_rect
from utils.utils_io import load_cfg

from preprocess.OrientAnything import (
    estimate_frame_pca1,
    estimate_frame_pca2,
    estimate_frame_vlm,
    get_crop_from_2d_kpts,
)


# ==============================================================================
# [Engine]
# ==============================================================================
class CamTriangulatorEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def triangulate_dlt(poses, Ks, obs_uv, vis):
        """
        Multi-view DLT triangulation with normalized coordinates (K removed).
        Args:
            poses: list of c2w (4x4)
            Ks:    list of K (3x3)
            obs_uv: (T,2) pixel locations
            vis:    (T,) 0/1 visibility flags
        """
        A = []
        for i in range(len(poses)):
            if vis[i] == 0:
                continue
            u, v = obs_uv[i]
            fx, fy = Ks[i][0, 0], Ks[i][1, 1]
            cx, cy = Ks[i][0, 2], Ks[i][1, 2]
            un = (u - cx) / fx
            vn = (v - cy) / fy

            T_w2c = np.linalg.inv(poses[i])
            P = T_w2c[:3, :]  # K removed

            A.append(un * P[2, :] - P[0, :])
            A.append(vn * P[2, :] - P[1, :])

        if len(A) < 4:
            return np.zeros(3), 0.0

        u_mat, s, vh = np.linalg.svd(np.array(A))
        cond = s[0] / s[-1] if s[-1] != 0 else 0.0

        X = vh[-1]
        p3d = X[:3] / X[3]
        return p3d, cond

    def ba_refine(self, p3d, poses, Ks, obs_uv, vis):
        """Point-only Bundle Adjustment refinement with Huber loss."""
        def res(x):
            pts =[]
            for i in range(len(poses)):
                if vis[i] == 0:
                    continue
                T_w2c = np.linalg.inv(poses[i])
                pc = T_w2c[:3, :3] @ x + T_w2c[:3, 3]
                if pc[2] < 1e-4:
                    # Keep the residual length fixed across finite-difference
                    # evaluations.  Dropping a view here makes SciPy's
                    # least_squares receive a different-sized vector when a
                    # trial point crosses the camera plane.
                    pts.extend((1e3, 1e3))
                    continue
                uv = Ks[i] @ pc
                pts.extend((uv[:2] / uv[2]) - obs_uv[i])
            # SciPy's least_squares expects a one-dimensional residual vector.
            # ``pts.extend`` adds one 2D reprojection residual per observation,
            # so flatten the (N, 2) array before applying the robust loss.
            return np.asarray(pts, dtype=np.float64).reshape(-1)

        sol = least_squares(
            res,
            p3d,
            loss="huber",
            f_scale=self.cfg.ba_f_scale,
            max_nfev=60
        )
        return sol.x

# ==============================================================================
# [Manager]
# ==============================================================================
class CamTriangulatorManager:
    """Manages multi-object triangulation, pose estimation, and visualization."""
    def __init__(self):
        self.objects_3d = None 
        self.T_o2c0 = None
        self.T_o2w_vis = None  
        self.engine = None
        self.cfg = None
        self.T_limit = 0
        self.vlm_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def process(self, image_path, cfg_path, frame_idx, all_image_paths, mps_path):
        # 1) INIT (Executes only once at sequence start)
        if self.objects_3d is None:
            cfg = load_cfg(cfg_path)
            self.cfg = cfg
            self.engine = CamTriangulatorEngine(cfg)

            # ---[ Pose Method Switch ] ---
            pose_method_cfg = getattr(cfg, "pose_method", "vlm")
            if isinstance(pose_method_cfg, dict) or hasattr(pose_method_cfg, "keys"):
                print(f"║ [Triangulator] Pose Estimation Methods: {dict(pose_method_cfg)}")
            else:
                print(f"║ [Triangulator] Pose Estimation Method (Global): {str(pose_method_cfg).upper()}")

            step_size = getattr(cfg, "step", 1)
            print(f"║[Triangulator] Using step size: {step_size}")

            tracks_path = os.path.join(mps_path, "preprocess", "cotracker_results.json")
            if not os.path.exists(tracks_path):
                raise FileNotFoundError(f"Missing cotracker results at {tracks_path}")

            with open(tracks_path, 'r') as f:
                cotracker_data = json.load(f)

            T_provided = len(all_image_paths)
            relevant_image_paths = all_image_paths
            
           # Read full-resolution frame 0 for VLM cropping if ANY object needs it
            needs_vlm = False
            if isinstance(pose_method_cfg, dict) or hasattr(pose_method_cfg, "keys"):
                pm_dict = dict(pose_method_cfg) if hasattr(pose_method_cfg, "keys") else pose_method_cfg
                if "vlm" in pm_dict.values() or pm_dict.get("default", "vlm") == "vlm":
                    needs_vlm = True
            elif str(pose_method_cfg).lower() == "vlm":
                needs_vlm = True

            cam0_rgb_full = None
            if needs_vlm:
                cam0_raw = cv2.imread(relevant_image_paths[0])
                if cam0_raw is not None:
                    cam0_rgb_full = cv2.cvtColor(cam0_raw, cv2.COLOR_BGR2RGB)
            
            # Extract all object keys (obj1, obj2...)
            obj_keys = sorted([k for k in cotracker_data.keys() if k.startswith("obj")])
            if not obj_keys:
                raise ValueError("No 'objx' keys found in cotracker_results.json!")

            # Define Camera0 as the anchor reference frame
            cam0_p = relevant_image_paths[0].replace("rgb.png", "aria_cam_rgb.json")
            with open(cam0_p, "r") as f:
                cam0_d = json.load(f)
            self.T_c02w = np.array(cam0_d["c2w"], dtype=np.float64)
            self.T_w2c0 = np.linalg.inv(self.T_c02w)
            R_w2c0, t_w2c0 = self.T_w2c0[:3, :3], self.T_w2c0[:3, 3]

            self.objects_3d = {}
            global_T_limit = 0
            anchor_key = obj_keys[0] # Primary Anchor Object
            anchor_center_cam = None

            # -----------------------------------------------------------------
            # Internal Helper 1: Crop object image (VLM mode only)
            # -----------------------------------------------------------------
            def get_crop_from_tracks(full_img, obj_key_name, pad=40):
                if full_img is None: return None
                tracks_0 = np.array(cotracker_data[obj_key_name]["tracks"][0]) 
                vis_0 = np.array(cotracker_data[obj_key_name]["visibility"][0])
                
                valid_pts = tracks_0[vis_0 > 0]
                if len(valid_pts) == 0: return Image.fromarray(full_img)

                x1, y1 = np.min(valid_pts, axis=0) - pad
                x2, y2 = np.max(valid_pts, axis=0) + pad
                
                h, w = full_img.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                return Image.fromarray(full_img[y1:y2, x1:x2])

            # -----------------------------------------------------------------
            # Internal Helper 2: Triangulation Engine Process
            # -----------------------------------------------------------------
            def process_single_object_tracks(obj_key_to_process):
                nonlocal global_T_limit
                print(f"║ [Triangulator] Processing {obj_key_to_process} 3D points...")
                tracks_all = np.array(cotracker_data[obj_key_to_process]["tracks"])[:T_provided]
                vis_all = np.array(cotracker_data[obj_key_to_process]["visibility"])[:T_provided]

                T_tracked = tracks_all.shape[0]
                N = tracks_all.shape[1]
                if T_tracked > global_T_limit: global_T_limit = T_tracked

                # A) Trajectory Smoothing
                for n in range(N):
                    for d in range(2):
                        tracks_all[:, n, d] = savgol_filter(tracks_all[:, n, d], cfg.smooth_window, cfg.smooth_polyorder)

                # B) Subsampling
                # A manipulated object is not a static multi-view target over
                # the whole recording.  Configurable per-object windows allow
                # us to triangulate it from the stable pre-grasp segment.
                tri_ranges = getattr(cfg, "triangulation_ranges", {})
                obj_range = tri_ranges.get(obj_key_to_process) if hasattr(tri_ranges, "get") else None
                range_start = 0
                range_end = T_tracked - 1
                if obj_range is not None:
                    range_start = int(obj_range.get("start", range_start))
                    range_end = int(obj_range.get("end", range_end))
                range_start = max(0, min(range_start, T_tracked - 1))
                range_end = max(range_start, min(range_end, T_tracked - 1))
                indices = np.arange(range_start, range_end + 1, step_size).tolist()
                last_idx = range_end
                if last_idx not in indices: indices.append(last_idx)
                indices = sorted(indices)

                # C) Load Geometry Information
                sub_poses, sub_Ks = [],[]
                for i in indices:
                    cam_p = relevant_image_paths[i].replace("rgb.png", "aria_cam_rgb.json")
                    with open(cam_p, 'r') as f:
                        cam_d = json.load(f)
                    sub_poses.append(np.array(cam_d["c2w"], dtype=np.float64))
                    sub_Ks.append(np.array(cam_d["k"], dtype=np.float64))

                sub_tracks = tracks_all[indices]
                sub_vis = vis_all[indices]

                # D) Triangulation & BA Refinement
                pts_3d_world =[]
                for n in range(N):
                    p3d_init, c = self.engine.triangulate_dlt(sub_poses, sub_Ks, sub_tracks[:, n], sub_vis[:, n])
                    if np.allclose(p3d_init, 0.0): continue
                    p3d_refined = self.engine.ba_refine(p3d_init, sub_poses, sub_Ks, sub_tracks[:, n], sub_vis[:, n])
                    if np.allclose(p3d_refined, 0.0): continue
                    pts_3d_world.append(p3d_refined)

                if len(pts_3d_world) < 3:
                    raise ValueError(f"║[Triangulator Error] {obj_key_to_process} tri-pts < 3!")

                pts_3d_world = np.array(pts_3d_world, dtype=np.float64)
                pts_3d_cam0 = (R_w2c0 @ pts_3d_world.T + t_w2c0[:, None]).T
                return pts_3d_world, pts_3d_cam0

            # -----------------------------------------------------------------
            # Core Loop: Compute poses for all objects
            # -----------------------------------------------------------------
            for i, obj_key in enumerate(obj_keys):
                is_anchor = (obj_key == anchor_key)
                # Regardless of the pose method, translation comes from point cloud triangulation
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
                        do_rm_bkg=True
                    )
                elif curr_pose_method == "pca1":
                    T_o2c0, info = estimate_frame_pca1(
                        pts_cam=pts_c0, 
                        is_anchor=is_anchor, 
                        anchor_center_cam=anchor_center_cam
                    )
                elif curr_pose_method == "pca2":
                    T_o2c0, info = estimate_frame_pca2(
                        pts_cam=pts_c0, 
                        is_anchor=is_anchor, 
                        anchor_center_cam=anchor_center_cam
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
                    "frame_info": info
                }

            self.T_limit = global_T_limit

            # ------------- Save Results & Export 3D -------------
            res_dict = {
                "cam0_c2w": self.T_c02w.tolist(),
                "objects": {}
            }
            for k, v in self.objects_3d.items():
                res_dict["objects"][k] = {
                    "points_3d_world": v["points_3d_world"].tolist(),
                    "points_3d_cam0": v["points_3d_cam0"].tolist(),
                    "object_to_cam0_matrix": v["T_o2c0"].tolist(),
                    "info": v["frame_info"]
                }
                
            json_path = os.path.join(mps_path, "preprocess", "camtriangulator_results.json")
            with open(json_path, 'w') as f:
                json.dump(res_dict, f, indent=4)

            self._export_camtriangulator_vis_ply(mps_path, cfg)

        # 2) Render QA Visualization Overlay per frame
        img = cv2.imread(image_path)
        if img is None or frame_idx >= self.T_limit:
            return img

        # Load camera projection parameters for the current frame
        cam_p = image_path.replace("rgb.png", "aria_cam_rgb.json")
        with open(cam_p, 'r') as f:
            cam_d = json.load(f)
        w2c = np.linalg.inv(np.array(cam_d["c2w"], dtype=np.float64))
        K = np.array(cam_d["k"], dtype=np.float64)

        vis_img = self._draw_qa(img, w2c, K, frame_idx)

        # Save the 2D preview image on the last frame
        if frame_idx == self.T_limit - 1:
            save_p = os.path.join(mps_path, "preprocess", "camtriangulator_vis.png")
            cv2.imwrite(save_p, vis_img)

        return vis_img


    def _export_camtriangulator_vis_ply(self, mps_path, cfg):
        """ Export PLY containing all objects in camera0 coordinates. """
        geos =[]
        # Colors for different objects: Green, Cyan, Magenta, Red...
        obj_colors = [[0.2, 0.9, 0.2], [0.2, 0.8, 0.9],[0.9, 0.2, 0.8], [0.9, 0.2, 0.2]]
        
        for idx, (obj_key, data) in enumerate(self.objects_3d.items()):
            pts = data["points_3d_cam0"]
            T = data["T_o2c0"]
            origin = T[:3, 3]
            base_color = obj_colors[idx % len(obj_colors)]

            # 1) Keypoints
            for pt in pts:
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=cfg.point_radius_m)
                sph.translate(pt)
                sph.paint_uniform_color(base_color)
                geos.append(sph)

            # 2) Origin marker
            center_sph = o3d.geometry.TriangleMesh.create_sphere(radius=cfg.point_radius_m * 1.6)
            center_sph.translate(origin)
            center_sph.paint_uniform_color([1.0, 0.8, 0.0]) # Yellow center for all
            geos.append(center_sph)

            # 3) Axes arrows
            axis_colors = [[1, 0, 0], [0, 1, 0],[0, 0, 1]]
            for i in range(3):
                axis_vec = T[:3, i]
                arrow = self._create_arrow(origin, origin + axis_vec * cfg.axes_len_m, axis_colors[i])
                geos.append(arrow)

        combined = o3d.geometry.TriangleMesh()
        for g in geos: combined += g
        save_p = os.path.join(mps_path, "preprocess", "camtriangulator_vis.ply")
        o3d.io.write_triangle_mesh(save_p, combined)
        print(f"║ [Triangulator] Multi-object PLY saved to: {save_p}")

    def _create_arrow(self, start, end, color):
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6: return o3d.geometry.TriangleMesh()
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cone_radius=0.005, cylinder_height=length * 0.8, cone_height=length * 0.2)
        arrow.paint_uniform_color(color)
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        rotation_matrix = self._get_rotation_matrix(z_axis, vec / length)
        arrow.rotate(rotation_matrix, center=(0, 0, 0))
        arrow.translate(start)
        return arrow

    def _get_rotation_matrix(self, vec1, vec2):
        a = (vec1 / (np.linalg.norm(vec1) + 1e-12)).reshape(3)
        b = (vec2 / (np.linalg.norm(vec2) + 1e-12)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-8: return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-12))

    
    def _draw_qa(self, img, w2c, K, frame_idx):
        def proj(p_w):
            pc = w2c[:3, :3] @ p_w + w2c[:3, 3]
            if pc[2] < 1e-4: return None
            uv = K @ pc
            return (int(uv[0] / uv[2]), int(uv[1] / uv[2]))

        # Draw multiple objects
        for obj_key, data in self.objects_3d.items():
            pts_w = data["points_3d_world"]
            T_o2w = data["T_o2w_vis"]
            
            # Draw point cloud
            for i, p in enumerate(pts_w):
                uv = proj(p)
                if uv:
                    cv2.circle(img, uv, 4, (0, 255, 0), -1, cv2.LINE_AA)

            # Draw coordinate axes
            origin_w = T_o2w[:3, 3]
            uv0 = proj(origin_w)
            if uv0:
                colors =[(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR: R, G, B
                labels = ["X", "Y", "Z"] 
                for i in range(3):
                    ax_w = origin_w + T_o2w[:3, i] * 0.12
                    uv1 = proj(ax_w)
                    if uv1:
                        cv2.arrowedLine(img, uv0, uv1, colors[i], 2, tipLength=0.2, line_type=cv2.LINE_AA)
                        # Add XYZ text labels
                        cv2.putText(img, labels[i], (uv1[0]+5, uv1[1]-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA)
                
                # Draw a yellow dot at the origin
                cv2.circle(img, uv0, 5, (0, 215, 255), -1)
                # Label the object key (obj1, obj2)
                cv2.putText(img, obj_key, (uv0[0]+10, uv0[1]+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        draw_glass_rect(img, (10, 10), (360, 70))
        cv2.putText(img, f"3D TRIANGULATOR QA ({len(self.objects_3d)} OBJS)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Frame: {frame_idx:04d}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img


_MANAGER_INSTANCE = CamTriangulatorManager()


def reset_camtriagulator():
    """Reset sequence state before processing a different recording."""
    global _MANAGER_INSTANCE
    _MANAGER_INSTANCE = CamTriangulatorManager()


def run_camtriagulator(image_path, cfg_path, frame_idx, all_image_paths=None, mps_path=None):
    return _MANAGER_INSTANCE.process(image_path, cfg_path, frame_idx, all_image_paths, mps_path)
