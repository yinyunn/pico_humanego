# -*- coding: utf-8 -*-
# @FileName: DatasetGen.py

"""
====================================================================================================
Project Aria Manipulation Dataset Generator (DatasetGen.py)
====================================================================================================

Description:
    This script acts as the final stage of the preprocessing pipeline. It consolidates all 
    multi-modal data (RGB, Masks, SLAM, Hands, and Object Poses) into a structured JSON 
    dataset (`training_data.json`) optimized for training Multi-Object Flow Matching policies.

Core Functionalities:
    1.  Virtual Static Anchor: Designates the primary object (e.g., 'obj1') as the global 
        coordinate origin, establishing an object-centric reference frame for the entire sequence.
    2.  Kinematic State Machine ("Latch & Propagate"): Monitors grasp states to dynamically 
        update the 6-DOF pose of the manipulated object by "latching" it to the hand's 
        kinematics. This effectively bypasses visual occlusion and motion blur during manipulation.
    3.  Consolidated Export: Outputs pure, ready-to-use 4x4 SE(3) transformation matrices 
        for all entities (Cameras, Hands, Objects) in the World frame.
    4.  Advanced Visualization: Generates high-fidelity Object-Centric 3D Point Clouds (.ply) 
        and Matplotlib trajectory plots (.png) for geometric verification and QA.

Generated Outputs:
    - [mps_path]/aria/all_data/[idx]/training_data.json: Per-frame dataloader target.
    - [mps_path]/aria/object_centric.ply: 3D point cloud and trajectory visualization.
    - [mps_path]/aria/object_centric.png: 2D Matplotlib diagnostic plot of the scene.
====================================================================================================
"""

import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Any
import open3d as o3d
import matplotlib
matplotlib.use("Agg")  # Run Matplotlib in headless mode
import matplotlib.pyplot as plt

from utils.utils_io import load_cfg



# Mapping from hand tracking method name → per-frame JSON filename
HAND_METHOD_JSON_MAP = {
    "aria_mps":   "aria_hands.json",
    "mediapipe":  "mediapipe_hands.json",
    "wilor":      "wilor_hands.json",
    "hamer":      "hamer_hands.json",
}

# Mapping from hand tracking method name → entity key in training_data.json
HAND_METHOD_ENTITY_KEY = {
    "aria_mps":   "hands",            # default / legacy key
    "mediapipe":  "hands_mediapipe",
    "wilor":      "hands_wilor",
    "hamer":      "hands_hamer",
}


class DatasetGen:
    """
    Main Dataset Generator class handling spatial transformations, state machine logic,
    and visualization exports.
    """
    def __init__(self, mps_path: str, cfg_path: str):
        self.mps_path = mps_path
        self.aria_dir = os.path.join(mps_path, "preprocess")
        self.all_data_dir = os.path.join(self.aria_dir, "all_data")
        self.cfg = load_cfg(cfg_path)

        # Load global metadata (Triangulated object poses, Camera 0 origins)
        self._load_objects_data()


    def _load_objects_data(self):
        """
        Parses CamTriangulator results. Designates the primary object as the Virtual 
        Static Anchor, extracting static 4x4 World Transformation Matrices for all objects.
        """
        tri_path = os.path.join(self.aria_dir, "camtriangulator_results.json")
        if not os.path.exists(tri_path):
            raise FileNotFoundError(f"Missing {tri_path}. Run CamTriangulator first.")
        
        with open(tri_path, "r") as f:
            tri_data = json.load(f)

        # Extract Camera 0 absolute pose as the local world reference baseline
        self.T_c02w = np.array(tri_data["cam0_c2w"], dtype=np.float64)
        self.T_w2c0 = np.linalg.inv(self.T_c02w)

        objects = tri_data["objects"]
        exclude_keys =["arm_and_obj", "timestamp", "info"]
        obj_keys = sorted([k for k in objects.keys() if k.startswith("obj") and k not in exclude_keys])
        
        if not obj_keys:
            raise KeyError("No valid objects found in Triangulation results.")

        # The anchor is task-dependent.  For PICO can-on-mousepad data the
        # mousepad (obj2) is static and the can (obj1) must remain dynamic.
        configured_anchor = getattr(self.cfg, "anchor_key", None)
        self.anchor_key = str(configured_anchor) if configured_anchor else obj_keys[0]
        if self.anchor_key not in obj_keys:
            raise KeyError(f"Configured anchor_key '{self.anchor_key}' is not in {obj_keys}")
        print(f"║[DatasetGen] Selected '{self.anchor_key}' as the Virtual Static Anchor.")

        self.objs_meta = {}
        for k, v in objects.items():
            T_ok2c0 = np.array(v["object_to_cam0_matrix"], dtype=np.float64)
            T_ok2w = self.T_c02w @ T_ok2c0 # Static World Pose
            
            pts_w = np.array(v.get("points_3d_world",[]), dtype=np.float64)
            pts_c0 = np.array(v.get("points_3d_cam0",[]), dtype=np.float64)
            
            # Map keypoints to the object's own local coordinate system
            T_c02ok = np.linalg.inv(T_ok2c0)
            pts_ok = (T_c02ok[:3, :3] @ pts_c0.T + T_c02ok[:3, 3][:, None]).T

            self.objs_meta[k] = {
                "T_ok2c0": T_ok2c0, # Stored for legacy visualization compatibility
                "T_ok2w_static": T_ok2w,
                "pts_w": pts_w,
                "pts_ok": pts_ok
            }

        # Cache inverses for high-speed spatial mapping
        self.T_anchor_2_w_static = self.objs_meta[self.anchor_key]["T_ok2w_static"]
        self.T_w_2_anchor_static = np.linalg.inv(self.T_anchor_2_w_static)


    def _get_hand_pose_world(self, hand_data: Optional[Dict], min_conf: float = None) -> Optional[np.ndarray]:
        """ Extracts the 4x4 SE(3) World Transformation Matrix for a given hand. """
        if hand_data is None: return None
        if min_conf is None:
            min_conf = self.cfg.min_hand_confidence
        if hand_data.get("confidence", 0.0) < min_conf: return None
        try:
            p_w = np.array(hand_data["midpoint_translation_opt_world"])
            R_w = np.array(hand_data["midpoint_orientation_opt_world"]).reshape(3, 3)
            T_h2w = np.eye(4, dtype=np.float64)
            T_h2w[:3, :3] = R_w
            T_h2w[:3, 3] = p_w
            return T_h2w
        except Exception:
            return None


    def _build_hands_entry_from_method(self, frame_dir: str, method: str) -> Optional[Dict]:
        """
        Reads a hand method JSON from the frame directory and builds a hands entry
        with the same structure as the default 'hands' key: {side: {T_hand_to_world, grasp}}.

        Returns None (written as JSON null) if the method JSON file does not exist,
        indicating the method was never run for this session. Returns a dict with
        per-side data (or empty dict if no hands detected) otherwise.
        """
        json_filename = HAND_METHOD_JSON_MAP.get(method)
        if json_filename is None:
            return None

        json_path = os.path.join(frame_dir, json_filename)
        if not os.path.exists(json_path):
            return None  # method not generated → null at top level

        try:
            with open(json_path, "r") as f:
                hands_d = json.load(f)
        except Exception:
            return None

        # Lower confidence threshold for image-based methods (they are less accurate)
        min_conf = self.cfg.min_hand_confidence * 0.5 if method != "aria_mps" else self.cfg.min_hand_confidence

        hands_out = {}
        for side_key, side_label in [("hand_l", "left"), ("hand_r", "right")]:
            h = hands_d.get(side_key)
            T_h2w = self._get_hand_pose_world(h, min_conf=min_conf)
            if T_h2w is not None:
                gr = h.get("grasp_state", 0) if h else 0
                hands_out[side_label] = {
                    "T_hand_to_world": T_h2w.tolist(),
                    "grasp": float(gr)
                }

        return hands_out


    # ============================================================
    #[Visualization Helpers: PLY & PNG]
    # ============================================================
    @staticmethod
    def _get_rotation_matrix(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """ Computes a rotation matrix that aligns vec1 to vec2. """
        a = (vec1 / (np.linalg.norm(vec1) + 1e-12)).reshape(3)
        b = (vec2 / (np.linalg.norm(vec2) + 1e-12)).reshape(3)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if s < 1e-8: return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1.0 - c) / (s**2 + 1e-12))


    def _create_cylinder_line(self, start: np.ndarray, end: np.ndarray, radius: float, color: List[float]):
        """ Generates a 3D cylinder bridging two points (used for drawing trajectory lines). """
        vec = end - start
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6: return o3d.geometry.TriangleMesh()
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=dist)
        cyl.paint_uniform_color(color)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        target = vec / dist
        R = self._get_rotation_matrix(z_axis, target)
        cyl.rotate(R, center=(0, 0, 0))
        cyl.translate((start + end) / 2.0)
        return cyl


    def _create_arrow(self, start: np.ndarray, end: np.ndarray, color: List[float], cyl_r: float, cone_r: float):
        """ Generates a 3D arrow for coordinate axis visualization. """
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6: return o3d.geometry.TriangleMesh()
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cyl_r, cone_radius=cone_r,
            cylinder_height=length * 0.8, cone_height=length * 0.2,
        )
        arrow.paint_uniform_color(color)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = self._get_rotation_matrix(z_axis, vec / length)
        arrow.rotate(R, center=(0, 0, 0))
        arrow.translate(start)
        return arrow


    def _add_axes_at_pose(self, geos: List, p: np.ndarray, R: np.ndarray, axis_len: float, cyl_r: float, cone_r: float, alpha: float = 1.0, axis_colors=None):
        """ Appends X (Red), Y (Green), Z (Blue) arrows to the geometry list at a specific pose. """
        if axis_colors is None: axis_colors = [[1, 0, 0],[0, 1, 0], [0, 0, 1]]
        for i in range(3):
            d = R[:, i]
            end = p + d * axis_len
            col = [float(axis_colors[i][0]) * alpha, float(axis_colors[i][1]) * alpha, float(axis_colors[i][2]) * alpha]
            geos.append(self._create_arrow(p, end, col, cyl_r, cone_r))


    def _export_object_and_traj_ply(self, traj_r: List, traj_l: List, axes_r: List, axes_l: List) -> None:
        """ Exports a comprehensive Open3D .ply file mapping all objects and hand trajectories to the Anchor frame. """
        if not self.cfg.export_obj_and_traj_ply: return
        geos =[]

        # A) Object keypoints (All objects mapped into the Static Anchor Frame)
        if bool(getattr(self.cfg, "ply_show_object_kpts", True)):
            obj_colors = [[0.2, 0.9, 0.2], [0.2, 0.8, 0.9],[0.9, 0.2, 0.8], [0.9, 0.2, 0.2]]
            for idx, (obj_key, meta) in enumerate(self.objs_meta.items()):
                # T_ok_in_anchor = T_world_to_anchor @ T_ok_to_world
                T_ok_2_anchor = self.T_w_2_anchor_static @ meta["T_ok2w_static"]
                R_rel, t_rel = T_ok_2_anchor[:3, :3], T_ok_2_anchor[:3, 3]
                
                pts_local = meta["pts_ok"] 
                pts_in_anchor = (R_rel @ pts_local.T + t_rel[:, None]).T
                color = obj_colors[idx % len(obj_colors)]
                
                for pt in pts_in_anchor:
                    sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(self.cfg.obj_kpt_radius_m))
                    sph.translate(pt)
                    sph.paint_uniform_color(color)
                    geos.append(sph)
                
                axis_len = float(self.cfg.obj_axes_len_m) * 0.6 
                self._add_axes_at_pose(geos, t_rel, R_rel, axis_len, 0.001, 0.003, alpha=0.8)

        # B) Static Anchor Origin Marker
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        center_sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(self.cfg.obj_kpt_radius_m) * 1.8)
        center_sph.translate(origin)
        center_sph.paint_uniform_color([1.0, 0.8, 0.0])  # Yellow Origin Marker
        geos.append(center_sph)
        self._add_axes_at_pose(geos, origin, np.eye(3), float(self.cfg.obj_axes_len_m), 0.002, 0.004)

        # C) Trajectories
        def _add_traj(traj, color_pt, color_line):
            if len(traj) == 0: return
            pts = [p for (p, _R) in traj]
            for p in pts:
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(self.cfg.traj_point_radius_m))
                sph.translate(p)
                sph.paint_uniform_color(color_pt)
                geos.append(sph)
            for i in range(len(pts) - 1):
                geos.append(self._create_cylinder_line(pts[i], pts[i + 1], float(self.cfg.traj_line_radius_m), color_line))

        _add_traj(traj_r,[0.2, 0.9, 0.9], [0.5, 0.9, 0.9])
        _add_traj(traj_l,[0.9, 0.2, 0.9], [0.9, 0.6, 0.9])

        # Draw trajectory mini-axes based on stride settings
        for (p, R) in axes_r: self._add_axes_at_pose(geos, p, R, float(self.cfg.traj_axes_len_m), 0.0008, 0.0018, 0.95)
        for (p, R) in axes_l: self._add_axes_at_pose(geos, p, R, float(self.cfg.traj_axes_len_m), 0.0008, 0.0018, 0.75)

        combined = o3d.geometry.TriangleMesh()
        for g in geos: combined += g
        save_p = os.path.join(self.aria_dir, "object_centric.ply")
        o3d.io.write_triangle_mesh(save_p, combined)
        print(f"║ [DatasetGen] Multi-Object PLY saved to: {save_p}")


    def _export_object_centric_png(self, traj_r: List, traj_l: List):
        """ Exports an object-centric 2D Matplotlib rendering of the scene and trajectories. """
        if not bool(getattr(self.cfg, "export_object_centric_png", True)): return
        
        elev = float(getattr(self.cfg, "png_view_elev", 20.0))
        azim = float(getattr(self.cfg, "png_view_azim", 10.0))
        dpi = int(getattr(self.cfg, "png_dpi", 220))
        obj_axis_len = float(getattr(self.cfg, "obj_axes_len_m", 0.10))
        fill_ratio = float(getattr(self.cfg, "png_fill_ratio", 0.80))

        out_path = os.path.join(self.mps_path, "preprocess", "object_centric.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig = plt.figure(figsize=(8, 8), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        mpl_colors =['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C'] 
        all_points_for_scaling =[]

        # Plot Objects
        for idx, (obj_key, meta) in enumerate(self.objs_meta.items()):
            T_ok_2_anchor = self.T_w_2_anchor_static @ meta["T_ok2w_static"]
            R_rel, t_rel = T_ok_2_anchor[:3, :3], T_ok_2_anchor[:3, 3]
            
            pts_local = meta["pts_ok"]
            pts_anchor = (R_rel @ pts_local.T + t_rel[:, None]).T
            all_points_for_scaling.append(pts_anchor)
            
            col = mpl_colors[idx % len(mpl_colors)]
            label = obj_key + (" (Anchor)" if obj_key == self.anchor_key else "")
            
            # Downsample if dense
            pts_plot = pts_anchor[::2] if len(pts_anchor) > 100 else pts_anchor
            ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], s=4, color=col, alpha=0.6, label=label)
            self._draw_axes_rgb_mpl(ax, t_rel, R_rel, length=0.03, lw=1.0)

        # Plot Trajectories
        def _get_traj_arr(traj): return np.array([p for p, _ in traj]) if traj else np.zeros((0,3))
        P2r, P2l = _get_traj_arr(traj_r), _get_traj_arr(traj_l)
        
        if len(P2r) > 0: 
            ax.plot(P2r[:,0], P2r[:,1], P2r[:,2], color="#19B5FE", lw=1.5, label="Right Hand")
            all_points_for_scaling.append(P2r)
        if len(P2l) > 0:
            ax.plot(P2l[:,0], P2l[:,1], P2l[:,2], color="#FF4FD8", lw=1.5, label="Left Hand")
            all_points_for_scaling.append(P2l)

        # Auto-Scaling Logic (Preserves uniform aspect ratio)
        if all_points_for_scaling:
            all_p = np.concatenate(all_points_for_scaling, axis=0)
            max_range = np.abs(all_p).max()
            r = max(max_range / fill_ratio, obj_axis_len * 1.2)
        else:
            r = 0.2

        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(loc="upper right", fontsize='small')
        ax.set_title(f"Multi-Object Scene ({self.anchor_key} as Static Origin)", pad=20)
        self._draw_axes_rgb_mpl(ax, np.zeros(3), np.eye(3), length=obj_axis_len, lw=2.5)

        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"║[DatasetGen] Multi-Object PNG saved: {out_path}")


    def _draw_axes_rgb_mpl(self, ax, p, R, length=0.1, lw=1.5):
        """ Draws RGB coordinate axes vectors in a Matplotlib 3D plot. """
        v = R * length
        ax.quiver(p[0], p[1], p[2], v[0,0], v[1,0], v[2,0], color='r', linewidth=lw)
        ax.quiver(p[0], p[1], p[2], v[0,1], v[1,1], v[2,1], color='g', linewidth=lw)
        ax.quiver(p[0], p[1], p[2], v[0,2], v[1,2], v[2,2], color='b', linewidth=lw)


    # ============================================================
    # [Main Processing Loop]
    # ============================================================
    def process(self, image_list: List[str]) -> Dict[str, Any]:
        """
        Iterates over the filtered sequence, applies Kinematic Latching, and writes 
        the definitive JSON dataset tailored for training multi-object flow matching models.
        """
        stats = {"total": len(image_list), "valid": 0, "discarded_no_hand": 0, "discarded_missing_file": 0}

        print(f"║ [DatasetGen] Processing and Kinematic Latching for {len(image_list)} frames...")

        # State Machine Variables (Independent for Left and Right hands)
        hand_states = {
            "left":  {"is_grasping": False, "latched_obj": None, "T_lock_h2obj": None, "has_released": False},
            "right": {"is_grasping": False, "latched_obj": None, "T_lock_h2obj": None, "has_released": False}
        }
        
        # Track dynamic pose of the objects (Initially identical to static world poses)
        dynamic_obj_poses = {k: v["T_ok2w_static"].copy() for k, v in self.objs_meta.items()}

        # Visualization Buffers
        traj_r, traj_l, axes_r, axes_l = [], [], [],[]
        stride = max(1, int(getattr(self.cfg, "traj_stride", 5)))
        max_pts = int(getattr(self.cfg, "traj_max_points", 5000))
        use_hand = str(getattr(self.cfg, "traj_use_hand", "right_first")).lower()

        for t, img_path in enumerate(tqdm(image_list, desc="Generating JSONs")):
            frame_dir = os.path.dirname(img_path)
            idx_str = os.path.basename(frame_dir)

            # 1) Check dependencies
            files = { 
                "cam": "aria_cam_rgb.json",
                "hands": "aria_hands.json", 
                "phases": "aria_phases.json", 
                "slam":  "aria_slam.json", 
            }
            paths = {k: os.path.join(frame_dir, v) for k, v in files.items()}
            if not all(os.path.exists(p) for p in paths.values()):
                stats["discarded_missing_file"] += 1
                continue

            with open(paths["cam"], "r") as f: 
                cam_d = json.load(f)
            with open(paths["hands"], "r") as f: 
                hands_d = json.load(f)
            with open(paths["phases"], "r") as f: 
                phases_d = json.load(f)

            # Determine if the current frame belongs to the 'FINISHED' phase
            is_finished = 1.0 if phases_d.get("mode", 0) == 4 else 0.0

            # Extract hand states
            h_l = hands_d.get("hand_l")
            h_r = hands_d.get("hand_r")
            gr_l = h_l.get("grasp_state", 0) if h_l else 0
            gr_r = h_r.get("grasp_state", 0) if h_r else 0
            T_hL2w = self._get_hand_pose_world(h_l)
            T_hR2w = self._get_hand_pose_world(h_r)

            # =================================================================
            # State Machine: Multi-Object Independent Bimanual Latching
            # =================================================================
            current_grasps = {"left": gr_l > 0.5, "right": gr_r > 0.5}
            current_hand_poses = {"left": T_hL2w, "right": T_hR2w}

            # Retrieve independent latching switches from config (default to False if not set in YAML)
            disable_latching_flags = {
                "left": getattr(self.cfg, "disable_kinematic_latching_left", False),
                "right": getattr(self.cfg, "disable_kinematic_latching_right", False)
            }

            for side in ["left", "right"]:
                state = hand_states[side]
                is_grasp = current_grasps[side]
                T_h2w = current_hand_poses[side]

                if not is_grasp:
                    state["has_released"] = True

                # Transition: Release -> Grasp (Trigger Latch)
                if is_grasp and not state["is_grasping"] and state["has_released"] and T_h2w is not None:
                    state["is_grasping"] = True
                    
                    # Heuristic: Find the closest object to the hand
                    closest_obj = None
                    min_dist = float('inf')
                    p_hand = T_h2w[:3, 3]
                    
                    for obj_k, obj_v in self.objs_meta.items():
                        # Skip the virtual static anchor (e.g., the sink/flower pot) to prevent moving the environment
                        if obj_k == self.anchor_key:
                            continue
                            
                        p_obj = dynamic_obj_poses[obj_k][:3, 3]
                        dist = float(np.linalg.norm(p_hand - p_obj))
                        if dist < min_dist:
                            min_dist = dist
                            closest_obj = obj_k
                            
                    # Distance threshold (e.g., 0.20 meters) to ensure we actually grasped something nearby
                    if closest_obj is not None and min_dist < 0.20:
                        state["latched_obj"] = closest_obj
                        # Compute rigid offset matrix: T_lock = Inv(T_hand) @ T_obj
                        state["T_lock_h2obj"] = np.linalg.inv(T_h2w) @ dynamic_obj_poses[closest_obj]
                    else:
                        state["latched_obj"] = None

                # Transition: Grasp -> Release (Drop)
                elif not is_grasp and state["is_grasping"]:
                    state["is_grasping"] = False
                    state["latched_obj"] = None
                    state["T_lock_h2obj"] = None

                # Update Dynamic Object Pose (Forward Kinematics Propagation)
                if state["is_grasping"] and state["latched_obj"] is not None and not disable_latching_flags[side]:
                    if T_h2w is not None:
                        dynamic_obj_poses[state["latched_obj"]] = T_h2w @ state["T_lock_h2obj"]

            # =================================================================
            # Trajectory Collection for Visualization (Mapped to Static Anchor)
            # =================================================================
            def _collect_for_vis(T_h2w):
                if T_h2w is None: return None
                # Transform hand pose into the Anchor's local coordinate system
                T_h_in_anchor = self.T_w_2_anchor_static @ T_h2w
                p = T_h_in_anchor[:3, 3]
                R = T_h_in_anchor[:3, :3]
                return p, R

            if (len(traj_r) + len(traj_l)) < max_pts:
                vR, vL = _collect_for_vis(T_hR2w), _collect_for_vis(T_hL2w)
                if use_hand == "both":
                    if vR: traj_r.append(vR); (axes_r.append(vR) if t % stride == 0 else None)
                    if vL: traj_l.append(vL); (axes_l.append(vL) if t % stride == 0 else None)
                elif use_hand == "left_first":
                    chosen = vL if vL else vR
                    if chosen: traj_l.append(chosen); (axes_l.append(chosen) if t % stride == 0 else None)
                else: 
                    chosen = vR if vR else vL
                    if chosen: traj_r.append(chosen); (axes_r.append(chosen) if t % stride == 0 else None)

            # =================================================================
            # Build Unified World Transform Dictionary
            # =================================================================
            world_transforms = {
                "cam0": self.T_c02w.tolist(),
                "virtual_static_anchor": self.T_anchor_2_w_static.tolist() 
            }
            
            # Dynamic Objects
            objects_out = {}
            for k, v in self.objs_meta.items():
                # Check if this specific object is currently latched by any active hand
                is_latched_by_left = (hand_states["left"]["latched_obj"] == k) and not disable_latching_flags["left"]
                is_latched_by_right = (hand_states["right"]["latched_obj"] == k) and not disable_latching_flags["right"]
                is_dynamic = is_latched_by_left or is_latched_by_right

                objects_out[k] = {
                    "T_obj_to_world": dynamic_obj_poses[k].tolist(),
                    "is_dynamic": bool(is_dynamic)
                }

            # Hands (default: aria_mps)
            hands_out = {}
            if T_hL2w is not None: hands_out["left"] = {"T_hand_to_world": T_hL2w.tolist(), "grasp": float(gr_l)}
            if T_hR2w is not None: hands_out["right"] = {"T_hand_to_world": T_hR2w.tolist(), "grasp": float(gr_r)}

            if not hands_out: stats["discarded_no_hand"] += 1

            # Alternative hand tracking methods (mediapipe, wilor, hamer)
            alt_hands = {}
            for method_name, entity_key in HAND_METHOD_ENTITY_KEY.items():
                if method_name == "aria_mps":
                    continue  # already handled above as default "hands"
                alt_hands[entity_key] = self._build_hands_entry_from_method(frame_dir, method_name)

            # Save Clean Architecture JSON (Ready for Dataloading)
            entities = {
                "hands": hands_out,
                "objects": objects_out
            }
            # Merge alternative hand method entries into entities
            entities.update(alt_hands)

            data = {
                "metadata": {
                    "idx": int(idx_str),
                    "ts": cam_d.get("ts"),
                    "w": cam_d.get("w"),
                    "h": cam_d.get("h"),
                    "fps": cam_d.get("fps"),
                    "k": cam_d.get("k"),
                    "c2w": cam_d.get("c2w"),
                    "anchor_key": self.anchor_key,
                    "is_finished": float(is_finished),
                    "world_transforms": world_transforms
                },
                "obs": {
                    "mask_arm_path": os.path.join(frame_dir, "mask_arm.png"),
                    "mask_obj_path": os.path.join(frame_dir, "mask_arm_and_obj.png"),

                    "rgb_path": os.path.join(frame_dir, "rgb.png"),
                    "rgb_WArmObjKpts_path": os.path.join(frame_dir, "rgb_WArmObjKpts.png"),
                    "rgb_WoArm_path": os.path.join(frame_dir, "rgb_WoArm.png"),
                    "rgb_WoArm_WArmObjKpts_path": os.path.join(frame_dir, "rgb_WoArm_WArmObjKpts.png"),
                },
                "entities": entities
            }

            stats["valid"] += 1
            with open(os.path.join(frame_dir, "training_data.json"), "w") as f:
                json.dump(data, f, indent=2)

        # Trigger Visualizations post-processing
        self._export_object_and_traj_ply(traj_r, traj_l, axes_r, axes_l)
        
        if use_hand == "both": 
            self._export_object_centric_png(traj_r, traj_l)
        elif use_hand == "left_first": 
            self._export_object_centric_png(traj_r=[], traj_l=(traj_l if traj_l else traj_r))
        else: 
            self._export_object_centric_png(traj_r=(traj_r if traj_r else traj_l), traj_l=[])

        return stats

# ==============================================================================
# [Public Interface] API for Pipeline Integration
# ==============================================================================
def run_datasetgen(image_list: List[str], mps_path: str, cfg_path: str) -> Dict[str, Any]:
    """Singleton entry point for triggering dataset generation from the main pipeline."""
    gen = DatasetGen(mps_path, cfg_path)
    return gen.process(image_list)
