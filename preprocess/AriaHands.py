# -*- coding: utf-8 -*-
# @FileName: AriaHands.py

"""
====================================================================================================
Project Aria Hand Tracking and Kinematics Pipeline (AriaHands.py)
====================================================================================================

Description:
    This script processes raw hand tracking data from Project Aria Machine Perception Services 
    (MPS). It extracts hand landmarks, computes orthonormal pose matrices (Wrist/Palm/Midpoint), 
    determines grasp states, and applies a multi-stage optimization pipeline to ensure 
    spatio-temporal consistency in world coordinates.

Core Functionalities:
    1.  Kinematic Extraction: Computes 6-DOF poses for wrists and a custom "gripper-like" 
        midpoint frame between the thumb and index fingertips.
    2.  Temporal Cleaning: Filters low-confidence detections, suppresses "ghost" segments 
        (short-lived noise), and interpolates missing frames.
    3.  Grasp Logic: Determines binary grasp states based on fingertip distances with 
        temporal smoothing and flicker suppression.
    4.  Hand Selection: Supports 'auto' selection based on confidence accumulation, 
        or forcing 'left'/'right' hands for specific tasks.
    5.  Kinematic Optimization: Utilizes Savitzky-Golay and EMA filtering to refine 
        trajectories in World Space.

Generated Outputs & File Descriptions:
    📁 [mps_path]/aria/
    ├── 📁 vis/
    │   └── 🎬 aria_hands_vis.mp4        # Visualization showing skeletons, poses, and grasp status.
    ├── 📁 all_data/
    │   └── 📁 [00000...idx]/
    │       └── 📄 aria_hands.json       # Per-frame kinematic and state metadata.
    └── 📄 aria_hands_analysis_[side].png # Kinematic analysis reports.

Technical Specifics:
    - Coordinate Frame: All "raw_world" outputs are in the MPS 'Closed Loop' World Frame.
    - Rotation Logic: Uses SVD-based robust rotation matrix orthogonalization.
====================================================================================================
"""

import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm
from typing import Optional, Tuple, Any
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Slerp

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from utils.utils_media import create_video_from_frames
from utils.utils_io import load_cfg

from preprocess.AriaCam import AriaCamGenerator
from preprocess.AriaCamTypes import AriaCam
from preprocess.AriaHandsTypes import (
    MidpointFrameBuilder, 
    AriaHandsJointAngles, 
    AriaHandData, 
    AriaHandsData, 
    AriaHands
)
from preprocess.AriaHandsOptimizer import AriaHandsOptimizer
from preprocess.AriaHandsOps import AriaHandsOps
from preprocess.AriaPhasesOps import AriaPhasesOps


class AriaHandsGenerator:
    """
    Generates refined hand tracking data by combining raw MPS results with camera parameters.
    Handles the transition from Device-space landmarks to World-space optimized kinematics.
    """

    def __init__(self, mps_path: str, cfg_path: str, aria_hands_mps: Any, aria_cam: AriaCam):
        """
        Initializes the generator with configuration and data providers.

        Args:
            mps_path (str): Path to the MPS data directory.
            cfg_path (str): Path to the AriaHands.yaml configuration file.
            aria_hands_mps: Raw hand tracking results from projectaria_tools.
            aria_cam (AriaCam): Processed camera metadata and image sequence.
        """
        self.mps_path = mps_path
        # Load configuration using the standardized YAML loader
        self.cfg = load_cfg(cfg_path)
        self.aria_hands_mps = aria_hands_mps
        self.aria_cam = aria_cam

        # Velocity/Pose Caching for differential computation
        self.prev_r_cache = None
        self.prev_l_cache = None
        self.prev_r_mid_cache = None
        self.prev_l_mid_cache = None
        
        self.prev_r_thumb_cache = None
        self.prev_l_thumb_cache = None
        self.prev_r_index_cache = None
        self.prev_l_index_cache = None

        # Orientation consistency caching for the Midpoint/Gripper frame
        self.prev_r_mid_R = None
        self.prev_l_mid_R = None
        self.mid_frame_builder = MidpointFrameBuilder()


    def get_aria_hands(self) -> Optional[AriaHands]:
        """
        Executes the full hand processing pipeline: extraction -> selection -> cleaning -> optimization.

        Returns:
            Optional[AriaHands]: Container with fully processed and smoothed hand data.
        """
        aria_hands = AriaHands(mps_path=self.mps_path)
        dt = 1.0 / self.aria_cam.fps

        # --- Phase 1: Raw Data Extraction & Velocity Computation ---
        for i, (hands_mps, cam_data) in enumerate(tqdm(zip(self.aria_hands_mps, self.aria_cam.cam), 
                                                     total=len(self.aria_cam), 
                                                     desc="Processing Aria Hands")):
            
            # Extract basic geometry and project landmarks to Camera/World frames
            aria_hands_data = self._get_aria_hands_data(
                hands_mps, np.linalg.inv(cam_data.c2d), cam_data.c2w, 
                cam_data.k, cam_data.h, cam_data.w, cam_data.idx, cam_data.ts
            )

            # Compute world-space velocities and assign to data object
            self._compute_and_assign_vel(aria_hands_data, cam_data.c2w, dt)

            aria_hands.hands.append(aria_hands_data)
            aria_hands.tss.append(cam_data.ts)

        # --- Phase 2: Selection and Temporal Cleaning ---
        # Select target hand based on confidence accumulation scores
        self._apply_hand_selection(aria_hands)

        # Filter by confidence and suppress noise (short fragments)
        self._filter_by_confidence(aria_hands, conf_th=self.cfg.hand_conf_threshold)
        self._suppress_short_hands(aria_hands, min_frames=self.cfg.hand_min_frames)
        
        # Interpolate short gaps to maintain trajectory continuity
        self._interpolate_hand_trajectories(aria_hands, max_gap=self.cfg.hand_interp_max_gap)
        
        # Initial pass for grasp state smoothing
        self._smooth_grasp_detection(aria_hands, size=self.cfg.grasp_smooth_win)

        # --- Phase 3: Kinematic Optimization ---
        # Apply Savitzky-Golay and EMA filtering via the Optimizer
        optimizer = AriaHandsOptimizer(self.cfg, dt)
        optimizer.run(aria_hands)
        
        # Final pass of grasp smoothing after pose optimization
        self._smooth_grasp_detection(aria_hands, size=self.cfg.grasp_smooth_win)

        # --- Phase 4: Reporting and Logging ---
        os.makedirs(os.path.join(self.mps_path, "preprocess"), exist_ok=True)
        AriaHandsOps.save_hands_analysis_plots_two(aria_hands, os.path.join(self.mps_path, "preprocess"), dt, self.cfg)
        AriaHandsOps.print_summary_and_eval(aria_hands)

        return aria_hands


    def _get_aria_hands_data(self, hands_mps: Any, d2c: np.ndarray, c2w: np.ndarray, 
                             k: np.ndarray, h: int, w: int, idx: int, ts: int) -> AriaHandsData:
        """
        Creates the combined AriaHandsData frame container.

        Args:
            hands_mps: Raw MPS tracking object.
            d2c: Device-to-Camera 4x4 matrix.
            c2w: Camera-to-World 4x4 matrix.
            k, h, w: Intrinsic matrix and image dimensions.
            idx, ts: Frame index and timestamp.

        Returns:
            AriaHandsData: Combined frame data for both hands.
        """
        hand_r = self._get_aria_hand_data(hands_mps.right_hand, d2c, c2w, k, h, w, is_right=True)
        hand_l = self._get_aria_hand_data(hands_mps.left_hand, d2c, c2w, k, h, w, is_right=False)
        return AriaHandsData(idx, ts, hand_r, hand_l)


    def _get_aria_hand_data(self, hand_mps: Any, d2c: np.ndarray, c2w: np.ndarray, 
                            k: np.ndarray, h: int, w: int, is_right: bool) -> Optional[AriaHandData]:
        """
        Extracts landmarks, computes poses, and determines grasp state for a single hand.

        Args:
            hand_mps: Hand-specific detection from MPS.
            is_right: Hand side indicator.

        Returns:
            Optional[AriaHandData]: Processed data or None if detection is invalid.
        """
        if hand_mps is None or not hand_mps.confidence:
            return None

        confidence = hand_mps.confidence
        if confidence <= 0.1: # Reject detections with near-zero confidence
            return None

        # 1. Extract raw normals and positions from Device Frame (IMU)
        palm_normal = hand_mps.wrist_and_palm_normal_device.palm_normal_device
        palm_position = hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.PALM_CENTER)]
        wrist_normal = hand_mps.wrist_and_palm_normal_device.wrist_normal_device
        wrist_position = hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.WRIST)]

        # 2. Heuristic Grasp Detection based on tip-to-tip distance
        thumb_tip = np.array(hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.THUMB_FINGERTIP)])
        index_tip = np.array(hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.INDEX_FINGERTIP)])
        distance = np.linalg.norm(thumb_tip - index_tip)
        grasp_state = 1 if distance < self.cfg.GRASP_THRESHOLD else 0

        # 3. Compute Local Orthonormal Pose Matrices
        palm_pose, wrist_pose = AriaHandsGenerator._compute_wrist_and_palm_pose(
            palm_normal, palm_position, wrist_normal, wrist_position, confidence, d2c
        )

        # 4. Projection and Biomechanical Angles
        keypoints_3d = AriaHandsGenerator._transform_keypoints_to_camera(hand_mps, d2c)
        keypoints_2d, _ = AriaHandsGenerator._project_points_rotated(keypoints_3d, k, h, w)
        joint_angles = AriaHandsJointAngles.from_keypoints_3d(keypoints_3d)

        return AriaHandData(
            d2c=d2c, c2w=c2w, is_right=is_right, confidence=confidence,
            wrist_pose=wrist_pose, palm_pose=palm_pose,
            hand_keypoints_3d=keypoints_3d, hand_keypoints_2d=keypoints_2d,
            grasp_state=grasp_state, joint_angles=joint_angles
        )


    @staticmethod
    def _compute_wrist_and_palm_pose(palm_n: np.ndarray, palm_p: np.ndarray, 
                                     wrist_n: np.ndarray, wrist_p: np.ndarray, 
                                     conf: float, T_cam_dev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs 4x4 pose matrices for the wrist and palm center in camera space.
        """
        if np.linalg.norm(wrist_n) == 0 or np.linalg.norm(palm_n) == 0:
            return None, None

        palm_wrist_vec = palm_p - wrist_p
        palm_wrist_vec /= (np.linalg.norm(palm_wrist_vec) + 1e-6)
        
        wrist_n = wrist_n / (np.linalg.norm(wrist_n) + 1e-6)
        palm_n = palm_n / (np.linalg.norm(palm_n) + 1e-6)

        def build_rotation_matrix(pos, normal, forward_vec):
            z_axis = normal
            y_axis = forward_vec
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= (np.linalg.norm(x_axis) + 1e-6)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= (np.linalg.norm(y_axis) + 1e-6)
            
            mat = np.eye(4)
            mat[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
            mat[:3, 3] = pos
            return mat

        palm_mat = build_rotation_matrix(palm_p, palm_n, palm_wrist_vec)
        wrist_mat = build_rotation_matrix(wrist_p, wrist_n, palm_wrist_vec)

        return T_cam_dev @ palm_mat, T_cam_dev @ wrist_mat


    @staticmethod
    def _transform_keypoints_to_camera(hand: Any, T_cam_dev: np.ndarray) -> np.ndarray:
        """Transforms 21 landmarks from Device Frame to Camera Frame."""
        Rm = T_cam_dev[:3, :3]
        t = T_cam_dev[:3, 3]
        kpts_dev = np.array(hand.landmark_positions_device)
        kpts_cam = np.einsum("ij, nj -> ni", Rm, kpts_dev) + t
        return kpts_cam


    @staticmethod
    def _project_points_rotated(points_cam: np.ndarray, k: np.ndarray, 
                                h_target: int, w_target: int) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D points in camera frame to 2D pixels with rotation awareness."""
        z = points_cam[:, 2]
        valid_mask = z > 1e-3
        if not np.any(valid_mask):
            return np.zeros((len(points_cam), 2)), np.zeros(len(points_cam), dtype=bool)

        homo_pix = (k @ points_cam.T).T
        u = homo_pix[:, 0] / (homo_pix[:, 2] + 1e-6)
        v = homo_pix[:, 1] / (homo_pix[:, 2] + 1e-6)

        projected = np.stack((u, v), axis=-1)
        in_bounds = (0 <= u) & (u < w_target) & (0 <= v) & (v < h_target)
        projected[~(valid_mask & in_bounds)] = 0.0
        
        return projected, (valid_mask & in_bounds)


    def _compute_and_assign_vel(self, hands_data: AriaHandsData, c2w: np.ndarray, dt: float) -> None:
        """
        Calculates raw world-space poses and derivatives (Linear/Angular velocity).
        Builds the custom "Gripper Frame" at the thumb-index midpoint.
        """
        def robust_rotation_from_matrix(matrix):
            try:
                return R.from_matrix(matrix)
            except ValueError:
                U, S, Vt = np.linalg.svd(matrix)
                d = np.linalg.det(U @ Vt)
                if d < 0:
                    U[:, -1] *= -1
                return R.from_matrix(U @ Vt)
    
        for is_right in [True, False]:
            h_data = hands_data.hand_r if is_right else hands_data.hand_l

            prev_cache = self.prev_r_cache if is_right else self.prev_l_cache
            prev_mid_cache = self.prev_r_mid_cache if is_right else self.prev_l_mid_cache
            prev_R = self.prev_r_mid_R if is_right else self.prev_l_mid_R

            if h_data and h_data.wrist_pose is not None:
                # --- Wrist: Transformation to World frame ---
                p_cam = h_data.wrist_pose[:3, 3]
                r_cam = h_data.wrist_pose[:3, :3]
                p_world = (c2w[:3, :3] @ p_cam) + c2w[:3, 3]
                r_world = c2w[:3, :3] @ r_cam

                h_data.wrist_pose_raw_world = np.eye(4)
                h_data.wrist_pose_raw_world[:3, :3] = r_world
                h_data.wrist_pose_raw_world[:3, 3] = p_world

                if prev_cache is not None:
                    h_data.wrist_lin_vel_raw_world = (p_world - prev_cache['pos']) / dt
                    rel_rot = prev_cache['rot'].T @ r_world
                    h_data.wrist_ang_vel_raw_world = robust_rotation_from_matrix(rel_rot).as_rotvec() / dt

                cache_val = {'pos': p_world, 'rot': r_world}
                if is_right: self.prev_r_cache = cache_val
                else: self.prev_l_cache = cache_val

                # --- Midpoint Kinematics (Node-based stability) ---
                if h_data.hand_keypoints_3d is not None and len(h_data.hand_keypoints_3d) >= 9: 
                    # Tip positions
                    thumb_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[0]) + c2w[:3, 3]
                    index_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[1]) + c2w[:3, 3]
                    # MCP Base positions for frame stability
                    thumb_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[6]) + c2w[:3, 3]
                    index_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[8]) + c2w[:3, 3]

                    h_data.thumb_translation_raw_world = thumb_w
                    h_data.index_translation_raw_world = index_w
                    h_data.thumb_base_raw_world = thumb_base_w
                    h_data.index_base_raw_world = index_base_w
                    
                    midpoint_w = (thumb_w + index_w) / 2.0
                    h_data.midpoint_translation_raw_world = midpoint_w

                    # Reconstruct Gripper frame orientation
                    R_mid = self.mid_frame_builder.build(
                        thumb_w=thumb_w, index_w=index_w,
                        thumb_base_w=thumb_base_w, index_base_w=index_base_w,
                        wrist_w=p_world, midpoint_w=midpoint_w, prev_R=prev_R
                    )

                    if R_mid is None: R_mid = prev_R if prev_R is not None else r_world.copy()

                    h_data.midpoint_pose_raw_world = np.eye(4)
                    h_data.midpoint_pose_raw_world[:3, :3] = R_mid
                    h_data.midpoint_pose_raw_world[:3, 3] = midpoint_w
                    h_data.midpoint_orientation_raw_world = R_mid.flatten()

                    if prev_mid_cache is not None:
                        h_data.midpoint_lin_vel_raw_world = (midpoint_w - prev_mid_cache['pos']) / dt
                        rel_rot_mid = prev_mid_cache['rot'].T @ R_mid
                        h_data.midpoint_ang_vel_raw_world = robust_rotation_from_matrix(rel_rot_mid).as_rotvec() / dt

                    cache_mid_val = {'pos': midpoint_w, 'rot': R_mid}
                    if is_right:
                        self.prev_r_mid_cache = cache_mid_val
                        self.prev_r_mid_R = R_mid
                    else:
                        self.prev_l_mid_cache = cache_mid_val
                        self.prev_l_mid_R = R_mid


    def _apply_hand_selection(self, aria_hands: AriaHands) -> None:
        """Determines which hand side to retain based on confidence scores."""
        selection = getattr(self.cfg, "hand_selection", "auto").lower()
        target_side = None 

        if selection == "left": target_side = "left"
        elif selection == "right": target_side = "right"
        elif selection == "auto":
            r_score = sum([h.hand_r.confidence for h in aria_hands.hands if h.hand_r])
            l_score = sum([h.hand_l.confidence for h in aria_hands.hands if h.hand_l])
            target_side = "right" if r_score >= l_score else "left"
            print(f"║ [Auto Hand Selection] Score R: {r_score:.1f}, L: {l_score:.1f} -> Selected: {target_side}")
        else:
            return

        for frame in aria_hands.hands:
            if target_side == "left": frame.hand_r = None 
            elif target_side == "right": frame.hand_l = None


    def _filter_by_confidence(self, aria_hands: AriaHands, conf_th: float = 0.25) -> None:
        """Removes frame detections with confidence below threshold."""
        for frame_data in aria_hands.hands:
            for hand_attr in ["hand_r", "hand_l"]:
                h = getattr(frame_data, hand_attr)
                if h and (h.confidence < conf_th):
                    setattr(frame_data, hand_attr, None)


    def _suppress_short_hands(self, aria_hands: AriaHands, min_frames: int = 5) -> None:
        """Removes short-duration 'flicker' detections."""
        for hand_attr in ["hand_r", "hand_l"]:
            presence = [getattr(h, hand_attr) is not None for h in aria_hands.hands]
            count, segments = 0, []
            for i, is_present in enumerate(presence):
                if is_present: count += 1
                else:
                    if 0 < count < min_frames: segments.append((i - count, i))
                    count = 0
            if 0 < count < min_frames: segments.append((len(presence) - count, len(presence)))
            for start, end in segments:
                for i in range(start, end): setattr(aria_hands.hands[i], hand_attr, None)


    def _interpolate_hand_trajectories(self, aria_hands: AriaHands, max_gap: int = 3) -> None:
        """Linearly interpolates position gaps and SLERP interpolates rotations."""
        for hand_attr in ["hand_r", "hand_l"]:
            presence = [getattr(h, hand_attr) is not None for h in aria_hands.hands]
            indices = np.where(presence)[0]
            if len(indices) < 2: continue
            for start_i, end_i in zip(indices[:-1], indices[1:]):
                gap = end_i - start_i - 1
                if 0 < gap <= max_gap:
                    h_start, h_end = getattr(aria_hands.hands[start_i], hand_attr), getattr(aria_hands.hands[end_i], hand_attr)
                    steps = np.linspace(0, 1, gap + 2)[1:-1]
                    for j, t in enumerate(steps):
                        curr_idx = start_i + j + 1
                        interp_kpts = (1 - t) * h_start.hand_keypoints_3d + t * h_end.hand_keypoints_3d
                        pos = (1 - t) * h_start.wrist_pose[:3, 3] + t * h_end.wrist_pose[:3, 3]
                        slerp_rot = Slerp([0, 1], R.from_matrix([h_start.wrist_pose[:3, :3], h_end.wrist_pose[:3, :3]]))(t).as_matrix()
                        wrist_p = np.eye(4); wrist_p[:3, :3] = slerp_rot; wrist_p[:3, 3] = pos
                        cam_data = self.aria_cam.cam[curr_idx]
                        interp_hand = AriaHandData(
                            d2c=h_start.d2c, c2w=cam_data.c2w, is_right=(hand_attr == "hand_r"),
                            confidence=float((1-t)*h_start.confidence + t*h_end.confidence),
                            wrist_pose=wrist_p, palm_pose=wrist_p,
                            hand_keypoints_3d=interp_kpts, joint_angles=AriaHandsJointAngles.from_keypoints_3d(interp_kpts)
                        )
                        interp_hand.hand_keypoints_2d, _ = AriaHandsGenerator._project_points_rotated(interp_kpts, cam_data.k, cam_data.h, cam_data.w)
                        setattr(aria_hands.hands[curr_idx], hand_attr, interp_hand)


    def _smooth_grasp_detection(self, aria_hands: AriaHands, size: int = 5) -> None:
        """Applies uniform filter and flicker suppression to binary grasp state."""
        size = max(int(size), 1)

        def _suppress_flicker_binary(x, max_len):
            x = x.copy().astype(np.int32); start = 0
            while start < len(x):
                val, end = x[start], start + 1
                while end < len(x) and x[end] == val: end += 1
                if (end - start) <= max_len and start > 0 and end < len(x):
                    if x[start - 1] == x[end]: x[start:end] = x[start - 1]
                start = end
            return x
    
        def _process_side(hand_attr):
            dists = np.array([np.linalg.norm(h.index_translation_raw_world - h.thumb_translation_raw_world) 
                             if (h and h.thumb_translation_raw_world is not None) else np.nan 
                             for h in [getattr(f, hand_attr) for f in aria_hands.hands]])
            v = ~np.isnan(dists)
            if np.sum(v) < 3: return
            d_i = np.interp(np.arange(len(dists)), np.where(v)[0], dists[v])
            g = _suppress_flicker_binary((uniform_filter1d(d_i, size=size, mode="nearest") < self.cfg.GRASP_THRESHOLD).astype(np.int32), self.cfg.grasp_flicker_max_len)
            for i, f in enumerate(aria_hands.hands):
                h = getattr(f, hand_attr)
                if h: h.grasp_state = int(g[i])

        _process_side("hand_r"); _process_side("hand_l")
    

    def draw_aria_hands_skeleton(self, img: np.ndarray, aria_hands_data: AriaHandsData, k: np.ndarray, d: np.ndarray, c2w: np.ndarray, full_skeleton: bool = False) -> np.ndarray:
        return AriaHandsOps.draw_aria_hands_skeleton(img, aria_hands_data, k, d, c2w, self.cfg.grasp_threshold, full_skeleton=full_skeleton)


    def draw_aria_hands_panel(self,img: np.ndarray, idx: int, aria_hands_data: AriaHandsData) -> np.ndarray:
        return AriaHandsOps.draw_aria_hands_panel(img, idx, aria_hands_data, self.cfg.opt_v_limit)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/AriaHands.yaml", help="Path to AriaHands.yaml")
    parser.add_argument("--no-video", action="store_false", dest="export_video", help="Disable video export")
    parser.add_argument("--no-gif", action="store_false", dest="export_gif", help="Disable GIF export")
    args = parser.parse_args()

    # Providers
    vrs_provider = data_provider.create_vrs_data_provider(os.path.join(args.mps_path, "sample.vrs"))
    mps_provider = MpsDataProvider(MpsDataPathsProvider(args.mps_path).get_data_paths())

    # Range
    aria_hands_mps = mps.hand_tracking.read_hand_tracking_results(os.path.join(args.mps_path, "hand_tracking", "hand_tracking_results.csv"))
    rgb_tss = vrs_provider.get_timestamps_ns(vrs_provider.get_stream_id_from_label("camera-rgb"), TimeDomain.DEVICE_TIME)
    start_idx = len(rgb_tss) - len(aria_hands_mps)
    end_idx = start_idx + len(aria_hands_mps) - 1
    
    # AriaCam
    aria_cam_rgb_generator = AriaCamGenerator(args.mps_path, os.path.join(os.path.dirname(args.cfg_path), "AriaCam.yaml"), vrs_provider, mps_provider, label='rgb')
    aria_cam_rgb = aria_cam_rgb_generator.get_aria_cam(start_idx, end_idx)
    aria_cam_rgb.save_aria_cam_json(label='rgb')
    aria_cam_rgb.save_aria_cam_video_orig(args.export_video, args.export_gif, label='rgb')

    # AriaHands
    aria_hands_generator = AriaHandsGenerator(args.mps_path, args.cfg_path, aria_hands_mps, aria_cam_rgb)
    aria_hands = aria_hands_generator.get_aria_hands()
    aria_hands.save_aria_hands_json()

    # Visualization
    if args.export_video:
        frames_all = []
        for idx, ts in enumerate(tqdm(aria_cam_rgb.tss, desc="Visualizing")):
            img = aria_cam_rgb.cam[idx].img.copy()
            
            # Overlay: Hands Skeleton
            img = aria_hands_generator.draw_aria_hands_skeleton(img, aria_hands.hands[idx], aria_cam_rgb.cam[idx].k, aria_cam_rgb.cam[idx].d, aria_cam_rgb.cam[idx].c2w)

            # Overlay: Hands HUD Panel
            img = aria_hands_generator.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])
            
            # Overlay: Phases HUD Panel
            img = AriaPhasesOps.draw_aria_phases_panel(img, idx, aria_cam_rgb.fps, len(aria_cam_rgb))

            frames_all.append(img)
        
        create_video_from_frames(frames_all, os.path.join(args.mps_path, "preprocess", "vis", "aria_hands_vis.mp4"), aria_cam_rgb.fps, args.export_gif)

# Execution Example:
# python -m preprocess.AriaHands --mps_path  "./data/test/test_0/mps_test_0_000_vrs/"