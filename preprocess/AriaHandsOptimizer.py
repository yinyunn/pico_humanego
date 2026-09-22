# -*- coding: utf-8 -*-
# @FileName: AriaHandsOptimizer.py

"""
====================================================================================================
Project Aria Hand Kinematics Optimizer (AriaHandsOptimizer.py)
====================================================================================================

Description:
    This module provides a multi-stage optimization pipeline to refine raw world-space 
    hand trajectories. It implements temporal smoothing for both positions (3D) and 
    orientations (SO3) to produce jitter-free kinematics suitable for training and 
    biomechanical analysis.

Core Functionalities:
    1.  Trajectory Smoothing: Uses Savitzky-Golay (SG) filters for linear positions 
        and Exponential Moving Average (EMA) for rotational basis vectors.
    2.  Gap Filling: Performs linear interpolation on short temporal gaps to maintain 
        data continuity before applying filters.
    3.  Pose Reconstruction: Rebuilds optimized "Gripper-like" midpoint frames using 
        smoothed MCP (Metacarpophalangeal) bases for high stability.
    4.  Velocity Estimation: Derives linear and angular velocities from optimized 
        trajectories using central differences and rotational log-mapping (Rotvec).

Technical Specifics:
    - Position Smoothing: Savitzky-Golay (Low-pass filtering).
    - Orientation Smoothing: Unit-vector EMA + Gram-Schmidt Re-orthonormalization.
    - Consistency: Enforces sign consistency (flipping prevention) across temporal steps.

====================================================================================================
"""

import numpy as np
from typing import Optional, List, Tuple, Any
from scipy.signal import savgol_filter

from preprocess.AriaHandsTypes import MidpointFrameBuilder, AriaHandData, AriaHands


class SimpleSmoother:
    """
    Utility class for trajectory-level smoothing. 
    Handles 1D and 3D signal filtering and gap management.
    """

    def __init__(
        self,
        dt: float,
        sg_window: int,
        sg_polyorder: int,
        min_valid_frames: int,
        fill_max_gap: int,
    ):
        """
        Initializes the smoother with filtering parameters.

        Args:
            dt (float): Time step between frames (1/FPS).
            sg_window (int): Savitzky-Golay window size (must be odd).
            sg_polyorder (int): Savitzky-Golay polynomial order.
            min_valid_frames (int): Minimum frames required to trigger optimization.
            fill_max_gap (int): Maximum consecutive NaNs to interpolate.
        """
        self.dt = float(dt)
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.min_valid_frames = min_valid_frames
        self.fill_max_gap = fill_max_gap


    def _interp_nans_1d(self, x: np.ndarray) -> np.ndarray:
        """
        Performs linear interpolation to fill all NaNs in a 1D array.

        Args:
            x (np.ndarray): 1D array with possible NaNs.
        Returns:
            np.ndarray: Interpolated 1D array.
        """
        x = x.copy()
        nan = np.isnan(x)
        if np.sum(~nan) < 2: 
            return x
        idx = np.arange(len(x))
        x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
        return x


    def _fill_gaps_xyz(self, xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Fills short temporal gaps in 3D trajectories using linear interpolation.

        Args:
            xyz (np.ndarray): (N, 3) trajectory array.
            valid (np.ndarray): (N,) boolean mask of valid frames.
        Returns:
            np.ndarray: Gap-filled trajectory.
        """
        out = xyz.copy()
        out[~valid] = np.nan
        idx_valid = np.where(valid)[0]
        if len(idx_valid) < 2: 
            return out
            
        for a, b in zip(idx_valid[:-1], idx_valid[1:]):
            gap = b - a - 1
            if 0 < gap <= self.fill_max_gap:
                for d in range(3):
                    out[a+1:b, d] = np.linspace(out[a, d], out[b, d], gap + 2)[1:-1]
        return out


    def optimize_positions(self, pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Main entry for position smoothing using Savitzky-Golay filtering.

        Args:
            pos (np.ndarray): (N, 3) raw position array.
            valid (np.ndarray): (N,) boolean mask.
        Returns:
            np.ndarray: Smoothed (N, 3) position array.
        """
        if len(pos) < self.min_valid_frames or np.sum(valid) < self.min_valid_frames:
            return pos.copy()
        
        # 1. Fill small gaps before filtering
        p = self._fill_gaps_xyz(pos, valid)
        out = p.copy()
        
        # 2. Window size adjustment (must be odd and <= length)
        w = self.sg_window
        if w % 2 == 0: w += 1
        w = min(w, len(p) if len(p) % 2 == 1 else len(p) - 1)
        
        if w < 5: 
            return np.nan_to_num(out, nan=0.0)

        # 3. Apply Savitzky-Golay axis-by-axis
        for d in range(3):
            xd = self._interp_nans_1d(out[:, d])
            try:
                out[:, d] = savgol_filter(xd, window_length=w, polyorder=min(self.sg_polyorder, w-2), mode="interp")
            except:
                out[:, d] = xd
        return out


class AriaHandsOptimizer:
    """
    Orchestrates the hand kinematics optimization process, handling both 
    hands across full sequences.
    """

    def __init__(self, cfg: Any, dt: float):
        """
        Args:
            cfg (ConfigBox/Namespace): Configuration containing SMOOTH_SG_*, etc.
            dt (float): Time step (1/FPS).
        """
        self.cfg = cfg
        self.dt = dt
        self.smoother = SimpleSmoother(dt,
                                       self.cfg.smooth_sg_window,
                                       self.cfg.smooth_sg_polyorder,
                                       self.cfg.smooth_min_valid_frames,
                                       self.cfg.smooth_fill_max_gap
                                       )
        self.mid_builder = MidpointFrameBuilder()


    def run(self, aria_hands: AriaHands) -> None:
        """
        Executes the full hand kinematics optimization pipeline.

        Args:
            aria_hands (AriaHands): The sequence-level hand container to be modified.
        """
        self._optimize_all_hands(aria_hands)
        print(f"[***] Smoothing pipeline finished using AriaHandsOptimizer.")


    def _optimize_all_hands(self, aria_hands: AriaHands) -> None:
        """
        Iterates through both hands and applies segmentation-based smoothing.
        """
        for hand_attr in ["hand_r", "hand_l"]:
            presence = np.array([getattr(f, hand_attr) is not None for f in aria_hands.hands], dtype=bool)
            if np.sum(presence) < self.cfg.smooth_min_valid_frames:
                continue

            # Extract continuous detection segments
            segments = self._extract_segments(presence)

            for (s, e) in segments:
                seg_len = e - s
                if seg_len < self.cfg.smooth_min_valid_frames:
                    continue

                # Prepare segment data
                frames = aria_hands.hands[s:e]
                hands = [getattr(fr, hand_attr) for fr in frames]
                valid_mask = np.array([h is not None for h in hands], dtype=bool)
                
                # --- Step 1: Position Smoothing (Savitzky-Golay) ---
                wrist_pos_raw = self._get_raw_pos_array(hands, "wrist_pose_raw_world")
                thumb_pos_raw = self._get_raw_pos_array(hands, "thumb_translation_raw_world")
                index_pos_raw = self._get_raw_pos_array(hands, "index_translation_raw_world")
                thumb_base_raw = self._get_raw_pos_array(hands, "thumb_base_raw_world")
                index_base_raw = self._get_raw_pos_array(hands, "index_base_raw_world")

                wrist_pos_opt = self.smoother.optimize_positions(wrist_pos_raw, valid_mask)
                thumb_pos_opt = self.smoother.optimize_positions(thumb_pos_raw, valid_mask)
                index_pos_opt = self.smoother.optimize_positions(index_pos_raw, valid_mask)
                thumb_base_opt = self.smoother.optimize_positions(thumb_base_raw, valid_mask)
                index_base_opt = self.smoother.optimize_positions(index_base_raw, valid_mask)
                
                mid_pos_opt = 0.5 * (thumb_pos_opt + index_pos_opt)

                # --- Step 2: Orientation Smoothing (EMA + Basis Re-ortho) ---
                # Wrist and Midpoint EMA Caches
                wrist_x_ema, wrist_y_ema = None, None
                mid_x_ema, mid_y_ema = None, None
                mid_prev_R = None 

                for k in range(seg_len):
                    h = hands[k]
                    if h is None: continue

                    # A. Update Wrist Pose (Smoothed Position + EMA Orientation)
                    h.wrist_pose_opt_world = np.eye(4)
                    h.wrist_pose_opt_world[:3, 3] = wrist_pos_opt[k]
                    
                    if h.wrist_pose_raw_world is not None:
                        wr_raw_R = h.wrist_pose_raw_world[:3, :3]
                        wr_x, wrist_x_ema = self._ema_unit_vec(wr_raw_R[:, 0], wrist_x_ema, alpha=self.cfg.smooth_ema_alpha_x)
                        wr_y, wrist_y_ema = self._ema_unit_vec(wr_raw_R[:, 1], wrist_y_ema, alpha=self.cfg.smooth_ema_alpha_y)
                        
                        # Gram-Schmidt Orthonormalization for Wrist
                        wr_z = np.cross(wr_x, wr_y)
                        wr_z /= (np.linalg.norm(wr_z) + 1e-6)
                        wr_y = np.cross(wr_z, wr_x)
                        h.wrist_pose_opt_world[:3, :3] = np.column_stack([wr_x, wr_y, wr_z])

                    # B. Update Smoothed Fingertips and MCP Bases
                    h.thumb_translation_opt_world = thumb_pos_opt[k]
                    h.index_translation_opt_world = index_pos_opt[k]
                    h.thumb_base_opt_world = thumb_base_opt[k]
                    h.index_base_opt_world = index_base_opt[k]

                    # C. Update Midpoint Pose (Smoothed Position + Gripper Frame Rebuild)
                    # Reconstruct Gripper Frame using smoothed rigid MCP bases
                    mid_R_rebuild = self.mid_builder.build(
                        thumb_w=thumb_pos_opt[k], 
                        index_w=index_pos_opt[k], 
                        thumb_base_w=thumb_base_opt[k], 
                        index_base_w=index_base_opt[k], 
                        wrist_w=wrist_pos_opt[k], 
                        midpoint_w=mid_pos_opt[k], 
                        prev_R=mid_prev_R
                    )

                    # Fallback to smoothed wrist orientation if construction fails
                    if mid_R_rebuild is None: 
                        mid_R_rebuild = mid_prev_R if mid_prev_R is not None else h.wrist_pose_opt_world[:3, :3]

                    # EMA Smoothing for Midpoint basis vectors
                    mid_x, mid_x_ema = self._ema_unit_vec(mid_R_rebuild[:, 0], mid_x_ema, alpha=self.cfg.smooth_ema_alpha_x)
                    mid_y, mid_y_ema = self._ema_unit_vec(mid_R_rebuild[:, 1], mid_y_ema, alpha=self.cfg.smooth_ema_alpha_y)
                    
                    # Gram-Schmidt Orthonormalization for Midpoint
                    mid_z = np.cross(mid_x, mid_y)
                    mid_z /= (np.linalg.norm(mid_z) + 1e-6)
                    mid_y = np.cross(mid_z, mid_x)
                    mid_R_opt = np.column_stack([mid_x, mid_y, mid_z])
                    
                    h.midpoint_translation_opt_world = mid_pos_opt[k]
                    h.midpoint_pose_opt_world = np.eye(4)
                    h.midpoint_pose_opt_world[:3, :3] = mid_R_opt
                    h.midpoint_pose_opt_world[:3, 3] = mid_pos_opt[k]
                    h.midpoint_orientation_opt_world = mid_R_opt.flatten()
                    mid_prev_R = mid_R_opt

                # --- Step 3 & 4: Velocity Computation (Finite Difference) ---
                self._assign_linear_vel_from_pos(hands, self.dt, key="wrist")
                self._assign_linear_vel_from_pos(hands, self.dt, key="midpoint")
                self._assign_angular_vel_from_rot(hands, self.dt, key="wrist")
                self._assign_angular_vel_from_rot(hands, self.dt, key="midpoint")


    @staticmethod
    def _get_raw_pos_array(hands: List[Optional[AriaHandData]], attr_name: str) -> np.ndarray:
        """Utility to extract position vectors or pose translations into a numpy array."""
        res = []
        for h in hands:
            val = getattr(h, attr_name) if h else None
            if val is not None and val.shape == (4, 4): 
                val = val[:3, 3]
            res.append(val if val is not None else np.zeros(3))
        return np.array(res)


    @staticmethod
    def _extract_segments(presence: np.ndarray) -> List[Tuple[int, int]]:
        """Identifies contiguous start/end indices of valid hand detections."""
        segs = []
        T, i = len(presence), 0
        while i < T:
            if not presence[i]:
                i += 1
                continue
            j = i + 1
            while j < T and presence[j]:
                j += 1
            segs.append((i, j))
            i = j
        return segs


    def _ema_unit_vec(self, v: np.ndarray, v_ema: Optional[np.ndarray], alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Exponential Moving Average to a unit vector with sign consistency.
        """
        v = np.asarray(v, dtype=np.float64)
        v /= (np.linalg.norm(v) + 1e-6)
        
        if v_ema is None: 
            return v, v.copy()
            
        # Ensure sign consistency to prevent 180-degree jumps during smoothing
        if float(np.dot(v, v_ema)) < 0.0: 
            v = -v
            
        v_new = (1.0 - float(alpha)) * v_ema + float(alpha) * v
        v_new /= (np.linalg.norm(v_new) + 1e-6)
        return v_new, v_new.copy()


    @staticmethod
    def _assign_linear_vel_from_pos(hands: List[AriaHandData], dt: float, key: str = "wrist") -> None:
        """Computes linear velocity v = (p_curr - p_prev) / dt."""
        prev_p = None
        for h in hands:
            if h is None:
                prev_p = None
                continue
            p = h.wrist_pose_opt_world[:3, 3] if key == "wrist" else h.midpoint_translation_opt_world
            if p is None:
                prev_p = None
                continue
            vel = (p - prev_p) / dt if prev_p is not None else np.zeros(3)
            if key == "wrist": h.wrist_lin_vel_opt_world = vel
            else: h.midpoint_lin_vel_opt_world = vel
            prev_p = p.copy()


    @staticmethod
    def _assign_angular_vel_from_rot(hands: List[AriaHandData], dt: float, key: str = "wrist") -> None:
        """
        Computes angular velocity using the rotational log-map: w = log(R_prev.T @ R_curr) / dt.
        """
        prev_R = None
        for h in hands:
            if h is None:
                prev_R = None
                continue
            
            curr_pose = h.wrist_pose_opt_world if key == "wrist" else h.midpoint_pose_opt_world
            if curr_pose is None:
                prev_R = None
                continue
                
            curr_R = curr_pose[:3, :3]
            
            if prev_R is None:
                ang_vel = np.zeros(3)
            else:
                try:
                    # Calculate relative rotation and map to rotation vector (Axis-Angle space)
                    from scipy.spatial.transform import Rotation as R_lib
                    rel_rot_mat = prev_R.T @ curr_R
                    ang_vel = R_lib.from_matrix(rel_rot_mat).as_rotvec() / dt
                except Exception:
                    ang_vel = np.zeros(3)
            
            if key == "wrist": h.wrist_ang_vel_opt_world = ang_vel
            else: h.midpoint_ang_vel_opt_world = ang_vel
                
            prev_R = curr_R.copy()