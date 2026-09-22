# -*- coding: utf-8 -*-
# @FileName: AriaHandTypes.py

"""
====================================================================================================
Project Aria Hand Tracking Data Types and Kinematic Utilities (AriaHandTypes.py)
====================================================================================================

Description:
    This module defines the fundamental data containers and geometric utilities for Project Aria 
    hand tracking. It manages the storage, calculation of joint angles, and serialization 
    of hand kinematics in both Camera and World coordinate systems.

Core Components:
    1. MidpointFrameBuilder: A robust utility to construct orthonormal "gripper-like" 
       coordinate frames (Pinch Frame) using MCP joints for stability.
    2. AriaHandsJointAngles: Computes a 20-DOF kinematic hand model (Flexion/Abduction).
    3. AriaHandData: Granular dataclass for per-hand poses, raw/optimized velocities, 
       and grasp states.
    4. AriaHands: Sequence-level container for multi-frame data aggregation and JSON export.

Technical Specifics:
    - Landmark Map: Based on the Project Aria 21-point hand tracking model.
    - Coordinate Frames: 'c' (Camera), 'd' (Device), 'w' (World).
    - Orthonormalization: Uses rigid MCP bases to prevent frame collapse during contact.

Generated Outputs:
    - [mps_path]/aria/all_data/[idx]/aria_hands.json (Complete kinematic state per frame)
====================================================================================================
"""


import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


class MidpointFrameBuilder:
    """
    Constructs an orthonormal "Gripper-like" orientation at the hand midpoint in World Space.
    Logic:
        x = normalize(index_base - thumb_base)
        y_proj = (midpoint - wrist) projected onto plane orthogonal to x
        z = x cross y
    """

    def __init__(
        self,
        eps_norm: float = 1e-6,
        eps_arm: float = 1e-5,
        eps_y: float = 1e-5,
        use_sign_consistency: bool = True
    ):
        """
        Initializes the builder with robustness thresholds.
        """
        self.eps_norm = float(eps_norm)
        self.eps_arm = float(eps_arm)
        self.eps_y = float(eps_y)
        self.use_sign_consistency = bool(use_sign_consistency)


    def _safe_normalize(self, v: np.ndarray) -> Optional[np.ndarray]:
        """Normalizes vector with epsilon check to avoid division by zero."""
        n = float(np.linalg.norm(v))
        if n < self.eps_norm:
            return None
        return v / n


    @staticmethod
    def _make_pose(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Assembles a 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = t
        return T


    def build(
        self,
        thumb_w: np.ndarray,
        index_w: np.ndarray,
        thumb_base_w: np.ndarray,   # Thumb MCP (index 6)
        index_base_w: np.ndarray,   # Index MCP (index 8)
        wrist_w: np.ndarray,
        midpoint_w: np.ndarray,
        prev_R: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Constructs a stable rotation matrix for the thumb-index midpoint.
        Uses MCP bases (index 6 and 8) to maintain a rigid X-axis during pinches.
        """
        # x = thumb_base -> index_base (Avoids singularity when tips touch)
        x_raw = index_base_w - thumb_base_w
        x = self._safe_normalize(x_raw)
        if x is None:
            return prev_R

        # y-axis using base_midpoint for rigid body assumption
        base_midpoint_w = (thumb_base_w + index_base_w) / 2.0
        arm = base_midpoint_w - wrist_w
        if float(np.linalg.norm(arm)) < self.eps_arm:
            return prev_R

        y_raw = arm 
        # Gram-Schmidt projection
        y_proj = y_raw - float(np.dot(y_raw, x)) * x
        y = self._safe_normalize(y_proj)
        if y is None:
            return prev_R

        z = self._safe_normalize(np.cross(x, y))
        if z is None:
            return prev_R

        y = self._safe_normalize(np.cross(z, x))
        if y is None:
            return prev_R

        # Sign consistency: prevent 180° flips
        if self.use_sign_consistency and prev_R is not None:
            if float(np.dot(prev_R[:, 0], x)) < 0.0:
                x, y = -x, -y
                z = np.cross(x, y)

        return np.column_stack([x, y, z])


@dataclass
class AriaHandsJointAngles:
    """
    Computes 20 joint angles in degrees from the 21-point Aria skeleton.
    Based on emg2pose definition for Flexion and Abduction.
    """
    data: Dict[str, float] = field(default_factory=dict)


    @classmethod
    def from_keypoints_3d(cls, kpts: np.ndarray):
        """Calculates angles based on bone vectors and palm plane projections."""
        if kpts is None or len(kpts) < 21:
            return cls(data={})

        angles = {}

        def get_angle(v1, v2):
            v1_n = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_n = v2 / (np.linalg.norm(v2) + 1e-6)
            return np.degrees(np.arccos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)))

        def get_abduction(bone_vec, ref_vec, plane_normal):
            def project(v): return v - np.dot(v, plane_normal) * plane_normal
            return get_angle(project(bone_vec), project(ref_vec))

        # Palm plane: Wrist(5), IndexMCP(8), MiddleMCP(11)
        v_w_m = kpts[11] - kpts[5]
        v_w_i = kpts[8] - kpts[5]
        palm_normal = np.cross(v_w_i, v_w_m)
        palm_normal /= (np.linalg.norm(palm_normal) + 1e-6)
        v_mid_prox_ref = kpts[12] - kpts[11]

        fingers_map = {
            'Index':  [8, 9, 10, 1], 'Middle': [11, 12, 13, 2],
            'Ring':   [14, 15, 16, 3], 'Pinky':  [17, 18, 19, 4]
        }

        for name, idxs in fingers_map.items():
            mcp, pip, dip, tip = idxs
            v_metacarpal = kpts[mcp] - kpts[5]
            v_prox, v_inter, v_dist = kpts[pip]-kpts[mcp], kpts[dip]-kpts[pip], kpts[tip]-kpts[dip]
            angles[f'{name}_MCP_Flex'] = get_angle(v_metacarpal, v_prox)
            angles[f'{name}_PIP_Flex'] = get_angle(v_prox, v_inter)
            angles[f'{name}_DIP_Flex'] = get_angle(v_inter, v_dist)
            angles[f'{name}_MCP_Abd'] = 0.0 if name == 'Middle' else get_abduction(v_prox, v_mid_prox_ref, palm_normal)

        v_thu_metacarpal, v_thu_prox, v_thu_dist = kpts[6]-kpts[5], kpts[7]-kpts[6], kpts[0]-kpts[7]
        angles['Thumb_CMC_Flex'] = get_angle(kpts[11]-kpts[5], v_thu_metacarpal)
        angles['Thumb_CMC_Abd']  = get_abduction(v_thu_metacarpal, v_mid_prox_ref, palm_normal)
        angles['Thumb_MCP_Flex'] = get_angle(v_thu_metacarpal, v_thu_prox)
        angles['Thumb_IP_Flex']  = get_angle(v_thu_prox, v_thu_dist)

        return cls(data=angles)


@dataclass
class AriaHandData:
    """Dataclass storing local tracking data and world-space kinematics for a single hand."""
    d2c: Optional[np.ndarray] = None
    c2w: Optional[np.ndarray] = None
    is_right: bool = None
    confidence: float = None
    wrist_pose: Optional[np.ndarray] = None
    palm_pose: Optional[np.ndarray] = None
    hand_keypoints_3d: Optional[np.ndarray] = None
    hand_keypoints_2d: Optional[np.ndarray] = None
    grasp_state: int = 0
    joint_angles: Optional[AriaHandsJointAngles] = None

    # Wrist Kinematics
    wrist_pose_raw_world: Optional[np.ndarray] = None
    wrist_pose_opt_world: Optional[np.ndarray] = None
    wrist_lin_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_ang_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_lin_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_ang_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Tips Kinematics
    index_translation_raw_world: Optional[np.ndarray] = None
    index_translation_opt_world: Optional[np.ndarray] = None
    thumb_translation_raw_world: Optional[np.ndarray] = None
    thumb_translation_opt_world: Optional[np.ndarray] = None

    # Midpoint Kinematics
    midpoint_translation_raw_world: Optional[np.ndarray] = None
    midpoint_orientation_raw_world: Optional[np.ndarray] = None
    midpoint_translation_opt_world: Optional[np.ndarray] = None
    midpoint_orientation_opt_world: Optional[np.ndarray] = None
    midpoint_pose_raw_world: Optional[np.ndarray] = None
    midpoint_pose_opt_world: Optional[np.ndarray] = None
    midpoint_lin_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midpoint_ang_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midpoint_lin_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midpoint_ang_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Rigid Bases
    thumb_base_raw_world: Optional[np.ndarray] = None
    index_base_raw_world: Optional[np.ndarray] = None
    thumb_base_opt_world: Optional[np.ndarray] = None
    index_base_opt_world: Optional[np.ndarray] = None


    @property
    def distance_midpoint2wrist_raw_world(self) -> Optional[float]:
        if self.wrist_pose_raw_world is None or self.midpoint_translation_raw_world is None: return None
        return float(np.linalg.norm(self.wrist_pose_raw_world[:3, 3] - self.midpoint_translation_raw_world))


    @property
    def distance_midpoint2wrist_opt_world(self) -> Optional[float]:
        if self.wrist_pose_opt_world is None or self.midpoint_pose_opt_world is None: return None
        return float(np.linalg.norm(self.wrist_pose_opt_world[:3, 3] - self.midpoint_pose_opt_world[:3, 3]))


@dataclass
class AriaHandsData:
    idx: int = 0
    ts: int = 0
    hand_r: Optional[AriaHandData] = None
    hand_l: Optional[AriaHandData] = None


@dataclass
class AriaHands:
    tss: List[int] = field(default_factory=list)
    hands: List[AriaHandsData] = field(default_factory=list)
    mps_path: str = None


    def __len__(self) -> int:
        return len(self.tss)


    @staticmethod
    def _safe_list(arr: Any) -> Any:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, (np.floating, np.integer)):
            return float(arr) if isinstance(arr, np.floating) else int(arr)
        return arr


    def save_aria_hands_json(self) -> None:
        """Serializes all hand tracking states into individual frame JSON files."""
        for i in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "preprocess", "all_data", f"{i:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            data = self.hands[i]

            def pack_hand(h: Optional[AriaHandData]):
                if h is None: return None
                # *** KEEPING ALL 31 FIELDS FROM ORIGINAL CODE ***
                return {
                    "d2c": self._safe_list(h.d2c),
                    "c2w": self._safe_list(h.c2w),
                    "confidence": h.confidence,
                    "grasp_state": h.grasp_state,
                    "wrist_pose": self._safe_list(h.wrist_pose),
                    "palm_pose": self._safe_list(h.palm_pose),
                    "kpts_3d": self._safe_list(h.hand_keypoints_3d),
                    "kpts_2d": self._safe_list(h.hand_keypoints_2d),
                    "joint_angles": h.joint_angles.data if h.joint_angles else {},
                    "wrist_pose_raw_world": self._safe_list(h.wrist_pose_raw_world),
                    "wrist_pose_opt_world": self._safe_list(h.wrist_pose_opt_world),
                    "wrist_lin_vel_raw_world": self._safe_list(h.wrist_lin_vel_raw_world),
                    "wrist_ang_vel_raw_world": self._safe_list(h.wrist_ang_vel_raw_world),
                    "wrist_lin_vel_opt_world": self._safe_list(h.wrist_lin_vel_opt_world),
                    "wrist_ang_vel_opt_world": self._safe_list(h.wrist_ang_vel_opt_world),
                    "index_translation_raw_world": self._safe_list(h.index_translation_raw_world),
                    "index_translation_opt_world": self._safe_list(h.index_translation_opt_world),
                    "thumb_translation_raw_world": self._safe_list(h.thumb_translation_raw_world),
                    "thumb_translation_opt_world": self._safe_list(h.thumb_translation_opt_world),
                    "midpoint_pose_raw_world": self._safe_list(h.midpoint_pose_raw_world),
                    "midpoint_pose_opt_world": self._safe_list(h.midpoint_pose_opt_world),
                    "midpoint_translation_raw_world": self._safe_list(h.midpoint_translation_raw_world),
                    "midpoint_orientation_raw_world": self._safe_list(h.midpoint_orientation_raw_world),
                    "midpoint_translation_opt_world": self._safe_list(h.midpoint_translation_opt_world),
                    "midpoint_orientation_opt_world": self._safe_list(h.midpoint_orientation_opt_world),
                    "midpoint_lin_vel_raw_world": self._safe_list(h.midpoint_lin_vel_raw_world),
                    "midpoint_ang_vel_raw_world": self._safe_list(h.midpoint_ang_vel_raw_world),
                    "midpoint_lin_vel_opt_world": self._safe_list(h.midpoint_lin_vel_opt_world),
                    "midpoint_ang_vel_opt_world": self._safe_list(h.midpoint_ang_vel_opt_world),
                    "distance_midpoint2wrist_raw_world": h.distance_midpoint2wrist_raw_world,
                    "distance_midpoint2wrist_opt_world": h.distance_midpoint2wrist_opt_world
                }

            json_data = {"idx": data.idx, "ts": data.ts, "hand_r": pack_hand(data.hand_r), "hand_l": pack_hand(data.hand_l)}
            with open(os.path.join(frame_dir, "aria_hands.json"), 'w') as f:
                json.dump(json_data, f, indent=4)


    def save_hands_json(self, filename: str = "aria_hands.json") -> None:
        """
        Generic serializer: saves hand tracking data to per-frame JSON files
        with a custom filename. This allows different hand detection methods
        (MediaPipe, WiLoR, HaMeR) to save their results in the same format
        but under different filenames (e.g., 'mediapipe_hands.json').
        """
        for i in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "preprocess", "all_data", f"{i:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            data = self.hands[i]

            sl = self._safe_list

            def pack_hand(h: Optional[AriaHandData]):
                if h is None: return None
                return {
                    "d2c": sl(h.d2c),
                    "c2w": sl(h.c2w),
                    "confidence": sl(h.confidence),
                    "grasp_state": sl(h.grasp_state),
                    "wrist_pose": sl(h.wrist_pose),
                    "palm_pose": sl(h.palm_pose),
                    "kpts_3d": sl(h.hand_keypoints_3d),
                    "kpts_2d": sl(h.hand_keypoints_2d),
                    "joint_angles": {k: sl(v) for k, v in (h.joint_angles.data if h.joint_angles else {}).items()},
                    "wrist_pose_raw_world": sl(h.wrist_pose_raw_world),
                    "wrist_pose_opt_world": sl(h.wrist_pose_opt_world),
                    "wrist_lin_vel_raw_world": sl(h.wrist_lin_vel_raw_world),
                    "wrist_ang_vel_raw_world": sl(h.wrist_ang_vel_raw_world),
                    "wrist_lin_vel_opt_world": sl(h.wrist_lin_vel_opt_world),
                    "wrist_ang_vel_opt_world": sl(h.wrist_ang_vel_opt_world),
                    "index_translation_raw_world": sl(h.index_translation_raw_world),
                    "index_translation_opt_world": sl(h.index_translation_opt_world),
                    "thumb_translation_raw_world": sl(h.thumb_translation_raw_world),
                    "thumb_translation_opt_world": sl(h.thumb_translation_opt_world),
                    "midpoint_pose_raw_world": sl(h.midpoint_pose_raw_world),
                    "midpoint_pose_opt_world": sl(h.midpoint_pose_opt_world),
                    "midpoint_translation_raw_world": sl(h.midpoint_translation_raw_world),
                    "midpoint_orientation_raw_world": sl(h.midpoint_orientation_raw_world),
                    "midpoint_translation_opt_world": sl(h.midpoint_translation_opt_world),
                    "midpoint_orientation_opt_world": sl(h.midpoint_orientation_opt_world),
                    "midpoint_lin_vel_raw_world": sl(h.midpoint_lin_vel_raw_world),
                    "midpoint_ang_vel_raw_world": sl(h.midpoint_ang_vel_raw_world),
                    "midpoint_lin_vel_opt_world": sl(h.midpoint_lin_vel_opt_world),
                    "midpoint_ang_vel_opt_world": sl(h.midpoint_ang_vel_opt_world),
                    "distance_midpoint2wrist_raw_world": sl(h.distance_midpoint2wrist_raw_world),
                    "distance_midpoint2wrist_opt_world": sl(h.distance_midpoint2wrist_opt_world)
                }

            json_data = {"idx": data.idx, "ts": sl(data.ts), "hand_r": pack_hand(data.hand_r), "hand_l": pack_hand(data.hand_l)}
            with open(os.path.join(frame_dir, filename), 'w') as f:
                json.dump(json_data, f, indent=4)