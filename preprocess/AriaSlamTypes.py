# -*- coding: utf-8 -*-
# @FileName: AriaSlamTypes.py

"""
====================================================================================================
Project Aria SLAM Data Structures and Kinematic Serialization (AriaSlamTypes.py)
====================================================================================================

Description:
    This module defines the data containers for Project Aria SLAM (Simultaneous Localization 
    and Mapping) outputs. It handles the storage of 6-DOF poses, relative transformations 
    from the sequence start, and calculated dynamic metrics such as linear speed and 
    angular velocity.

Core Components:
    1. AriaSlamFrame: A dataclass representing the spatial and dynamic state of the device 
       at a specific timestamp.
    2. AriaSlam: A high-level container for a sequence of frames, providing automated 
       serialization to per-frame JSON files.

Generated Outputs & File Descriptions:
    📁 [mps_path]/aria/all_data/
    └── 📁 [00000...idx]/
        └── 📄 aria_slam.json
              - idx: Global frame index.
              - ts: Device timestamp in nanoseconds.
              - t_world: Absolute translation in World Frame (x, y, z).
              - rpy_deg: Absolute orientation in Euler angles (Roll, Pitch, Yaw).
              - delta_t_world: Translation relative to the first frame.
              - delta_rpy_deg: Rotation relative to the first frame.
              - linear_speed_mps: Instantaneous linear velocity magnitude (m/s).
              - angular_speed_rps: Instantaneous yaw rate (rad/s).
              - yaw_unwrapped_deg: Continuous yaw angle to prevent 0-360 jumps.

Technical Specifics:
    - Coordinate Frame: World coordinates follow the MPS 'Closed Loop' reference.
    - Units: Meters for translation, Degrees for Euler angles (except for rad/s velocity).
====================================================================================================
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class AriaSlamFrame:
    """
    Represents the spatial and dynamic state of a single SLAM-synchronized frame.

    Attributes:
        idx (int): Frame index.
        ts (int): Timestamp in nanoseconds.
        c2w (np.ndarray): 4x4 Camera-to-World transformation matrix.
        k (np.ndarray): 3x3 Intrinsic matrix.
        t_world (np.ndarray): 3D translation vector in world coordinates.
        rpy_deg (np.ndarray): Euler angles (Roll, Pitch, Yaw) in degrees.
        delta_t (np.ndarray): Translation relative to the sequence start.
        delta_rpy_deg (np.ndarray): Rotation relative to the sequence start.
        v (float): Linear speed magnitude in meters per second.
        w (float): Angular speed (yaw rate) in radians per second.
        yaw_unwrapped_deg (float): Continuous yaw angle in degrees.
    """
    idx: int
    ts: int
    c2w: np.ndarray
    k: np.ndarray
    
    # Absolute Pose Metadata
    t_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Relative Pose (Baseline: First Frame)
    delta_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_rpy_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Kinematic Dynamics
    v: float = 0.0          
    w: float = 0.0          
    yaw_unwrapped_deg: float = 0.0


@dataclass
class AriaSlam:
    """
    A sequence-level container for AriaSlamFrame objects.

    Attributes:
        mps_path (str): Root directory for the Project Aria MPS data.
        frames (List[AriaSlamFrame]): Chronological list of processed SLAM frames.
    """
    mps_path: str
    frames: List[AriaSlamFrame] = field(default_factory=list)


    def save_aria_slam_json(self) -> None:
        """
        Serializes each SLAM frame into a standardized JSON file.
        Files are organized by frame index in the [mps_path]/aria/all_data/ directory.
        """
        # Ensure the target directories exist
        os.makedirs(os.path.join(self.mps_path, "preprocess", "all_data"), exist_ok=True)
        all_data_dir = os.path.join(self.mps_path, "preprocess", "all_data")

        for f in self.frames:
            # Construct per-frame directory path
            frame_dir = os.path.join(all_data_dir, f"{f.idx:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            p = os.path.join(frame_dir, "aria_slam.json")
            
            # Map dataclass attributes to JSON dictionary
            data = {
                "idx": int(f.idx),
                "ts": int(f.ts),
                "t_world": f.t_world.tolist(),
                "rpy_deg": f.rpy_deg.tolist(),
                "delta_t_world": f.delta_t.tolist(),
                "delta_rpy_deg": f.delta_rpy_deg.tolist(),
                "linear_speed_mps": float(f.v),
                "angular_speed_rps": float(f.w),
                "yaw_unwrapped_deg": float(f.yaw_unwrapped_deg),
            }
            
            # Export with consistent indentation
            with open(p, "w") as jf:
                json.dump(data, jf, indent=2)