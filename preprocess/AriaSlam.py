# -*- coding: utf-8 -*-
# @FileName: AriaSlam.py

"""
====================================================================================================
Project Aria SLAM Kinematics and Trajectory Extraction (AriaSlam.py)
====================================================================================================

Description:
    This script extracts and processes 6-DOF device trajectories from Project Aria Machine 
    Perception Services (MPS). It transforms rotation matrices into continuous Euler angles 
    (Roll, Pitch, Yaw), calculates relative motion from the sequence start, and derives 
    instantaneous linear and angular velocities in World Space.

Core Functionalities:
    1.  Pose Decomposition: Converts 4x4 Camera-to-World (c2w) matrices into 3D translation 
        vectors and Z-Y-X convention Euler angles.
    2.  Yaw Unwrapping: Applies phase unwrapping to the yaw component to ensure a continuous 
        rotational trajectory, preventing 0-360 degree jumps.
    3.  Kinematic Computation: Calculates linear speed (m/s) and angular yaw-rate (rad/s) 
        using finite differences and timestamps.
    4.  Diagnostic Reporting: Generates professional 4x3 diagnostic plots comparing 
        absolute and relative motion metrics.
    5.  Trajectory Visualization: Projects the future device path onto 2D image frames and 
        renders a dynamic HUD showing real-time spatial parameters.

Generated Outputs:
    📁 [mps_path]/aria/
    ├── 🎬 aria_slam_vis.mp4          # Video overlay with trajectory projection and HUD.
    ├── 🖼️ aria_slam_analysis.png     # Professional 12-panel diagnostic chart.
    └── 📁 all_data/[idx]/
        └── 📄 aria_slam.json         # Per-frame kinematic metadata.

Technical Note:
    - Orientation: Uses RPY (Roll, Pitch, Yaw) Z-Y-X convention.
    - Coordinate Frame: World coordinates follow the MPS 'Closed Loop' reference.
====================================================================================================
"""

import os
import json
import math
import argparse
import numpy as np
import cv2
from typing import Tuple
from tqdm import tqdm

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from preprocess.AriaCam import AriaCamGenerator
from preprocess.AriaCamTypes import AriaCam
from preprocess.AriaSlamTypes import AriaSlam, AriaSlamFrame
from preprocess.AriaSlamOps import AriaSlamOps
from preprocess.AriaPhasesOps import AriaPhasesOps

from utils.utils_media import create_video_from_frames
from utils.utils_vis import draw_glass_rect
from utils.utils_io import load_cfg


class AriaSlamGenerator:
    """
    Handles the transformation of raw camera poses into refined SLAM kinematic data.
    """

    def __init__(self, mps_path: str, cfg_path: str , aria_cam: AriaCam):
        """
        Initializes the generator with project paths and pre-processed camera data.

        Args:
            mps_path (str): Path to the MPS output directory.
            cfg: Configuration object (ConfigBox).
            aria_cam (AriaCam): Container with camera poses and intrinsics.
        """
        self.mps_path = mps_path
        self.cfg_path = cfg_path
        self.aria_cam = aria_cam

        self.cfg = load_cfg(self.cfg_path)


    @staticmethod
    def rotmat_to_rpy_zyx(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Converts a 3x3 rotation matrix to Roll, Pitch, and Yaw angles (Z-Y-X convention).

        Args:
            R (np.ndarray): 3x3 Rotation matrix.
        Returns:
            Tuple[float, float, float]: (Roll, Pitch, Yaw) in radians.
        """
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy > 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw


    def get_aria_slam(self) -> AriaSlam:
        """
        Processes the sequence of camera poses to generate a full SLAM kinematic report.
        
        Steps:
            1. Pose Extraction: Decouples c2w matrices into T and RPY.
            2. Phase Unwrapping: Ensures continuous rotation for Yaw.
            3. Derivation: Computes delta poses and speeds.

        Returns:
            AriaSlam: Container populated with processed AriaSlamFrames.
        """
        aria_slam = AriaSlam(mps_path=self.mps_path)
        dt = 1.0 / max(1e-6, self.aria_cam.fps)
        
        # --- 1. Raw Extraction & RPY Conversion ---
        yaws_raw = []
        for i, cam_f in enumerate(self.aria_cam.cam):
            r, p, y = self.rotmat_to_rpy_zyx(cam_f.c2w[:3, :3])
            
            f = AriaSlamFrame(idx=cam_f.idx, ts=cam_f.ts, c2w=cam_f.c2w, k=cam_f.k)
            f.t_world = cam_f.c2w[:3, 3].astype(np.float64)
            f.rpy_deg = np.degrees([r, p, y])
            
            yaws_raw.append(y)
            aria_slam.frames.append(f)

        if not aria_slam.frames: 
            return aria_slam

        # --- 2. Yaw Phase Unwrapping ---
        # Ensures that rotation past 180/-180 becomes continuous (e.g., 181, 182...)
        yaws_unwrapped = np.unwrap(yaws_raw)
        
        # --- 3. Differential Kinematics and Delta Calculation ---
        t0 = aria_slam.frames[0].t_world.copy()
        rpy0 = aria_slam.frames[0].rpy_deg.copy()
        
        for i, f in enumerate(aria_slam.frames):
            f.yaw_unwrapped_deg = math.degrees(yaws_unwrapped[i])
            
            # Relative translation from origin (first frame)
            f.delta_t = f.t_world - t0
            
            # Relative rotation with wrap-around correction
            dy = f.rpy_deg - rpy0
            dy[2] = math.degrees(((math.radians(dy[2]) + math.pi) % (2*math.pi)) - math.pi)
            f.delta_rpy_deg = dy
            
            # Instantaneous Speed and Yaw Rate
            if i > 0:
                prev = aria_slam.frames[i-1]
                f.v = np.linalg.norm(f.t_world - prev.t_world) / dt
                f.w = abs(yaws_unwrapped[i] - yaws_unwrapped[i-1]) / dt

        # --- 4. Diagnostic Visualization ---
        aria_dir = os.path.join(self.mps_path, "preprocess")
        os.makedirs(aria_dir, exist_ok=True)
        AriaSlamOps.save_professional_plot(aria_slam, os.path.join(aria_dir, "aria_slam_analysis.png"))

        return aria_slam
    

    def draw_aria_slam_panel(self, img: np.ndarray, frame: AriaSlamFrame, aria_slam: AriaSlam) -> np.ndarray:
        return AriaSlamOps.draw_aria_slam_panel(img, frame, aria_slam, self.cfg.sensitivity_range)


    def draw_future_traj_on_image(self, img: np.ndarray, idx: int, slam: AriaSlam) -> np.ndarray:
        return AriaSlamOps.draw_future_traj_on_image(img, idx, slam, self.cfg.traj_future_len, self.cfg.traj_step, self.cfg.ground_offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/AriaSlam.yaml", help="Path to AriaSlam.yaml")
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

    # AriaSlam
    aria_slam_generator = AriaSlamGenerator(args.mps_path, args.cfg_path, aria_cam_rgb)
    aria_slam = aria_slam_generator.get_aria_slam()
    aria_slam.save_aria_slam_json()
    
    # Visualization
    if args.export_video:
        frames_all = []
        for idx, ts in enumerate(tqdm(aria_cam_rgb.tss, desc="Visualizing")):
            img = aria_cam_rgb.cam[idx].img.copy()

            # Overlay: Slam 3D Trajectory Projection
            img = aria_slam_generator.draw_future_traj_on_image(img, idx, aria_slam)

            # Overlay: Slam HUD Panel
            img = aria_slam_generator.draw_aria_slam_panel(img, aria_slam.frames[idx], aria_slam,)
            
            # Overlay: Phases HUD Panel
            img = AriaPhasesOps.draw_aria_phases_panel(img, idx, aria_cam_rgb.fps, len(aria_cam_rgb))

            frames_all.append(img)

        create_video_from_frames(frames_all, os.path.join(args.mps_path, "preprocess", "vis", "aria_slam_vis.mp4"), aria_cam_rgb.fps, args.export_gif)

# Execution Example:
# python -m preprocess.AriaSlam --mps_path  "./data/test/test_0/mps_test_0_000_vrs/"