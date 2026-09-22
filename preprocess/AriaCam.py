# -*- coding: utf-8 -*-
# @FileName: AriaCam.py

"""
====================================================================================================
Project Aria Camera Data Preprocessing Pipeline (AriaCam.py)
====================================================================================================

Description:
    This script serves as a specialized generator for processing Project Aria camera streams 
    (RGB and SLAM). It bridges raw sensor data from .vrs files with Machine Perception 
    Services (MPS) outputs to create a standardized, rectified dataset.

Core Functionalities:
    1.  Image Rectification: Transforms raw, distorted Fisheye frames into Pinhole 
        (linear) projections.
    2.  Artifact Correction: Applies devignetting masks to ensure uniform brightness 
        across the frame.
    3.  Orientation Handling: Corrects the physical 90-degree sensor tilt typical of 
        Aria hardware by applying consistent image and calibration rotations.
    4.  Spatial Mapping: Calculates per-frame Intrinsics (K) and Extrinsics (T) 
        transforming between Camera, Device, and World coordinate systems.
    5.  Temporal Alignment: Synchronizes camera frames using MPS-corrected timestamps 
        to ensure consistency with hand-tracking and pose data.

Generated Outputs & File Descriptions:
    The script populates the source MPS directory with a new 'aria_cam' subdirectory:
    
    📁 [mps_path]/aria_cam/
    ├── 📄 aria_cam_[label].json
    │     - The primary metadata file.
    │     - Contains global camera parameters (FPS, Resolution, FOV).
    │     - Contains per-frame lists:
    │         - 'tss': Device timestamps in nanoseconds.
    │         - 'k', 'd': Intrinsic matrix (3x3) and distortion (zeroed for pinhole).
    │         - 'c2w', 'd2w', 'c2d': 4x4 Transformation matrices (Camera-to-World, etc.).
    │
    ├── 🎬 aria_cam_[label].mp4
    │     - The processed video stream.
    │     - Visual properties: Undistorted, Rotated 90° CW, and BGR color-corrected.
    │     - Used for visual verification and temporal feature extraction.
    │
    └── 🎞️ aria_cam_[label].gif (Optional)
          - A lightweight preview animation for quick data inspection.

Input Requirements:
    - VRS File: Located at [mps_path]/sample.vrs.
    - MPS Data: Closed-loop poses and hand tracking results (required for frame range clipping).
    - YAML Config: Resolution and FOV settings (default: ./cfg/preprocess/AriaCam.yaml).

Technical Note:
    - Coordination System: World coordinates follow the MPS 'Closed Loop' frame.
    - Image Processing: Uses Bilinear Interpolation for undistortion to balance speed and quality.
====================================================================================================
"""

import os
import cv2
import argparse
import subprocess
import numpy as np
import math
from tqdm import tqdm
from typing import List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP

from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
from projectaria_tools.core.image import InterpolationMethod

from utils.utils_io import load_cfg
from preprocess.AriaCamTypes import AriaCam, AriaCamData


class AriaCamGenerator:
    """
    A class to manage Project Aria camera data extraction, calibration, and image processing.
    It handles undistortion, devignetting, rotation, and coordinate transformations (Intrinsics/Extrinsics).
    """

    def __init__(self, mps_path: str, cfg_path: str, vrs_provider: VrsDataProvider, mps_provider: MpsDataProvider, label: str):
        """
        Initializes the generator with data providers and configuration.

        Args:
            mps_path (str): Path to the Machine Perception Services (MPS) output directory.
            cfg_path (str): Path to the YAML configuration file.
            vrs_provider (VrsDataProvider): Provider for raw VRS sensor data.
            mps_provider (MpsDataProvider): Provider for processed MPS data (poses, etc.).
            label (str): Camera identifier (e.g., 'rgb', 'slam_l', 'slam_r').
        """
        self.mps_path = mps_path
        self.cfg_path = cfg_path
        self.vrs_provider = vrs_provider
        self.mps_provider = mps_provider
        self.label = label

        self._init_cam()


    def _init_cam(self) -> None:
        """
        Loads configuration and triggers the initialization of masks and calibrations.
        """
        self.cfg = load_cfg(self.cfg_path)
        self.cam_label = self.cfg.cam_labels[self.label]
        self.cam_cfg = self.cfg.cam_size[self.label]

        self._init_devignetting_mask()
        self._init_cam_calib()


    def _init_devignetting_mask(self) -> None:
        """
        Sets the devignetting mask folder path and loads the specific mask for the current camera.
        """
        self.device_calib = self.vrs_provider.get_device_calibration()
        self.device_calib.set_devignetting_mask_folder_path(self._init_devignetting_mask_folder_path())
        self.devignetting_mask = self.device_calib.load_devignetting_mask(self.cam_label)


    def _init_cam_calib(self) -> None:
        """
        Initializes stream IDs and sets up both Fisheye (raw) and Pinhole (rectified) calibrations.
        """
        self.cam_stream_id = self.vrs_provider.get_stream_id_from_label(self.cam_label)
        self._init_cam_fisheye_calib()
        self._init_cam_pinhole_calib() 


    def _init_cam_fisheye_calib(self) -> None:
        """
        Retrieves the raw fisheye camera calibration from the VRS device calibration.
        """
        self.cam_fisheye_calib = self.device_calib.get_camera_calib(self.cam_label)


    def _init_cam_pinhole_calib(self) -> None:
        """
        Creates linear (pinhole) camera calibrations for undistortion. 
        Includes a standard version and a rotated version to match the 90-degree sensor orientation.
        """
        # Standard Pinhole Calibration
        self.cam_pinhole_calib = calibration.get_linear_camera_calibration(
            image_width = self.cam_cfg.w, 
            image_height = self.cam_cfg.h,
            focal_length = self.cam_cfg.w / (2 * math.tan(math.radians(self.cam_cfg.fov / 2))), 
            label = self.cam_label, 
            T_Device_Camera = self.cam_fisheye_calib.get_transform_device_camera(),
        )
        # Apply 90-degree clockwise rotation to the calibration model
        self.cam_pinhole_calib = calibration.rotate_camera_calib_cw90deg(self.cam_pinhole_calib)

        # Rotated Pinhole Calibration (Swapped Width/Height)
        self.cam_pinhole_calib_rot = calibration.get_linear_camera_calibration(
                image_width = self.cam_cfg.h, 
                image_height = self.cam_cfg.w,
                focal_length = self.cam_cfg.w / (2 * math.tan(math.radians(self.cam_cfg.fov / 2))), 
                label = self.cam_label, 
                T_Device_Camera = self.cam_fisheye_calib.get_transform_device_camera(),
            )


    def _init_devignetting_mask_folder_path(self) -> str:
        """
        Locates the devignetting masks. Downloads them from Project Aria servers if not found locally.

        Returns:
            str: The local directory path containing devignetting masks.
        """
        devignetting_mask_folder_path = os.path.join(os.path.dirname(__file__), "aria_devignetting_masks")
        if not os.path.exists(devignetting_mask_folder_path):
            print(f"Directory {devignetting_mask_folder_path} not found. Downloading devignetting masks...")
            try:
                download_cmd = 'curl -L -o devignetting_masks.zip "https://www.projectaria.com/async/sample/download/?bucket=core&filename=devignetting_masks_bin.zip"'
                subprocess.run(download_cmd, shell=True, check=True)
                unzip_cmd = f'unzip -o ./devignetting_masks.zip -d {devignetting_mask_folder_path}'
                subprocess.run(unzip_cmd, shell=True, check=True)
                os.remove("./devignetting_masks.zip")
                print("Successfully downloaded and extracted devignetting masks.")
            except Exception as e:
                print(f"Error downloading masks: {e}")
        return devignetting_mask_folder_path


    def _calculate_fps(self, tss: List[int]) -> int:
        """
        Calculates the average Frames Per Second (FPS) based on a list of timestamps.

        Args:
            tss (List[int]): A list of timestamps in nanoseconds.

        Returns:
            int: The calculated FPS, rounded to the nearest integer.
        """
        if len(tss) < 2:
            return 10
        first_ts = tss[0]
        last_ts = tss[-1]
        duration_sec = (last_ts - first_ts) / 1e9
        fps = (len(tss) - 1) / duration_sec
        fps = int(Decimal(str(fps)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        return fps


    def _get_aria_cam_data(self, idx: int) -> Optional[AriaCamData]:
        """
        Processes a single frame: fetches data, corrects vignetting, undistorts, rotates, and recolors.

        Args:
            idx (int): The index of the frame in the VRS stream.

        Returns:
            Optional[AriaCamData]: An object containing processed image and spatial metadata.
        """
        ts = self._get_ts(idx)

        fov = self.cam_cfg.fov
        h, w = self.cam_cfg.h, self.cam_cfg.w
        k, d = self._get_intrinsics(self.cam_pinhole_calib_rot)
        c2w, c2d, d2w = self._get_extrinsics(ts, self.cam_pinhole_calib)

        img = self._get_orig_img(idx)
        img = self._devignette_img(img)
        img = self._undistort_img(img)
        img = self._rotate_img(img)
        img = self._recolor_img(img)

        return AriaCamData(ts=ts,
                           img=img, fov=fov,
                           h=h, w=w, k=k, d=d,
                           c2w=c2w, c2d=c2d, d2w=d2w)


    def _get_ts(self, idx: int) -> int:
        """
        Retrieves the corrected timestamp for a given frame index, aligned with the RGB stream.

        Args:
            idx (int): Frame index.

        Returns:
            int: Corrected capture timestamp in nanoseconds.
        """
        # use ts in rgb_data as the common reference timestamp
        rgb_data = self.vrs_provider.get_image_data_by_index(self.vrs_provider.get_stream_id_from_label("camera-rgb"), idx)
        ts = rgb_data[1].capture_timestamp_ns
        ts = self.mps_provider.get_rgb_corrected_timestamp_ns(ts, TimeQueryOptions.CLOSEST)
        return ts


    def _get_orig_img(self, idx: int) -> np.ndarray:
        """
        Fetches the raw image array from the VRS provider.

        Args:
            idx (int): Frame index.

        Returns:
            np.ndarray: Raw image data.
        """
        data = self.vrs_provider.get_image_data_by_index(self.cam_stream_id, idx)
        img = np.copy(data[0].to_numpy_array())
        return img


    def _devignette_img(self, img: np.ndarray) -> np.ndarray:
        """
        Applies devignetting to correct uneven brightness across the image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Devignetted image.
        """
        img = calibration.devignetting(img, self.devignetting_mask).astype(np.uint8)
        return img


    def _undistort_img(self, img: np.ndarray) -> np.ndarray:
        """
        Rectifies the fisheye image into a pinhole projection.

        Args:
            img (np.ndarray): Fisheye image.

        Returns:
            np.ndarray: Undistorted pinhole image.
        """
        img = calibration.distort_by_calibration(img, self.cam_pinhole_calib, self.cam_fisheye_calib, InterpolationMethod.BILINEAR)
        return img


    def _rotate_img(self, img: np.ndarray) -> np.ndarray:
        """
        Rotates the image to correct the physical sensor orientation.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Rotated image (90 degrees clockwise).
        """
        img = np.rot90(img, k=3)
        img = np.ascontiguousarray(img)
        return img


    def _recolor_img(self, img: np.ndarray) -> np.ndarray:
        """
        Converts the image to BGR format for OpenCV compatibility.

        Args:
            img (np.ndarray): Input image (RGB or Gray).

        Returns:
            np.ndarray: BGR image.
        """
        if self.label == 'rgb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    def _get_intrinsics(self, cam_calib) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the intrinsic matrix K and distortion coefficients D.
        Note: The K matrix is adjusted for the 90-degree sensor rotation.

        Args:
            cam_calib: The Project Aria camera calibration object.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (3x3 Intrinsic matrix, Distortion coefficients).
        """
        fx, fy = cam_calib.get_focal_lengths()
        cx, cy = cam_calib.get_principal_point()
        # Adjusted K matrix for rotated coordinate system
        k = np.array([
            [fy,           0,            self.cam_cfg.w - cy], 
            [0,            fx,           cx], 
            [0,            0,            1]
        ])
        d = np.zeros(5) # Distortion is zeroed as we are using a rectified pinhole model
        return k, d


    def _get_extrinsics(self, ts: int, cam_calib) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the extrinsic transformations for a specific timestamp.

        Args:
            ts (int): Timestamp in nanoseconds.
            cam_calib: Camera calibration object.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (T_c2w, T_c2d, T_d2w) 4x4 transformation matrices.
        """
        # T_d2w: Device to World
        T_d2w = self._get_T_d2w(ts)
        if np.array_equal(T_d2w, np.eye(4)):
            return np.eye(4), np.eye(4), np.eye(4)
        
        # T_c2d: Camera to Device
        T_c2d = cam_calib.get_transform_device_camera().to_matrix()

        # T_c2w: Camera to World
        T_c2w = T_d2w @ T_c2d
        return T_c2w, T_c2d, T_d2w


    def _get_T_d2w(self, ts: int) -> np.ndarray:
        """
        Retrieves the 4x4 Device-to-World transformation matrix from MPS poses.

        Args:
            ts (int): Timestamp in nanoseconds.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        pose = self.mps_provider.get_interpolated_closed_loop_pose(ts)
        if pose is not None:
            T_d2w = pose.transform_world_device.to_matrix()
        else:
            fallback_pose = self.mps_provider.get_closed_loop_pose(ts, TimeQueryOptions.CLOSEST)
            if fallback_pose:
                T_d2w = fallback_pose.transform_world_device.to_matrix()
            else:
                T_d2w = np.eye(4)
        return T_d2w


    def get_aria_cam(self, start_idx: int, end_idx: int) -> Optional[AriaCam]:
        """
        Processes a sequence of frames and encapsulates them into an AriaCam object.

        Args:
            start_idx (int): Start index for frame processing.
            end_idx (int): End index for frame processing.

        Returns:
            Optional[AriaCam]: A container object containing all processed frames and metadata.
        """
        aria_cam = AriaCam()
        for idx in tqdm(range(start_idx, end_idx + 1), desc=f"Processing Aria Cam [{self.label}]"):
            aria_cam_data = self._get_aria_cam_data(idx)
            aria_cam_data.idx = idx - start_idx # Re-index starting from 0
            aria_cam.tss.append(aria_cam_data.ts)
            aria_cam.cam.append(aria_cam_data)
        
        # Populate global camera properties from the first frame
        aria_cam.fps = self._calculate_fps(aria_cam.tss)
        aria_cam.fov = aria_cam.cam[0].fov
        aria_cam.h = aria_cam.cam[0].h
        aria_cam.w = aria_cam.cam[0].w
        aria_cam.k = aria_cam.cam[0].k
        aria_cam.d = aria_cam.cam[0].d
        aria_cam.first_ts = aria_cam.cam[0].ts
        aria_cam.mps_path = self.mps_path
        return aria_cam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/AriaCam.yaml", help="Path to AriaCam.yaml")
    parser.add_argument("--no-video", action="store_false", dest="export_video", help="Disable video export")
    parser.add_argument("--no-gif", action="store_false", dest="export_gif", help="Disable GIF export")
    args = parser.parse_args()
    
    # Initialize data providers
    mps_path = args.mps_path
    vrs_path = os.path.join(mps_path, "sample.vrs")
    vrs_provider = data_provider.create_vrs_data_provider(vrs_path)
    mps_provider = MpsDataProvider(MpsDataPathsProvider(mps_path).get_data_paths())
    cfg_path = args.cfg_path

    # Determine frame range based on MPS hand tracking availability
    hand_tracking_results_path = os.path.join(mps_path, "hand_tracking", "hand_tracking_results.csv")
    rgb_timestamps_ns = vrs_provider.get_timestamps_ns(vrs_provider.get_stream_id_from_label("camera-rgb"), TimeDomain.DEVICE_TIME)
    aria_hands_mps = mps.hand_tracking.read_hand_tracking_results(hand_tracking_results_path)
    
    total_rgb_frames = len(rgb_timestamps_ns)
    mps_not_detected_number = total_rgb_frames - len(aria_hands_mps)
    print(f'[***] Total RGB frames: {total_rgb_frames}, Aria Hands MPS detected: {len(aria_hands_mps)}, Not detected: {mps_not_detected_number}')
    
    start_idx = mps_not_detected_number
    num_frames = len(aria_hands_mps)
    end_idx = start_idx + num_frames - 1
    
    ######### RGB #########
    aria_cam_rgb_generator = AriaCamGenerator(mps_path, cfg_path, vrs_provider, mps_provider, label='rgb')
    aria_cam_rgb = aria_cam_rgb_generator.get_aria_cam(start_idx, end_idx)
    aria_cam_rgb.save_aria_cam_json(label='rgb')
    aria_cam_rgb.save_aria_cam_video_orig(args.export_video, args.export_gif, label='rgb')

    # ######### SLAM-LEFT #########
    # aria_cam_slam_l_generator = AriaCamGenerator(mps_path, cfg_path, vrs_provider, mps_provider, label='slam_l')
    # aria_cam_slam_l = aria_cam_slam_l_generator.get_aria_cam(start_idx, end_idx)
    # aria_cam_slam_l.save_aria_cam_json(label='slam_l')
    # aria_cam_slam_l.save_aria_cam_video_orig(args.export_video, args.export_gif, label='slam_l')
    
    # ######### SLAM-RIGHT #########
    # aria_cam_slam_r_generator = AriaCamGenerator(mps_path, cfg_path, vrs_provider, mps_provider, label='slam_r')
    # aria_cam_slam_r = aria_cam_slam_r_generator.get_aria_cam(start_idx, end_idx)
    # aria_cam_slam_r.save_aria_cam_json(label='slam_r')
    # aria_cam_slam_r.save_aria_cam_video_orig(args.export_video, args.export_gif, label='slam_r')

# Execution Example:
# python -m preprocess.AriaCam --mps_path  "./data/test/test_0/mps_test_0_000_vrs/"