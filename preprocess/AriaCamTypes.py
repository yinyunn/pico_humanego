# -*- coding: utf-8 -*-
# @FileName: AriaCamTypes.py

"""
====================================================================================================
Project Aria Camera Data Structure and Serialization (AriaCamTypes.py)
====================================================================================================

Description:
    This module defines the data containers and serialization logic for processed Project Aria 
    camera data. It utilizes Python dataclasses to store per-frame multi-modal information 
    (images, intrinsics, extrinsics) and provides methods for exporting data to disk in 
    standardized formats (JSON, PNG, MP4).

Core Components:
    1. AriaCamData: A dataclass representing a single synchronized camera frame, including 
       its spatial transformations (Camera-to-World, etc.) and the processed image.
    2. AriaCam: A container class for a sequence of AriaCamData frames, managing global 
       metadata like FPS and providing export utilities.

Generated Outputs & File Descriptions:
    The methods in this script populate the following structure within the MPS directory:

    📁 [mps_path]/aria/
    ├── 📄 aria_cam_[label]_config.json      # Summary metadata (FPS, global K, total frames).
    ├── 📁 vis/
    │   └── 🎬 aria_cam_[label].mp4          # Processed visualization video.
    └── 📁 all_data/
        └── 📁 [00000...idx]/
            ├── 🖼️ [label].png               # Rectified and rotated frame image.
            └── 📄 aria_cam_[label].json     # Per-frame spatial-temporal metadata.

Technical Specifics:
    - Extrinsics: Follows the relation T_c2w = T_d2w @ T_c2d.
    - Coordinate Frames: 'c' (Camera), 'd' (Device), 'w' (World).
====================================================================================================
"""

import os
import cv2
import json
import numpy as np

from dataclasses import dataclass, field
from typing import List, Any

from utils.utils_media import create_video_from_frames


@dataclass
class AriaCamData:
    """
    Represents the spatial and visual state of a single camera frame.

    Attributes:
        idx (int): Global frame index within the sequence.
        ts (int): Corrected capture timestamp in nanoseconds.
        img (np.ndarray): Processed image array (H, W, 3) in BGR format.
        fov (float): Vertical Field of View in degrees.
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        k (np.ndarray): 3x3 Intrinsic matrix.
        d (np.ndarray): Distortion coefficients (usually zeroed for pinhole).
        c2w (np.ndarray): 4x4 Extrinsic matrix: Camera-to-World (T_c2w).
        c2d (np.ndarray): 4x4 Extrinsic matrix: Camera-to-Device (T_c2d).
        d2w (np.ndarray): 4x4 Extrinsic matrix: Device-to-World (T_d2w).
    """
    idx: int = 0
    ts: int = 0
    img: np.ndarray = None
    fov: float = 0.0
    h: int = 0
    w: int = 0
    k: np.ndarray = None
    d: np.ndarray = None
    c2w: np.ndarray = None
    c2d: np.ndarray = None
    d2w: np.ndarray = None


@dataclass
class AriaCam:
    """
    A sequence-level container for AriaCamData, including export and serialization logic.

    Attributes:
        tss (List[int]): List of timestamps for all frames.
        cam (List[AriaCamData]): List of per-frame data objects.
        fps (int): Calculated average frames per second.
        first_ts (int): Timestamp of the first frame.
        fov (float): Vertical FOV.
        h (int): Global image height.
        w (int): Global image width.
        k (np.ndarray): Global intrinsic matrix (from first frame).
        d (np.ndarray): Global distortion coefficients.
        c2d (np.ndarray): Camera-to-Device transformation.
        mps_path (str): Root directory path for the MPS data.
    """
    tss: List[int] = field(default_factory=list)
    cam: List[AriaCamData] = field(default_factory=list)
    
    fps: int = 0
    first_ts: int = 0
    fov: float = 0.0
    h: int = 0
    w: int = 0
    k: np.ndarray = None
    d: np.ndarray = None
    c2d: np.ndarray = None
    
    mps_path: str = None


    def __len__(self) -> int:
        """Returns the number of frames stored."""
        return len(self.tss)


    @staticmethod
    def _safe_list(arr: Any) -> Any:
        """
        Converts a numpy array to a nested list for JSON serialization.
        
        Args:
            arr (Any): Input array or list.
        Returns:
            Any: Python list if input was np.ndarray, else original input.
        """
        return arr.tolist() if isinstance(arr, np.ndarray) else arr


    def save_aria_cam_json(self, label: str) -> None:
        """
        Saves individual frame images and per-frame JSON metadata to the filesystem.
        The data is saved in [mps_path]/aria/all_data/[idx]/.

        Args:
            label (str): Identifier for the camera stream (e.g., 'rgb').
        """
        for idx in range(len(self.tss)):
            # Define directory for the specific frame
            frame_dir = os.path.join(self.mps_path, "preprocess", "all_data", f"{idx:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            img_path = os.path.join(frame_dir, f"{label}.png")

            cam = self.cam[idx]
            # Save the processed image
            cv2.imwrite(img_path, cam.img)
        
            # Compile per-frame metadata
            json_data = {
                "idx": cam.idx,
                "ts": cam.ts,
                "fov": cam.fov,
                "h": cam.h, 
                "w": cam.w,
                "k": self._safe_list(cam.k), 
                "d": self._safe_list(cam.d), 
                "c2w": self._safe_list(cam.c2w),
                "c2d": self._safe_list(cam.c2d),
                "d2w": self._safe_list(cam.d2w),
                f"{label}_path": os.path.join("preprocess", "all_data", f"{idx:05d}", f"{label}.png"),
                "fps": self.fps
            }
            
            # Write individual JSON metadata
            with open(os.path.join(frame_dir, f"aria_cam_{label}.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

        # Trigger summary config save
        self._save_aria_cam_config_json(label)


    def _save_aria_cam_config_json(self, label: str) -> None:
        """
        Saves a global configuration summary for the camera stream.

        Args:
            label (str): Identifier for the camera stream.
        """
        save_path = os.path.join(self.mps_path, "preprocess", f"aria_cam_{label}_config.json")
        summary_data = {
            "total_frames": len(self),
            "fps": self.fps,
            "first_ts": self.first_ts,
            "h": self.h,
            "w": self.w,
            "k": self._safe_list(self.k),
            "d": self._safe_list(self.d),
            "c2d": self._safe_list(self.c2d),
        }
        with open(save_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"[***] JSON Summary saved to: {save_path}")


    def save_aria_cam_video_orig(self, export_video: bool, export_gif: bool, label: str) -> None:
        """
        Aggregates frames into a video or GIF for visualization purposes.

        Args:
            export_video (bool): Whether to generate an MP4 file.
            export_gif (bool): Whether to generate a GIF file.
            label (str): Identifier for the camera stream.
        """
        frames_all = []
        for idx in range(len(self.tss)):
            cam = self.cam[idx]
            frames_all.append(cam.img)
            
        if export_video:
            save_path = os.path.join(self.mps_path, "preprocess", "vis", f"aria_cam_{label}.mp4")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            create_video_from_frames(
                frames=frames_all, 
                save_path=save_path,
                fps=self.fps,
                export_gif=export_gif,
                ratio=10
            )