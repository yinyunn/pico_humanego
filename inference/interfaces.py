# -*- coding: utf-8 -*-
# @FileName: interfaces.py
"""
Hardware & perception interfaces for HumanEgo real-world inference.

This is a *reference template*. It will NOT run out of the box on your setup —
it is meant to show the standard structure of a HumanEgo inference stack so you
can drop in your own camera, robot and perception and reuse the rest unchanged.

The whole inference loop (`run_inference.py`) is written against the three
abstract interfaces below, so the policy is completely decoupled from hardware:

    Camera      -> any calibrated RGB-D camera   (paper example: CamRS.py / Intel RealSense)
    RobotArm    -> any arm with a parallel jaw   (paper example: RobotArmTrossen.py / Trossen)
    Perception  -> any object-pose + clean-image module
                   (paper example: DINO-SAM detection + LaMa inpainting + PCA pose)

To port HumanEgo to your robot you only implement these three; `policy.py`,
`controller.py` and the main loop stay the same.

────────────────────────────────────────────────────────────────────────────
FRAME & UNIT CONVENTIONS  (most porting bugs live here — read carefully)
────────────────────────────────────────────────────────────────────────────
  * Every pose is a 4x4 homogeneous SE(3) matrix (np.float32/64).
  * Suffix `_in_cam`  == expressed in the CAMERA OPTICAL frame
        (OpenCV convention: +x right, +y down, +z forward / into the scene).
  * The camera optical frame is the single shared "world" frame for one episode;
    ICTs are built relative to it (or to an anchor object inside it).
  * Positions are in METERS. Rotations are proper 3x3 rotation matrices.
  * Grasp / gripper scalars are in [0, 1]:  0.0 = fully OPEN, 1.0 = fully CLOSED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np


# =====================================================================
# Data containers
# =====================================================================

@dataclass
class Frame:
    """One synchronized RGB-D sample from the camera."""
    rgb: np.ndarray      # (H, W, 3) uint8, BGR (OpenCV order)
    depth_m: np.ndarray  # (H, W)    float32, metric depth in meters (0 = invalid)
    K: np.ndarray        # (3, 3)    pinhole intrinsics matching `rgb`'s resolution


@dataclass
class ObjectState:
    """Per-object scene description produced once by Perception.estimate_objects()."""
    T_in_cam: np.ndarray                              # (4, 4) object 6DoF pose in camera frame
    kpts_local: np.ndarray = field(                   # (N, 3) keypoints in the object's OWN frame
        default_factory=lambda: np.zeros((0, 3), np.float32)
    )                                                 # (only needed if the policy uses PCD features)


# =====================================================================
# Hardware interfaces  — implement these for your own robot / camera
# =====================================================================

class Camera(Protocol):
    """A calibrated RGB-D camera.  Paper example: `CamRS` (Intel RealSense).

    `CamRS` already exposes `get_rgbd()` -> CamRSData(rgb, depth_m, ...) and a
    `k_rgb` intrinsics attribute; wrap it in a tiny adapter (see run_inference.py)
    to satisfy this interface.
    """

    def get_frame(self) -> Frame:
        """Grab one aligned RGB-D frame + intrinsics."""
        ...

    def close(self) -> None:
        ...


class RobotArm(Protocol):
    """One robot arm + parallel gripper, hand-eye calibrated to the camera.
    Paper example: `RobotArmTrossen` (Trossen ViperX / WidowX).

    The single most important quantity you must provide is the hand-eye
    extrinsic `T_base_in_cam` (4x4): it places the robot base in the camera
    frame and is what lets the policy command poses the arm can reach.
    Obtain it from a hand-eye calibration; a wrong extrinsic is the #1 cause
    of "the robot moves to the wrong place".
    """

    T_base_in_cam: np.ndarray  # (4, 4) robot base -> camera optical frame

    def get_T_ee_in_cam(self) -> np.ndarray:
        """Current end-effector pose (4x4) in the camera frame (FK ∘ extrinsic)."""
        ...

    def move_ee_in_cam(
        self, T_ee_in_cam: np.ndarray, duration: float, blocking: bool = False
    ) -> bool:
        """Servo the EE to a Cartesian target (camera frame). IK is the driver's job.
        Returns False if the target is unreachable / IK failed."""
        ...

    def get_gripper(self) -> float:
        """Current gripper opening, normalized to [0, 1]  (0 open .. 1 closed)."""
        ...

    def set_gripper(self, value: float, blocking: bool = False) -> None:
        """Command gripper in [0, 1] (0 open .. 1 closed)."""
        ...

    def go_home(self, blocking: bool = True) -> None:
        ...

    def close(self) -> None:
        ...


# =====================================================================
# Perception interface  — the heaviest part to port
# =====================================================================

class Perception(Protocol):
    """Turns raw RGB-D into (a) object 6DoF poses and (b) a clean, embodiment-
    agnostic RGB image — the two things the policy needs from the world.

    Paper reference implementation (see preprocess/ and run_inference.py):
        estimate_objects : open-vocab detect+segment (DINO-SAM) -> lift mask
                           pixels to 3D via depth -> robust filter -> PCA 6DoF pose.
        make_clean_image : inpaint the real arm out of the frame (LaMa), then
                           render a virtual gripper + object keypoints in its
                           place, matching how the model was TRAINED.

    These are deliberately swappable: any detector/pose-estimator that returns
    object poses, and any module that produces an image matching your training
    visualization, will work. Keep the OUTPUT contract identical to training.
    """

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        """One-time, at episode start. Pass several frames for robustness.
        Returns {object_key: ObjectState}. The anchor object key (e.g. "obj1")
        defines the object-centric reference frame used by the ICTs."""
        ...

    def make_clean_image(
        self,
        frame: Frame,
        ee_poses_in_cam: Dict[str, np.ndarray],
        grippers: Dict[str, float],
    ) -> np.ndarray:
        """Per-step. Produce the embodiment-agnostic RGB the model expects:
        real arm inpainted away, a virtual gripper rendered at each EE pose.
        Returns (H, W, 3) BGR uint8. If your model was trained 'state-only'
        (no image), this can return a black image and will be ignored."""
        ...
