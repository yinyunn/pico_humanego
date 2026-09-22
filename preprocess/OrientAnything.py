# -*- coding: utf-8 -*-
# @FileName: OrientAnything.py

"""
====================================================================================================
Object Pose Estimation Methods (OrientAnything.py)
====================================================================================================

Description:
    Contains all 6-DOF object pose estimation methods, extracted from CamTriangulator.py
    for cleaner separation of concerns. Three methods are available:

    1. pca1 — PCA-based Object-Centric Pose Estimator with Asymmetric Axis Assignment.
    2. pca2 — Object-Centric + Relational Pose Estimator (vertical/horizontal aware).
    3. vlm  — Semantic Pose Estimator using Orient-Anything V2 (NeurIPS 2025 Spotlight).

    All methods share the same interface:
        Input:  3D point cloud (pca) or cropped image (vlm), anchor/context flag
        Output: 4x4 SE(3) transformation matrix (T_o2c) + diagnostics dict

Dependencies:
    - pca1, pca2: numpy only (always available)
    - vlm: Orient-Anything V2 (https://github.com/SpatialVision/Orient-Anything-V2)
           Installed as pip package: orient-anything
           Model: Auto-downloaded from HuggingFace (Viglong/OriAnyV2_ckpt, ~5GB)

    Orient-Anything V2 License: CC-BY Attribution 4.0
    Paper: https://openreview.net/pdf?id=n3armuTFit
====================================================================================================
"""

import os
import numpy as np
from typing import Optional, Tuple
from PIL import Image

import torch


# ==============================================================================
# Orient-Anything V2 availability check (pip package: orient-anything)
# ==============================================================================
ORIENT_ANYTHING_AVAILABLE = False
_ORIENT_ANYTHING_IMPORT_ERROR = None

try:
    from orient_anything.vision_tower import VGGT_OriAny_Ref  # noqa: F401
    ORIENT_ANYTHING_AVAILABLE = True
except ImportError as e:
    _ORIENT_ANYTHING_IMPORT_ERROR = (
        f"Orient-Anything V2 import failed: {e}\n"
        "  Install: pip install -e /path/to/orient-anything\n"
        "  Source: https://github.com/SpatialVision/Orient-Anything-V2"
    )


# ==============================================================================
# VLM Model Singleton
# ==============================================================================
_VLM_MODEL_INSTANCE = None

def _get_vlm_model():
    """Internal singleton factory for downloading and initializing the VLM model."""
    global _VLM_MODEL_INSTANCE
    if _VLM_MODEL_INSTANCE is not None:
        return _VLM_MODEL_INSTANCE

    if not ORIENT_ANYTHING_AVAILABLE:
        raise ImportError(_ORIENT_ANYTHING_IMPORT_ERROR)

    print("║ [VLM] Initializing Orient-Anything V2 Instance...")
    from orient_anything.vision_tower import VGGT_OriAny_Ref
    from huggingface_hub import hf_hub_download

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)

    ckpt_p = hf_hub_download(
        repo_id="Viglong/OriAnyV2_ckpt",
        filename="demo_ckpts/rotmod_realrotaug_best.pt",
        repo_type="model"
    )

    model.load_state_dict(torch.load(ckpt_p, map_location='cpu'))
    model.eval().to(device)
    _VLM_MODEL_INSTANCE = model
    print("║ [VLM] Model is ready.")
    return _VLM_MODEL_INSTANCE


# ==============================================================================
# Helper Functions
# ==============================================================================

def angles_to_rot_matrix(az_deg: float, el_deg: float, ro_deg: float) -> np.ndarray:
    """
    Converts Euler angles predicted by Orient-Anything into a 3x3 rotation matrix
    (relative to the local camera coordinate system).

    Rotation order: Yaw (Y) -> Pitch (X) -> Roll (Z)
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    ro = np.radians(ro_deg)

    R_y = np.array([[np.cos(az),  0, np.sin(az)],[0,           1,          0],[-np.sin(az), 0, np.cos(az)]
    ])
    R_x = np.array([[1,          0,           0],[0, np.cos(el), -np.sin(el)],[0, np.sin(el),  np.cos(el)]
    ])
    R_z = np.array([[np.cos(ro), -np.sin(ro), 0],[np.sin(ro),  np.cos(ro), 0],[0,                    0, 1]
    ])

    R = R_z @ R_x @ R_y
    return R


def get_crop_from_2d_kpts(rgb_img: np.ndarray, kpts_2d: np.ndarray, pad: int = 40) -> Image.Image:
    """
    Crops the object from the image based on the bounding box of 2D keypoints.

    Args:
        rgb_img: BGR image (OpenCV format).
        kpts_2d: (N, 2) 2D keypoints in pixel coordinates.
        pad: Padding around the bounding box.

    Returns:
        PIL.Image in RGB format.
    """
    import cv2
    if len(kpts_2d) == 0:
        return Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

    x1, y1 = np.min(kpts_2d, axis=0) - pad
    x2, y2 = np.max(kpts_2d, axis=0) + pad

    h, w = rgb_img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    # Convert BGR (OpenCV) to RGB (PIL) for Orient-Anything
    return Image.fromarray(cv2.cvtColor(rgb_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


# ==============================================================================
# Pose Estimation Method 1: PCA Star-shaped Relational
# ==============================================================================

def estimate_frame_pca1(
    pts_cam: np.ndarray,
    is_anchor: bool = True,
    anchor_center_cam: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    PCA-based Object-Centric Pose Estimator with Asymmetric Axis Assignment.

    Anchor Object (the object being manipulated):
        - Y-Axis: v2 (middle PCA variance), sign -> cam_down.
        - X-Axis: v1 (longest PCA variance) if elongated, or Camera-Right if symmetric.
        - Z-Axis: cross(X, Y), completes right-handed frame.

    Context Object (environmental reference):
        - Y-Axis: v3 (shortest PCA variance / surface normal), sign -> cam_down.
        - X-Axis: Points toward the Anchor's center, projected onto plane perp to Y.
        - Z-Axis: cross(X, Y), completes right-handed frame.

    Args:
        pts_cam:           (N, 3) 3D keypoints in camera frame.
        is_anchor:         True for the manipulated object, False for context objects.
        anchor_center_cam: (3,) Anchor object center in camera frame (required if !is_anchor).

    Returns:
        T_o2c: (4, 4) Object-to-Camera transformation matrix.
        info:  Dict with 'pca_evals' and 'method' for diagnostics.
    """
    assert pts_cam.ndim == 2 and pts_cam.shape[1] == 3
    t = pts_cam.mean(axis=0)

    # 1. PCA: extract principal axes sorted by variance (descending)
    Q = pts_cam - t[None, :]
    C = Q.T @ Q
    evals, evecs = np.linalg.eigh(C)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    v1 = evecs[:, 0]  # Longest variance
    v2 = evecs[:, 1]  # Middle variance
    v3 = evecs[:, 2]  # Shortest variance

    # Camera reference directions (OpenCV convention: X-Right, Y-Down, Z-Forward)
    cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if is_anchor:
        y_axis = v2.copy()
        if np.dot(y_axis, cam_up) > 0:
            y_axis = -y_axis

        lam_a, lam_b = evals[0], evals[1]
        anisotropy = (lam_a - lam_b) / (lam_a + 1e-12)

        if anisotropy > 0.15:
            x_axis = v1.copy()
            if np.dot(x_axis, cam_right) < 0:
                x_axis = -x_axis
            method_used = f"anchor_pca (aniso:{anisotropy:.2f})"
        else:
            x_proj = cam_right - np.dot(cam_right, y_axis) * y_axis
            x_axis = x_proj / (np.linalg.norm(x_proj) + 1e-12)
            method_used = f"anchor_symmetric (aniso:{anisotropy:.2f})"

    else:
        if anchor_center_cam is None:
            raise ValueError("anchor_center_cam MUST be provided for context objects!")

        y_axis = v3.copy()
        if np.dot(y_axis, cam_up) > 0:
            y_axis = -y_axis

        vec_to_anchor = anchor_center_cam - t
        x_proj = vec_to_anchor - np.dot(vec_to_anchor, y_axis) * y_axis
        norm_x = np.linalg.norm(x_proj)

        if norm_x > 1e-4:
            x_axis = x_proj / norm_x
            method_used = "context_relational_aligned"
        else:
            x_proj = cam_right - np.dot(cam_right, y_axis) * y_axis
            x_axis = x_proj / (np.linalg.norm(x_proj) + 1e-12)
            method_used = "context_stacked_fallback"

    # Orthogonalization & Z-Axis
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    if np.linalg.det(R) < 0:
        x_axis = -x_axis
        R = np.stack([x_axis, y_axis, z_axis], axis=1)

    T_o2c = np.eye(4, dtype=np.float64)
    T_o2c[:3, :3] = R
    T_o2c[:3, 3] = t

    info = {
        "pca_evals": evals.tolist(),
        "method": method_used
    }
    return T_o2c, info


# ==============================================================================
# Pose Estimation Method 2: PCA Object-Centric + Relational
# ==============================================================================

def estimate_frame_pca2(
    pts_cam: np.ndarray,
    is_anchor: bool = True,
    anchor_center_cam: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Object-Centric + Relational Pose Estimator.
    Relies purely on the object's PCA principal axes, utilizing the camera/anchor
    as a "compass" to assign axis identities and signs.
    Handles both vertical and horizontal object geometries.
    """
    assert pts_cam.ndim == 2 and pts_cam.shape[1] == 3
    t = pts_cam.mean(axis=0)

    Q = pts_cam - t[None, :]
    C = Q.T @ Q
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    v_long, v_mid, v_short = evecs[:, 0], evecs[:, 1], evecs[:, 2]

    cam_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    is_vertical = abs(np.dot(v_long, cam_down)) > abs(np.dot(v_long, cam_right))

    method_used = ""

    if is_vertical:
        y_axis = v_long.copy()
        if np.dot(y_axis, cam_down) < 0:
            y_axis = -y_axis

        if is_anchor:
            if abs(np.dot(v_mid, cam_right)) > abs(np.dot(v_short, cam_right)):
                x_axis = v_mid.copy()
            else:
                x_axis = v_short.copy()
            if np.dot(x_axis, cam_right) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_vertical_anchor"
        else:
            if anchor_center_cam is None:
                raise ValueError("anchor_center_cam MUST be provided for context objects!")
            vec_to_anchor = anchor_center_cam - t
            if abs(np.dot(v_mid, vec_to_anchor)) > abs(np.dot(v_short, vec_to_anchor)):
                x_axis = v_mid.copy()
            else:
                x_axis = v_short.copy()
            if np.dot(x_axis, vec_to_anchor) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_vertical_context"

    else:
        x_axis = v_long.copy()

        if is_anchor:
            if np.dot(x_axis, cam_right) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_horizontal_anchor"
        else:
            if anchor_center_cam is None:
                raise ValueError("anchor_center_cam MUST be provided for context objects!")
            vec_to_anchor = anchor_center_cam - t
            if np.dot(x_axis, vec_to_anchor) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_horizontal_context"

        if abs(np.dot(v_mid, cam_down)) > abs(np.dot(v_short, cam_down)):
            y_axis = v_mid.copy()
        else:
            y_axis = v_short.copy()
        if np.dot(y_axis, cam_down) < 0:
            y_axis = -y_axis

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    T_o2c = np.eye(4, dtype=np.float64)
    T_o2c[:3, :3] = R
    T_o2c[:3, 3] = t

    info = {
        "pca_evals": evals.tolist(),
        "method": method_used
    }
    return T_o2c, info


# ==============================================================================
# Pose Estimation Method 3: VLM (Orient-Anything V2)
# ==============================================================================

@torch.no_grad()
def estimate_frame_vlm(
    image: Image.Image,
    t_cam: np.ndarray,
    is_anchor: bool = True,
    anchor_center_cam: Optional[np.ndarray] = None,
    do_rm_bkg: bool = True,
    model: Optional[torch.nn.Module] = None,
) -> tuple[np.ndarray, dict]:
    """
    VLM-driven Relational Pose Estimator using Orient-Anything V2.

    Anchor: Completely trusts the visual pose prediction from the VLM.
    Context: Trusts VLM's Y-axis (up/down) but forces X-axis toward Anchor.

    Args:
        image: PIL.Image of the cropped object.
        t_cam: 3D translation vector (3,) in camera coordinates.
        is_anchor: If True, completely trusts the visual pose prediction.
        anchor_center_cam: 3D center of the Anchor object (required if is_anchor=False).
        do_rm_bkg: Whether to remove the background before VLM inference.
        model: Pre-loaded Orient-Anything model instance.

    Returns:
        T_o2c: 4x4 SE(3) transformation matrix.
        info: Dictionary containing predicted angles and symmetries.
    """
    assert t_cam.shape == (3,), "t_cam must be a 3D numpy array"
    if model is None:
        model = _get_vlm_model()

    # Import Orient-Anything utilities from pip package
    from orient_anything.utils.app_utils import (
        background_preprocess, inf_single_case
    )

    # 1. Image Preprocessing
    pil_img = image.copy().convert("RGB")
    if do_rm_bkg:
        pil_img = background_preprocess(pil_img, True)

    # 2. Orient-Anything V2 Inference
    try:
        ans_dict = inf_single_case(model, pil_img, None)
    except Exception as e:
        print(f"║ [VLM Error] Orient-Anything inference failed: {e}")
        return np.eye(4), {"error": str(e)}

    # Extract predicted values
    az = float(ans_dict.get('ref_az_pred', 0.0))
    el = float(ans_dict.get('ref_el_pred', 0.0))
    ro = float(ans_dict.get('ref_ro_pred', 0.0))
    alpha = int(ans_dict.get('ref_alpha_pred', 1))

    # 3. Convert predicted angles to a base rotation matrix
    R_base = angles_to_rot_matrix(az, el, ro)

    x_axis_base = R_base[:, 0]
    y_axis_base = R_base[:, 1]
    z_axis_base = R_base[:, 2]

    method_used = ""

    # 4. Apply Semantic Relational Logic
    if is_anchor:
        R_final = R_base
        method_used = f"vlm_anchor (alpha:{alpha})"
    else:
        if anchor_center_cam is None:
            raise ValueError("anchor_center_cam MUST be provided for context objects!")

        y_axis = y_axis_base.copy()

        vec_to_anchor = anchor_center_cam - t_cam
        x_proj = vec_to_anchor - np.dot(vec_to_anchor, y_axis) * y_axis
        norm_x = np.linalg.norm(x_proj)

        if norm_x > 1e-4:
            x_axis = x_proj / norm_x
            method_used = "vlm_context_relational"
        else:
            x_axis = x_axis_base
            method_used = "vlm_context_stacked_fallback"

        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

        R_final = np.stack([x_axis, y_axis, z_axis], axis=1)

    # 5. Build 4x4 Output Matrix
    T_o2c = np.eye(4, dtype=np.float64)
    T_o2c[:3, :3] = R_final
    T_o2c[:3, 3] = t_cam

    info = {
        "vlm_azimuth": az,
        "vlm_elevation": el,
        "vlm_rotation": ro,
        "vlm_symmetry_alpha": alpha,
        "method": method_used
    }

    return T_o2c, info
