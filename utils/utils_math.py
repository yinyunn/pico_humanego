import numpy as np
import torch
import time
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from utils.utils_vis import Color
import torch
import torch.nn.functional as F
import math


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz": R = Rz @ Ry @ Rx
    elif order == "xzy": R = Ry @ Rz @ Rx
    elif order == "yxz": R = Rz @ Rx @ Ry
    elif order == "yzx": R = Rx @ Rz @ Ry
    elif order == "zyx": R = Rx @ Ry @ Rz
    elif order == "zxy": R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (torch.tensor([0, 0, 0, 1], device=R.device)
              .reshape(*(1,) * len(dims), 1, 4).expand(*dims, 1, 4))
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=torch.float32)


def roty(theta):
    return torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.float32)


def rotz(theta):
    return torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.float32)

    
# Pretty labels for the Aria preprocess pipeline stages (drives the `Stage k/N`
# banner). Any @time_it-decorated function not listed here keeps the plain banner.
_PREPROCESS_STAGES = {
    "preprocess_aria":            "Aria extraction (RGB · MPS hands · SLAM · phases)",
    "preprocess_indices":         "Object-centric window",
    "preprocess_dinosam":         "DINO + SAM2 detection & segmentation",
    "preprocess_kptsselector":    "Keypoint selection",
    "preprocess_cotracker":       "CoTracker 2D tracking",
    "preprocess_camtriangulator": "3D triangulation",
    "preprocess_lama":            "LaMa arm inpainting",
    "preprocess_visualkpts":      "Keypoint rendering",
    "preprocess_datasetgen":      "Dataset generation (training_data.json)",
}
_STAGE_ORDER = list(_PREPROCESS_STAGES)
try:
    from rich.console import Console as _RichConsole
    _RC = _RichConsole()
except Exception:
    _RC = None


def _fmt_dur(sec: float) -> str:
    if sec > 3600:
        return f"{sec / 3600:.2f}h"
    if sec > 60:
        return f"{sec / 60:.2f}m"
    return f"{sec:.2f}s"


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        class_name = "Preprocess"
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__

        stage = _PREPROCESS_STAGES.get(func.__name__)
        if stage and _RC is not None:
            idx = _STAGE_ORDER.index(func.__name__) + 1
            _RC.rule(f"[bold cyan]Stage {idx}/{len(_STAGE_ORDER)}[/]  [bold]{stage}[/]",
                     style="cyan")
        else:
            print(f"\n{Color.HEADER}>>>>>>>>> Starting {class_name}.{func.__name__}{Color.END}")

        result = func(*args, **kwargs)

        time_str = _fmt_dur(time.perf_counter() - start_time)
        if stage and _RC is not None:
            idx = _STAGE_ORDER.index(func.__name__) + 1
            _RC.print(f"[green]✓[/] Stage {idx}/{len(_STAGE_ORDER)} "
                      f"[bold]{stage}[/] — done in [yellow]{time_str}[/]")
        else:
            print(f"{Color.OKGREEN}<<<<<<<<< Finished {class_name}.{func.__name__} in {time_str}{Color.END}")

        return result
    return wrapper


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def rotmat_to_o6d(Rm: np.ndarray) -> np.ndarray:
    """Convert R(3,3) to 6D by flattening first two columns."""
    Rm = np.array(Rm, dtype=np.float32).reshape(3, 3)
    return Rm[:, :2].reshape(-1).astype(np.float32)


def o6d_to_rotmat(o6d: np.ndarray) -> np.ndarray:
    """Convert 6D to R(3,3) using Gram-Schmidt process."""
    o6d = np.array(o6d, dtype=np.float32).reshape(3, 2)
    a1 = o6d[:, 0]
    a2 = o6d[:, 1]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    Rm = np.stack([b1, b2, b3], axis=1).astype(np.float32)
    return Rm


def normalize_o6d(o6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalization for 6D rotation."""
    is_1d = (o6d.ndim == 1)
    x = o6d.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-8)
    out = np.stack([b1, b2], axis=2).reshape(-1, 6)
    return out[0].astype(np.float32) if is_1d else out.astype(np.float32)


def normalize_pos(pos: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Axis-wise normalization: (x - mean) / std"""
    return (pos - mean) / (std + 1e-8)


def unnormalize_pos(pos_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert back to physical units: x_norm * std + mean"""
    return pos_norm * std + mean


def interpolate_pose(pos0, R0_mat, pos1, R1_mat, alpha):
    """Interpolates position and rotation given an alpha in [0,1]."""
    interp_pos = pos0 + (pos1 - pos0) * alpha
    rots = R.from_matrix([R0_mat, R1_mat])
    slerp = Slerp([0, 1], rots)
    interp_R_mat = slerp([alpha])[0].as_matrix()
    return interp_pos.astype(np.float32), interp_R_mat.astype(np.float32)


def get_rrc_params(height, width, scale_range, ratio_range, rng):
    """Calculates parameters for random resized crop (y, x, h, w)."""
    area = height * width
    for _ in range(10):
        target_area = rng.uniform(*scale_range) * area
        log_ratio = (np.log(ratio_range[0]), np.log(ratio_range[1]))
        aspect_ratio = np.exp(rng.uniform(*log_ratio))

        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            y = rng.randint(0, height - h + 1)
            x = rng.randint(0, width - w + 1)
            return y, x, h, w

    return 0, 0, height, width


def apply_photometric_aug(rgb01: np.ndarray, rng: np.random.RandomState, AUG_IMG_PROB: float = 0.8, AUG_BRIGHTNESS_DELTA: float = 0.20, AUG_CONTRAST_DELTA: float = 0.20, AUG_GAMMA_DELTA: float = 0.15, AUG_NOISE_STD: float = 0.02, AUG_BLUR_PROB: float = 0.15, AUG_BLUR_KSIZE: int = 3, AUG_GRAY_PROB: float = 0.10, AUG_HUE_DELTA: float = 10, AUG_SAT_RANGE: tuple = (0.6, 1.4)) -> np.ndarray:
    """Applies photometric augmentations to float32 RGB [0,1]."""
    if rng.rand() > AUG_IMG_PROB:
        return rgb01

    x = rgb01.copy()

    # Grayscale
    if rng.rand() < AUG_GRAY_PROB:
        g = (0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]).astype(np.float32)
        x = np.stack([g, g, g], axis=-1)

    # Hue & Saturation
    if rng.rand() < 0.5:
        img_bgr_u8 = (x[..., ::-1] * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_noise = (rng.rand() * 2.0 - 1.0) * AUG_HUE_DELTA
        hsv[..., 0] = (hsv[..., 0] + h_noise) % 180
        s_scale = rng.uniform(AUG_SAT_RANGE[0], AUG_SAT_RANGE[1])
        hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
        img_bgr_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = img_bgr_aug[..., ::-1].astype(np.float32) / 255.0

    # Brightness
    b = (rng.rand() * 2.0 - 1.0) * AUG_BRIGHTNESS_DELTA
    x = x + b

    # Contrast
    c = 1.0 + (rng.rand() * 2.0 - 1.0) * AUG_CONTRAST_DELTA
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * c + mean

    # Gamma
    g = 1.0 + (rng.rand() * 2.0 - 1.0) * AUG_GAMMA_DELTA
    g = max(0.5, min(1.5, g))
    x = np.power(clip01(x), g)

    # Noise
    if AUG_NOISE_STD > 0:
        n = rng.randn(*x.shape).astype(np.float32) * AUG_NOISE_STD
        x = x + n

    # Blur
    if rng.rand() < AUG_BLUR_PROB and AUG_BLUR_KSIZE in (3, 5, 7):
        x_u8 = (clip01(x) * 255.0).astype(np.uint8)
        x_u8 = cv2.GaussianBlur(x_u8, (AUG_BLUR_KSIZE, AUG_BLUR_KSIZE), 0)
        x = x_u8.astype(np.float32) / 255.0

    return clip01(x).astype(np.float32)


def apply_random_erasing(img_f: np.ndarray, rng: np.random.RandomState, AUG_CUTOUT_PROB: float = 0.5, AUG_CUTOUT_N_HOLES: tuple = (3, 8), AUG_CUTOUT_SIZE: tuple = (0.05, 0.2)) -> np.ndarray:
    """Applies Cutout / Random Erasing to images."""
    if rng.rand() > AUG_CUTOUT_PROB:
        return img_f

    h_img, w_img = img_f.shape[:2]
    n_holes = rng.randint(AUG_CUTOUT_N_HOLES[0], AUG_CUTOUT_N_HOLES[1] + 1)
    
    for _ in range(n_holes):
        h_hole = int(h_img * rng.uniform(AUG_CUTOUT_SIZE[0], AUG_CUTOUT_SIZE[1]))
        w_hole = int(w_img * rng.uniform(AUG_CUTOUT_SIZE[0], AUG_CUTOUT_SIZE[1]))
        y0 = rng.randint(0, h_img - h_hole + 1)
        x0 = rng.randint(0, w_img - w_hole + 1)
        img_f[y0:y0+h_hole, x0:x0+w_hole, :] = 0.0
        
    return img_f


def rot6d_to_R_batch(o6d: torch.Tensor) -> torch.Tensor:
    x = o6d.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=1, eps=1e-8)
    a2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=1)
    R = torch.stack([b1, b2, b3], dim=2) 
    return R


def geodesic_deg_from_R(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    Rt = torch.matmul(R1.transpose(1, 2), R2)
    tr = Rt[:, 0, 0] + Rt[:, 1, 1] + Rt[:, 2, 2]
    c = ((tr - 1.0) / 2.0).clamp(-0.999999, 0.999999)
    ang = torch.acos(c) * (180.0 / math.pi)
    return ang

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



####### for infernce

def mirror_pose(T_mat: np.ndarray) -> np.ndarray:
    """
    Mathematically reflects a real Left-Arm 4x4 pose matrix into a
    virtual Right-Arm pose matrix (or vice versa).
    
    Args:
        T_mat (np.ndarray): Original 4x4 transformation matrix.
        
    Returns:
        np.ndarray: Mirrored 4x4 transformation matrix.
    """
    if T_mat is None:
        return None
    
    # S_cam: Inverts the X-axis due to flipped image projection.
    S_cam = np.array([
        [-1,  0,  0,  0],[ 0,  1,  0,  0],
        [ 0,  0,  1,  0],[ 0,  0,  0,  1]
    ], dtype=np.float64)
    
    # S_local: Flips the local X-axis so that the Y-axis still points inwards,
    # ensuring X and Z perfectly align with the native right-hand kinematic model.
    S_local = np.array([
        [-1,  0,  0,  0],[ 0,  1,  0,  0],
        [ 0,  0,  1,  0],[ 0,  0,  0,  1]
    ], dtype=np.float64)
    
    return S_cam @ T_mat @ S_local


def mirror_intrinsics(K: np.ndarray, img_w: int) -> np.ndarray:
    """
    Flips the camera intrinsics matrix (horizontally flips the optical center c_x).
    
    Args:
        K (np.ndarray): 3x3 Camera intrinsics matrix.
        img_w (int): Image width in pixels.
        
    Returns:
        np.ndarray: Mirrored 3x3 camera intrinsics matrix.
    """
    if K is None:
        return None
    K_mirrored = K.copy()
    K_mirrored[0, 2] = img_w - K_mirrored[0, 2]
    return K_mirrored