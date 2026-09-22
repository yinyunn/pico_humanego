# -*- coding: utf-8 -*-
# @FileName: HaMeRHands.py

"""
====================================================================================================
HaMeR Hand Mesh Recovery Pipeline (HaMeRHands.py)
====================================================================================================

Description:
    Image-based hand detection and 3D mesh recovery using HaMeR (Hand Mesh Recovery).
    Produces AriaHands-compatible output so it can be directly swapped with Aria MPS hand
    tracking or MediaPipe-based tracking for ablation studies.

Pipeline:
    1. Read RGB frames from the existing AriaCam sequence
    2. Stage 1: Use MediaPipe Hands to detect hand bounding boxes
    3. Stage 2: Use HaMeR to estimate MANO 3D mesh from each hand crop
    4. Recover absolute 3D keypoints in camera frame:
       - Use 2D wrist position + camera intrinsics for back-projection direction
       - Use HaMeR camera-space 3D joints for relative structure
       - Estimate depth via physical hand size constraint
    5. Remap HaMeR 21-point ordering (same as MediaPipe after reordering) to Aria MPS ordering
    6. Transform from camera frame to world frame via c2w
    7. Build midpoint "gripper" frame using MidpointFrameBuilder
    8. Detect grasp state via thumb-index fingertip distance
    9. Apply temporal smoothing (Savitzky-Golay + EMA) via AriaHandsOptimizer
   10. Save per-frame JSON in identical format to aria_hands.json

Generated Outputs:
    [mps_path]/aria/all_data/[idx]/hamer_hands.json  (per-frame, same schema as aria_hands.json)
    [mps_path]/aria/vis/hamer_hands_vis.mp4           (skeleton visualization)
    [mps_path]/aria/hamer_hands_analysis_r.png
    [mps_path]/aria/hamer_hands_analysis_l.png

Requirements:
    pip install mediapipe            (for Stage 1 hand detection)
    pip install hamer                (for Stage 2 mesh recovery, or local checkpoint)
====================================================================================================
"""

import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R

from utils.utils_io import load_cfg
from utils.utils_media import create_video_from_frames

from preprocess.AriaCamTypes import AriaCam
from preprocess.AriaHandsTypes import (
    MidpointFrameBuilder,
    AriaHandsJointAngles,
    AriaHandData,
    AriaHandsData,
    AriaHands,
)
from preprocess.AriaHandsOptimizer import AriaHandsOptimizer
from preprocess.AriaHandsOps import AriaHandsOps

# ==================================================================
# HaMeR availability check
# ==================================================================
HAMER_AVAILABLE = False
try:
    from hamer.models import HAMER
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    HAMER_AVAILABLE = True
except ImportError:
    pass

if not HAMER_AVAILABLE:
    try:
        import importlib
        _hamer_spec = importlib.util.find_spec("hamer")
        if _hamer_spec is not None:
            import hamer
            HAMER_AVAILABLE = True
    except Exception:
        pass

if not HAMER_AVAILABLE:
    print(
        "[HaMeR] WARNING: HaMeR package is not installed.\n"
        "  To install: pip install --no-deps hamer@git+https://github.com/geopavlakos/hamer.git\n"
        "  HaMeRHands will fall back to MediaPipe-only mode or return empty results."
    )


# ==================================================================
# HaMeR checkpoint auto-download via HuggingFace Hub
# ==================================================================
# HaMeR checkpoints (MIT License, (c) 2023 UC Regents, Georgios Pavlakos)
# Hosted at: https://huggingface.co/Leo-TX/hamer
HAMER_HF_REPO = "Leo-TX/hamer"
# MediaPipe hand_landmarker.task (Apache 2.0, (c) Google)
# Hosted at: https://huggingface.co/Leo-TX/mediapipe-hand
MEDIAPIPE_HF_REPO = "Leo-TX/mediapipe-hand"

# Use ~/.cache/hamer/ instead of the hamer package's default "./_DATA"
# so the project directory stays clean (no _DATA/ folder created).
HAMER_CACHE_DIR = os.path.expanduser("~/.cache/hamer")


def _patch_hamer_cache_dir():
    """
    Monkey-patch hamer.configs.CACHE_DIR_HAMER to use ~/.cache/hamer/
    instead of the default "./_DATA". This prevents the hamer package
    from creating an _DATA/ directory in the project root.
    """
    try:
        import hamer.configs
        hamer.configs.CACHE_DIR_HAMER = HAMER_CACHE_DIR
    except ImportError:
        pass


# Apply the patch immediately when this module is loaded
if HAMER_AVAILABLE:
    _patch_hamer_cache_dir()


def _ensure_hamer_ckpts() -> str:
    """
    Ensure HaMeR checkpoint and config files are available.
    Downloads from HuggingFace Hub to ~/.cache/hamer/ if not present.
    Also ensures MANO_RIGHT.pkl is present (from WiLoR's HF repo).

    HaMeR is released under the MIT License by UC Regents / Georgios Pavlakos.
    See: https://github.com/geopavlakos/hamer

    Returns:
        Path to hamer.ckpt
    """
    cache_dir = HAMER_CACHE_DIR

    local_ckpt = os.path.join(cache_dir, "hamer_ckpts", "checkpoints", "hamer.ckpt")
    if os.path.isfile(local_ckpt):
        # Local files exist, ensure MANO is also present
        _ensure_mano(cache_dir)
        return local_ckpt

    # Download from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        print("[HaMeR] Downloading checkpoints from HuggingFace Hub...")

        hf_files = {
            "hamer.ckpt":          os.path.join(cache_dir, "hamer_ckpts", "checkpoints", "hamer.ckpt"),
            "model_config.yaml":   os.path.join(cache_dir, "hamer_ckpts", "model_config.yaml"),
            "dataset_config.yaml": os.path.join(cache_dir, "hamer_ckpts", "dataset_config.yaml"),
            "mano_mean_params.npz": os.path.join(cache_dir, "data", "mano_mean_params.npz"),
        }

        for hf_path, local_path in hf_files.items():
            if not os.path.isfile(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                cached = hf_hub_download(repo_id=HAMER_HF_REPO, filename=hf_path)
                # Symlink from HF cache to expected location
                if not os.path.isfile(local_path):
                    os.symlink(cached, local_path)
                print(f"  ✓ {os.path.basename(hf_path)}")

        print("[HaMeR] Checkpoint download complete")
    except Exception as e:
        print(f"[HaMeR] WARNING: HuggingFace download failed: {e}")
        print(f"[HaMeR] Please manually place hamer.ckpt at: {local_ckpt}")

    _ensure_mano(cache_dir)
    return local_ckpt


def _ensure_mano(cache_dir: str) -> None:
    """
    Ensure MANO_RIGHT.pkl is present in cache_dir/data/mano/.
    Sources (in order): WiLoR's HF cache → download from warmshao/WiLoR-mini.

    NOTE: MANO license prohibits redistribution, so we fetch from the
    original author's distribution (WiLoR bundles it under their license).
    """
    mano_dst = os.path.join(cache_dir, "data", "mano", "MANO_RIGHT.pkl")
    if os.path.isfile(mano_dst):
        return

    os.makedirs(os.path.dirname(mano_dst), exist_ok=True)

    # Try to find it in WiLoR's HuggingFace cache
    import glob
    patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/models--warmshao--WiLoR-mini/snapshots/*/pretrained_models/MANO_RIGHT.pkl"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            import shutil
            shutil.copy2(matches[0], mano_dst)
            print(f"[HaMeR] Copied MANO_RIGHT.pkl from WiLoR cache")
            return

    # Try triggering WiLoR download
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id="warmshao/WiLoR-mini",
            subfolder="pretrained_models",
            filename="MANO_RIGHT.pkl",
        )
        import shutil
        shutil.copy2(cached, mano_dst)
        print(f"[HaMeR] Downloaded MANO_RIGHT.pkl via WiLoR's HF repo")
    except Exception:
        print(f"[HaMeR] WARNING: MANO_RIGHT.pkl not found at {mano_dst}")
        print(f"  Please download from https://mano.is.tue.mpg.de/ and place it there.")


# ==================================================================
# HaMeR 21-point ordering (after common reordering) is the SAME as
# MediaPipe's 21-point ordering. Use identical mapping to Aria MPS.
# ==================================================================
# MediaPipe / HaMeR ordering:
#   0=Wrist, 1=ThumbCMC, 2=ThumbMCP, 3=ThumbIP, 4=ThumbTip,
#   5=IndexMCP, 6=IndexPIP, 7=IndexDIP, 8=IndexTip,
#   9=MiddleMCP, 10=MiddlePIP, 11=MiddleDIP, 12=MiddleTip,
#   13=RingMCP, 14=RingPIP, 15=RingDIP, 16=RingTip,
#   17=PinkyMCP, 18=PinkyPIP, 19=PinkyDIP, 20=PinkyTip
#
# Aria MPS ordering:
#   0=ThumbTip, 1=IndexTip, 2=MiddleTip, 3=RingTip, 4=PinkyTip,
#   5=Wrist, 6=ThumbMCP, 7=ThumbIP, 8=IndexMCP, 9=IndexPIP, 10=IndexDIP,
#   11=MiddleMCP, 12=MiddlePIP, 13=MiddleDIP,
#   14=RingMCP, 15=RingPIP, 16=RingDIP,
#   17=PinkyMCP, 18=PinkyPIP, 19=PinkyDIP,
#   20=PalmCenter (approximated as mean of Wrist, IndexMCP, MiddleMCP)
#
# MP_TO_ARIA[aria_idx] = hamer/mediapipe_idx
MP_TO_ARIA = [
    4,   # Aria 0  = ThumbTip     <- HaMeR 4
    8,   # Aria 1  = IndexTip     <- HaMeR 8
    12,  # Aria 2  = MiddleTip    <- HaMeR 12
    16,  # Aria 3  = RingTip      <- HaMeR 16
    20,  # Aria 4  = PinkyTip     <- HaMeR 20
    0,   # Aria 5  = Wrist        <- HaMeR 0
    2,   # Aria 6  = ThumbMCP     <- HaMeR 2
    3,   # Aria 7  = ThumbIP      <- HaMeR 3
    5,   # Aria 8  = IndexMCP     <- HaMeR 5
    6,   # Aria 9  = IndexPIP     <- HaMeR 6
    7,   # Aria 10 = IndexDIP     <- HaMeR 7
    9,   # Aria 11 = MiddleMCP    <- HaMeR 9
    10,  # Aria 12 = MiddlePIP    <- HaMeR 10
    11,  # Aria 13 = MiddleDIP    <- HaMeR 11
    13,  # Aria 14 = RingMCP      <- HaMeR 13
    14,  # Aria 15 = RingPIP      <- HaMeR 14
    15,  # Aria 16 = RingDIP      <- HaMeR 15
    17,  # Aria 17 = PinkyMCP     <- HaMeR 17
    18,  # Aria 18 = PinkyPIP     <- HaMeR 18
    19,  # Aria 19 = PinkyDIP     <- HaMeR 19
    -1,  # Aria 20 = PalmCenter   <- computed (mean of Wrist, IndexMCP, MiddleMCP)
]

# Average adult hand: wrist to middle MCP ~ 0.085m
HAND_SIZE_WRIST_TO_MIDDLE_MCP_M = 0.085


def remap_mp_to_aria(kpts_mp_21: np.ndarray) -> np.ndarray:
    """
    Remap 21 HaMeR/MediaPipe keypoints to Aria MPS 21-point ordering.

    Args:
        kpts_mp_21: (21, 3) keypoints in HaMeR/MediaPipe ordering.

    Returns:
        (21, 3) keypoints in Aria ordering. Index 20 (PalmCenter) is computed.
    """
    kpts_aria = np.zeros((21, 3), dtype=kpts_mp_21.dtype)
    for aria_idx in range(20):
        mp_idx = MP_TO_ARIA[aria_idx]
        kpts_aria[aria_idx] = kpts_mp_21[mp_idx]
    # Aria 20 = PalmCenter ~ mean(Wrist=0, IndexMCP=5, MiddleMCP=9)
    kpts_aria[20] = (kpts_mp_21[0] + kpts_mp_21[5] + kpts_mp_21[9]) / 3.0
    return kpts_aria


# ==================================================================
# HaMeR model wrapper
# ==================================================================

class HaMeRModel:
    """
    Thin wrapper around the HaMeR hand mesh recovery model.
    Handles loading the pretrained checkpoint and running inference
    on hand image crops.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cfg = None

        if not HAMER_AVAILABLE:
            return

        try:
            from hamer.models import load_hamer
            # Ensure checkpoint + MANO files are present (download from HF if needed)
            ckpt_path = _ensure_hamer_ckpts()
            self.model, self.cfg = load_hamer(ckpt_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[HaMeR] Model loaded on {self.device}")
        except Exception as e:
            print(f"[HaMeR] WARNING: Failed to load HaMeR model: {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    @staticmethod
    def _compute_hamer_confidence(
        pred_kpts_3d_rel: np.ndarray,
        joints_cam: np.ndarray,
        joints_2d: np.ndarray,
        img_w: int,
        img_h: int,
        bbox: np.ndarray,
        scaled_focal_length,
    ) -> float:
        """
        Compute per-detection confidence for HaMeR from reconstruction quality.

        Uses three signals:
          1. Depth plausibility: wrist Z should be 0.1-2.0m
          2. 2D coverage: projected joints should cover most of the detection bbox
          3. 3D compactness: hand keypoints in MANO space should have reasonable spread

        Returns: confidence in [0.1, 0.99]
        """
        try:
            # 1. Depth plausibility (wrist Z)
            wrist_z = float(joints_cam[0, 2])
            if wrist_z < 0.05 or wrist_z > 3.0:
                return 0.15
            depth_score = 1.0
            if wrist_z < 0.1:
                depth_score = 0.5
            elif wrist_z > 2.0:
                depth_score = 0.6

            # 2. 2D coverage: joints should span a reasonable fraction of bbox
            bx1, by1, bx2, by2 = bbox[:4]
            bbox_w = max(bx2 - bx1, 1.0)
            bbox_h = max(by2 - by1, 1.0)
            j2d_valid = joints_2d[(joints_2d[:, 0] > 0) & (joints_2d[:, 1] > 0)]
            if len(j2d_valid) > 5:
                j_span_x = j2d_valid[:, 0].max() - j2d_valid[:, 0].min()
                j_span_y = j2d_valid[:, 1].max() - j2d_valid[:, 1].min()
                coverage = (j_span_x / bbox_w + j_span_y / bbox_h) / 2.0
                coverage_score = float(np.clip(coverage, 0.1, 1.0))
            else:
                coverage_score = 0.3

            # 3. 3D compactness: MANO hand size should be ~0.15-0.25m
            hand_span = float(np.linalg.norm(
                pred_kpts_3d_rel.max(axis=0) - pred_kpts_3d_rel.min(axis=0)
            ))
            if hand_span < 0.05 or hand_span > 0.5:
                compact_score = 0.3
            else:
                compact_score = 1.0

            confidence = 0.95 * depth_score * coverage_score * compact_score
            return float(np.clip(confidence, 0.1, 0.99))

        except Exception:
            return 0.50

    @torch.no_grad()
    def predict_from_crop(
        self,
        img_rgb: np.ndarray,
        bbox: np.ndarray,
        is_right: int = 1,
        focal_length: float = 500.0,
    ) -> Optional[dict]:
        """
        Run HaMeR inference on a hand image crop.

        Args:
            img_rgb: Full RGB image (H, W, 3), uint8.
            bbox: Bounding box [x1, y1, x2, y2] in pixel coords.
            is_right: 1 for right hand, 0 for left hand.
            focal_length: Approximate focal length for the crop.

        Returns:
            Dictionary with:
                'joints_3d': (21, 3) camera-space 3D joints in meters
                'joints_2d': (21, 2) projected 2D joints in pixel coords
                'confidence': float reconstruction confidence
            Or None if inference fails.
        """
        if not self.is_available:
            return None

        try:
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.utils.renderer import cam_crop_to_full
            from hamer.utils import recursive_to

            x1, y1, x2, y2 = bbox.astype(int)
            bbox_size = max(x2 - x1, y2 - y1)

            if bbox_size < 10:
                return None

            img_h, img_w = img_rgb.shape[:2]

            # ViTDetDataset expects BGR image, boxes (N,4), right (N,) as 0/1
            dataset = ViTDetDataset(
                self.cfg,
                img_cv2=img_rgb[:, :, ::-1],  # RGB -> BGR
                boxes=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                right=np.array([is_right], dtype=np.float32),
            )

            if len(dataset) == 0:
                return None

            # Use DataLoader for proper batching (handles numpy→tensor + collation)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
            batch = next(iter(dataloader))
            batch = recursive_to(batch, self.device)

            out = self.model(batch)

            # Extract predictions (following demo.py logic)
            pred_cam = out['pred_cam']  # (1, 3) tensor
            pred_keypoints_3d = out['pred_keypoints_3d'][0].cpu().numpy()  # (21, 3)

            # Mirror x-axis for left hands (same as demo.py: multiplier = 2*right - 1)
            multiplier = (2 * batch['right'] - 1)  # +1 for right, -1 for left
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            # Convert crop camera to full-image camera translation
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()[0]  # (3,) = [tx, ty, tz]

            # For left hand, un-mirror the x-axis of keypoints
            if is_right == 0:
                pred_keypoints_3d[:, 0] = -pred_keypoints_3d[:, 0]

            # 3D joints in full-image camera space:
            # pred_keypoints_3d are relative to hand root in MANO space
            # pred_cam_t_full gives the camera translation [tx, ty, tz]
            joints_cam = pred_keypoints_3d + pred_cam_t_full[np.newaxis, :]

            # Project 3D -> 2D using perspective projection
            joints_2d = np.zeros((21, 2), dtype=np.float32)
            if np.all(joints_cam[:, 2] > 0):
                fl = float(scaled_focal_length.cpu()) if isinstance(scaled_focal_length, torch.Tensor) else float(scaled_focal_length)
                joints_2d[:, 0] = joints_cam[:, 0] / joints_cam[:, 2] * fl + img_w / 2.0
                joints_2d[:, 1] = joints_cam[:, 1] / joints_cam[:, 2] * fl + img_h / 2.0

            # Confidence: combine model quality signals.
            # HaMeR doesn't output explicit per-keypoint confidence.
            # Instead, compute a reprojection-based quality score from
            # 3D→2D projection consistency: compare joints_2d (from HaMeR 3D)
            # with the crop center/size to measure how well the prediction
            # matches the detected bounding box. The reconstruction loss
            # provides a useful proxy via keypoint spread and depth stability.
            confidence = self._compute_hamer_confidence(
                pred_keypoints_3d, joints_cam, joints_2d,
                img_w, img_h, bbox, scaled_focal_length
            )

            return {
                'joints_3d': joints_cam.astype(np.float32),
                'joints_2d': joints_2d.astype(np.float32),
                'confidence': confidence,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None


# ==================================================================
# MediaPipe hand detector (Stage 1: bounding box detection)
# ==================================================================

class MediaPipeHandDetector:
    """
    Lightweight wrapper for MediaPipe Hands used only for detection
    (bounding boxes + handedness classification) in the two-stage pipeline.
    Uses the new mp.tasks.vision.HandLandmarker API (mediapipe v0.10+).
    """

    def __init__(self):
        import mediapipe as mp
        self._mp = mp

        # Locate the hand_landmarker.task model file
        # Priority: local weights/ dir → HuggingFace Hub auto-download
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "weights", "mediapipe", "hand_landmarker.task"
        )
        if not os.path.isfile(model_path):
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id=MEDIAPIPE_HF_REPO,
                    filename="hand_landmarker.task",
                )
                print(f"[MediaPipe] Downloaded hand_landmarker.task from HuggingFace Hub")
            except Exception:
                raise FileNotFoundError(
                    f"MediaPipe hand_landmarker.task not found at {model_path}. "
                    "Download from: https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def detect(self, img_rgb: np.ndarray) -> list:
        """
        Detect hands and return bounding boxes with handedness.

        Args:
            img_rgb: RGB image (H, W, 3), uint8.

        Returns:
            List of dicts with keys:
                'bbox': np.array([x1, y1, x2, y2])
                'label': "Left" or "Right"
                'is_right_int': 1 for Right, 0 for Left (for ViTDetDataset)
                'confidence': float
                'landmarks_2d': (21, 2) pixel coords (for fallback)
                'world_landmarks': (21, 3) hand-centered meters (for fallback)
        """
        h_img, w_img = img_rgb.shape[:2]
        mp_img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=img_rgb
        )
        results = self.landmarker.detect(mp_img)
        detections = []

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, hand_world_lms, handedness_list in zip(
                results.hand_landmarks,
                results.hand_world_landmarks,
                results.handedness,
            ):
                label = handedness_list[0].category_name  # "Left" or "Right"
                confidence = handedness_list[0].score

                # Extract 2D landmarks (normalized → pixel coords)
                kpts_2d = np.array(
                    [[lm.x * w_img, lm.y * h_img] for lm in hand_landmarks],
                    dtype=np.float32,
                )

                # Extract world landmarks (for fallback 3D recovery)
                kpts_world = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_world_lms],
                    dtype=np.float32,
                )

                # Compute bounding box with padding
                x_min, y_min = kpts_2d.min(axis=0)
                x_max, y_max = kpts_2d.max(axis=0)
                pad_x = (x_max - x_min) * 0.3
                pad_y = (y_max - y_min) * 0.3
                bbox = np.array([
                    max(0, x_min - pad_x),
                    max(0, y_min - pad_y),
                    min(w_img, x_max + pad_x),
                    min(h_img, y_max + pad_y),
                ], dtype=np.float32)

                detections.append({
                    'bbox': bbox,
                    'label': label,
                    'is_right_int': 1 if label == "Right" else 0,
                    'confidence': confidence,
                    'landmarks_2d': kpts_2d,
                    'world_landmarks': kpts_world,
                })

        return detections


class ViTPoseHandDetector:
    """
    Hand detection via ViTPose wholebody keypoints (official HaMeR demo approach).
    Pipeline: YOLO body detection → ViTPose wholebody (133 kpts) → hand bbox from hand keypoints.

    This matches the official HaMeR demo.py which uses ViTDet+ViTPose.
    We use easy_ViTPose (no mmpose/detectron2 dependency) with YOLO for body detection.

    COCO-WholeBody keypoint indices:
      0-16:   body (17)
      17-22:  foot (6)
      23-90:  face (68)
      91-111: left hand (21)
      112-132: right hand (21)
    """

    VITPOSE_AVAILABLE = False

    # Hand keypoint index ranges in COCO-WholeBody (133 keypoints)
    LEFT_HAND_SLICE = slice(91, 112)   # 21 keypoints
    RIGHT_HAND_SLICE = slice(112, 133)  # 21 keypoints

    # HuggingFace Hub repo and filenames for auto-download
    HF_REPO = "JunkyByte/easy_ViTPose"
    # Prefer huge, fall back to smaller variants
    VITPOSE_VARIANTS = [
        ("h", "torch/wholebody/vitpose-h-wholebody.pth"),
        ("l", "torch/wholebody/vitpose-l-wholebody.pth"),
        ("b", "torch/wholebody/vitpose-b-wholebody.pth"),
        ("s", "torch/wholebody/vitpose-s-wholebody.pth"),
    ]
    YOLO_HF_PATH = "yolov8/yolov8s.pt"

    def __init__(self, device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from easy_ViTPose import VitInference

        # Download ViTPose checkpoint (try huge → large → base → small)
        vitpose_model = None
        model_name = None
        for variant, hf_path in self.VITPOSE_VARIANTS:
            try:
                vitpose_model = hf_hub_download(
                    repo_id=self.HF_REPO, filename=hf_path,
                )
                model_name = variant
                break
            except Exception:
                continue

        if vitpose_model is None:
            raise FileNotFoundError(
                f"Could not download any ViTPose wholebody model from {self.HF_REPO}. "
                "Check your internet connection or install manually."
            )

        # Download YOLOv8s for body detection
        yolo_model = hf_hub_download(
            repo_id=self.HF_REPO, filename=self.YOLO_HF_PATH,
        )

        self.model = VitInference(
            model=vitpose_model,
            yolo=yolo_model,
            model_name=model_name,
            dataset="wholebody",
            device=device,
        )
        self._model_name = model_name
        print(f"[ViTPose] Loaded ViTPose-{model_name.upper()} wholebody + YOLOv8s (from HuggingFace Hub)")
        ViTPoseHandDetector.VITPOSE_AVAILABLE = True

    def detect(self, img_rgb: np.ndarray, kpt_conf_thr: float = 0.3,
               min_valid_kpts: int = 4, bbox_pad_ratio: float = 0.3) -> list:
        """
        Detect hands via ViTPose wholebody keypoints.

        Returns list of dicts compatible with MediaPipeHandDetector.detect():
            'bbox': np.array([x1, y1, x2, y2])
            'label': "Left" or "Right"
            'is_right_int': 0 or 1
            'confidence': float (mean keypoint confidence)
            'landmarks_2d': (21, 2) pixel coords  [x, y]
            'world_landmarks': None (not available from ViTPose)

        For egocentric images where YOLO cannot detect a person body,
        automatically falls back to full-image ViTPose inference with
        a lower confidence threshold.
        """
        h_img, w_img = img_rgb.shape[:2]

        # --- Try normal YOLO-based person detection first ---
        keypoints = self.model.inference(img_rgb)  # {person_id: (133, 3)}

        # --- Egocentric fallback: no person detected, run full-image ViTPose ---
        ego_mode = False
        if len(keypoints) == 0:
            from easy_ViTPose.vit_utils.inference import pad_image
            img_pad, (left_pad, top_pad) = pad_image(img_rgb, 3 / 4)
            raw_kpts = self.model._inference(img_pad)[0]  # (133, 3) [y, x, conf]
            raw_kpts[:, :2] -= [top_pad, left_pad]
            keypoints = {0: raw_kpts}
            ego_mode = True
            # Lower confidence threshold for egocentric (no body context → noisier)
            kpt_conf_thr = min(kpt_conf_thr, 0.15)

        detections = []
        for pid, kpts in keypoints.items():
            # kpts shape: (133, 3) where each row is [y, x, confidence]
            candidates = []
            for hand_slice, label, is_right_int in [
                (self.LEFT_HAND_SLICE,  "Left",  0),
                (self.RIGHT_HAND_SLICE, "Right", 1),
            ]:
                hand_kpts = kpts[hand_slice]  # (21, 3) = [y, x, conf]
                conf = hand_kpts[:, 2]
                valid = conf > kpt_conf_thr

                if valid.sum() < min_valid_kpts:
                    continue

                # Extract 2D landmarks in [x, y] format
                # easy_ViTPose returns [y, x, conf], convert to [x, y]
                landmarks_2d = np.stack([hand_kpts[:, 1], hand_kpts[:, 0]], axis=1).astype(np.float32)

                # Build bbox from valid keypoints
                valid_pts = landmarks_2d[valid]
                x_min, y_min = valid_pts.min(axis=0)
                x_max, y_max = valid_pts.max(axis=0)
                pad_x = (x_max - x_min) * bbox_pad_ratio
                pad_y = (y_max - y_min) * bbox_pad_ratio
                bbox = np.array([
                    max(0, x_min - pad_x),
                    max(0, y_min - pad_y),
                    min(w_img, x_max + pad_x),
                    min(h_img, y_max + pad_y),
                ], dtype=np.float32)

                # Confidence = mean of per-keypoint confidence (valid ones)
                mean_conf = float(conf[valid].mean())

                # Skip tiny detections (likely noise from low-confidence keypoints)
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                if bbox_w < 20 or bbox_h < 20:
                    continue

                candidates.append({
                    'bbox': bbox,
                    'label': label,
                    'is_right_int': is_right_int,
                    'confidence': mean_conf,
                    'landmarks_2d': landmarks_2d,
                    'world_landmarks': None,  # ViTPose doesn't output 3D world landmarks
                    'ego_mode': ego_mode,
                })

            # --- Deduplicate overlapping left/right detections ---
            # ViTPose always outputs BOTH left and right hand keypoint groups
            # (COCO-WholeBody 133 format). In egocentric views, only one hand
            # may be visible but both groups fire on the same image region.
            # If two detections have high IoU, keep only the higher-confidence one.
            if len(candidates) == 2:
                b1, b2 = candidates[0]['bbox'], candidates[1]['bbox']
                # Compute IoU
                xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
                xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                union = a1 + a2 - inter
                iou = inter / max(union, 1e-6)

                if iou > 0.3:
                    # Same hand detected as both left and right — keep higher confidence
                    best = max(candidates, key=lambda c: c['confidence'])
                    detections.append(best)
                else:
                    detections.extend(candidates)
            else:
                detections.extend(candidates)

        return detections


class HaMeRHandsGenerator:
    """
    Generates Aria-compatible hand tracking data using HaMeR (Hand Mesh Recovery).
    Uses a two-stage pipeline:
      Stage 1: ViTPose wholebody for hand detection (official approach), or
               MediaPipe as fallback
      Stage 2: HaMeR for MANO-based 3D mesh recovery

    Produces the same AriaHands data structure and JSON format for seamless integration.
    """

    def __init__(self, mps_path: str, cfg_path: str, aria_cam: AriaCam):
        """
        Args:
            mps_path: Root data directory (e.g., data/mps_serve_bread_000_vrs).
            cfg_path: Path to AriaHands.yaml config (reuses same config).
            aria_cam: Processed camera sequence with per-frame k, c2w, images.
        """
        self.mps_path = mps_path
        self.cfg = load_cfg(cfg_path)
        self.aria_cam = aria_cam

        # Stage 1: Hand detector (prefer ViTPose, fall back to MediaPipe)
        try:
            self.detector = ViTPoseHandDetector(device="cuda")
            self._detector_name = "ViTPose"
        except (ImportError, FileNotFoundError, Exception) as e:
            print(f"[HaMeR] ViTPose not available ({e}), falling back to MediaPipe detector")
            self.detector = MediaPipeHandDetector()
            self._detector_name = "MediaPipe"

        # Stage 2: HaMeR model
        self.hamer_model = HaMeRModel()

        if not self.hamer_model.is_available:
            print(
                "[HaMeR] WARNING: HaMeR model not available. "
                "Falling back to MediaPipe-only 3D recovery."
            )

        # Caches for velocity computation
        self.prev_r_cache = None
        self.prev_l_cache = None
        self.prev_r_mid_cache = None
        self.prev_l_mid_cache = None
        self.prev_r_mid_R = None
        self.prev_l_mid_R = None
        self.mid_frame_builder = MidpointFrameBuilder()

    def get_aria_hands(self) -> AriaHands:
        """
        Full pipeline: extract -> clean -> optimize -> return AriaHands.
        """
        aria_hands = AriaHands(mps_path=self.mps_path)
        dt = 1.0 / self.aria_cam.fps

        use_hamer = self.hamer_model.is_available

        # Phase 1: Per-frame detection
        for i, cam_data in enumerate(tqdm(self.aria_cam.cam, total=len(self.aria_cam),
                                          desc="HaMeR Hands")):
            # Load RGB image
            img_bgr = cam_data.img
            if img_bgr is None:
                img_path = os.path.join(self.mps_path, "preprocess", "all_data",
                                        f"{cam_data.idx:05d}", "rgb.png")
                if os.path.isfile(img_path):
                    img_bgr = cv2.imread(img_path)
                else:
                    aria_hands.hands.append(AriaHandsData(cam_data.idx, cam_data.ts))
                    aria_hands.tss.append(cam_data.ts)
                    continue

            if img_bgr is None:
                aria_hands.hands.append(AriaHandsData(cam_data.idx, cam_data.ts))
                aria_hands.tss.append(cam_data.ts)
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_img, w_img = img_bgr.shape[:2]
            k = cam_data.k
            c2w = cam_data.c2w

            # Stage 1: Detect hands via MediaPipe
            detections = self.detector.detect(img_rgb)

            hand_r = None
            hand_l = None

            fx = k[0, 0]
            fy = k[1, 1]
            focal = (fx + fy) / 2.0

            for det in detections:
                label = det['label']
                mp_confidence = det['confidence']

                if use_hamer:
                    # Stage 2: HaMeR 3D mesh recovery from crop
                    hamer_result = self.hamer_model.predict_from_crop(
                        img_rgb, det['bbox'],
                        is_right=det['is_right_int'],
                        focal_length=focal,
                    )

                    if hamer_result is not None:
                        kpts_cam_mp = hamer_result['joints_3d']   # (21, 3) in camera space
                        kpts_2d_mp = hamer_result['joints_2d']    # (21, 2) pixel coords
                        # Combine detector confidence with HaMeR reconstruction quality.
                        # For ViTPose in ego mode, keypoint confidence is low (~0.2-0.5)
                        # even for clearly visible hands, so use max(mp, floor) to
                        # avoid unfairly killing confidence.
                        hamer_conf = hamer_result['confidence']
                        det_conf = max(float(mp_confidence), 0.5) if det.get('ego_mode') else float(mp_confidence)
                        confidence = float(det_conf * hamer_conf)

                        # Validate depth: ensure wrist Z is reasonable
                        wrist_z = kpts_cam_mp[0, 2]
                        if wrist_z < 0.05 or wrist_z > 3.0:
                            # Depth from HaMeR is unreliable (likely due to focal length
                            # mismatch — HaMeR assumes f≈5000 but Aria has f≈320).
                            # Re-estimate absolute depth from pixel-size + real focal.
                            kpts_cam_mp = self._recover_absolute_3d_from_hamer(
                                kpts_cam_mp, det['landmarks_2d'], k, h_img, w_img,
                            )
                            if kpts_cam_mp is None:
                                continue
                            # Recalculate confidence now that depth is corrected —
                            # the original hamer_conf was 0.15 due to bad depth.
                            # Use a generous confidence: depth recovery succeeded
                            # and HaMeR MANO structure is valid. Don't penalize
                            # by low ViTPose keypoint scores (which reflect 2D
                            # localization uncertainty, not absence of a hand).
                            confidence = max(float(mp_confidence), 0.50)
                    else:
                        # HaMeR failed on this crop; fall back to MediaPipe 3D recovery
                        # (only possible when detector provides world_landmarks, i.e., MediaPipe)
                        if det.get('world_landmarks') is not None:
                            kpts_cam_mp = self._recover_absolute_3d(
                                det['landmarks_2d'], det['world_landmarks'], k, h_img, w_img,
                            )
                            kpts_2d_mp = det['landmarks_2d']
                            confidence = mp_confidence
                            if kpts_cam_mp is None:
                                continue
                        else:
                            # ViTPose detector: no world_landmarks for fallback, skip
                            continue
                else:
                    # Pure MediaPipe fallback (no HaMeR)
                    if det.get('world_landmarks') is not None:
                        kpts_cam_mp = self._recover_absolute_3d(
                            det['landmarks_2d'], det['world_landmarks'], k, h_img, w_img,
                        )
                        kpts_2d_mp = det['landmarks_2d']
                        confidence = mp_confidence
                        if kpts_cam_mp is None:
                            continue
                    else:
                        continue

                # Remap to Aria ordering
                kpts_cam_aria = remap_mp_to_aria(kpts_cam_mp)
                kpts_2d_aria = remap_mp_to_aria(
                    np.column_stack([kpts_2d_mp, np.zeros(21)])
                )[:, :2]

                # Build AriaHandData
                h_data = self._build_hand_data(
                    kpts_cam_aria, kpts_2d_aria, confidence,
                    c2w, k, h_img, w_img,
                    is_right=(label == "Right"),
                )

                # Note: MediaPipe "Left"/"Right" labels are from the camera's perspective
                # (mirrored). For egocentric Aria video (non-mirrored), "Right" in
                # MediaPipe = the user's right hand.
                if label == "Right":
                    if hand_r is None or confidence > hand_r.confidence:
                        hand_r = h_data
                else:
                    if hand_l is None or confidence > hand_l.confidence:
                        hand_l = h_data

            frame_data = AriaHandsData(cam_data.idx, cam_data.ts, hand_r, hand_l)

            # Compute velocities and midpoint frame
            self._compute_and_assign_vel(frame_data, c2w, dt)

            aria_hands.hands.append(frame_data)
            aria_hands.tss.append(cam_data.ts)

        # Phase 2: Temporal cleaning
        # For image-based detectors (ViTPose/HaMeR), detections are noisier
        # and have more intermittent gaps than Aria MPS, so:
        # 1) Filter low-confidence detections
        # 2) Interpolate gaps FIRST (fill missing frames before judging segment length)
        # 3) Then suppress short segments
        self._filter_by_confidence(aria_hands, conf_th=0.3)
        self._interpolate_hand_trajectories(aria_hands, max_gap=self.cfg.hand_interp_max_gap * 2)
        self._suppress_short_hands(aria_hands, min_frames=max(5, self.cfg.hand_min_frames // 3))
        self._smooth_grasp_detection(aria_hands, size=self.cfg.grasp_smooth_win)

        # Phase 3: Kinematic optimization
        optimizer = AriaHandsOptimizer(self.cfg, dt)
        optimizer.run(aria_hands)
        self._smooth_grasp_detection(aria_hands, size=self.cfg.grasp_smooth_win)

        # Phase 4: Reports
        os.makedirs(os.path.join(self.mps_path, "preprocess"), exist_ok=True)
        try:
            AriaHandsOps.save_hands_analysis_plots_two(
                aria_hands, os.path.join(self.mps_path, "preprocess"), dt, self.cfg
            )
        except Exception as e:
            print(f"[HaMeR] Warning: analysis plots failed: {e}")
        AriaHandsOps.print_summary_and_eval(aria_hands)

        return aria_hands

    # ==================================================================
    # 3D Recovery (HaMeR with depth re-estimation)
    # ==================================================================

    def _recover_absolute_3d_from_hamer(
        self,
        kpts_3d_hamer: np.ndarray,    # (21, 3) HaMeR camera-space joints
        kpts_2d_mp: np.ndarray,        # (21, 2) MediaPipe 2D detections for depth estimation
        k: np.ndarray,                 # (3, 3) camera intrinsics
        h_img: int, w_img: int,
    ) -> Optional[np.ndarray]:
        """
        Re-estimate absolute depth for HaMeR 3D joints using the pinhole model.

        HaMeR outputs camera-space 3D joints but the absolute depth/scale may
        be inaccurate. We re-estimate using the known physical hand size constraint
        and 2D detections from MediaPipe, then offset HaMeR relative structure
        to the corrected wrist position.

        Returns:
            (21, 3) keypoints in camera frame (meters), or None if invalid.
        """
        # Wrist=0, MiddleMCP=9 in HaMeR/MediaPipe ordering
        wrist_2d = kpts_2d_mp[0]
        middle_mcp_2d = kpts_2d_mp[9]

        # Physical distance from HaMeR 3D joints
        physical_dist = float(np.linalg.norm(kpts_3d_hamer[9] - kpts_3d_hamer[0]))
        if physical_dist < 0.01:
            physical_dist = HAND_SIZE_WRIST_TO_MIDDLE_MCP_M

        # 2D pixel distance
        pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
        if pixel_dist < 5.0:
            return None

        fx = k[0, 0]
        fy = k[1, 1]
        focal = (fx + fy) / 2.0
        z_wrist = focal * physical_dist / pixel_dist

        if z_wrist < 0.05 or z_wrist > 3.0:
            return None

        # Back-project wrist 2D -> 3D camera frame
        cx, cy = k[0, 2], k[1, 2]
        x_wrist = (wrist_2d[0] - cx) * z_wrist / fx
        y_wrist = (wrist_2d[1] - cy) * z_wrist / fy
        wrist_cam = np.array([x_wrist, y_wrist, z_wrist], dtype=np.float32)

        # Use HaMeR relative structure offset from wrist
        offsets = kpts_3d_hamer - kpts_3d_hamer[0:1]
        kpts_cam = wrist_cam[np.newaxis, :] + offsets

        if np.any(kpts_cam[:, 2] < 0.01):
            kpts_cam[:, 2] = np.clip(kpts_cam[:, 2], 0.01, None)

        return kpts_cam.astype(np.float32)

    # ==================================================================
    # 3D Recovery (MediaPipe fallback, same as MediaPipeHands.py)
    # ==================================================================

    def _recover_absolute_3d(
        self,
        kpts_2d_mp: np.ndarray,      # (21, 2) pixel coords
        kpts_world_mp: np.ndarray,    # (21, 3) hand-centered meters
        k: np.ndarray,                # (3, 3) camera intrinsics
        h_img: int, w_img: int,
    ) -> Optional[np.ndarray]:
        """
        Recover absolute 3D keypoints in camera frame from MediaPipe outputs.
        Fallback path when HaMeR is unavailable or fails on a particular crop.

        Strategy:
            1. Measure known physical distance (wrist->middle_MCP) from world_landmarks
            2. Measure same distance in 2D pixels
            3. Estimate wrist depth: z = focal * physical_dist / pixel_dist
            4. Back-project wrist to camera frame
            5. Add relative 3D offsets from world_landmarks

        Returns:
            (21, 3) keypoints in camera frame (meters), or None if invalid.
        """
        wrist_2d = kpts_2d_mp[0]
        middle_mcp_2d = kpts_2d_mp[9]

        wrist_world = kpts_world_mp[0]
        middle_mcp_world = kpts_world_mp[9]
        physical_dist = float(np.linalg.norm(middle_mcp_world - wrist_world))

        if physical_dist < 0.01:
            physical_dist = HAND_SIZE_WRIST_TO_MIDDLE_MCP_M

        pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
        if pixel_dist < 5.0:
            return None

        fx = k[0, 0]
        fy = k[1, 1]
        focal = (fx + fy) / 2.0
        z_wrist = focal * physical_dist / pixel_dist

        if z_wrist < 0.05 or z_wrist > 3.0:
            return None

        cx, cy = k[0, 2], k[1, 2]
        x_wrist = (wrist_2d[0] - cx) * z_wrist / fx
        y_wrist = (wrist_2d[1] - cy) * z_wrist / fy
        wrist_cam = np.array([x_wrist, y_wrist, z_wrist], dtype=np.float32)

        offsets = kpts_world_mp - kpts_world_mp[0:1]
        kpts_cam = wrist_cam[np.newaxis, :] + offsets

        if np.any(kpts_cam[:, 2] < 0.01):
            kpts_cam[:, 2] = np.clip(kpts_cam[:, 2], 0.01, None)

        return kpts_cam.astype(np.float32)

    # ==================================================================
    # Build AriaHandData from camera-frame keypoints
    # ==================================================================

    def _build_hand_data(
        self,
        kpts_cam_aria: np.ndarray,   # (21, 3) Aria ordering, camera frame
        kpts_2d_aria: np.ndarray,    # (21, 2) Aria ordering, pixel coords
        confidence: float,
        c2w: np.ndarray,
        k: np.ndarray,
        h_img: int, w_img: int,
        is_right: bool,
    ) -> AriaHandData:
        """Build AriaHandData from camera-frame 21 keypoints."""

        # Wrist pose in camera frame (simple: use wrist position + palm orientation)
        wrist_pos_cam = kpts_cam_aria[5]  # Aria 5 = Wrist
        palm_center_cam = kpts_cam_aria[20]  # Aria 20 = PalmCenter
        index_mcp_cam = kpts_cam_aria[8]  # Aria 8 = IndexMCP
        middle_mcp_cam = kpts_cam_aria[11]  # Aria 11 = MiddleMCP

        # Build wrist frame: Z = palm normal, Y = wrist->palm direction
        v_wrist_palm = palm_center_cam - wrist_pos_cam
        v_wrist_palm_norm = np.linalg.norm(v_wrist_palm)
        if v_wrist_palm_norm < 1e-6:
            wrist_pose = None
        else:
            y_axis = v_wrist_palm / v_wrist_palm_norm
            v_lateral = index_mcp_cam - middle_mcp_cam
            x_axis = np.cross(y_axis, v_lateral)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                wrist_pose = None
            else:
                x_axis /= x_norm
                z_axis = np.cross(x_axis, y_axis)
                z_axis /= (np.linalg.norm(z_axis) + 1e-6)
                y_axis = np.cross(z_axis, x_axis)

                wrist_pose = np.eye(4, dtype=np.float64)
                wrist_pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
                wrist_pose[:3, 3] = wrist_pos_cam

        # Grasp detection: ratio-based (scale-invariant)
        # thumb tip (Aria 0) vs index tip (Aria 1), normalized by palm size
        thumb_tip = kpts_cam_aria[0]
        index_tip = kpts_cam_aria[1]
        wrist = kpts_cam_aria[5]        # Aria 5 = Wrist
        mid_mcp = kpts_cam_aria[11]     # Aria 11 = MiddleMCP
        distance = float(np.linalg.norm(thumb_tip - index_tip))
        palm_size = float(np.linalg.norm(mid_mcp - wrist))
        if palm_size > 0.01:
            grasp_ratio = distance / palm_size
            grasp_state = 1 if grasp_ratio < 1.0 else 0
        else:
            grasp_threshold = getattr(self.cfg, 'grasp_threshold', 0.105)
            grasp_state = 1 if distance < grasp_threshold else 0

        # Joint angles
        joint_angles = AriaHandsJointAngles.from_keypoints_3d(kpts_cam_aria)

        # Use identity for d2c since we don't have device->camera for image-based methods
        d2c = np.eye(4, dtype=np.float64)

        return AriaHandData(
            d2c=d2c,
            c2w=c2w,
            is_right=is_right,
            confidence=confidence,
            wrist_pose=wrist_pose,
            palm_pose=wrist_pose,  # Approximate: same as wrist
            hand_keypoints_3d=kpts_cam_aria,
            hand_keypoints_2d=kpts_2d_aria,
            grasp_state=grasp_state,
            joint_angles=joint_angles,
        )

    # ==================================================================
    # Velocity & Midpoint computation (mirrors AriaHands.py logic)
    # ==================================================================

    def _compute_and_assign_vel(self, hands_data: AriaHandsData,
                                c2w: np.ndarray, dt: float) -> None:
        """Compute world-space poses, velocities, and midpoint gripper frame."""
        def robust_rot(matrix):
            try:
                return R.from_matrix(matrix)
            except ValueError:
                U, S, Vt = np.linalg.svd(matrix)
                d = np.linalg.det(U @ Vt)
                if d < 0: U[:, -1] *= -1
                return R.from_matrix(U @ Vt)

        for is_right in [True, False]:
            h_data = hands_data.hand_r if is_right else hands_data.hand_l
            prev_cache = self.prev_r_cache if is_right else self.prev_l_cache
            prev_mid_cache = self.prev_r_mid_cache if is_right else self.prev_l_mid_cache
            prev_R = self.prev_r_mid_R if is_right else self.prev_l_mid_R

            if h_data and h_data.wrist_pose is not None:
                # Wrist -> World
                p_cam = h_data.wrist_pose[:3, 3]
                r_cam = h_data.wrist_pose[:3, :3]
                p_world = (c2w[:3, :3] @ p_cam) + c2w[:3, 3]
                r_world = c2w[:3, :3] @ r_cam

                h_data.wrist_pose_raw_world = np.eye(4)
                h_data.wrist_pose_raw_world[:3, :3] = r_world
                h_data.wrist_pose_raw_world[:3, 3] = p_world

                if prev_cache is not None:
                    h_data.wrist_lin_vel_raw_world = (p_world - prev_cache['pos']) / dt
                    rel = prev_cache['rot'].T @ r_world
                    h_data.wrist_ang_vel_raw_world = robust_rot(rel).as_rotvec() / dt

                cache_val = {'pos': p_world, 'rot': r_world}
                if is_right: self.prev_r_cache = cache_val
                else: self.prev_l_cache = cache_val

                # Midpoint gripper frame
                if h_data.hand_keypoints_3d is not None and len(h_data.hand_keypoints_3d) >= 9:
                    thumb_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[0]) + c2w[:3, 3]
                    index_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[1]) + c2w[:3, 3]
                    thumb_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[6]) + c2w[:3, 3]
                    index_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[8]) + c2w[:3, 3]

                    h_data.thumb_translation_raw_world = thumb_w
                    h_data.index_translation_raw_world = index_w
                    h_data.thumb_base_raw_world = thumb_base_w
                    h_data.index_base_raw_world = index_base_w

                    midpoint_w = (thumb_w + index_w) / 2.0
                    h_data.midpoint_translation_raw_world = midpoint_w

                    R_mid = self.mid_frame_builder.build(
                        thumb_w=thumb_w, index_w=index_w,
                        thumb_base_w=thumb_base_w, index_base_w=index_base_w,
                        wrist_w=p_world, midpoint_w=midpoint_w, prev_R=prev_R,
                    )
                    if R_mid is None:
                        R_mid = prev_R if prev_R is not None else r_world.copy()

                    h_data.midpoint_pose_raw_world = np.eye(4)
                    h_data.midpoint_pose_raw_world[:3, :3] = R_mid
                    h_data.midpoint_pose_raw_world[:3, 3] = midpoint_w
                    h_data.midpoint_orientation_raw_world = R_mid.flatten()

                    if prev_mid_cache is not None:
                        h_data.midpoint_lin_vel_raw_world = (midpoint_w - prev_mid_cache['pos']) / dt
                        rel = prev_mid_cache['rot'].T @ R_mid
                        h_data.midpoint_ang_vel_raw_world = robust_rot(rel).as_rotvec() / dt

                    cache_mid = {'pos': midpoint_w, 'rot': R_mid}
                    if is_right:
                        self.prev_r_mid_cache = cache_mid
                        self.prev_r_mid_R = R_mid
                    else:
                        self.prev_l_mid_cache = cache_mid
                        self.prev_l_mid_R = R_mid

    # ==================================================================
    # Temporal cleaning (mirrors AriaHands.py logic)
    # ==================================================================

    def _filter_by_confidence(self, aria_hands: AriaHands, conf_th: float = 0.3) -> None:
        for frame_data in aria_hands.hands:
            for attr in ["hand_r", "hand_l"]:
                h = getattr(frame_data, attr)
                if h and (h.confidence < conf_th):
                    setattr(frame_data, attr, None)

    def _suppress_short_hands(self, aria_hands: AriaHands, min_frames: int = 5) -> None:
        for attr in ["hand_r", "hand_l"]:
            presence = [getattr(h, attr) is not None for h in aria_hands.hands]
            count, segments = 0, []
            for i, is_present in enumerate(presence):
                if is_present:
                    count += 1
                else:
                    if 0 < count < min_frames:
                        segments.append((i - count, i))
                    count = 0
            if 0 < count < min_frames:
                segments.append((len(presence) - count, len(presence)))
            for start, end in segments:
                for i in range(start, end):
                    setattr(aria_hands.hands[i], attr, None)

    def _interpolate_hand_trajectories(self, aria_hands: AriaHands, max_gap: int = 3) -> None:
        from scipy.spatial.transform import Slerp
        for attr in ["hand_r", "hand_l"]:
            presence = [getattr(h, attr) is not None for h in aria_hands.hands]
            indices = np.where(presence)[0]
            if len(indices) < 2:
                continue
            for start_i, end_i in zip(indices[:-1], indices[1:]):
                gap = end_i - start_i - 1
                if 0 < gap <= max_gap:
                    h_start = getattr(aria_hands.hands[start_i], attr)
                    h_end = getattr(aria_hands.hands[end_i], attr)
                    if h_start.wrist_pose is None or h_end.wrist_pose is None:
                        continue
                    steps = np.linspace(0, 1, gap + 2)[1:-1]
                    for j, t in enumerate(steps):
                        fill_idx = start_i + j + 1
                        h_new = AriaHandData(
                            d2c=h_start.d2c, c2w=h_start.c2w,
                            is_right=h_start.is_right,
                            confidence=(1.0 - t) * h_start.confidence + t * h_end.confidence,
                        )
                        # Interpolate wrist pose
                        pos_interp = (1.0 - t) * h_start.wrist_pose[:3, 3] + t * h_end.wrist_pose[:3, 3]
                        try:
                            rots = R.from_matrix([h_start.wrist_pose[:3, :3], h_end.wrist_pose[:3, :3]])
                            slerp = Slerp([0, 1], rots)
                            rot_interp = slerp(t).as_matrix()
                        except Exception:
                            rot_interp = h_start.wrist_pose[:3, :3]
                        T_interp = np.eye(4)
                        T_interp[:3, :3] = rot_interp
                        T_interp[:3, 3] = pos_interp
                        h_new.wrist_pose = T_interp
                        h_new.palm_pose = T_interp

                        # Interpolate keypoints
                        if h_start.hand_keypoints_3d is not None and h_end.hand_keypoints_3d is not None:
                            h_new.hand_keypoints_3d = (1.0 - t) * h_start.hand_keypoints_3d + t * h_end.hand_keypoints_3d
                        if h_start.hand_keypoints_2d is not None and h_end.hand_keypoints_2d is not None:
                            h_new.hand_keypoints_2d = (1.0 - t) * h_start.hand_keypoints_2d + t * h_end.hand_keypoints_2d

                        h_new.grasp_state = h_start.grasp_state
                        setattr(aria_hands.hands[fill_idx], attr, h_new)

    def _smooth_grasp_detection(self, aria_hands: AriaHands, size: int = 5) -> None:
        from scipy.ndimage import uniform_filter1d
        for attr in ["hand_r", "hand_l"]:
            states = []
            for h in aria_hands.hands:
                hand = getattr(h, attr)
                states.append(hand.grasp_state if hand else 0)
            g = np.array(states, dtype=np.float32)
            g = uniform_filter1d(g, size=size)
            g = (g > 0.5).astype(int)

            # Flicker suppression
            flicker_max = getattr(self.cfg, 'grasp_flicker_max_len', 5)
            for flip_val in [0, 1]:
                count = 0
                for i in range(len(g)):
                    if g[i] == flip_val:
                        count += 1
                    else:
                        if 0 < count <= flicker_max:
                            for j in range(i - count, i):
                                g[j] = 1 - flip_val
                        count = 0

            for i, h in enumerate(aria_hands.hands):
                hand = getattr(h, attr)
                if hand:
                    hand.grasp_state = int(g[i])

    # ==================================================================
    # Visualization (delegate to AriaHandsOps)
    # ==================================================================

    def draw_aria_hands_skeleton(self, img, data, k, d, c2w):
        """Draw hand skeleton overlay -- compatible with existing vis pipeline."""
        return AriaHandsOps.draw_aria_hands_skeleton(
            img, data, k, d, c2w,
            getattr(self.cfg, 'grasp_threshold', 0.105)
        )

    def draw_aria_hands_panel(self, img, idx, data):
        return AriaHandsOps.draw_aria_hands_panel(
            img, idx, data, getattr(self.cfg, 'opt_v_limit', 0.6)
        )


# ==================================================================
# Standalone runner
# ==================================================================

def run_hamer_hands(mps_path: str, cfg_path: str, aria_cam=None,
                    export_video: bool = False, export_gif: bool = False) -> AriaHands:
    """
    Entry point for Preprocess.py integration.

    Args:
        mps_path: Root data directory.
        cfg_path: Path to AriaHands.yaml config.
        aria_cam: Optional pre-built AriaCam object. If None, reconstructs from disk.
        export_video: Whether to export visualization video.
        export_gif: Whether to export GIF alongside the video.
    """
    from preprocess.MediaPipeHands import _build_aria_cam_from_disk

    if aria_cam is None:
        aria_cam = _build_aria_cam_from_disk(mps_path)

    gen = HaMeRHandsGenerator(mps_path, cfg_path, aria_cam)
    aria_hands = gen.get_aria_hands()

    # Save per-frame JSONs with method-specific filename
    aria_hands.save_hands_json(filename="hamer_hands.json")
    print(f"[HaMeR] Saved hamer_hands.json for {len(aria_hands)} frames")

    # ── Visualization video (skeleton + HUD overlay) ──
    if export_video and len(aria_cam.cam) > 0:
        import cv2
        from tqdm import tqdm
        print(f"[HaMeR] Generating visualization video …")
        vis_frames = []
        for idx in tqdm(range(len(aria_cam.cam)), desc="HaMeR Vis"):
            cam_d = aria_cam.cam[idx]
            img = cam_d.img
            if img is None:
                img_path = os.path.join(mps_path, "preprocess", "all_data",
                                        f"{cam_d.idx:05d}", "rgb.png")
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
            if img is None:
                continue
            img = img.copy()

            if idx < len(aria_hands.hands):
                img = gen.draw_aria_hands_skeleton(
                    img, aria_hands.hands[idx],
                    cam_d.k, getattr(cam_d, 'd', np.zeros(8)), cam_d.c2w
                )
                img = gen.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])

            vis_frames.append(img)

        if vis_frames:
            vis_dir = os.path.join(mps_path, "preprocess", "vis")
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, "hamer_hands_vis.mp4")
            create_video_from_frames(vis_frames, save_path, aria_cam.fps, export_gif)
            print(f"[HaMeR] Saved visualization → {save_path}")

    return aria_hands


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HaMeR Hand Mesh Recovery Tracking")
    parser.add_argument("--mps_path", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/base/AriaHands.yaml")
    parser.add_argument("--export_video", action="store_true")
    parser.add_argument("--export_gif", action="store_true")
    args = parser.parse_args()
    print(f"[HaMeR] mps_path={args.mps_path}")
    run_hamer_hands(args.mps_path, args.cfg_path,
                    export_video=args.export_video, export_gif=args.export_gif)
