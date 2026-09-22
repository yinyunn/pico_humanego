# -*- coding: utf-8 -*-
# @FileName: MediaPipeHands.py

"""
====================================================================================================
MediaPipe Hand Tracking Pipeline (MediaPipeHands.py)
====================================================================================================

Description:
    Image-based hand detection using Google's MediaPipe Hands. Produces AriaHands-compatible
    output so it can be directly swapped with Aria MPS hand tracking for ablation studies.

Pipeline:
    1. Read RGB frames from the existing AriaCam sequence
    2. Run MediaPipe Hands to get 21 2D/3D landmarks per hand
    3. Recover absolute 3D keypoints in camera frame:
       - Use 2D wrist position + camera intrinsics for back-projection direction
       - Use MediaPipe world_landmarks (hand-centered, metric) for relative structure
       - Estimate depth via physical hand size constraint
    4. Remap MediaPipe 21-point ordering to Aria MPS 21-point ordering
    5. Transform from camera frame to world frame via c2w
    6. Build midpoint "gripper" frame using MidpointFrameBuilder
    7. Detect grasp state via thumb-index fingertip distance
    8. Apply temporal smoothing (Savitzky-Golay + EMA) via AriaHandsOptimizer
    9. Save per-frame JSON in identical format to aria_hands.json

Generated Outputs:
    [mps_path]/aria/all_data/[idx]/MediaPipeHands.json  (per-frame, same schema as aria_hands.json)
    [mps_path]/aria/vis/mediapipe_hands_vis.mp4          (skeleton visualization)
    [mps_path]/aria/mediapipe_hands_analysis_r.png
    [mps_path]/aria/mediapipe_hands_analysis_l.png

Requirements:
    pip install mediapipe
====================================================================================================
"""

import os
import json
import cv2
import numpy as np
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
# MediaPipe 21-point → Aria MPS 21-point index mapping
# ==================================================================
# MediaPipe ordering:
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
# MP_TO_ARIA[aria_idx] = mediapipe_idx
MP_TO_ARIA = [
    4,   # Aria 0  = ThumbTip     ← MP 4
    8,   # Aria 1  = IndexTip     ← MP 8
    12,  # Aria 2  = MiddleTip    ← MP 12
    16,  # Aria 3  = RingTip      ← MP 16
    20,  # Aria 4  = PinkyTip     ← MP 20
    0,   # Aria 5  = Wrist        ← MP 0
    2,   # Aria 6  = ThumbMCP     ← MP 2  (ThumbMCP in Aria = ThumbMCP in MP)
    3,   # Aria 7  = ThumbIP      ← MP 3
    5,   # Aria 8  = IndexMCP     ← MP 5
    6,   # Aria 9  = IndexPIP     ← MP 6
    7,   # Aria 10 = IndexDIP     ← MP 7
    9,   # Aria 11 = MiddleMCP    ← MP 9
    10,  # Aria 12 = MiddlePIP    ← MP 10
    11,  # Aria 13 = MiddleDIP    ← MP 11
    13,  # Aria 14 = RingMCP      ← MP 13
    14,  # Aria 15 = RingPIP      ← MP 14
    15,  # Aria 16 = RingDIP      ← MP 15
    17,  # Aria 17 = PinkyMCP     ← MP 17
    18,  # Aria 18 = PinkyPIP     ← MP 18
    19,  # Aria 19 = PinkyDIP     ← MP 19
    -1,  # Aria 20 = PalmCenter   ← computed (mean of Wrist, IndexMCP, MiddleMCP)
]

# Average adult hand: wrist to middle MCP ≈ 0.085m
HAND_SIZE_WRIST_TO_MIDDLE_MCP_M = 0.085


def remap_mp_to_aria(kpts_mp_21: np.ndarray) -> np.ndarray:
    """
    Remap 21 MediaPipe keypoints to Aria MPS 21-point ordering.

    Args:
        kpts_mp_21: (21, 3) keypoints in MediaPipe ordering.

    Returns:
        (21, 3) keypoints in Aria ordering. Index 20 (PalmCenter) is computed.
    """
    kpts_aria = np.zeros((21, 3), dtype=kpts_mp_21.dtype)
    for aria_idx in range(20):
        mp_idx = MP_TO_ARIA[aria_idx]
        kpts_aria[aria_idx] = kpts_mp_21[mp_idx]
    # Aria 20 = PalmCenter ≈ mean(Wrist=MP0, IndexMCP=MP5, MiddleMCP=MP9)
    kpts_aria[20] = (kpts_mp_21[0] + kpts_mp_21[5] + kpts_mp_21[9]) / 3.0
    return kpts_aria


def create_hand_landmarker(*, model_path=None, video_mode=False, allow_download=True):
    """Shared model/options for Aria and PICO; PICO smoke uses local weights only."""
    import mediapipe as mp
    if model_path is not None:
        model_path = os.fspath(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
    else:
        # Locate the hand_landmarker.task model file
        # Priority: local weights/ dir → HuggingFace Hub auto-download
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "weights", "mediapipe", "hand_landmarker.task"
        )
        if not os.path.isfile(model_path) and not allow_download:
            raise FileNotFoundError(f"Existing MediaPipe model not found: {model_path}")
        if not os.path.isfile(model_path):
            try:
                from huggingface_hub import hf_hub_download
                # MediaPipe hand_landmarker.task (Apache 2.0, (c) Google)
                # Hosted at: https://huggingface.co/Leo-TX/mediapipe-hand
                model_path = hf_hub_download(
                    repo_id="Leo-TX/mediapipe-hand",
                    filename="hand_landmarker.task",
                )
                print(f"[MediaPipe] Downloaded hand_landmarker.task from HuggingFace Hub")
            except Exception:
                raise FileNotFoundError(
                    f"MediaPipe hand_landmarker.task not found at {model_path}. "
                    "Download from: https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                )
    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=(mp.tasks.vision.RunningMode.VIDEO if video_mode
                          else mp.tasks.vision.RunningMode.IMAGE),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


class MediaPipeHandsGenerator:
    """
    Generates Aria-compatible hand tracking data using Google MediaPipe Hands.
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

        # MediaPipe Hands model (new tasks API, v0.10+)
        import mediapipe as mp
        self._mp = mp  # Store for later use in get_aria_hands

        self.mp_landmarker = create_hand_landmarker()

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
        Full pipeline: extract → clean → optimize → return AriaHands.
        """
        aria_hands = AriaHands(mps_path=self.mps_path)
        dt = 1.0 / self.aria_cam.fps

        # Phase 1: Per-frame detection
        for i, cam_data in enumerate(tqdm(self.aria_cam.cam, total=len(self.aria_cam),
                                          desc="MediaPipe Hands")):
            # Load RGB image
            img_bgr = cam_data.img
            if img_bgr is None:
                img_path = os.path.join(self.mps_path, "preprocess", "all_data",
                                        f"{cam_data.idx:05d}", "aria_cam_rgb.jpg")
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

            # Run MediaPipe (new tasks API)
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=img_rgb
            )
            results = self.mp_landmarker.detect(mp_img)

            hand_r = None
            hand_l = None

            if results.hand_landmarks and results.handedness:
                for hand_landmarks, hand_world_lms, handedness_list in zip(
                    results.hand_landmarks,
                    results.hand_world_landmarks,
                    results.handedness,
                ):
                    # Determine side (MediaPipe mirrors: "Right" in image = user's right)
                    label = handedness_list[0].category_name  # "Left" or "Right"
                    confidence = handedness_list[0].score

                    # Extract 2D landmarks (normalized → pixel coords)
                    kpts_2d_mp = np.array(
                        [[lm.x * w_img, lm.y * h_img] for lm in hand_landmarks],
                        dtype=np.float32,
                    )

                    # Extract world landmarks (hand-centered, meters)
                    kpts_world_mp = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_world_lms],
                        dtype=np.float32,
                    )

                    # Recover absolute 3D in camera frame
                    kpts_cam_mp = self._recover_absolute_3d(
                        kpts_2d_mp, kpts_world_mp, k, h_img, w_img
                    )

                    if kpts_cam_mp is None:
                        continue

                    # Re-project 3D → 2D for consistency with recovered depth
                    # (matches HaMeR/WiLoR pipelines which also re-project)
                    fx, fy = k[0, 0], k[1, 1]
                    cx, cy = k[0, 2], k[1, 2]
                    kpts_2d_reproj = np.zeros((21, 2), dtype=np.float32)
                    if np.all(kpts_cam_mp[:, 2] > 0.01):
                        kpts_2d_reproj[:, 0] = kpts_cam_mp[:, 0] / kpts_cam_mp[:, 2] * fx + cx
                        kpts_2d_reproj[:, 1] = kpts_cam_mp[:, 1] / kpts_cam_mp[:, 2] * fy + cy
                    kpts_2d_mp = kpts_2d_reproj

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
        self._filter_by_confidence(aria_hands, conf_th=0.3)  # MediaPipe has its own confidence
        self._suppress_short_hands(aria_hands, min_frames=self.cfg.hand_min_frames)
        self._interpolate_hand_trajectories(aria_hands, max_gap=self.cfg.hand_interp_max_gap)
        self._smooth_grasp_detection(aria_hands, size=self.cfg.grasp_smooth_win)

        # Phase 3: Kinematic optimization
        optimizer = AriaHandsOptimizer(self.cfg, dt)
        optimizer.run(aria_hands)
        self._smooth_grasp_detection(aria_hands, size=self.cfg.grasp_smooth_win)

        # Close the MediaPipe landmarker
        try:
            self.mp_landmarker.close()
        except Exception:
            pass

        # Phase 4: Reports
        os.makedirs(os.path.join(self.mps_path, "preprocess"), exist_ok=True)
        try:
            AriaHandsOps.save_hands_analysis_plots_two(
                aria_hands, os.path.join(self.mps_path, "preprocess"), dt, self.cfg
            )
        except Exception as e:
            print(f"║ [MediaPipe] Warning: analysis plots failed: {e}")
        AriaHandsOps.print_summary_and_eval(aria_hands)

        return aria_hands

    # ==================================================================
    # 3D Recovery
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

        Strategy:
            1. Measure known physical distance (wrist→middle_MCP) from world_landmarks
            2. Measure same distance in 2D pixels
            3. Estimate wrist depth: z = focal * physical_dist / pixel_dist
            4. Back-project wrist to camera frame
            5. Add relative 3D offsets from world_landmarks

        Returns:
            (21, 3) keypoints in camera frame (meters), or None if invalid.
        """
        # MediaPipe indices: Wrist=0, MiddleMCP=9
        wrist_2d = kpts_2d_mp[0]
        middle_mcp_2d = kpts_2d_mp[9]

        # Physical distance from world_landmarks
        wrist_world = kpts_world_mp[0]
        middle_mcp_world = kpts_world_mp[9]
        physical_dist = float(np.linalg.norm(middle_mcp_world - wrist_world))

        if physical_dist < 0.01:
            # Fallback: use known average hand size
            physical_dist = HAND_SIZE_WRIST_TO_MIDDLE_MCP_M

        # 2D pixel distance
        pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
        if pixel_dist < 5.0:
            return None  # Hand too small in image or detection noise

        # Estimate depth using pinhole model: pixel_dist = focal * physical_dist / z
        fx = k[0, 0]
        fy = k[1, 1]
        focal = (fx + fy) / 2.0
        z_wrist = focal * physical_dist / pixel_dist

        if z_wrist < 0.05 or z_wrist > 3.0:
            return None  # Unreasonable depth

        # Back-project wrist 2D → 3D camera frame
        cx, cy = k[0, 2], k[1, 2]
        x_wrist = (wrist_2d[0] - cx) * z_wrist / fx
        y_wrist = (wrist_2d[1] - cy) * z_wrist / fy
        wrist_cam = np.array([x_wrist, y_wrist, z_wrist], dtype=np.float32)

        # All 21 points in camera frame:
        # kpt_cam[i] = wrist_cam + (world_landmark[i] - world_landmark[wrist])
        offsets = kpts_world_mp - kpts_world_mp[0:1]  # relative to wrist
        kpts_cam = wrist_cam[np.newaxis, :] + offsets

        # Sanity check: all Z should be positive
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

        # Build wrist frame: Z = palm normal, Y = wrist→palm direction
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

        # Grasp detection: thumb tip (Aria 0) vs index tip (Aria 1)
        # Normalize by palm size (wrist→middle_MCP) to make threshold scale-invariant.
        # MediaPipe 3D recovery produces systematically smaller distances than Aria MPS
        # (mean ~0.078 vs ~0.100), so absolute thresholds don't transfer well.
        thumb_tip = kpts_cam_aria[0]
        index_tip = kpts_cam_aria[1]
        wrist = kpts_cam_aria[5]       # Aria 5 = Wrist
        mid_mcp = kpts_cam_aria[11]    # Aria 11 = MiddleMCP
        distance = float(np.linalg.norm(thumb_tip - index_tip))
        palm_size = float(np.linalg.norm(mid_mcp - wrist))
        if palm_size > 0.01:
            # Relative ratio: grasp when thumb-index < ~1.0× palm_size
            # Aria MPS: mean ratio ≈ 1.04, 63% below 1.0.  MediaPipe produces
            # systematically smaller distances, so 1.0 yields ~88% pre-smoothing,
            # which temporal smoothing + flicker suppression bring to ≈70-75%.
            grasp_ratio = distance / palm_size
            grasp_state = 1 if grasp_ratio < 1.0 else 0
        else:
            # Fallback to absolute threshold if palm size is degenerate
            grasp_threshold = getattr(self.cfg, 'grasp_threshold', 0.105)
            grasp_state = 1 if distance < grasp_threshold else 0

        # Joint angles
        joint_angles = AriaHandsJointAngles.from_keypoints_3d(kpts_cam_aria)

        # Use identity for d2c since we don't have device→camera for image-based methods
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
                # Wrist → World
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
        """Draw hand skeleton overlay — compatible with existing vis pipeline."""
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

def _build_aria_cam_from_disk(mps_path: str):
    """
    Reconstruct a lightweight AriaCam object from per-frame JSON files on disk.
    This allows running MediaPipeHands standalone (without the full Preprocess pipeline).
    Returns an AriaCam with cam[] having img, k, c2w per frame.
    """
    from preprocess.AriaCamTypes import AriaCam, AriaCamData
    import cv2

    all_data_dir = os.path.join(mps_path, "preprocess", "all_data")
    if not os.path.isdir(all_data_dir):
        raise FileNotFoundError(f"all_data directory not found: {all_data_dir}")

    frame_dirs = sorted([d for d in os.listdir(all_data_dir) if d.isdigit()])
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {all_data_dir}")

    cam = AriaCam()
    cam.mps_path = mps_path

    for fn in frame_dirs:
        frame_dir = os.path.join(all_data_dir, fn)
        cam_json_path = os.path.join(frame_dir, "aria_cam_rgb.json")
        rgb_path = os.path.join(frame_dir, "rgb.png")

        if not os.path.exists(cam_json_path) or not os.path.exists(rgb_path):
            continue

        with open(cam_json_path, 'r') as f:
            cam_d = json.load(f)

        img = cv2.imread(rgb_path)  # Keep as BGR to match AriaCam convention

        frame = AriaCamData(
            idx=int(fn),
            ts=cam_d.get('ts', 0),
            img=img,
            h=img.shape[0] if img is not None else 0,
            w=img.shape[1] if img is not None else 0,
            k=np.array(cam_d['k'], dtype=np.float64) if cam_d.get('k') is not None else np.eye(3),
            c2w=np.array(cam_d['c2w'], dtype=np.float64) if cam_d.get('c2w') is not None else np.eye(4),
            d=np.zeros(8),       # Distortion coefficients (images are already rectified)
            c2d=np.eye(4),       # Camera-to-Device (identity for rectified)
            d2w=np.eye(4),       # Device-to-World placeholder
        )

        cam.tss.append(frame.ts)
        cam.cam.append(frame)

    if cam.cam:
        cam.k = cam.cam[0].k
        cam.h = cam.cam[0].h
        cam.w = cam.cam[0].w
        cam.fps = 30  # Default Aria RGB FPS; overridden if aria_cam_rgb.json has fps field
        if len(cam.cam) > 0:
            first_cam_json = os.path.join(all_data_dir, frame_dirs[0], "aria_cam_rgb.json")
            if os.path.exists(first_cam_json):
                with open(first_cam_json, 'r') as f:
                    first_d = json.load(f)
                cam.fps = first_d.get('fps', 30)

    print(f"[MediaPipeHands] Loaded {len(cam.cam)} frames from disk")
    return cam


def run_mediapipe_hands(mps_path: str, cfg_path: str, aria_cam=None,
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
    if aria_cam is None:
        aria_cam = _build_aria_cam_from_disk(mps_path)

    gen = MediaPipeHandsGenerator(mps_path, cfg_path, aria_cam)
    aria_hands = gen.get_aria_hands()

    # Save per-frame JSONs with method-specific filename
    aria_hands.save_hands_json(filename="mediapipe_hands.json")
    print(f"[MediaPipeHands] Saved mediapipe_hands.json for {len(aria_hands)} frames")

    # ── Visualization video (skeleton + HUD overlay, same style as AriaHands) ──
    if export_video and len(aria_cam.cam) > 0:
        print(f"[MediaPipeHands] Generating visualization video …")
        vis_frames = []
        for idx in tqdm(range(len(aria_cam.cam)), desc="MediaPipe Vis"):
            cam_d = aria_cam.cam[idx]
            img = cam_d.img
            if img is None:
                img_path = os.path.join(mps_path, "preprocess", "all_data",
                                        f"{cam_d.idx:05d}", "aria_cam_rgb.jpg")
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
            if img is None:
                continue
            img = img.copy()

            if idx < len(aria_hands.hands):
                # Skeleton overlay (keypoints, bones, grasp indicator, axis gizmo)
                img = gen.draw_aria_hands_skeleton(
                    img, aria_hands.hands[idx],
                    cam_d.k, getattr(cam_d, 'd', np.zeros(8)), cam_d.c2w
                )
                # HUD panel (confidence bar, joint angles, velocity gauge)
                img = gen.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])

            vis_frames.append(img)

        if vis_frames:
            vis_dir = os.path.join(mps_path, "preprocess", "vis")
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, "mediapipe_hands_vis.mp4")
            create_video_from_frames(vis_frames, save_path, aria_cam.fps, export_gif)
            print(f"[MediaPipeHands] Saved visualization → {save_path}")

    return aria_hands


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MediaPipe Hand Tracking")
    parser.add_argument("--mps_path", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/base/AriaHands.yaml")
    parser.add_argument("--export_video", action="store_true")
    parser.add_argument("--export_gif", action="store_true")
    args = parser.parse_args()
    print(f"[MediaPipeHands] mps_path={args.mps_path}")
    run_mediapipe_hands(args.mps_path, args.cfg_path,
                        export_video=args.export_video, export_gif=args.export_gif)
