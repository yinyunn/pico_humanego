# -*- coding: utf-8 -*-
# @FileName: AriaPhases.py

"""
====================================================================================================
Project Aria Task Phase Segmentation Pipeline (AriaPhases.py)
====================================================================================================

Description:
    This script identifies and segments high-level temporal task phases (e.g., STOP/MANIP, 
    FORWARD, ROTATE, TRANSITION) from continuous Project Aria SLAM kinematics and hand tracking data.
    It applies heuristic thresholds, temporal smoothing, and multi-modal refinement to output 
    clean, jitter-free phase masks.

Core Functionalities:
    1.  Kinematic State Detection: Derives base motion states using linear (v) and angular (w) 
        velocities with configurable thresholds.
    2.  Temporal Smoothing: Fills short gaps and suppresses transient noise in the phase 
        predictions using morphological rules and median filtering.
    3.  Multi-Modal Refinement: Incorporates hand velocity data to refine 'MANIP' (manipulation) 
        and 'TRANSITION' phases, ensuring biomechanical consistency.
    4.  Diagnostic Reporting: Generates summary statistics and professional timeline plots 
        to evaluate phase transitions.

Generated Outputs:
    📁 [mps_path]/aria/
    ├── 🎬 vis/aria_phases_vis.mp4       # Video visualization with Phase HUD overlays.
    ├── 🖼️ aria_phases_analysis.png      # Professional diagnostic plot mapping phases to kinematics.
    ├── 📄 aria_phases_results.json      # Overall sequence summary and phase distributions.
    └── 📁 all_data/[idx]/
        └── 📄 aria_phases.json          # Per-frame temporal mode and kinematic metadata.

Technical Specifics:
    - Mode Encoding: 0=STOP/MANIP, 1=FORWARD, 2=ROTATE, 3=TRANSITION.
    - Configuration: Uses a YAML-based config mapping for customizable speed thresholds.
====================================================================================================
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, Any

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from preprocess.AriaCam import AriaCamGenerator
from preprocess.AriaSlam import AriaSlamGenerator
from preprocess.AriaHands import AriaHandsGenerator
from preprocess.AriaPhasesTypes import AriaPhases, AriaPhasesFrame
from preprocess.AriaPhasesOps import AriaPhasesOps

from utils.utils_media import create_video_from_frames
from utils.utils_io import load_cfg


class AriaPhasesGenerator:
    """
    Handles the orchestration of the temporal phase segmentation process, utilizing both
    SLAM kinematics and optional Hand tracking kinematics for accurate state estimation.
    """

    def __init__(self, mps_path: str, cfg_path: str, aria_cam: Any, aria_slam: Any, aria_hands: Optional[Any] = None):
        """
        Initializes the phase generator with configuration and pre-processed modalities.

        Args:
            mps_path (str): Path to the MPS data directory.
            cfg_path (str): Path to the YAML configuration file.
            aria_cam (AriaCam): Processed camera metadata and frame info.
            aria_slam (AriaSlam): Processed SLAM kinematic sequences.
            aria_hands (Optional[AriaHands]): Processed hand tracking sequence (used for manipulation refinement).
        """
        self.mps_path = mps_path
        self.cfg_path = cfg_path
        self.cfg = load_cfg(self.cfg_path)
        
        self.aria_cam = aria_cam
        self.aria_slam = aria_slam
        self.aria_hands = aria_hands


    def get_aria_phases(self) -> AriaPhases:
        """
        Executes the multi-stage phase segmentation pipeline to classify each frame's task state.

        Steps:
            1. Stop Detection: Threshold-based stop logic using linear/angular velocities.
            2. Mask Smoothing: Gap filling and short-run suppression.
            3. Kinematic Vetoes: Overrides stops if significant yaw rotation occurs.
            4. Mode Classification: Assigns multi-class states (Stop/Forward/Rotate/Transition).
            5. Hand Refinement: Uses hand velocity to accurately extend manipulation boundaries.

        Returns:
            AriaPhases: A sequence container populated with per-frame phase classifications.
        """
        n = len(self.aria_slam.frames)
        v = np.array([f.v for f in self.aria_slam.frames])
        w = np.array([f.w for f in self.aria_slam.frames])
        yaw_u = np.array([f.yaw_unwrapped_deg for f in self.aria_slam.frames])
        tss = [f.ts for f in self.aria_slam.frames]

        # --- 1. Initial Stop Detection (Pure Kinematics) ---
        stop = AriaPhasesOps.compute_stop_mask_from_vw(
            v, w, self.cfg.v_stop_thresh, self.cfg.w_stop_thresh,
            self.cfg.stop_hold_frames, self.cfg.stop_debounce_frames
        )

        # --- 2. Temporal Smoothing & Gap Filling ---
        stop = AriaPhasesOps.close_binary_mask(stop, kernel_size=self.cfg.stop_debounce_frames)

        # --- 3. Post-Process Stop Mask (Short-run Removal & Yaw Veto) ---
        stop = AriaPhasesOps.postprocess_stop_mask(
            stop, yaw_u, self.cfg.stop_min_on_frames, self.cfg.stop_yaw_veto_deg
        )

        # --- 4. Apply Stop Offset Expansion ---
        stop = AriaPhasesOps.apply_stop_offset(stop, self.cfg.stop_offset_frames)

        # --- 5. Base Mode Classification (0=STOP, 1=FORWARD, 2=ROTATE) ---
        mode, _, mode_stats = AriaPhasesOps.compute_mode_from_stop_vw(stop, v, w, yaw_u, self.cfg.w_rot_thresh, self.cfg.v_rot_max, self.cfg)

        # --- 6. Hand-Kinematics Refinement for Manipulation Phases ---
        if self.aria_hands is not None:
            vel_th = getattr(self.cfg, "manip_clean_vel_thresh", 0.15)
            wait_frames = getattr(self.cfg, "manip_clean_wait_frames", 5)
            manual_offset = getattr(self.cfg, "manip_clean_manual_offset", 15)
            mode, mode_stats = AriaPhasesOps.refine_manip_phases_with_hand_vel(mode, self.aria_hands, vel_th, wait_frames, manual_offset)

        # --- 7. Inject FINISHED Phase for Task Termination ---
        # Marks the last N of the final hold as 'FINISHED'
        n_finish_frames = getattr(self.cfg, "finished_frames", 60)
        mode = AriaPhasesOps.inject_finished_phase(mode, n_finish_frames)
        
        # --- 8. Construct Sequence Container ---
        phases_obj = AriaPhases(mps_path=self.mps_path)
        for i in range(n):
            mv = int(mode[i])
            ms_map = {0: "STOP", 1: "FORWARD", 2: "ROTATE", 3: "TRANSITION", 4: "FINISHED"}
            ms = ms_map.get(mv, "UNKNOWN")
            
            phases_obj.frames.append(AriaPhasesFrame(
                idx=self.aria_slam.frames[i].idx, 
                ts=tss[i], 
                mode=mv,  
                stop=int(np.ravel(stop)[i]),
                mode_str=ms, 
                v=float(v[i]), 
                w=float(w[i]), 
                yaw_u_deg=float(yaw_u[i])
            ))

        # --- 9. Compile Summary Statistics ---
        duration_s = (tss[-1] - tss[0]) * 1e-9 if n > 1 else 0
        
        # Determine hyperparams dict safely (handle ConfigBox or Namespace)
        if hasattr(self.cfg, '__dict__'):
            hp_dict = vars(self.cfg)
        elif hasattr(self.cfg, 'keys'):
            hp_dict = dict(self.cfg)
        else:
            hp_dict = {}

        phases_obj.summary = {
            "total_frames": n, 
            "fps_median": self.aria_cam.fps, 
            "duration_s": duration_s,
            "segment_summary_opt": AriaPhasesOps.segment_summary_from_masks(stop, v, w, self.cfg.v_walk_thresh),
            "stage_window_check": AriaPhasesOps.stage_window_check(mode, [4, 3, 2, 1, 0], True, False),
            "mode_stats_opt": mode_stats,
            "hyperparams": hp_dict
        }
        return phases_obj


    def draw_aria_phases_panel(self, img: np.ndarray, idx: int, aria_phases: AriaPhases) -> np.ndarray:
        """
        Renders the Phase HUD overlay dynamically onto the image.

        Args:
            img (np.ndarray): The input image frame.
            idx (int): Current frame index.
            aria_phases (AriaPhases): The populated phase sequence container.

        Returns:
            np.ndarray: The image with the HUD panel drawn in-place.
        """
        fps = self.aria_cam.fps
        n_frames = len(self.aria_cam)
        mode_str = aria_phases.frames[idx].mode_str
        return AriaPhasesOps.draw_aria_phases_panel(img, idx, fps, n_frames, mode_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/AriaPhases.yaml", help="Path to AriaPhases.yaml")
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


    # AriaHands
    aria_hands_generator = AriaHandsGenerator(args.mps_path, os.path.join(os.path.dirname(args.cfg_path), "AriaHands.yaml"), aria_hands_mps, aria_cam_rgb)
    aria_hands = aria_hands_generator.get_aria_hands()
    aria_hands.save_aria_hands_json()

    # AriaSlam
    aria_slam_generator = AriaSlamGenerator(args.mps_path, os.path.join(os.path.dirname(args.cfg_path), "AriaSlam.yaml"), aria_cam_rgb)
    aria_slam = aria_slam_generator.get_aria_slam()
    aria_slam.save_aria_slam_json()

    # AriaPhases
    aria_phases_generator = AriaPhasesGenerator(args.mps_path, args.cfg_path, aria_cam_rgb, aria_slam, aria_hands)
    aria_phases = aria_phases_generator.get_aria_phases()
    aria_phases.save_aria_phases_json()
    
    # Visualization
    if args.export_video:
        frames_all =[]
        for idx, ts in enumerate(tqdm(aria_cam_rgb.tss, desc="Visualizing")):
            img = aria_cam_rgb.cam[idx].img.copy()

            # Overlay: Hands Skeleton
            img = aria_hands_generator.draw_aria_hands_skeleton(img, aria_hands.hands[idx], aria_cam_rgb.cam[idx].k, aria_cam_rgb.cam[idx].d, aria_cam_rgb.cam[idx].c2w)

            # Overlay: Hands HUD Panel
            img = aria_hands_generator.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])
            
            # Overlay: Slam 3D Trajectory Projection
            img = aria_slam_generator.draw_future_traj_on_image(img, idx, aria_slam)

            # Overlay: Slam HUD Panel
            img = aria_slam_generator.draw_aria_slam_panel(img, aria_slam.frames[idx], aria_slam)
            
            # Overlay: Phases HUD Panel
            img = aria_phases_generator.draw_aria_phases_panel(img, idx, aria_phases)

            frames_all.append(img)

        create_video_from_frames(frames_all, os.path.join(args.mps_path, "preprocess", "vis", "aria_phases_vis.mp4"), aria_cam_rgb.fps, args.export_gif)

# Execution Example:
# python -m preprocess.AriaPhases --mps_path  "./data/test/test_0/mps_test_0_000_vrs/"