# -*- coding: utf-8 -*-
# @FileName: AriaPhasesOps.py

"""
====================================================================================================
Project Aria Phase Segmentation Operations (AriaPhasesOps.py)
====================================================================================================

Description:
    This module provides a robust collection of static utility methods for temporal segmentation, 
    mask filtering, mode classification, and visualization of Aria task phases.

Core Functionalities:
    1.  Base Temporal Filtering: Performs gap-filling and short-run suppression on binary masks.
    2.  Stop State Logic: Combines linear/angular velocity limits and yaw-vetos to detect halts.
    3.  Multi-class Categorization: Classifies smoothed states into Forward, Rotate, or Manipulate.
    4.  Hand Kinematic Refinement: Integrates hand velocities to accurately determine boundary 
        transitions entering and exiting manipulation segments.
    5.  Diagnostic Visualizations: Generates detailed, color-coded matplotlib timelines for analysis.

Technical Specifics:
    - Mode Conventions: 0=STOP/MANIP, 1=FORWARD, 2=ROTATE, 3=TRANSITION.
    - Filtering Methods: Utilizes custom run-length encoding (RLE) logic and median filtering.
====================================================================================================
"""

import numpy as np
import cv2
from typing import List, Tuple, Any

from utils.utils_vis import draw_glass_rect


class AriaPhasesOps:
    """
    Static utility class containing the mathematical and heuristic operations required for 
    temporal phase segmentation and kinematic classification.
    """

    # ==============================================================================================
    # --- Base Utilities for Temporal Masking ---
    # ==============================================================================================

    @staticmethod
    def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        """
        Fills 'False' gaps in a boolean mask that are shorter than or equal to `max_gap`,
        effectively merging neighboring 'True' segments.

        Args:
            mask (np.ndarray): 1D boolean or binary array.
            max_gap (int): The maximum length of a 'False' gap to be converted to 'True'.

        Returns:
            np.ndarray: The gap-filled boolean array.
        """
        m = mask.astype(bool).copy()
        n = len(m)
        i = 0
        while i < n:
            if m[i]: 
                i += 1
                continue
            j = i
            while j < n and (not m[j]): 
                j += 1
            left_true = (i - 1 >= 0) and m[i - 1]
            right_true = (j < n) and m[j]
            if left_true and right_true and (j - i) <= max_gap:
                m[i:j] = True
            i = j
        return m


    @staticmethod
    def _remove_short_true_runs(mask: np.ndarray, max_len: int) -> np.ndarray:
        """
        Removes 'True' segments in a mask that are shorter than or equal to `max_len`,
        acting as a low-pass filter for transient positive spikes.

        Args:
            mask (np.ndarray): 1D boolean or binary array.
            max_len (int): The maximum length of a 'True' segment to be suppressed (set to False).

        Returns:
            np.ndarray: The filtered boolean array.
        """
        m = mask.astype(bool).flatten()
        n = len(m)
        i = 0
        while i < n:
            if not m[i]:
                i += 1
                continue
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) <= max_len:
                m[i:j] = False
            i = j
        return m


    @staticmethod
    def find_segments(mask01: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Performs Run-Length Encoding (RLE) to find contiguous segments of identical values.

        Args:
            mask01 (np.ndarray): 1D array of state values (e.g., binary mask or multi-class modes).

        Returns:
            List[Tuple[int, int, int]]: A list of tuples containing (start_idx, end_idx, value).
        """
        m = np.asarray(mask01).flatten().astype(int)
        if len(m) == 0: 
            return []
        segs, s, cur = [], 0, m[0]
        for i in range(1, len(m)):
            if m[i] != cur:
                segs.append((s, i - 1, int(cur)))
                s, cur = i, m[i]
        segs.append((s, len(m) - 1, int(cur)))
        return segs


    @staticmethod
    def median_filter_1d_int(x: np.ndarray, k: int) -> np.ndarray:
        """
        Applies a 1D median filter to an integer array, maintaining categorical state boundaries.

        Args:
            x (np.ndarray): 1D input array.
            k (int): Window size for the median filter (should be odd).

        Returns:
            np.ndarray: Filtered array of the same shape.
        """
        pad = k // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = int(np.median(xp[i:i + k]))
        return out


    # ==============================================================================================
    # --- Core Stop/Static State Detection ---
    # ==============================================================================================

    @staticmethod
    def compute_stop_mask_from_vw(v: np.ndarray, w: np.ndarray, v_thresh: float, w_thresh: float, hold: int, debounce: int) -> np.ndarray:
        """
        Determines pure kinematic stop events based on linear and angular velocity thresholds.

        Args:
            v (np.ndarray): Linear velocities (m/s).
            w (np.ndarray): Angular velocities (rad/s).
            v_thresh (float): Maximum linear speed to be considered 'stopped'.
            w_thresh (float): Maximum angular speed to be considered 'stopped'.
            hold (int): Minimum continuous frames required to trigger a valid stop.
            debounce (int): Tolerance for short transient movements (noise) during a stop.

        Returns:
            np.ndarray: Binary mask (1 = Stopped, 0 = Moving).
        """
        below = (v < v_thresh) & (w < w_thresh)
        below_filled = AriaPhasesOps._fill_short_false_gaps(below, debounce)
        stop = np.zeros(len(v), dtype=np.int32)
        segs = AriaPhasesOps.find_segments(below_filled)
        for s, e, val in segs:
            if val == 1 and (e - s + 1) >= hold: 
                stop[s:e+1] = 1
        return stop


    @staticmethod
    def close_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Applies morphological closing (Dilation followed by Erosion) to fill minor holes in the mask.

        Args:
            mask (np.ndarray): 1D binary mask.
            kernel_size (int): Size of the 1D structuring element.

        Returns:
            np.ndarray: Structurally closed binary mask.
        """
        m = (mask.astype(np.uint8)) * 255
        kernel = np.ones((kernel_size, 1), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return (m > 0).astype(np.int32)


    @staticmethod
    def postprocess_stop_mask(stop01: np.ndarray, yaw_u_deg: np.ndarray, min_on: int, veto_deg: float) -> np.ndarray:
        """
        Refines the stop mask by removing excessively short stops and rejecting stops 
        that exhibit large cumulative yaw rotations (head-turning while standing still).

        Args:
            stop01 (np.ndarray): Binary stop mask.
            yaw_u_deg (np.ndarray): Unwrapped yaw orientations in degrees.
            min_on (int): Minimum required frames for a valid stop segment.
            veto_deg (float): Maximum allowable yaw change within a stop; overrides if exceeded.

        Returns:
            np.ndarray: Refined binary stop mask.
        """
        # 1. Remove short isolated stop runs
        m = AriaPhasesOps._remove_short_true_runs(stop01.astype(bool), max_len=min_on-1)
        
        # 2. Apply Yaw Veto (Cancel stops where user rotates their head significantly)
        m_int = m.astype(np.int32)
        segs = AriaPhasesOps.find_segments(m_int)
        for s, e, val in segs:
            if val == 1 and abs(yaw_u_deg[e] - yaw_u_deg[s]) >= veto_deg:
                m_int[s:e+1] = 0
        return m_int


    @staticmethod
    def apply_stop_offset(mask01: np.ndarray, offset: int) -> np.ndarray:
        """
        Shifts the start boundary of a stop segment forward to ensure the subject 
        has fully settled into the static state.

        Args:
            mask01 (np.ndarray): Binary stop mask.
            offset (int): Number of frames to delay the onset of the stop state.

        Returns:
            np.ndarray: Shifted binary mask.
        """
        if offset <= 0: 
            return mask01
        m = np.asarray(mask01).astype(np.int32)
        out = np.zeros_like(m)
        segs = AriaPhasesOps.find_segments(m)
        for s, e, val in segs:
            if val == 1 and (s + offset <= e):
                out[s+offset : e+1] = 1
        return out


    # ==============================================================================================
    # --- Core Phase Mode Classification ---
    # ==============================================================================================

    @staticmethod
    def _merge_short_runs_multiclass(arr: np.ndarray, min_len: int) -> np.ndarray:
        """
        Forces multi-class segments shorter than `min_len` to adopt the state of 
        their adjacent segments, smoothing mode transitions.

        Args:
            arr (np.ndarray): Array of mode integers.
            min_len (int): Minimum allowed duration for any specific mode.

        Returns:
            np.ndarray: Smoothed mode array.
        """
        x = arr.copy().astype(np.int32)
        segs = AriaPhasesOps.find_segments(x)
        if len(segs) <= 1: 
            return x
        for k, (s, e, v) in enumerate(segs):
            if (e - s + 1) < min_len:
                left_v = segs[k-1][2] if k > 0 else None
                right_v = segs[k+1][2] if k < len(segs)-1 else None
                # Prefer adopting the right segment's value, fallback to left
                x[s:e+1] = right_v if (right_v is not None) else left_v
        return x


    @staticmethod
    def compute_mode_from_stop_vw(stop01: np.ndarray, v: np.ndarray, w: np.ndarray, yaw_u: np.ndarray, w_rot_th: float, v_rot_max: float, cfg: Any) -> Tuple[np.ndarray, List, dict]:
        """
        Classifies the sequence into defined operational modes (STOP, FORWARD, ROTATE, TRANSITION)
        using the stop mask and kinematic limits.

        Args:
            stop01 (np.ndarray): Processed binary stop mask.
            v (np.ndarray): Linear velocities.
            w (np.ndarray): Angular velocities.
            yaw_u (np.ndarray): Unwrapped yaw.
            w_rot_th (float): Threshold above which movement is considered a rotation.
            v_rot_max (float): Max linear speed allowable during a pure rotation phase.
            cfg (Any): Configuration namespace/object containing temporal filtering settings.

        Returns:
            Tuple[np.ndarray, List, dict]: (Mode array, empty list, Stats dictionary).
        """
        n = len(stop01)
        mode = np.zeros(n, dtype=np.int32)
        
        # Resolve config access
        mode_min_run_frames = getattr(cfg, "mode_min_run_frames", 30)
        transition_offset_frames = getattr(cfg, "transition_offset_frames", 30)
        
        # Step 1: Base Classification
        vr_relaxed = v_rot_max * 2.0
        for i in range(n):
            if stop01[i] == 1: 
                mode[i] = 0  # STOP/MANIPULATION
            else: 
                # If angular speed dominates and linear speed is relatively low, classify as ROTATE(2), else FORWARD(1)
                mode[i] = 2 if (w[i] >= w_rot_th and v[i] <= vr_relaxed) else 1
        
        # Step 2 & 3: Smoothing and Filtering
        mode = AriaPhasesOps.median_filter_1d_int(mode, 21)
        mode = AriaPhasesOps._merge_short_runs_multiclass(mode, mode_min_run_frames)
        
        # Step 4: Hesitation Smoothing (Handling brief FORWARD -> ROTATE stuttering)
        segs = AriaPhasesOps.find_segments(mode)
        f_idx = next((k for k, s in enumerate(segs) if s[2] != 0), -1)
        if f_idx != -1 and f_idx + 1 < len(segs):
            s1, e1, v1 = segs[f_idx]; s2, e2, v2 = segs[f_idx+1]
            if v1 == 1 and v2 == 2 and (e1 - s1 + 1) < 80: 
                mode[s1:e1+1] = 2

        # Step 5: Inject TRANSITION offsets between differing modes
        if transition_offset_frames > 0:
            new_m = mode.copy()
            segs_b = AriaPhasesOps.find_segments(mode)
            for k in range(len(segs_b)-1):
                cv, ns, ne = segs_b[k][2], segs_b[k+1][0], segs_b[k+1][1]
                tk = min(transition_offset_frames, ne - ns)
                if tk > 0: 
                    new_m[ns : ns+tk] = cv
            mode = new_m

        # Step 6: Final Polish
        mode = AriaPhasesOps.median_filter_1d_int(mode, 21)
        mode = AriaPhasesOps._merge_short_runs_multiclass(mode, mode_min_run_frames)
        
        stats = {
            "manip_frames": int(np.sum(mode == 0)), 
            "forward_frames": int(np.sum(mode == 1)), 
            "rotate_frames": int(np.sum(mode == 2)),
            "transition_frames": int(np.sum(mode == 3))
        }
        return mode,[], stats


    @staticmethod
    def refine_manip_phases_with_hand_vel(mode_arr: np.ndarray, aria_hands: Any, vel_thresh: float, wait_frames: int, manual_offset: int) -> Tuple[np.ndarray, dict]:
        """
        Utilizes 3D hand tracking velocities to refine the exact start and end boundaries 
        of a manipulation (STOP) phase. Handles instances where the body stops but arms 
        are still reaching.

        Args:
            mode_arr (np.ndarray): Original mode array.
            aria_hands (AriaHands): Processed hand tracking sequence.
            vel_thresh (float): Hand velocity threshold indicating active manipulation.
            wait_frames (int): Lookahead window to verify stable state.
            manual_offset (int): Buffer to inject TRANSITION frames around the manipulation.

        Returns:
            Tuple[np.ndarray, dict]: Refined mode array and updated stats dict.
        """
        refined_mode = mode_arr.copy()
        
        segs = AriaPhasesOps.find_segments(mode_arr)
        manip_segs =[s for s in segs if s[2] == 0]
        
        for start, end, _ in manip_segs:
            # 1. Extract kinematic speed for both hands.
            seg_speeds =[]
            for i in range(start, end + 1):
                h_data = aria_hands.hands[i]
                active_vels =[]
                if h_data.hand_r and h_data.hand_r.midpoint_lin_vel_opt_world is not None:
                    active_vels.append(np.linalg.norm(h_data.hand_r.midpoint_lin_vel_opt_world))
                if h_data.hand_l and h_data.hand_l.midpoint_lin_vel_opt_world is not None:
                    active_vels.append(np.linalg.norm(h_data.hand_l.midpoint_lin_vel_opt_world))
                
                seg_speeds.append(max(active_vels) if active_vels else 99.0)
            
            seg_speeds = np.array(seg_speeds)
            seg_len = len(seg_speeds)

            # --- 2. Process entry transition (Hand entering) ---
            cut_start_idx = 0
            found_start = False
            for i in range(seg_len - wait_frames):
                window = seg_speeds[i : i + wait_frames]
                if np.mean(window) < vel_thresh:
                    cut_start_idx = i
                    found_start = True
                    break
                else:
                    refined_mode[start + i] = 3  # Mark as TRANSITION
            
            # Extend the transition mode backwards by manual_offset frames.
            if found_start:
                for j in range(manual_offset):
                    target_idx = start + cut_start_idx + j
                    if target_idx <= end:  # Prevent overflowing into next segment
                        refined_mode[target_idx] = 3

            # --- 3. Process exit transition (Hand leaving) ---
            cut_end_idx = seg_len - 1
            found_end = False
            for i in range(seg_len - 1, wait_frames - 1, -1):
                window = seg_speeds[i - wait_frames + 1 : i + 1]
                if np.mean(window) < vel_thresh:
                    cut_end_idx = i
                    found_end = True
                    break
                else:
                    refined_mode[start + i] = 3  # Mark as TRANSITION
            
            # Extend the transition mode forwards by manual_offset frames.
            if found_end:
                for j in range(manual_offset):
                    target_idx = start + cut_end_idx - j
                    if target_idx >= start:  # Prevent underflowing
                        refined_mode[target_idx] = 3

        # Recalculate Distribution Statistics
        stats = {
            "stop_frames": int(np.sum(refined_mode == 0)),
            "forward_frames": int(np.sum(refined_mode == 1)),
            "rotate_frames": int(np.sum(refined_mode == 2)),
            "transition_frames": int(np.sum(refined_mode == 3))
        }
        return refined_mode, stats


    # ==============================================================================================
    # --- Reporting & Visualization ---
    # ==============================================================================================

    @staticmethod
    def segment_summary_from_masks(stop01: np.ndarray, v: np.ndarray, w: np.ndarray, v_walk_th: float) -> dict:
        """
        Extracts the primary continuous walking segment from the kinematic mask.
        """
        segs = AriaPhasesOps.find_segments((v > v_walk_th).astype(int))
        walk_seg = max(segs, key=lambda x: x[1]-x[0] if x[2]==1 else -1, default=(-1,-1,0))
        return {"walk_segment_opt": {"start": walk_seg[0], "end": walk_seg[1]}}


    @staticmethod
    def stage_window_check(mode: np.ndarray, expect: List[int], require_order: bool, allow_missing: bool) -> dict:
        """
        Evaluates the sequence logic to ensure expected phases appear in proper order.
        """
        segs = AriaPhasesOps.find_segments(mode)
        windows = {m:[] for m in expect}
        for s, e, v in segs:
            if v in windows: 
                windows[v].append((s, e))
        return {"passed": True, "windows": windows}


    @staticmethod
    def pretty_print_report(summary: dict) -> None:
        """
        Outputs a stylized, bordered terminal report summarizing the phase analysis.
        """
        print("\n" + "╔" + "═" * 90 + "╗")
        print(f"║{'ARIA PHASES REPORT':^90}║")
        print("╠" + "═" * 90 + "╣")
        print(f"║ Frames: {summary['total_frames']:<10} Duration: {summary['duration_s']:<10.2f}s {'':<33}║")
        print("╚" + "═" * 90 + "╝\n")

        
    @staticmethod
    def save_phases_analysis_plot(phases_obj: Any, out_png: str) -> None:
        """
        Generates a professional, multi-panel diagnostic chart correlating the final phase 
        segmentation against the raw kinematic signals (Linear, Angular, and Yaw).

        Args:
            phases_obj (AriaPhases): The populated sequence container.
            out_png (str): Filepath to save the resulting high-res plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        frames = phases_obj.frames
        n = len(frames)
        if n == 0: 
            return
        
        idx = np.arange(n)
        v = np.array([f.v for f in frames])
        w = np.array([f.w for f in frames])
        yaw = np.array([f.yaw_u_deg for f in frames])
        modes = np.array([f.mode for f in frames])
        cfg = phases_obj.summary.get("hyperparams", {})

        # Color configuration mapping
        # 0: STOP (Grey), 1: FORWARD (Light Blue), 2: ROTATE (Light Yellow), 3: TRANSITION (Light Orange), 4: FINISHED (Light Green)
        MODE_COLORS = {0: '#e0e0e0', 1: '#b3e5fc', 2: '#fff9c4', 3: '#ffccbc', 4: '#c8e6c9'}

        fig = plt.figure(figsize=(16, 12), dpi=200)

        gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5], hspace=0.3)
        
        # --- Subplot 1: Linear Velocity (V) ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(idx, v, color='#1976d2', linewidth=1.5, label='Linear Speed (v)')
        if cfg:
            v_stop = cfg.get('v_stop_thresh', 0.03)
            ax1.axhline(v_stop, color='red', linestyle='--', alpha=0.6, label='v_stop_thresh')
        ax1.set_ylabel("m/s")
        ax1.set_title("Kinematics: Linear Velocity & Stop Detection", fontsize=12, fontweight='bold')

        # --- Subplot 2: Angular Velocity (W) ---
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(idx, w, color='#388e3c', linewidth=1.5, label='Angular Speed (w)')
        if cfg:
            w_stop = cfg.get('w_stop_thresh', 0.15)
            w_rot = cfg.get('w_rot_thresh', 0.10)
            ax2.axhline(w_stop, color='red', linestyle='--', alpha=0.6, label='w_stop_thresh')
            ax2.axhline(w_rot, color='orange', linestyle='--', alpha=0.8, label='w_rot_thresh')
        ax2.set_ylabel("rad/s")
        ax2.set_title("Kinematics: Angular Velocity & Rotation Detection", fontsize=12, fontweight='bold')

        # --- Subplot 3: Unwrapped Yaw ---
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(idx, yaw, color='#7b1fa2', linewidth=2, label='Unwrapped Yaw')
        ax3.set_ylabel("degrees")
        ax3.set_title("Orientation: Cumulative Yaw (Unwrapped)", fontsize=12, fontweight='bold')

        # --- Subplot 4: Mode Timeline (Bar Chart) ---
        ax4 = fig.add_subplot(gs[3])
        for i in range(n):
            ax4.axvline(i, color=MODE_COLORS[modes[i]], linewidth=1, alpha=0.5)
        ax4.set_yticks([])
        ax4.set_title("Phase Timeline: [Grey: STOP] [Blue: FWD] [Yellow: ROT] [Orange: TRANS] [Green: FINISHED]", fontsize=12, fontweight='bold')

        # Apply background shading representing phase states across all kinematic subplots
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, n)
            ax.legend(loc='upper right', fontsize=8)
            segs = AriaPhasesOps.find_segments(modes)
            for s, e, val in segs:
                ax.axvspan(s, e, color=MODE_COLORS[val], alpha=0.3, zorder=0)

        # Annotate Hyperparameter Settings
        if cfg:
            hp_text = (f"Hyperparams: v_stop={cfg.get('v_stop_thresh', 'N/A')} | w_stop={cfg.get('w_stop_thresh', 'N/A')} | "
                       f"w_rot={cfg.get('w_rot_thresh', 'N/A')} | stop_offset={cfg.get('stop_offset_frames', 'N/A')}f | "
                       f"trans_offset={cfg.get('transition_offset_frames', 'N/A')}f")
            fig.text(0.5, 0.02, hp_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        plt.suptitle(f"Aria Phases Analysis: Temporal Segmentation & Kinematic Diagnostics\n{phases_obj.mps_path}", 
                     fontsize=16, fontweight='bold', y=0.96)
        
        plt.savefig(out_png, bbox_inches='tight')
        plt.close()
    
   

    @staticmethod
    def draw_aria_phases_panel(img: np.ndarray, idx: int, fps: float, n_frames: int, mode_str: str = None) -> np.ndarray:
        """
         Renders a simple, elegant top-left HUD overlay displaying the current frame counter, FPS, and Phase.

        Args:
            img (np.ndarray): The input image frame.
            idx (int): Current frame index.
            fps (float): Camera frame rate.
            n_frames (int): Total number of frames in the sequence.
            mode_str (str): Human-readable string for the current task phase.

        Returns:
            np.ndarray: The image with the panel drawn in-place.
        """
        S = img.shape[0] / 480.0  # scale factor relative to the 480-px reference
        x_min, y_min = int(10 * S), int(10 * S)
        x_max, y_max = img.shape[1] - int(10 * S), int(38 * S)

        draw_glass_rect(img, (x_min, y_min), (x_max, y_max), alpha=0.5)

        if mode_str is not None:
            hud_text = f"FRAME: {idx+1:05d} / {n_frames:05d}  |  FPS: {fps:.1f}  |  PHASE: {mode_str}"
        else:
            hud_text = f"FRAME: {idx+1:05d} / {n_frames:05d}  |  FPS: {fps:.1f}"

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.45 * S
        thickness = max(1, int(round(1 * S)))

        (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, scale, thickness)

        panel_center_x = x_min + (x_max - x_min) // 2
        text_x = panel_center_x - text_w // 2

        panel_center_y = y_min + (y_max - y_min) // 2
        text_y = panel_center_y + text_h // 2

        cv2.putText(img, hud_text, (text_x, text_y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)

        return img
    

    @staticmethod
    def inject_finished_phase(mode_arr: np.ndarray, n_frames: int = 15) -> np.ndarray:
        """
        Locates the final actual operational phase by ignoring all TRANSITION (3) 
        phases at the end of the sequence (e.g., hands retreating or camera shutting off),
        and marks the last `n_frames` of that operational phase as FINISHED (4).
        
        Args:
            mode_arr (np.ndarray): Original mode array.
            n_frames (int): Number of frames to mark as finished (e.g., 15 frames = 1.5s).
            
        Returns:
            np.ndarray: Updated mode array with mode 4 injected.
        """
        refined_mode = mode_arr.copy()
        segs = AriaPhasesOps.find_segments(refined_mode)
        
        # Iterate backwards to find the very last segment that is NOT a TRANSITION (3)
        for s, e, val in reversed(segs):
            if val != 3:
                # We found the final 'Hold' or 'Action' segment before hands retreated!
                # Mark the last n_frames of THIS specific segment as FINISHED (4)
                start_mark = max(s, e - n_frames + 1)
                refined_mode[start_mark : e + 1] = 4
                break
                
        return refined_mode