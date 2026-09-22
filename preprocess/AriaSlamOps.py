# -*- coding: utf-8 -*-
# @FileName: AriaSlamOps.py

"""
====================================================================================================
Project Aria SLAM Visualization and Diagnostic Operations (AriaSlamOps.py)
====================================================================================================

Description:
    This module provides visualization utilities for Project Aria SLAM data. It includes 
    real-time Head-Up Display (HUD) overlays for image frames, 3D trajectory projection 
    using "future-path" logic, and high-quality Matplotlib-based diagnostic reporting 
    for kinematic evaluation.

Core Functionalities:
    1.  HUD Rendering (Panels): Draws glass-morphism style panels for Speed (Linear/Angular), 
        Position (Radar-style), and Orientation (Compass-style).
    2.  Trajectory Projection: Projects the 3D future path of the device onto the 2D image 
        plane with ground-offset and Jet-colormap temporal encoding.
    3.  Diagnostic Plotting: Generates a 4x3 professional analysis report covering 
        XYZ translations, RPY rotations, speeds, and trajectory distributions.

Technical Specifics:
    - Projection: Uses the standard pinhole model with distortion coefficients.
    - Coordinate Space: Visualizes data in the MPS World 'Closed Loop' frame.
    - Aesthetics: Implements adaptive scaling for different image resolutions.

Generated Outputs:
    - Real-time video overlays (processed via AriaSlam.py).
    - [mps_path]/aria/aria_slam_analysis.png (Multi-panel kinematic report).
====================================================================================================
"""

import math
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from preprocess.AriaSlamTypes import AriaSlam, AriaSlamFrame
from utils.utils_vis import draw_glass_rect, draw_arc_gauge


class AriaSlamOps:
    """
    Diagnostic and visualization operations for Aria SLAM trajectories and kinematics.
    """


    @staticmethod
    def draw_aria_slam_panel(img: np.ndarray, frame: AriaSlamFrame, aria_slam: AriaSlam, sensitivity_range: float) -> np.ndarray:
        """
        Renders the upper HUD panels for real-time SLAM parameter tracking.

        Layout:
            - Left Panel: Arc gauges for Linear Velocity (V) and Angular Velocity (W).
            - Right Panel: XY Mini-Radar for position and a Compass for Yaw orientation.

        Args:
            img (np.ndarray): The input BGR image.
            frame (AriaSlamFrame): Data for the current time step.
            aria_slam (AriaSlam): Full SLAM sequence for scale normalization.

        Returns:
            np.ndarray: Image with HUD panels overlaid.
        """
        # --- 0. Adaptive Scaling Calculation ---
        # Reference width 840 → sc=1.52 at 1280x960. This is the layout the
        # panel was designed for: leaves a gap between this top-area panel
        # and the bottom-area hand panels, so they don't collide.
        img_h, img_w = img.shape[:2]
        sc = img_w / 840.0

        def S(val): return int(val * sc)
        def F(val): return max(0.3, val * sc)
        def T(val): return max(1, int(val * sc))

        # Statistical Max for Gauge Calibration (95th percentile)
        v_all = [f.v for f in aria_slam.frames]
        w_all = [f.w for f in aria_slam.frames]
        max_v = float(np.percentile(v_all, 95)) if v_all else 1.0
        max_w = float(np.percentile(w_all, 95)) if w_all else 1.0
        max_v, max_w = max(0.5, max_v), max(0.5, max_w)

        # Panel Dimensions (Aligned with Hand Panel Width)
        panel_w = S(200)
        panel_h = S(280) 
        margin_x = S(10)
        margin_y = S(60)

        x_left = margin_x
        x_right = img_w - panel_w - margin_x

        LABEL_COLOR = (200, 200, 200)
        VAL_COLOR = (0, 255, 255)
        DELTA_COLOR = (255, 150, 50)
        
        def get_center_x(text, font_scale, thickness, base_x, box_width):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            return base_x + (box_width - tw) // 2

        # ========================================================
        # [LEFT PANEL] V & W (Kinematic Speeds)
        # ========================================================
        draw_glass_rect(img, (x_left, margin_y), (x_left + panel_w, margin_y + panel_h), alpha=0.62)
        cx_left = x_left + panel_w // 2

        # --- 1. Linear Velocity V (Top) ---
        draw_arc_gauge(img, (cx_left, margin_y + S(50)), frame.v, max_v, "V")
        y_text = margin_y + S(110)
        t_lbl = "LINEAR SPEED"
        cv2.putText(img, t_lbl, (get_center_x(t_lbl, F(0.4), T(1), x_left, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.4), LABEL_COLOR, T(1), cv2.LINE_AA)
        y_text += S(25)
        t_val = f"{frame.v:>4.2f} m/s"
        cv2.putText(img, t_val, (get_center_x(t_val, F(0.5), T(1), x_left, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.5), VAL_COLOR, T(1), cv2.LINE_AA)

        # --- 2. Angular Velocity W (Bottom) ---
        y_gauge2 = margin_y + S(185)
        draw_arc_gauge(img, (cx_left, y_gauge2), frame.w, max_w, "W")
        y_text = y_gauge2 + S(60)
        t_lbl = "ANGULAR SPEED"
        cv2.putText(img, t_lbl, (get_center_x(t_lbl, F(0.4), T(1), x_left, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.4), LABEL_COLOR, T(1), cv2.LINE_AA)
        y_text += S(25)
        t_val = f"{frame.w:>4.2f} r/s"
        cv2.putText(img, t_val, (get_center_x(t_val, F(0.5), T(1), x_left, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.5), VAL_COLOR, T(1), cv2.LINE_AA)


        # ========================================================
        # [RIGHT PANEL] Position & Orientation (Spatial Tracking)
        # ========================================================
        draw_glass_rect(img, (x_right, margin_y), (x_right + panel_w, margin_y + panel_h), alpha=0.62)
        cx_right = x_right + panel_w // 2

        # --- 3. XY Mini Radar (Top) ---
        r_size = S(60) 
        bx1, bx2 = cx_right - r_size // 2, cx_right + r_size // 2
        by1, by2 = margin_y + S(20), margin_y + S(20) + r_size
        
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (80, 80, 80), T(1), cv2.LINE_AA)
        cv2.line(img, (cx_right, by1), (cx_right, by2), (55, 55, 55), T(1))
        cy_radar = (by1 + by2) // 2
        cv2.line(img, (bx1, cy_radar), (bx2, cy_radar), (55, 55, 55), T(1))
        
        # Coordinate Mapping to Radar UI
        px = int(cx_right + (frame.delta_t[0] / sensitivity_range) * (r_size // 2))
        py = int(cy_radar - (frame.delta_t[1] / sensitivity_range) * (r_size // 2))
        px, py = np.clip(px, bx1+S(2), bx2-S(2)), np.clip(py, by1+S(2), by2-S(2))
        cv2.circle(img, (px, py), S(4), VAL_COLOR, -1, cv2.LINE_AA)

        y_text = by2 + S(25)
        t_lbl = "POSITION (X, Y)"
        cv2.putText(img, t_lbl, (get_center_x(t_lbl, F(0.38), T(1), x_right, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.38), LABEL_COLOR, T(1), cv2.LINE_AA)
        y_text += S(20)
        t_val = f"Abs: {frame.t_world[0]:>5.2f}, {frame.t_world[1]:>5.2f}m"
        cv2.putText(img, t_val, (get_center_x(t_val, F(0.35), T(1), x_right, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.35), VAL_COLOR, T(1), cv2.LINE_AA)
        y_text += S(18)
        t_dlt = f"Dlt: {frame.delta_t[0]:>5.2f}, {frame.delta_t[1]:>5.2f}m"
        cv2.putText(img, t_dlt, (get_center_x(t_dlt, F(0.35), T(1), x_right, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.35), DELTA_COLOR, T(1), cv2.LINE_AA)

        # --- 4. Compass (Bottom) ---
        c_cy, c_r = margin_y + S(185), S(28)
        cv2.circle(img, (cx_right, c_cy), c_r, (80, 80, 80), T(1), cv2.LINE_AA)
        
        # Draw Yaw Heading Needle
        angle_rad = math.radians(frame.delta_rpy_deg[2]) - math.pi/2
        tx = int(cx_right + c_r * math.cos(angle_rad))
        ty = int(c_cy + c_r * math.sin(angle_rad))
        cv2.line(img, (cx_right, c_cy), (tx, ty), DELTA_COLOR, T(2), cv2.LINE_AA)
        
        y_text = c_cy + c_r + S(20)
        t_lbl = "ORIENTATION (YAW)"
        cv2.putText(img, t_lbl, (get_center_x(t_lbl, F(0.38), T(1), x_right, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.38), LABEL_COLOR, T(1), cv2.LINE_AA)
        y_text += S(20)
        t_val = f"Abs: {frame.rpy_deg[2]:>5.1f} deg"
        cv2.putText(img, t_val, (get_center_x(t_val, F(0.35), T(1), x_right, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.35), VAL_COLOR, T(1), cv2.LINE_AA)
        y_text += S(18)
        t_dlt = f"Dlt: {frame.delta_rpy_deg[2]:>+5.1f} deg"
        cv2.putText(img, t_dlt, (get_center_x(t_dlt, F(0.35), T(1), x_right, panel_w), y_text), 
                    cv2.FONT_HERSHEY_DUPLEX, F(0.35), DELTA_COLOR, T(1), cv2.LINE_AA)

        return img
    

    @staticmethod
    def draw_future_traj_on_image(img: np.ndarray, idx: int, slam: AriaSlam, traj_future_len: int, traj_step: int, ground_offset: float) -> np.ndarray:
        """
        Projects and draws the future path of the device onto the camera frame.

        Args:
            img (np.ndarray): Current image frame.
            idx (int): Current frame index.
            slam (AriaSlam): Container for the full trajectory sequence.

        Returns:
            np.ndarray: Image with colored trajectory dots.
        """
        if idx >= len(slam.frames): 
            return img
            
        f_curr = slam.frames[idx]
        T_w2c = np.linalg.inv(f_curr.c2w)
        
        # Collect sample points from the future trajectory
        pts_world = []
        for i in range(idx, min(len(slam.frames), idx + traj_future_len), traj_step):
            p = slam.frames[i].t_world.copy()
            p[2] -= ground_offset  # Virtual floor offset for visual ground contact
            pts_world.append(p)
        
        if len(pts_world) < 2: 
            return img
        pts_world = np.array(pts_world)
        
        # Project 3D World Points to 2D Pixel Coordinates
        p_cam = (T_w2c[:3, :3] @ pts_world.T).T + T_w2c[:3, 3]
        mask = p_cam[:, 2] > 0.1  # Depth culling (only project points in front of camera)
        
        uv_h = (f_curr.k @ p_cam.T).T
        u = uv_h[:, 0] / (uv_h[:, 2] + 1e-6)
        v = uv_h[:, 1] / (uv_h[:, 2] + 1e-6)
        
        # Render trajectory using Jet Colormap (Time-encoded colors)
        radius = max(2, int(round(4 * img.shape[0] / 480.0)))
        for i in range(len(u)):
            if mask[i]:
                color_idx = int(i / len(u) * 255)
                color = cv2.applyColorMap(np.array([color_idx], dtype=np.uint8), cv2.COLORMAP_JET)[0][0].tolist()
                cv2.circle(img, (int(u[i]), int(v[i])), radius, color, -1, cv2.LINE_AA)
        return img


    @staticmethod
    def save_professional_plot(slam: AriaSlam, out_png: str) -> None:
        """
        Generates a 12-panel high-fidelity diagnostic report for SLAM quality analysis.

        Args:
            slam (AriaSlam): The SLAM sequence data.
            out_png (str): Path to save the resulting image.
        """
        n = len(slam.frames)
        idx = np.arange(n)
        
        # Decompose attributes for plotting
        data = {
            "X": [f.t_world[0] for f in slam.frames],
            "Y": [f.t_world[1] for f in slam.frames],
            "Z": [f.t_world[2] for f in slam.frames],
            "rx": [f.rpy_deg[0] for f in slam.frames],
            "ry": [f.rpy_deg[1] for f in slam.frames],
            "rz": [f.yaw_unwrapped_deg for f in slam.frames],
            "dyaw": [f.delta_rpy_deg[2] for f in slam.frames],
            "v": [f.v for f in slam.frames],
            "w": [f.w for f in slam.frames],
        }

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 3, figure=fig, wspace=0.25, hspace=0.35)
        titles = [
            "X (meters)", "Y (meters)", "Z (meters)", 
            "rx (roll, deg)", "ry (pitch, deg)", "rz (yaw, UNWRAPPED deg)", 
            "Delta yaw vs frame0", "XY trajectory", "XYZ trajectory", 
            "Linear speed v (m/s)", "Angular speed w (rad/s)", "Future 10-frame yaw hist"
        ]

        # 1-D Time Series Plots
        for i, key in enumerate(["X", "Y", "Z", "rx", "ry", "rz", "dyaw"]):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.plot(idx, data[key], color='tab:orange', linewidth=2)
            ax.set_title(titles[i]); ax.grid(True, alpha=0.3)

        # 2-D Spatial Trajectory
        ax_xy = fig.add_subplot(gs[2, 1])
        ax_xy.plot(data["X"], data["Y"], color='tab:orange')
        ax_xy.scatter(data["X"][0], data["Y"][0], marker='o', label='start')
        ax_xy.scatter(data["X"][-1], data["Y"][-1], marker='x', label='end')
        ax_xy.set_title(titles[7]); ax_xy.set_aspect('equal'); ax_xy.grid(True, alpha=0.3)

        # 3-D Spatial Trajectory
        ax_3d = fig.add_subplot(gs[2, 2], projection='3d')
        ax_3d.plot(data["X"], data["Y"], data["Z"], color='tab:orange')
        ax_3d.set_title(titles[8])

        # Dynamic Metrics (Speed/Yaw Rate)
        ax_v = fig.add_subplot(gs[3, 0]); ax_v.plot(idx, data["v"], color='tab:orange'); ax_v.set_title(titles[9]); ax_v.grid(True, alpha=0.3)
        ax_w = fig.add_subplot(gs[3, 1]); ax_w.plot(idx, data["w"], color='tab:orange'); ax_w.set_title(titles[10]); ax_w.grid(True, alpha=0.3)

        # Delta Yaw Distribution (Noise/Stability Check)
        ax_h = fig.add_subplot(gs[3, 2])
        future_dyaw = np.diff(data["rz"])[::10]
        ax_h.hist(future_dyaw, bins=30, color='tab:blue')
        ax_h.set_title(titles[11]); ax_h.grid(True, alpha=0.3)

        plt.suptitle("Aria SLAM Kinematics & Trajectory Diagnostics", fontsize=16, y=0.95)
        plt.savefig(out_png, dpi=200); plt.close()