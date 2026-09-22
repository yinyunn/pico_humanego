# -*- coding: utf-8 -*-
# @FileName: VisualKpts.py

"""
====================================================================================================
Visual Keypoints Rendering Pipeline (VisualKpts.py)
====================================================================================================

Description:
    This script is responsible for rendering highly visual, 2D representations of 3D spatial 
    keypoints onto image frames. It draws physical wireframe models of the user's hand (gripper) 
    and tracks multi-object movement, utilizing dynamic color gradients, fading historical trails, 
    and explicit visual cues to distinguish between "reaching" (open) and "transporting" (closed) states.

Core Functionalities:
    1.  Physical Gripper Projection: Constructs a 3D wireframe of the hand based on physical 
        dimensions (meters) and projects it onto the 2D camera plane.
    2.  Dynamic Grasp Cues: Instantly switches the color palette of the hand wireframe 
        (e.g., from warm to cool tones) to explicitly signal to visual models that a grasp has occurred.
    3.  Historical Trajectories (Trails): Draws fading "comet tail" trajectories for both 
        the hand and tracked objects, with tapering thickness and alpha blending.
    4.  Multi-Object Rendering: Automatically cycles through distinct color palettes to 
        render multiple tracked objects simultaneously without color clashing.

Generated Outputs:
    - Augmented RGB images saved alongside the original frames (e.g., *_WArmObjKpts.png).
    - Can be composed into full video visualizations by upstream managers.

Technical Specifics:
    - Color Format: Strictly relies on BGR format for OpenCV compatibility.
    - Performance: Pre-loads all required kinematics and extrinsics into memory before 
      the rendering loop begins, ensuring smooth and fast sequential processing.
====================================================================================================
"""

import os
import cv2
import json
import numpy as np
from typing import Optional, List, Tuple

from utils.utils_io import load_cfg

# ==============================================================================
# [Helper] Linear Color Interpolation
# ==============================================================================
def interpolate_bgr(color_start: List[int], color_end: List[int], ratio: float) -> List[int]:
    """
    Linearly interpolates between two BGR colors.
    
    Args:
        color_start: The starting BGR color (ratio = 0.0).
        color_end: The ending BGR color (ratio = 1.0).
        ratio: Interpolation factor[0.0, 1.0].
        
    Returns:
        Interpolated BGR color as a list of integers.
    """
    ratio = np.clip(ratio, 0, 1)
    return[
        int(color_start[i] + (color_end[i] - color_start[i]) * ratio)
        for i in range(3)
    ]


# ==============================================================================
# [Manager] VisualKpts Engine
# ==============================================================================
class VisualKptsEngine:
    """
    Core engine managing the projection, styling, and rendering of physical hand 
    keypoints and object tracking trails.
    """
    def __init__(self, cfg_path: str = None):
        self.cfg = load_cfg(cfg_path) if cfg_path else None
        self.cotracker_data = None
        self.global_image_list = []
        self.preloaded_hands =[]
        self.preloaded_cams =[]
        
        # Internal counter for automatic color picking when processing sequential objects
        self._obj_render_counter = 0

    def _load_global_data(self, mps_path: str, cfg_path: str, all_image_paths: List[str]):
        """Preloads camera intrinsics, extrinsics, and hand kinematics for the entire sequence."""
        self.cfg = load_cfg(cfg_path)
        self.global_image_list = all_image_paths

        cotracker_json = os.path.join(mps_path, "preprocess", "cotracker_results.json")
        if os.path.exists(cotracker_json):
            with open(cotracker_json, 'r') as f:
                self.cotracker_data = json.load(f)

        self.preloaded_hands = []
        self.preloaded_cams = []
        print(f"║ [VisualKpts] Preloading metadata for smooth rendering...")

        # Detect source: Aria (aria_hands.json) vs Robot (robot_state.json)
        sample_dir = os.path.dirname(all_image_paths[0])
        is_robot = os.path.exists(os.path.join(sample_dir, "robot_state.json")) and \
                   not os.path.exists(os.path.join(sample_dir, "aria_hands.json"))

        if is_robot:
            # Robot mode: load session_meta once for static camera
            meta_path = os.path.join(mps_path, "preprocess", "session_meta.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if "k" in meta:
                k_val = meta["k"]
            elif "camera" in meta:
                k_val = meta["camera"].get("k_rgb", meta["camera"].get("k"))
            else:
                raise KeyError("session_meta.json missing intrinsics ('k' or 'camera.k_rgb')")
            static_cam = {"k": k_val, "c2w": np.eye(4).tolist()}

            # _get_segments_3d_oriented_binary expects: X=finger spread, Y=approach (root at -Y, tip at 0)
            # Trossen ee: X=finger spread, Z=approach (tip at +Z)
            # Mapping: Trossen X→X (spread), Trossen Z→-Y (approach flipped), Trossen Y→Z
            R_z2y = np.eye(4)
            R_z2y[:3, :3] = np.array([[1, 0,  0],
                                       [0, 0, -1],
                                       [0, 1,  0]], dtype=np.float64)

            for img_p in all_image_paths:
                frame_dir = os.path.dirname(img_p)
                self.preloaded_cams.append(static_cam)
                try:
                    with open(os.path.join(frame_dir, "robot_state.json"), 'r') as f:
                        rs = json.load(f)
                    hand_dict = {}
                    side_map = {"right": "hand_r", "left": "hand_l"}
                    for arm_name, hand_key in side_map.items():
                        arm = rs.get("arms", {}).get(arm_name)
                        if arm and "T_ee_in_cam" in arm:
                            T = np.array(arm["T_ee_in_cam"]) @ R_z2y
                            gripper_q = arm.get("gripper_q", 0.0)
                            hand_dict[hand_key] = {
                                "midpoint_pose_opt_world": T.flatten().tolist(),
                                "grasp_state": 1.0 - float(np.clip(gripper_q, 0.0, 1.0)),
                            }
                    self.preloaded_hands.append(hand_dict if hand_dict else None)
                except:
                    self.preloaded_hands.append(None)
        else:
            # Aria mode: original loading
            for img_p in all_image_paths:
                frame_dir = os.path.dirname(img_p)
                try:
                    with open(os.path.join(frame_dir, "aria_cam_rgb.json"), 'r') as f:
                        self.preloaded_cams.append(json.load(f))
                    with open(os.path.join(frame_dir, "aria_hands.json"), 'r') as f:
                        self.preloaded_hands.append(json.load(f))
                except:
                    self.preloaded_cams.append(None)
                    self.preloaded_hands.append(None)

    def _get_segments_3d_oriented_binary(self, hand_json: dict) -> tuple[List[Tuple[np.ndarray, np.ndarray]], bool]:
        """
        Constructs the 3D local geometry of the gripper based on physical dimensions 
        and the current grasp state (Open/Closed).
        """
        pose_w = np.array(hand_json["midpoint_pose_opt_world"]).reshape(4, 4)
        is_close = hand_json.get("grasp_state", 0) > 0.5
        
        # Adjust gripper width dynamically based on the grasp state
        width = self.cfg.visualkpts_gripper_width_min if is_close else self.cfg.visualkpts_gripper_width_max
        half_w = width / 2.0
        
        fl = self.cfg.visualkpts_gripper_finger_len
        rd = self.cfg.visualkpts_gripper_root_depth
        
        # Define local coordinates for a simple two-finger gripper
        p_l_tip_l  = np.array([-half_w, 0, 0, 1])    
        p_r_tip_l  = np.array([half_w, 0, 0, 1])     
        p_l_base_l = np.array([-half_w, -fl, 0, 1])  
        p_r_base_l = np.array([half_w, -fl, 0, 1])   
        p_root_l   = np.array([0, -(fl + rd), 0, 1]) 

        def to_w(p_l): return (pose_w @ p_l)[:3]

        p_root = to_w(p_root_l)
        p_l_base = to_w(p_l_base_l)
        p_r_base = to_w(p_r_base_l)
        p_l_tip = to_w(p_l_tip_l)
        p_r_tip = to_w(p_r_tip_l)
        
        segments =[
            (p_root, p_l_base),   # Wrist to Left Base
            (p_root, p_r_base),   # Wrist to Right Base
            (p_l_base, p_l_tip),  # Left Base to Left Tip
            (p_r_base, p_r_tip)   # Right Base to Right Tip
        ]
        return segments, is_close
    
    def _draw_kpt_with_core(self, canvas: np.ndarray, u: float, v: float, color: List[int], radius: int):
        """Draws a solid colored circle with a contrasting white inner core."""
        cv2.circle(canvas, (int(u), int(v)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(u), int(v)), self.cfg.visualkpts_white_core_radius, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_advanced_trails(self, canvas: np.ndarray, g_idx: int, w2c: np.ndarray, K: np.ndarray):
        """Renders fading historical trajectories (comet tails) for both the gripper and objects."""
        trail_len = self.cfg.visualkpts_trail_len
        if trail_len <= 0: return
        
        start_t = max(0, g_idx - trail_len)
        overlay = canvas.copy()
        
        # Base colors for the standard open hand
        c1, c2, c3, c4 = self.cfg.color_hand_1_line, self.cfg.color_hand_2_line, \
                         self.cfg.color_hand_3_line, self.cfg.color_hand_4_line

        for t in range(start_t, g_idx, self.cfg.visualkpts_trail_step):
            h_c = self.preloaded_hands[t]
            h_n = self.preloaded_hands[min(t + self.cfg.visualkpts_trail_step, g_idx)]
            if h_c is None or h_n is None: continue
            
            # Temporal progression ratio (0.0 = oldest, 1.0 = newest)
            time_ratio = (t - start_t) / (g_idx - start_t + 1e-6)
            thickness = int(self.cfg.visualkpts_trail_thick_min + 
                            (self.cfg.visualkpts_trail_thick_max - self.cfg.visualkpts_trail_thick_min) * time_ratio)

            # Draw Hand Trails
            for side in["hand_r", "hand_l"]:
                hc_side = h_c.get(side)
                hn_side = h_n.get(side)
                if not hc_side or not hn_side: continue
                
                seg_c, _ = self._get_segments_3d_oriented_binary(hc_side)
                seg_n, _ = self._get_segments_3d_oriented_binary(hn_side)
                
                for s_idx in range(4):
                    s_col, e_col = (c4, c3) if s_idx < 2 else (c2, c1)
                    
                    pts_c_3d = np.linspace(seg_c[s_idx][0], seg_c[s_idx][1], self.cfg.visualkpts_pts_per_line)
                    pts_n_3d = np.linspace(seg_n[s_idx][0], seg_n[s_idx][1], self.cfg.visualkpts_pts_per_line)
                    
                    def project(p3d):
                        pc = (w2c[:3, :3] @ p3d.T).T + w2c[:3, 3]
                        uvh = (K @ pc.T).T
                        return np.column_stack([uvh[:,0]/uvh[:,2], uvh[:,1]/uvh[:,2], pc[:,2]])

                    uvz_c = project(pts_c_3d)
                    uvz_n = project(pts_n_3d)

                    for p_i in range(len(uvz_c)):
                        if uvz_c[p_i, 2] > 0.1 and uvz_n[p_i, 2] > 0.1: # Depth check
                            spatial_ratio = p_i / (len(uvz_c) - 1)
                            base_col = interpolate_bgr(s_col, e_col, spatial_ratio)
                            
                            # Fade color intensity based on age
                            draw_col =[int(c * (0.2 + 0.8 * time_ratio)) for c in base_col]
                            
                            cv2.line(overlay, (int(uvz_c[p_i,0]), int(uvz_c[p_i,1])), 
                                     (int(uvz_n[p_i,0]), int(uvz_n[p_i,1])), draw_col, thickness, cv2.LINE_AA)

            # Draw Object Trails
            if self.cotracker_data:
                tracks = np.array(self.cotracker_data["tracks"])
                vis = np.array(self.cotracker_data["visibility"])
                if g_idx < len(tracks):
                    # Only draw trail if the object has moved beyond the threshold
                    motion = np.linalg.norm(tracks[g_idx] - tracks[start_t], axis=-1)
                    for n in range(tracks.shape[1]):
                        if motion[n] > self.cfg.visualkpts_obj_motion_th and vis[t, n] > 0:
                            obj_trail_col =[int(c * (0.2 + 0.8 * time_ratio)) for c in self.cfg.color_object_trail]
                            cv2.line(overlay, (int(tracks[t,n,0]), int(tracks[t,n,1])), 
                                     (int(tracks[min(t+1, g_idx),n,0]), int(tracks[min(t+1, g_idx),n,1])), 
                                     obj_trail_col, thickness, cv2.LINE_AA)

        # Alpha blending for the full overlay
        cv2.addWeighted(overlay, self.cfg.visualkpts_trail_alpha_max, canvas, 1 - self.cfg.visualkpts_trail_alpha_max, 0, canvas)

    def _draw_xyz_axes(self, canvas: np.ndarray, pose_w: np.ndarray, w2c: np.ndarray, K: np.ndarray, length: float = 0.06):
        """Projects and renders 3D coordinate axes onto the 2D image plane."""
        if not isinstance(pose_w, np.ndarray):
            pose_w = np.array(pose_w).reshape(4, 4)
        elif pose_w.shape == (16,):
            pose_w = pose_w.reshape(4, 4)

        T_cam = w2c @ pose_w
        origin_cam = T_cam[:3, 3]
        if origin_cam[2] <= 0.01: return 

        def project(p_cam):
            u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
            v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
            return (u, v)

        uv_o = project(origin_cam)
        axis_info = [
            {"dir": T_cam[:3, 0], "color": (0, 0, 255), "label": "X"}, # BGR Red
            {"dir": T_cam[:3, 1], "color": (0, 255, 0), "label": "Y"}, # BGR Green
            {"dir": T_cam[:3, 2], "color": (255, 0, 0), "label": "Z"}  # BGR Blue
        ]

        for info in axis_info:
            p_tip = origin_cam + info["dir"] * length
            uv_tip = project(p_tip)
            cv2.arrowedLine(canvas, uv_o, uv_tip, info["color"], 2, tipLength=0.3, line_type=cv2.LINE_AA)
            p_label = origin_cam + info["dir"] * (length * 1.25)
            uv_label = project(p_label)
            
            # Draw label with shadow for readability
            cv2.putText(canvas, info["label"], (uv_label[0]+1, uv_label[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, info["label"], uv_label, cv2.FONT_HERSHEY_DUPLEX, 0.5, info["color"], 1, cv2.LINE_AA)

        cv2.circle(canvas, uv_o, 3, (255, 255, 255), -1, cv2.LINE_AA)


    def process(self, image_path: str, cfg_path: str, frame_idx: int, all_image_paths: List[str], mps_path: str, draw_axes: bool = False):
        """Main entry point for full-frame processing."""
        if self.cfg is None or self.global_image_list != all_image_paths:
            self._load_global_data(mps_path, cfg_path, all_image_paths)

        try: 
            g_idx = self.global_image_list.index(image_path)
        except ValueError: 
            return cv2.imread(image_path)

        frame_dir = os.path.dirname(image_path)

        # Only generate WArmObjKpts for rgb and rgb_WoArm bases.
        # Skipped: rgb_WoArmObj base, all WArmKpts variants (hand-only, no objects).
        for img_name in ['rgb', 'rgb_WoArm']:
            bg_path = os.path.join(frame_dir, f'{img_name}.png')
            canvas = cv2.imread(bg_path) if os.path.exists(bg_path) else cv2.imread(image_path)
            if canvas is None: return None

            curr_cam = self.preloaded_cams[g_idx]
            if curr_cam:
                K = np.array(curr_cam["k"])
                w2c = np.linalg.inv(np.array(curr_cam["c2w"]))

                self._draw_advanced_trails(canvas, g_idx, w2c, K)

                curr_h = self.preloaded_hands[g_idx]
                if curr_h:
                    for side in ["hand_r", "hand_l"]:
                        hand = curr_h.get(side)
                        if hand and hand.get("midpoint_pose_opt_world"):
                            if draw_axes:
                                pose_w = np.array(hand["midpoint_pose_opt_world"]).reshape(4, 4)
                                self._draw_xyz_axes(canvas, pose_w, w2c, K)

                            segments, is_close = self._get_segments_3d_oriented_binary(hand)

                            # --- Dynamic Color Switching based on Grasp State ---
                            if is_close:
                                c1, c2, c3, c4 = self.cfg.color_hand_closed_1, self.cfg.color_hand_closed_2, \
                                                 self.cfg.color_hand_closed_3, self.cfg.color_hand_closed_4
                            else:
                                c1, c2, c3, c4 = self.cfg.color_hand_1_line, self.cfg.color_hand_2_line, \
                                                 self.cfg.color_hand_3_line, self.cfg.color_hand_4_line

                            for s_idx, (p1, p2) in enumerate(segments):
                                s_col, e_col = (c4, c3) if s_idx < 2 else (c2, c1)
                                line_w = np.linspace(p1, p2, self.cfg.visualkpts_pts_per_line)
                                pc = (w2c[:3, :3] @ line_w.T).T + w2c[:3, 3]
                                uvh = (K @ pc.T).T

                                for p_i, pt in enumerate(np.column_stack([uvh[:,0]/uvh[:,2], uvh[:,1]/uvh[:,2], pc[:,2]])):
                                    if pt[2] > 0.1:
                                        spatial_ratio = p_i / (self.cfg.visualkpts_pts_per_line - 1)
                                        curr_col = interpolate_bgr(s_col, e_col, spatial_ratio)
                                        self._draw_kpt_with_core(canvas, pt[0], pt[1], curr_col, self.cfg.visualkpts_radius_current)

            # Draw object keypoints and save WArmObjKpts only
            if self.cotracker_data:
                palette = self.cfg.obj_colors
                c_idx = 0
                for obj_key, obj_data in self.cotracker_data.items():
                    if (not obj_key.startswith("obj")) or (obj_key == "obj_and_arm"):
                        continue

                    if g_idx < len(obj_data["tracks"]):
                        tracks = np.array(obj_data["tracks"][g_idx])
                        color = palette[c_idx % len(palette)]

                        for pt in tracks:
                            if not np.isnan(pt[0]):
                                self._draw_kpt_with_core(canvas, pt[0], pt[1], color, self.cfg.visualkpts_radius_current + 1)
                    c_idx += 1
            save_path = os.path.join(os.path.dirname(image_path), f"{img_name}_WArmObjKpts.png")
            cv2.imwrite(save_path, canvas)

        return canvas

    def reset_obj_counter(self):
        """Resets the internal object color counter at the start of a new frame processing."""
        self._obj_render_counter = 0

    def process_single_obj(self, img: np.ndarray, kpts_2d: np.ndarray, force_color: Optional[List[int]] = None) -> np.ndarray:
        """
        Renders 2D keypoints for a single object.
        Automatically cycles through the `obj_colors` palette if `force_color` is not provided.
        """
        if img is None:
            return None
        
        canvas = img.copy()
        
        # Determine Color
        if force_color is not None:
            color = force_color
        else:
            palette = self.cfg.obj_colors
            color = palette[self._obj_render_counter % len(palette)]
            self._obj_render_counter += 1
            
        radius = self.cfg.visualkpts_radius_current + 1
        
        if kpts_2d is None:
            return canvas

        pts = np.atleast_2d(kpts_2d)

        # Render points
        for pt in pts:
            u, v = pt[0], pt[1]
            if np.isnan(u) or np.isnan(v) or (u == 0 and v == 0):
                continue
            if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
                self._draw_kpt_with_core(canvas, int(u), int(v), color, radius)
        
        return canvas
    
    def process_single_gripper(self, img: np.ndarray, pose_cam: np.ndarray, is_grasping: bool, K: np.ndarray) -> np.ndarray:
        """ 
        Renders the wireframe of the gripper for a single frame, switching color palettes 
        based on the grasp state.
        """
        if pose_cam.shape == (6,):
            T = pose_cam[:3]
            R_vec = pose_cam[3:]
            R_mat, _ = cv2.Rodrigues(R_vec)
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = R_mat
            pose_matrix[:3, 3] = T
            pose_cam = pose_matrix
        
        canvas = img.copy()
        
        current_width = self.cfg.visualkpts_gripper_width_min if is_grasping else self.cfg.visualkpts_gripper_width_max
        half_w = current_width / 2.0
        fl = self.cfg.visualkpts_gripper_finger_len
        rd = self.cfg.visualkpts_gripper_root_depth

        p_l_tip_l = np.array([-half_w, 0, 0, 1])    
        p_r_tip_l = np.array([half_w, 0, 0, 1])     
        p_l_base_l = np.array([-half_w, 0, -fl, 1]) 
        p_r_base_l = np.array([half_w, 0, -fl, 1])  
        p_root_l = np.array([0, 0, -(fl + rd), 1])  

        segments_local =[
            (p_root_l, p_l_base_l),  
            (p_root_l, p_r_base_l),  
            (p_l_base_l, p_l_tip_l), 
            (p_r_base_l, p_r_tip_l)  
        ]

        # --- Dynamic Color Shifting for Strong Visual Cue ---
        if not is_grasping:
            # Open state: Use warm gradient (Red -> Yellow)
            c1, c2, c3, c4 = self.cfg.color_hand_1_line, self.cfg.color_hand_2_line, \
                             self.cfg.color_hand_3_line, self.cfg.color_hand_4_line
        else:
            # Closed state: Snap to cool gradient (Purple -> White)
            c1, c2, c3, c4 = self.cfg.color_hand_closed_1, self.cfg.color_hand_closed_2, \
                             self.cfg.color_hand_closed_3, self.cfg.color_hand_closed_4

        for s_idx, (p1_l, p2_l) in enumerate(segments_local):
            s_col, e_col = (c4, c3) if s_idx < 2 else (c2, c1)
            pts_local = np.linspace(p1_l, p2_l, self.cfg.visualkpts_pts_per_line) 
            pts_cam_transformed = (pose_cam @ pts_local.T).T 
            
            for i in range(len(pts_cam_transformed)):
                p_cam = pts_cam_transformed[i][:3]
                if p_cam[2] <= 0.001: continue 
                
                uvh = K @ p_cam
                u, v = uvh[0] / uvh[2], uvh[1] / uvh[2]
                
                if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
                    spatial_ratio = i / (self.cfg.visualkpts_pts_per_line - 1)
                    curr_col = interpolate_bgr(s_col, e_col, spatial_ratio)
                    self._draw_kpt_with_core(canvas, u, v, curr_col, self.cfg.visualkpts_radius_current)

        return canvas

# ==============================================================================
#[Public Interface] API for Pipeline Integration
# ==============================================================================
_VIS_INSTANCE = VisualKptsEngine()


def reset_visualkpts():
    """Reset the singleton so a new session gets a fresh state."""
    global _VIS_INSTANCE
    _VIS_INSTANCE = VisualKptsEngine()


def run_visualkpts(image_path: str, cfg_path: str, frame_idx: int, all_image_paths: List[str] = None, mps_path: str = None) -> np.ndarray:
    """Singleton entry point for triggering the visual keypoints rendering."""
    return _VIS_INSTANCE.process(image_path, cfg_path, frame_idx, all_image_paths, mps_path)