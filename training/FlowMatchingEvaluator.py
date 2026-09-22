# -*- coding: utf-8 -*-
# @FileName: FlowMatchingEvaluator.py
"""
Teacher-forced Professional Visualization Evaluator for FlowMatchingModel
GT vs Pred Hand Trajectory Edition.

This script evaluates the Flow Matching Model by:
1. Extracting K-step future 3D hand trajectories.
2. Projecting 3D traces to 2D using camera intrinsics.
3. Saving a stacked GT-vs-Pred hand-trajectory video (GT on top, Pred on bottom).
"""

from __future__ import annotations
import os
import json
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
import imageio.v2 as imageio

from rich.console import Console
console = Console()             

from training.FlowMatchingDataloader import (
    FlowMatchingDataloader,
    MPSSessions,
)
from utils.utils_math import (
    unnormalize_pos,
    rot6d_to_R_batch
)
from training.FlowMatchingModel import FlowMatchingModel

from utils.utils_media import create_video_from_frames

# =================================================================================================
# 1. IO & Geometry Helpers
# =================================================================================================

def safe_imread_rgb(path: str, fallback_hw=(640, 640)) -> np.ndarray:
    """Safely reads an RGB image, resized to fallback_hw. Returns a blank image if path is invalid."""
    if path and os.path.exists(path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            # Always resize to target panel size for consistent grid layout
            if bgr.shape[0] != fallback_hw[0] or bgr.shape[1] != fallback_hw[1]:
                bgr = cv2.resize(bgr, (fallback_hw[1], fallback_hw[0]), interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.zeros((fallback_hw[0], fallback_hw[1], 3), dtype=np.uint8)

def _read_json(path: str) -> Optional[dict]:
    """Safely reads a JSON file."""
    try:
        with open(path, "r") as f: 
            return json.load(f)
    except Exception: 
        return None

def find_K(d: dict) -> Optional[np.ndarray]:
    """Extracts 3x3 Camera Intrinsics matrix."""
    K = d.get("metadata", {}).get("k", None)
    return np.array(K, dtype=np.float32).reshape(3, 3) if K is not None else None

def find_T_c2w(d: dict) -> Optional[np.ndarray]:
    """Extracts 4x4 Camera-to-World transformation matrix."""
    T = d.get("metadata", {}).get("c2w", None)
    return np.array(T, dtype=np.float32).reshape(4, 4) if T is not None else None

def project_points_world_to_image(Pw: np.ndarray, K: np.ndarray, T_c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D world coordinates to 2D image pixel coordinates using Pinhole Camera Model.
    Returns:
        uv: (N, 2) pixel coordinates.
        valid: (N,) boolean mask for points in front of the camera (Z > 0).
    """
    if len(Pw) == 0: 
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    
    R_wc, t_wc = T_c2w[:3, :3], T_c2w[:3, 3:4]
    
    # World to Camera conversion
    Pc = ((R_wc.T @ Pw.T) - (R_wc.T @ t_wc)).T  
    
    z = np.maximum(Pc[:, 2], 1e-6) # Prevent division by zero
    valid = Pc[:, 2] > 1e-6
    
    # Perspective projection
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    
    return np.stack([u, v], axis=1), valid


# =================================================================================================
# 2. Entity State Extractors
# =================================================================================================

def get_gt_hand_world(d_t: dict, side: str) -> Tuple[Optional[np.ndarray], float]:
    """Extracts ground-truth 4x4 Hand Pose and grasp state."""
    hands = d_t.get("entities", {}).get("hands", {})
    if side in hands:
        return np.array(hands[side]["T_hand_to_world"], dtype=np.float32), float(hands[side]["grasp"])
    return None, 0.0

def extract_anchor_uv(state_tensor: torch.Tensor, use_region_attn: bool) -> Optional[torch.Tensor]:
    """Dynamically finds the Object Anchor Token (TypeID==3.0) to serve as Spatial Attention Bias."""
    if not use_region_attn: 
        return None
        
    B = state_tensor.shape[0]
    anchor_uvs =[]
    
    for b in range(B):
        idx = (state_tensor[b, :, 0] == 3.0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            anchor_uvs.append(state_tensor[b, idx[0], 1:3])
        else:
            anchor_uvs.append(torch.zeros(2, device=state_tensor.device))
            
    anchor_uv = torch.stack(anchor_uvs, dim=0)
    return torch.clamp(anchor_uv * 0.5 + 0.5, 0.0, 1.0)


# =================================================================================================
# 3. Artistic Rendering Tools
# =================================================================================================

def draw_polyline_uv(img: np.ndarray, uv: np.ndarray, valid: np.ndarray, color: Tuple[int,int,int], thickness=4, point_r=6) -> np.ndarray:
    """Draws connected 3D-to-2D projected points as a robust polyline."""
    out = img.copy()
    idx = np.where(valid)[0]
    
    if idx.size == 0: 
        return out
        
    # Clamp to prevent OpenCV drawing errors
    uvv = np.clip(uv[idx], [0, 0], [img.shape[1]-1, img.shape[0]-1]).astype(np.int32)
    
    if len(uvv) >= 2: 
        cv2.polylines(out, [uvv.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
        
    for p in uvv: 
        cv2.circle(out, tuple(p), point_r, color, -1, cv2.LINE_AA)
        
    return out

def put_title(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Adds a semi-transparent top bar with elegant titles."""
    out = img.copy()
    H, W = out.shape[:2]
    bar_h = max(48, int(H * 0.08))
    
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    
    fs1, fs2 = max(0.8, H / 900.0), max(0.6, H / 1200.0)
    cv2.putText(out, title, (16, int(bar_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, fs1, (245, 245, 245), 2, cv2.LINE_AA)
    
    if subtitle: 
        cv2.putText(out, subtitle, (16, int(bar_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, fs2, (185, 185, 185), 1, cv2.LINE_AA)
        
    return out


# =================================================================================================
# 4. Model Inference Core
# =================================================================================================

@torch.no_grad()
def solve_ode_trajectory(
    model: FlowMatchingModel, 
    x_rgb: torch.Tensor, 
    x_ict: torch.Tensor, 
    ict_mask: torch.Tensor,
    x_pcd: Optional[torch.Tensor], 
    device: str, 
    action_dim: int, 
    pred_horizon: int,
    num_inference_steps: int = 10, 
    use_region_attn: bool = True, 
    fixed_noise: Optional[torch.Tensor] = None 
) -> Dict[str, torch.Tensor]:
    """
    Executes the ODE integration for Flow Matching to generate multi-step trajectories.
    """
    B = x_rgb.size(0)
    
    # Init from noise (use fixed noise for video consistency)
    if fixed_noise is not None:
        x_t = fixed_noise.clone().to(device)
    else:
        x_t = torch.randn(B, pred_horizon, action_dim, device=device)
        
    dt = 1.0 / num_inference_steps
    anchor_uv = extract_anchor_uv(x_ict, use_region_attn)
    
    out = None
    for i in range(num_inference_steps):
        t_tensor = torch.full((B, 1), i * dt, device=device)
        out = model(
            x_rgb=x_rgb, 
            x_ict=x_ict, 
            ict_mask=ict_mask, 
            x_t=x_t, 
            t=t_tensor, 
            x_pcd=x_pcd, 
            anchor_uv=anchor_uv
        )
        x_t = x_t + out["v_pred"] * dt
        
    # Inject final physical trajectory back into the dictionary
    out["action_trajectory"] = x_t
    return out

def decode_o6d_to_mat(o6d_array: np.ndarray) -> np.ndarray:
    """Safe wrapper to decode a 6D rotation array to a 3x3 Rotation Matrix."""
    o6d_t = torch.from_numpy(o6d_array).float()
    if o6d_t.ndim == 1: 
        o6d_t = o6d_t.unsqueeze(0)
    return rot6d_to_R_batch(o6d_t)[0].detach().cpu().numpy()


# =================================================================================================
# 5. The Ultimate 6-Grid Evaluator
# =================================================================================================

@torch.no_grad()
def run_teacher_forced_vis(
    model: FlowMatchingModel, ckpt_path: str, mps_path: str, out_dir: str,
    single_hand_side: str, pred_horizon: int, image_size: Tuple[int, int],
    device: str, centric_mode: str, frame_mode: str, action_mode: str, max_ict: int,
    stats: Dict, num_inference_steps: int = 10, max_frames: Optional[int] = None,
    make_video: bool = True, video_fps: int = 30,
    use_done_in_flow: bool = False,
):
    model.eval()
    
    # ------------------ Initialization ------------------
    action_dim = model.action_dim
    single_hand = model.single_hand
    
    dummy_ds = FlowMatchingDataloader(
        sessions=[MPSSessions(mps_path)],
        image_size=image_size, pred_horizon=pred_horizon,
        single_hand=single_hand, single_hand_side=single_hand_side,
        centric_mode=centric_mode, frame_mode=frame_mode, action_mode=action_mode, max_ict=max_ict,
        use_pcd_features=model.use_pcd_features,
        use_aux_obj_dynamics=model.use_aux_obj_dynamics,
        use_aux_visual_foresight=model.use_aux_visual_foresight,
        use_aux_temporal_contrastive=False, # We don't need contrastive for rendering
        enable_augmentation=False, stats=stats
    )

    aria_dir = os.path.join(mps_path, "preprocess", "all_data")
    if not os.path.exists(aria_dir):
        return
        
    frame_dirs = sorted([d for d in os.listdir(aria_dir) if d.isdigit()])

    vis_dir = os.path.join(out_dir, "teacher_forced_vis")
    frames_dir = os.path.join(vis_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    saved_count = 0
    vis_frames =[]
    
    # BGR Color Palette
    C_MAP = {
        "left": (255, 100, 50),   # Blue
        "right": (50, 150, 255),  # Orange
        "obj": (255, 50, 255)     # Purple
    }
    
    hand_sides = [single_hand_side] if single_hand else ["left", "right"]
    num_hands = len(hand_sides)
    base_dim = 10 * num_hands

    # Generate ONE fixed noise for silky-smooth video coherence
    fixed_x0 = torch.randn(1, pred_horizon, action_dim, device=device)

    # ------------------ Main Rendering Loop ------------------
    for i in range(len(frame_dirs)):
        if max_frames and saved_count >= max_frames: break
        if i + pred_horizon >= len(frame_dirs): break

        fd = frame_dirs[i]
        json_path_curr = os.path.join(aria_dir, fd, "training_data.json")
        d_t = _read_json(json_path_curr)
        
        if d_t is None: continue

        # Ensure primary hand exists in this frame
        primary_h, _ = get_gt_hand_world(d_t, single_hand_side if single_hand else "left")
        if primary_h is None: continue

        # Extract Camera Parameters
        K_cam = find_K(d_t)
        T_c2w = find_T_c2w(d_t)
        if K_cam is None or T_c2w is None: continue

        # Scale K to match panel resolution (image_size) instead of native resolution
        md = d_t.get("metadata", {})
        orig_h = int(md.get("h", 480) or 480)
        orig_w = int(md.get("w", 640) or 640)
        scale_x = image_size[1] / orig_w   # panel_w / native_w
        scale_y = image_size[0] / orig_h   # panel_h / native_h
        K_cam = K_cam.copy()
        K_cam[0, :] *= scale_x   # fx, cx
        K_cam[1, :] *= scale_y   # fy, cy

        # Build Inputs using DataLoader components
        x_rgb = dummy_ds._load_image_tensor(d_t, 0).unsqueeze(0).to(device)
        T_w2ref = dummy_ds._get_T_w2ref(d_t)
        state_np, pcd_np, ict_mask_np = dummy_ds._build_ict(d_t, T_w2ref) 

        x_ict = torch.from_numpy(state_np).unsqueeze(0).to(device)
        ict_mask = torch.from_numpy(ict_mask_np).unsqueeze(0).to(device)
        x_pcd = torch.from_numpy(pcd_np).unsqueeze(0).to(device) if model.use_pcd_features else None

        # --- A. Model Inference ---
        out_dict = solve_ode_trajectory(
            model, x_rgb, x_ict, ict_mask, x_pcd, device, 
            action_dim, pred_horizon, num_inference_steps, 
            use_region_attn=model.use_region_attn, fixed_noise=fixed_x0
        )
        pred_action_full = out_dict["action_trajectory"][0].cpu().numpy() 

        # --- B. Get Ground Truth ---
        targets_gt = dummy_ds._build_targets(json_path_curr, T_w2ref, 0, 1, d_t)

        # Get GT Done flag at the end of the horizon
        gt_done_flag = targets_gt.get("y_done", torch.zeros(pred_horizon, 1))[-1, 0].item()

        # Get Predicted Done probability
        if use_done_in_flow:
            # Done is last dim of action trajectory
            pred_done_prob = float(pred_action_full[-1, -1])
            pred_done_prob = 1.0 / (1.0 + np.exp(-pred_done_prob))  # sigmoid
        else:
            # Independent BCE head
            pred_done_logit = out_dict.get("done_logit", torch.zeros(1, 1))
            pred_done_prob = torch.sigmoid(pred_done_logit)[0, 0].item()

        # --- C. Setup 6 Canvases ---
        # Derive image path from json_path (always valid) instead of stored rgb_path (may be stale)
        frame_dir = os.path.dirname(json_path_curr)
        raw_rgb_path = os.path.join(frame_dir, "rgb.png")
        raw_rgb_img = safe_imread_rgb(raw_rgb_path, fallback_hw=image_size)
        raw_rgb_bgr = cv2.cvtColor(raw_rgb_img, cv2.COLOR_RGB2BGR) # Switch to BGR for OpenCV
        
        p_hand_gt = raw_rgb_bgr.copy()
        p_hand_pr = raw_rgb_bgr.copy()

        T_ref2w = np.linalg.inv(T_w2ref)

        # --- D. Draw Hands (Panel 1 & 4) ---
        for h_idx, side in enumerate(hand_sides):
            c_hand = C_MAP[side]
            
            # GT Hand Rendering
            gt_pos_world =[]
            for k in range(1, pred_horizon + 1):
                dk = _read_json(dummy_ds._get_future_json_path(json_path_curr, k))
                if dk:
                    T_h_w_k, _ = get_gt_hand_world(dk, side)
                    if T_h_w_k is not None: gt_pos_world.append(T_h_w_k[:3, 3])
                    
            if len(gt_pos_world) > 0:
                uv_gt, v_gt = project_points_world_to_image(np.array(gt_pos_world), K_cam, T_c2w)
                p_hand_gt = draw_polyline_uv(p_hand_gt, uv_gt, v_gt, c_hand)

            # PRED Hand Rendering
            T_h_w_0, _ = get_gt_hand_world(d_t, side)
            if T_h_w_0 is not None:
                T_h_ref_base = T_w2ref @ T_h_w_0
                # Action layout: [pos_all_hands | o6d_all_hands | g_all_hands]
                # e.g. dual-hand: [L_pos(3), R_pos(3), L_o6d(6), R_o6d(6), L_g(1), R_g(1)]
                pos_start = h_idx * 3
                rot_start = num_hands * 3 + h_idx * 6
                pred_pos_world =[]

                for k in range(pred_horizon):
                    p_norm = pred_action_full[k, pos_start : pos_start + 3]
                    o6d_norm = pred_action_full[k, rot_start : rot_start + 6]
                    
                    T_delta = np.eye(4)
                    T_delta[:3, :3] = decode_o6d_to_mat(o6d_norm)
                    T_delta[:3, 3] = unnormalize_pos(p_norm, dummy_ds.pos_mean, dummy_ds.pos_std)
                    
                    T_h_ref_k = T_h_ref_base @ T_delta if action_mode == 'delta' else T_delta
                    T_h_w = T_ref2w @ T_h_ref_k
                    pred_pos_world.append(T_h_w[:3, 3])

                if len(pred_pos_world) > 0:
                    uv_pr, v_pr = project_points_world_to_image(np.array(pred_pos_world), K_cam, T_c2w)
                    p_hand_pr = draw_polyline_uv(p_hand_pr, uv_pr, v_pr, c_hand)

        # --- E. Assemble the GT (top) vs Pred (bottom) Grid ---
        gt_title_suffix = " [FINISHED]" if gt_done_flag > 0.5 else ""
        pr_title_suffix = f" [DONE: {pred_done_prob:.2f}]"

        # Draw explicit visual cues if the model predicts the task is finished
        if pred_done_prob > 0.5:
            cv2.rectangle(p_hand_pr, (0, 0), (p_hand_pr.shape[1], p_hand_pr.shape[0]), (0, 255, 0), 8)
            cv2.putText(p_hand_pr, "TASK FINISHED", (int(p_hand_pr.shape[1]*0.5 - 110), int(p_hand_pr.shape[0]*0.85)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        p_hand_gt = put_title(p_hand_gt, f"Frame {int(fd)}{gt_title_suffix}", "GT Hand Traj")
        p_hand_pr = put_title(p_hand_pr, f"Pred{pr_title_suffix}", "PRED Hand Traj")

        canvas = np.vstack([p_hand_gt, p_hand_pr])
        
        # if int(fd) % 10 == 0:
        #     cv2.imwrite(os.path.join(frames_dir, f"frame_{int(fd):05d}.png"), canvas)
        
        if make_video: vis_frames.append(canvas)
        saved_count += 1

    # --- H. Export Final Assets ---
    if make_video and vis_frames:
        create_video_from_frames(vis_frames, os.path.join(vis_dir, "evaluation_vis.mp4"), fps=30, export_gif=True, ratio=10)
    
    console.print(f"[OK] Evaluator: GT-vs-Pred Hand Traj Teacher-forced ({action_mode}) Visualization finished.")


if __name__ == "__main__":
    pass