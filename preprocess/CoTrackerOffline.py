# -*- coding: utf-8 -*-
# @FileName: CoTrackerOffline.py

"""
====================================================================================================
Project Aria CoTracker3 Offline Tracking Pipeline (CoTrackerOffline.py)
====================================================================================================

Description:
    This script performs dense, offline point tracking across video sequences using Facebook's 
    CoTracker3. It takes initial keypoints (typically extracted from a reference frame via 
    DINO+SAM2 and KptsSelector) and tracks them robustly both forward and backward in time.

Core Functionalities:
    1.  Letterboxing Preprocessing: Resizes and pads images to a square resolution (e.g., 640x640) 
        while maintaining the aspect ratio, which is optimal for CoTracker's grid-based attention.
    2.  Chunk-based Inference: Processes video frames in overlapping chunks to manage VRAM 
        usage efficiently for long sequences.
    3.  Bidirectional Tracking: Tracks points forward (from ref_idx to end) and backward 
        (from ref_idx to start) to ensure full-sequence trajectory coverage.
    4.  Multi-Object Support: Aggregates keypoints from multiple objects, tracks them 
        simultaneously in a single batch pass, and splits the results back per object.
    5.  Visualization: Generates HUD-style trajectory trails and active point counters.

Generated Outputs:
    - [mps_path]/aria/cotracker_results.json: Full trajectory coordinates and visibility flags.
    - Rendered Visualization frames (handled by the caller or standalone __main__).

Technical Specifics:
    - Mixed Precision: Uses torch.amp.autocast for memory-efficient inference.
    - Coordinate Mapping: Handles seamless mapping between original image dimensions and 
      the padded square inference resolution.
====================================================================================================
"""

import os
import cv2
import json
import time
import torch
import numpy as np
import gc
import argparse
from typing import List
from huggingface_hub import hf_hub_download

from cotracker.predictor import CoTrackerPredictor
from utils.utils_vis import draw_glass_rect
from utils.utils_io import load_cfg, NumpyEncoder


# ==============================================================================
# [Engine] CoTracker Inference Engine
# ==============================================================================
class CoTrackerOfflineEngine:
    """
    Core engine for running CoTracker3 offline inference.
    Handles device allocation, letterbox scaling, and chunked processing.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._get_device()
        print(f"║ [CoTracker] Initializing on {self.device}...")
        
        ckpt = hf_hub_download(repo_id="facebook/cotracker3", filename="scaled_offline.pth")
        self.model = CoTrackerPredictor(checkpoint=ckpt).to(self.device)

    def _get_device(self):
        """Determines the optimal hardware accelerator (CUDA/MPS/CPU)."""
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def _letterbox(self, img_bgr):
        """
        Resizes and pads an image to a square resolution while maintaining aspect ratio.
        Provides the metadata needed to reverse this transformation later.
        """
        h, w = img_bgr.shape[:2]
        res = self.cfg.cotracker_res
        scale = res / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (new_w, new_h))
        
        pad_w, pad_h = res - new_w, res - new_h
        pad_l, pad_t = pad_w // 2, pad_h // 2
        
        img_sq = cv2.copyMakeBorder(resized, pad_t, pad_h-pad_t, pad_l, pad_w-pad_l, 
                                    cv2.BORDER_CONSTANT, value=(0,0,0))
        meta = {"scale": scale, "pad_l": pad_l, "pad_t": pad_t, "orig_w": w, "orig_h": h}
        return img_sq, meta

    def _unletterbox_points(self, tracks_sq, meta):
        """
        Maps tracked coordinates from the letterboxed square resolution back to 
        the original image resolution.
        """
        s, px, py = meta["scale"], meta["pad_l"], meta["pad_t"]
        out = tracks_sq.copy()
        out[..., 0] = (out[..., 0] - px) / (s + 1e-12)
        out[..., 1] = (out[..., 1] - py) / (s + 1e-12)
        return out

    def _process_single_chunk(self, frames_sq, queries_at_first_frame):
        """
        Internal function: Processes a continuous video chunk.
        Args:
            frames_sq: List of square-padded image arrays.
            queries_at_first_frame: List of [x, y] coordinates in the square resolution.
        """
        video_np = np.stack(frames_sq).astype(np.uint8)
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(self.device)
        
        # Construct queries tensor (Assumes all points start at frame 0 of the chunk)
        # Query format:[b, n, 3] -> (batch_idx, frame_idx, x, y)
        qs = [[0, x, y] for (x, y) in queries_at_first_frame]
        queries_tensor = torch.tensor([qs], device=self.device).float()

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_tracks, pred_vis = self.model(video_tensor, queries=queries_tensor)
        
        tracks = pred_tracks[0].detach().cpu().numpy()
        vis = pred_vis[0].detach().cpu().numpy()
        
        del video_tensor, queries_tensor, pred_tracks, pred_vis
        torch.cuda.empty_cache()
        return tracks, vis
    
    def run_inference(self, image_paths: List[str], init_kpts: List[List[float]], ref_idx: int):
        """
        Executes bidirectional tracking across the entire sequence.
        """
        num_frames = len(image_paths)
        chunk_size = self.cfg.cotracker_chunk_size
        
        # 1. Resolve reference frame index
        if ref_idx < 0: ref_idx = num_frames + ref_idx
        ref_idx = max(0, min(ref_idx, num_frames - 1))

        # 2. Preprocess all frames (Read and letterbox; keep in CPU memory)
        all_frames_sq =[]
        lb_meta = None
        for p in image_paths:
            img = cv2.imread(p)
            sq, meta = self._letterbox(img)
            all_frames_sq.append(sq)
            if lb_meta is None: lb_meta = meta

        # 3. Map initial keypoints to the letterboxed square coordinate system
        s, px, py = lb_meta["scale"], lb_meta["pad_l"], lb_meta["pad_t"]
        init_queries_sq = [[x * s + px, y * s + py] for (x, y) in init_kpts]

        # Prepare result containers
        final_tracks_sq = np.zeros((num_frames, len(init_kpts), 2), dtype=np.float32)
        final_vis = np.zeros((num_frames, len(init_kpts)), dtype=np.float32)

        # ------------------------------------------------------
        # A. Forward Tracking (from ref_idx to the end of the sequence)
        # ------------------------------------------------------
        print(f"║ [CoTracker] Tracking Forward from frame {ref_idx}...")
        curr_queries = init_queries_sq
        for start_f in range(ref_idx, num_frames - 1, chunk_size - 1):
            end_f = min(start_f + chunk_size, num_frames)
            chunk_frames = all_frames_sq[start_f:end_f]
            
            print(f"  → Chunk: {start_f} to {end_f-1}")
            t_chunk, v_chunk = self._process_single_chunk(chunk_frames, curr_queries)
            
            # Populate results
            final_tracks_sq[start_f:end_f] = t_chunk
            final_vis[start_f:end_f] = v_chunk
            
            # Update queries for the next chunk using the last frame's predictions
            curr_queries = t_chunk[-1].tolist()

        # ------------------------------------------------------
        # B. Backward Tracking (from ref_idx down to 0)
        # ------------------------------------------------------
        if ref_idx > 0:
            print(f"║[CoTracker] Tracking Backward from frame {ref_idx}...")
            # We slice and reverse the video segment, effectively turning this into a "forward" pass
            # Example: If ref_idx=50, the sequence becomes[50, 49, 48 ... 0]
            backward_indices = list(range(ref_idx, -1, -1))
            curr_queries = init_queries_sq
            
            for i in range(0, len(backward_indices) - 1, chunk_size - 1):
                idx_chunk = backward_indices[i : i + chunk_size]
                chunk_frames = [all_frames_sq[idx] for idx in idx_chunk]
                
                print(f"  ← Chunk: {idx_chunk[0]} to {idx_chunk[-1]}")
                t_chunk, v_chunk = self._process_single_chunk(chunk_frames, curr_queries)
                
                # Populate results (mapping back to original global indices)
                for local_i, global_idx in enumerate(idx_chunk):
                    final_tracks_sq[global_idx] = t_chunk[local_i]
                    final_vis[global_idx] = v_chunk[local_i]
                
                # Update queries
                curr_queries = t_chunk[-1].tolist()

        # 4. Inverse transformation back to original image coordinates
        tracks = self._unletterbox_points(final_tracks_sq, lb_meta)
        return tracks, final_vis

    def cleanup(self):
        """Release PyTorch model and VRAM."""
        self.model.to("cpu")
        del self.model
        gc.collect()
        if "cuda" in self.device: torch.cuda.empty_cache()


# ==============================================================================
# [Visualization]
# ==============================================================================
class CoTrackerVisualizer:
    """Renders CoTracker trajectories and HUD overlays onto video frames."""
    def __init__(self):
        self.colors =[
            (52, 152, 219), (46, 204, 113), (231, 76, 60),
            (241, 196, 15), (155, 89, 182)
        ]

    def render_frame(self, img_bgr, tracks, vis, frame_idx, trail_len):
        canvas = img_bgr.copy()
        T, N, _ = tracks.shape
        
        # Draw HUD Panel
        draw_glass_rect(canvas, (canvas.shape[1]-200, 10), (canvas.shape[1]-10, 80))
        cv2.putText(canvas, "COTRACKER", (canvas.shape[1]-190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,215,0), 1, cv2.LINE_AA)
        
        visible_cnt = np.sum(vis[frame_idx] > 0)
        cv2.putText(canvas, f"Points: {visible_cnt}/{N}", (canvas.shape[1]-190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        # Draw Point Tracks and Trails
        for n in range(N):
            if vis[frame_idx, n] == 0: continue
            
            col = self.colors[n % len(self.colors)]
            c_bgr = (col[2], col[1], col[0])  # Convert to BGR format

            # Render Historical Trail
            start_t = max(0, frame_idx - trail_len)
            pts =[]
            for t in range(start_t, frame_idx + 1):
                if vis[t, n] > 0:
                    pts.append(tracks[t, n].astype(np.int32))
                else:
                    if len(pts) > 1:
                        cv2.polylines(canvas, [np.array(pts)], False, c_bgr, 2, cv2.LINE_AA)
                    pts =[]
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts)], False, c_bgr, 2, cv2.LINE_AA)

            # Render Tracking Head
            x, y = int(tracks[frame_idx, n, 0]), int(tracks[frame_idx, n, 1])
            cv2.circle(canvas, (x, y), 4, c_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 6, (255,255,255), 1, cv2.LINE_AA)
            
        return canvas


# ==============================================================================
#[Manager] Singleton State Manager
# ==============================================================================
class CoTrackerOfflineManager:
    """
    Manages state across sequence iterations, performing one-time full-sequence 
    inference upon the first call, and rendering visualizations sequentially thereafter.
    """
    def __init__(self):
        self.tracks = None
        self.vis = None
        self.engine = None
        self.visualizer = CoTrackerVisualizer()

        self.inference_time = 0
        self.total_frames = 0
        self.num_points = 0

    def process(self, image_path, cfg_path, frame_idx, all_image_paths, mps_path):
        # 1. INIT Inference (Triggered only on the first frame)
        if self.tracks is None:
            cfg = load_cfg(cfg_path)
            self.cfg = cfg 
            self.engine = CoTrackerOfflineEngine(cfg)
            
            print(f"║ [CoTracker] Actual frames for tracking: {len(all_image_paths)}")

            # Load Multi-Object Keypoints JSON
            kpts_path = os.path.join(mps_path, "preprocess", "kptsselector_results.json")
            if not os.path.exists(kpts_path):
                print(f"║ [Error] Keypoints JSON not found: {kpts_path}")
                return cv2.imread(image_path)
            
            with open(kpts_path, 'r') as f:
                kpts_data = json.load(f)
            
            #[MODIFIED FOR MULTI-OBJECT] Concatenate points from all objects
            objects_dict = kpts_data.get("objects", {})
            init_kpts =[]
            self.obj_slices = {} # Track list slicing indices: { "obj_1": (start, end), ... }
            
            curr_idx = 0
            for obj_key, pts in objects_dict.items():
                init_kpts.extend(pts)
                self.obj_slices[obj_key] = (curr_idx, curr_idx + len(pts))
                curr_idx += len(pts)
            
            self.num_points = len(init_kpts)
            self.total_frames = len(all_image_paths)

            if not init_kpts or not all_image_paths:
                print("║[Error] No keypoints or no image list.")
                return cv2.imread(image_path)

            # Execute full-sequence inference (tracking all object points simultaneously)
            t_inf_start = time.perf_counter()
            self.tracks, self.vis = self.engine.run_inference(all_image_paths, init_kpts, cfg.ref_idx)
            self.inference_time = time.perf_counter() - t_inf_start

            # [MODIFIED FOR MULTI-OBJECT] Split and save results per object
            res = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "info": f"Tracked first {len(all_image_paths)} frames for {len(objects_dict)} objects"
            }
            # Unpack the global track array into individual object arrays based on slice indices
            for obj_key, (start, end) in self.obj_slices.items():
                res[obj_key] = {
                    "tracks": self.tracks[:, start:end, :],
                    "visibility": self.vis[:, start:end]
                }

            res_path = os.path.join(mps_path, "preprocess", "cotracker_results.json")
            with open(res_path, 'w') as f:
                json.dump(res, f, cls=NumpyEncoder, indent=4)
            print(f"║ [CoTracker] Multi-object results saved to {res_path}")

            # Release model VRAM
            self.engine.cleanup()
            self.engine = None 

        # 2. Render Current Frame (Visualizer handles the full global track array)
        img = cv2.imread(image_path)
        if self.tracks is not None and frame_idx < len(self.tracks):
            viz_img = self.visualizer.render_frame(
                img, self.tracks, self.vis, frame_idx, self.cfg.cotracker_viz_trail_len
            )
            return viz_img
        else:
            return img

# ==========================================================

_COTRACKER_INSTANCE = CoTrackerOfflineManager()


def reset_cotracker_offline():
    """Reset the singleton so a new session gets a fresh inference run."""
    global _COTRACKER_INSTANCE
    _COTRACKER_INSTANCE = CoTrackerOfflineManager()


def run_cotracker_offline(image_path, cfg_path, frame_idx, all_image_paths=None, mps_path=None, save_path=None):
    """
    Public entry point for the CoTracker pipeline.
    Note: On the first call (frame_idx=0), `all_image_paths` and `mps_path` MUST be provided
    to trigger the full-sequence offline tracking batch.
    """
    viz = _COTRACKER_INSTANCE.process(image_path, cfg_path, frame_idx, all_image_paths, mps_path)
    
    if save_path:
        cv2.imwrite(save_path, viz)
        
    return viz

def print_cotracker_offline_stats():
    """Prints cumulative performance and throughput statistics."""
    instance = _COTRACKER_INSTANCE
    if instance.total_frames > 0:
        inf_time = instance.inference_time
        total_f = instance.total_frames
        total_p = instance.num_points
        
        print(f"╔" + "═"*60)
        print(f"║ [CoTracker Offline Summary]")
        print(f"║ ⚡ Tracked Frames   : {total_f}")
        print(f"║ ⚡ Tracked Points   : {total_p}")
        print(f"╠" + "═"*60)
        print(f"║ 🚀 Inference Time   : {inf_time:.2f}s (Total Chunk Processing)")
        print(f"║ 🚀 Inference Speed  : {total_f/inf_time:.2f} FPS")
        print(f"╚" + "═"*60)
    else:
        print("║ [CoTracker] No tracking data to summarize.")

# ==========================================================
#[Main] Standalone Execution Entrypoint
# ==========================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to preprocess_cfg.json or CoTracker.yaml")
    args = parser.parse_args()

    # 1. Load configuration
    cfg = load_cfg(args.cfg_path)
    mps_path = args.mps_path

    # 2. Parse Phase JSON to retrieve operational stage index boundaries
    phases_json_path = os.path.join(mps_path, "preprocess", "aria_phases_results.json")
    if not os.path.exists(phases_json_path):
        print(f"║ [Error] Phases JSON not found: {phases_json_path}. Please run AriaPhases first.")
        exit(1)
        
    with open(phases_json_path, 'r') as f: 
        phases_data = json.load(f)
    
    windows_dict = phases_data.get("stage_window_check", {}).get("windows", {})
    
    # Helper to reconstruct the image list matching the preprocess.py logic
    def get_image_list_from_keys(keys):
        allowed_idx =[]
        for key in keys:
            wins = windows_dict.get(key,[])
            for win in wins:
                if not isinstance(win, (list, tuple)) or len(win) != 2:
                    continue
                s, e = int(win[0]), int(win[1])
                r_start, r_end = (s, e) if s <= e else (e, s)
                allowed_idx.extend(range(r_start, r_end + 1))
        allowed_idx = sorted(set(allowed_idx))
        res_paths =[]
        for i in allowed_idx:
            img_path = os.path.join(mps_path, "preprocess", "all_data", f"{i:05d}", "rgb.png")
            if os.path.exists(img_path):
                res_paths.append(img_path)
        return res_paths

    # 3. Construct base list for the MANIP phase (Phase 0)
    manip_image_list = get_image_list_from_keys(["0"])
    
    # Reconstruct the complete all_image_list
    all_data_dir = os.path.join(mps_path, "preprocess", "all_data")
    all_indices = sorted([int(d) for d in os.listdir(all_data_dir) if d.isdigit()])
    all_image_list =[os.path.join(all_data_dir, f"{i:05d}", "rgb.png") for i in all_indices]

    # 4. Reconstruct Object-Centric Frame Selection Logic
    if not manip_image_list:
        print("║ [Error] manip_image_list (Phase 0) is empty. Cannot determine tracking range.")
        exit(1)

    first_manip = manip_image_list[0]
    try:
        split_idx = all_image_list.index(first_manip)
    except ValueError:
        split_idx = 0
    
    # Fetch look-back frames (Nav segment) leading up to manipulation
    max_pre = getattr(cfg, "object_centric_max_frames", 100)
    min_pre = getattr(cfg, "object_centric_min_frames", 30)
    
    pre_frames_count = min(split_idx, max_pre)
    object_centric_image_list = all_image_list[split_idx - pre_frames_count : split_idx]
    
    # Padding logic if pre-manip frames are insufficient
    if len(object_centric_image_list) < min_pre:
        needed = min_pre - len(object_centric_image_list)
        supplementary_frames = manip_image_list[:needed]
        object_centric_image_list = object_centric_image_list + supplementary_frames
        print(f"║ [Info] Pre-manip frames only {pre_frames_count}, supplemented {len(supplementary_frames)} frames from manip.")

    # 5. Compile the final tracking list
    # Logic: Object-Centric (Approach/Nav) + Manip (Interaction)
    cotracker_image_list = object_centric_image_list + manip_image_list
    
    print(f"║ [CoTracker Subprocess] Object-Centric: {len(object_centric_image_list)} frames")
    print(f"║ [CoTracker Subprocess] Manip: {len(manip_image_list)} frames")
    print(f"║ [CoTracker Subprocess] Total sequence to track: {len(cotracker_image_list)} frames")

    # 6. Execute Tracking
    # Note: Calling with frame_idx=0 triggers the Manager's one-time full-sequence batch run
    run_cotracker_offline(
        image_path=cotracker_image_list[0],
        cfg_path=args.cfg_path,
        frame_idx=0,
        all_image_paths=cotracker_image_list,
        mps_path=args.mps_path
    )

    print_cotracker_offline_stats()
    print("║ [CoTracker Subprocess] Done.")