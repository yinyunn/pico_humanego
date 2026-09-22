# -*- coding: utf-8 -*-
# @FileName: DINOSAM.py

"""
====================================================================================================
Project Aria DINO-SAM2 Segmentation Pipeline (DINOSAM.py)
====================================================================================================

Description:
    This script processes RGB frames using Grounding DINO for object detection and SAM2 
    for mask generation. It supports multi-object prompts and generates combined masks 
    for downstream tasks.

Technical Specifics:
    - Grounding DINO: Text-to-Box detection.
    - SAM2: Box-to-Mask segmentation.
====================================================================================================
"""

import os
import sys
import subprocess
import cv2
import torch
import numpy as np
import gc
import argparse
from tqdm import tqdm
from PIL import Image
import time
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.utils_media import create_video_from_frames
from utils.utils_vis import draw_glass_rect, draw_status_bar, C_CYAN, C_GREEN, C_RED, C_GOLD, C_WHITE, C_GRAY
from utils.utils_io import load_cfg


def run_dinosam_subprocess(mps_path: str, cfg_path: str, export_video: bool = True, export_gif: bool = True):
    """
    Utility function to run this script as a subprocess.
    """
    cmd = [
        sys.executable, "-m", "preprocess.DINOSAM",
        "--input", mps_path,
        "--cfg_path", cfg_path,
    ]
    if not export_video:
        cmd.append("--no-video")
    if not export_gif:
        cmd.append("--no-gif")

    env = os.environ.copy()
    print("║ [Subprocess] Running:", " ".join(cmd))
    r = subprocess.run(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"DINOSAM subprocess failed with return code {r.returncode}")

class DINOSAMEngine:
    """Model Engine for DINO and SAM2."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"║ [System] Initializing Models on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.cfg.dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.dino_model_id).to(self.device)
        
        ckpt_path = hf_hub_download(repo_id=self.cfg.sam2_repo_id, filename=self.cfg.sam2_checkpoint_name)
        self.predictor = SAM2ImagePredictor(build_sam2(self.cfg.sam2_config, ckpt_path, device=self.device))

    def predict_frame_internal(self, image_np, text_prompt, max_boxes=None):
        """Internal prediction function using pre-set image in predictor."""
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        W, H = image_pil.size
        
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        logits = outputs.logits.sigmoid()[0]
        boxes = outputs.pred_boxes[0]
        
        mask_filter = logits.max(-1)[0] > self.cfg.box_threshold
        filtered_logits = logits[mask_filter]
        filtered_boxes = boxes[mask_filter]
        
        if len(filtered_boxes) == 0:
            return None, 0.0, None, None

        box_scores = filtered_logits.max(-1)[0]
        if max_boxes is not None and int(max_boxes) > 0 and len(filtered_boxes) > int(max_boxes):
            keep = torch.topk(box_scores, k=int(max_boxes), largest=True).indices
            filtered_boxes = filtered_boxes[keep]
            box_scores = box_scores[keep]

        confidences = box_scores.cpu().numpy()
        avg_conf = np.mean(confidences)
        
        pixel_boxes = filtered_boxes * torch.Tensor([W, H, W, H]).to(self.device)
        cx, cy, w, h = pixel_boxes.unbind(-1)
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        input_boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()

        masks, _, _ = self.predictor.predict(box=input_boxes, multimask_output=False)
        
        combined_mask = np.any(masks.squeeze(), axis=0) if masks.ndim > 3 else masks.squeeze()
        if combined_mask.ndim > 2:
             combined_mask = np.any(combined_mask, axis=0)

        return (combined_mask.astype(np.uint8) * 255), avg_conf, input_boxes, confidences

    def cleanup(self):
        """Release resources."""
        print("║ [Cleanup] Releasing Engine Resources...")
        if hasattr(self, 'dino_model'): self.dino_model.to("cpu"); del self.dino_model
        if hasattr(self, 'predictor'): self.predictor.model.to("cpu"); del self.predictor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

class DINOSAM:
    """Inference and Visualization Manager."""
    def __init__(self, cfg_path):
        self.cfg = load_cfg(cfg_path)
        self.engine = DINOSAMEngine(self.cfg)

    def _render_vis(self, img, mask, boxes, box_confs, avg_conf, latency, text_prompt):
        """Render side-by-side visualization."""
        left_vis = img.copy()
        num_objects = 0
        if boxes is not None:
            num_objects = len(boxes)
            for box, b_conf in zip(boxes, box_confs):
                bx1, by1, bx2, by2 = box.astype(int)
                cv2.rectangle(left_vis, (bx1, by1), (bx2, by2), C_GREEN, 2)
                cv2.putText(left_vis, f"{b_conf:.2f}", (bx1, by1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GREEN, 1, cv2.LINE_AA)
        
        mask_vis = mask if mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        heatmap_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        
        draw_glass_rect(heatmap_vis, (10, 10), (350, 240))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        header_col = C_GREEN if avg_conf > self.cfg.box_threshold else C_GRAY
        cv2.putText(heatmap_vis, "DINO-SAM2 ANALYZER", (20, 30), font, 0.5, C_GOLD, 1, cv2.LINE_AA)
        status = "[SIGNAL ACTIVE]" if avg_conf > 0 else "[SEARCHING...]"
        cv2.putText(heatmap_vis, status, (20, 50), font, 0.4, header_col, 1, cv2.LINE_AA)
        cv2.putText(heatmap_vis, f"OBJECTS: {num_objects}", (20, 120), font, 0.5, C_GOLD, 1, cv2.LINE_AA)
        
        draw_status_bar(heatmap_vis, (20, 170), 300, avg_conf, 1.0, f"Conf: {avg_conf:.2f}", C_CYAN)
        cv2.putText(heatmap_vis, f"LATENCY: {latency*1000:.1f}ms", (20, 210), font, 0.4, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(heatmap_vis, f"PROMPT: {text_prompt[:20]}...", (20, 225), font, 0.3, C_GRAY, 1, cv2.LINE_AA)

        return cv2.hconcat([left_vis, heatmap_vis])


    def process_single(self, img, prompt, save_path=None, max_boxes=None):
        """Process a single image or numpy array."""
        if isinstance(img, str):
            image_np = cv2.imread(img)
        else:
            image_np = img

        if image_np is None: return None

        self.engine.predictor.set_image(image_np)
        t_start = time.perf_counter()
        mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(
            image_np, prompt, max_boxes=max_boxes
        )
        
        if self.engine.device == "cuda": torch.cuda.synchronize()
        latency = time.perf_counter() - t_start

        mask_out = mask if mask is not None else np.zeros(image_np.shape[:2], dtype=np.uint8)
        if save_path: cv2.imwrite(save_path, mask_out)
        return mask_out
    

    def process_and_save(self, image_path, prompts_dict):
        """Process multiple prompts and save individual/combined masks."""
        if not os.path.exists(image_path): return None, 0
        
        img = cv2.imread(image_path)
        base_dir = os.path.dirname(image_path)
        combined_all_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        self.engine.predictor.set_image(img)
        last_vis = None
        prompts_count = 0 

        for key, prompt in prompts_dict.items():
            if not prompt.strip(): continue
            prompts_count += 1 

            t_start = time.perf_counter()
            max_boxes_cfg = getattr(self.cfg, "max_boxes_per_prompt", {})
            max_boxes = max_boxes_cfg.get(key) if hasattr(max_boxes_cfg, "get") else None
            mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(
                img, prompt, max_boxes=max_boxes
            )
            
            if self.engine.device == "cuda": torch.cuda.synchronize()
            latency = time.perf_counter() - t_start

            mask_out = mask if mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.imwrite(os.path.join(base_dir, f"mask_{key}.png"), mask_out)
            combined_all_mask = cv2.bitwise_or(combined_all_mask, mask_out)

            last_vis = self._render_vis(img, combined_all_mask, boxes, box_confs, avg_conf, latency, prompt)
            
        cv2.imwrite(os.path.join(base_dir, "mask_arm_and_obj.png"), combined_all_mask)
        return last_vis, prompts_count

    def cleanup(self):
        self.engine.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="MPS path OR Image path")
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/DINOSAM.yaml")
    parser.add_argument("--save_path", type=str, default=None, help="Explicit save path (for single img) or Video output dir")
    parser.add_argument("--no-video", action="store_false", dest="export_video", help="Disable video export")
    parser.add_argument("--no-gif", action="store_false", dest="export_gif", help="Disable GIF export")
    parser.set_defaults(export_video=True, export_gif=True)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg_path)
    
    dinosam = DINOSAM(args.cfg_path)

    t_global_start = time.perf_counter()
    processed_count = 0

    frames_all_dinosam = []
    subdirs = sorted([d for d in os.listdir(os.path.join(args.input, "preprocess", "all_data")) if d.isdigit()])
    image_list = [os.path.join(args.input, "preprocess", "all_data", d, "rgb.png") for d in subdirs]
    print(f"║ [Info] DINOSAM Processing {len(image_list)} frames...")
    for i, img_path in enumerate(tqdm(image_list, desc="DINO+SAM2 Pipeline")):
        if not os.path.exists(img_path): continue
        vis_dinosam = dinosam.process_and_save(img_path, cfg.dinosam_prompt)
        processed_count += 1
        if args.export_video:
            frames_all_dinosam.append(vis_dinosam)

    if args.export_video:
        create_video_from_frames(
            frames=frames_all_dinosam,
            save_path=os.path.join(args.input, "preprocess", "vis", "mask_vis.mp4"),
            fps=cfg.fps,
            export_gif=args.export_gif
        )
    t_global_end = time.perf_counter()
    if processed_count > 0:
        total_t = t_global_end - t_global_start
        print(f"╔" + "═"*60)
        print(f"║ [DINOSAM Standalone Summary]")
        print(f"║ ⚡ Processed Frames : {processed_count}")
        print(f"║ ⚡ Total Time      : {total_t:.2f}s")
        print(f"║ ⚡ Average FPS     : {processed_count/total_t:.2f} FPS")
        print(f"╚" + "═"*60)

    dinosam.cleanup()
