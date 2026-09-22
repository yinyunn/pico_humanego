# -*- coding: utf-8 -*-
# @FileName: DINOSAMOps.py

"""
====================================================================================================
Project Aria DINO-SAM2 Persistent Worker Operations (DINOSAMOps.py)
====================================================================================================

Description:
    This module manages a persistent worker subprocess for DINO-SAM2 inference. It utilizes 
    a request-response loop via stdin/stdout to avoid the overhead of re-initializing heavy 
    DINO and SAM2 models for every frame.

Core Functionalities:
    1. Persistent Subprocess Management: Launches and monitors a worker process.
    2. Inter-process Communication: Exchanges JSON-serialized requests and results.
    3. Performance Monitoring: Tracks throughput (FPS), model efficiency, and latency.
    4. Memory Optimization: Configures PyTorch VRAM allocation for background execution.

Technical Specifics:
    - Communication: JSON over non-buffered stdin/stdout.
    - Resource Handling: Automatic cleanup via atexit.
====================================================================================================
"""

import os
import sys
import json
import uuid
import time
import atexit
import tempfile
import subprocess
import cv2
import numpy as np

from utils.utils_io import load_cfg

# =========================================================
# Persistent DINOSAM Worker
# =========================================================

_WORKER = None

class _DinoSamWorker:
    """
    Manages a long-running subprocess to perform GPU-intensive segmentation.
    """
    def __init__(self, cfg_path: str, force_cpu: bool = False):
        self.cfg_path = cfg_path
        env = os.environ.copy()

        # Cumulative statistics
        self.total_frames = 0  
        self.total_prompts = 0
        self.total_time = 0    
        
        # Optimize VRAM allocation to prevent sub-processes from hogging all memory
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""

        print(f"║ [DINOSAM] Launching persistent worker subprocess...")
        
        # Use -u to ensure output is unbuffered
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "preprocess.DINOSAMOps", "--worker", "--cfg_path", cfg_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr, # Keep error output visible in main terminal
            text=True,
            bufsize=1,
            env=env,
        )

        # Wait for READY or handle initialization errors
        init_error = None
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                break
            
            line = line.strip()
            # Attempt to parse as JSON in case sub-process returns an error packet
            if line.startswith("{"):
                try:
                    resp = json.loads(line)
                    if not resp.get("ok", True):
                        init_error = resp.get("error", "Unknown init error")
                        break
                except:
                    pass
            
            if line == "READY":
                print("║ [DINOSAM] Worker subprocess is READY.")
                return 
        
        # Termination logic if initialization fails
        self.proc.terminate()
        error_msg = init_error if init_error else "Worker process exited unexpectedly during initialization."
        raise RuntimeError(f"DINOSAM worker failed: {error_msg}")

    def process(self, image_path, prompts_dict):
        """
        Sends a processing request to the persistent worker.
        """
        t_start = time.perf_counter()

        # Temporary path for visualization output
        tmp_vis = os.path.join(tempfile.gettempdir(), f"dino_vis_{uuid.uuid4().hex}.png")

        req = {
            "cmd": "process",
            "image_path": image_path,
            "prompts_dict": prompts_dict,
            "vis_path": tmp_vis,
        }

        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Worker process died during processing.")
            
            resp = json.loads(line)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error", "DINOSAM worker error"))

            # Read visualization result back
            vis = cv2.imread(tmp_vis)

            duration = time.perf_counter() - t_start
            self.total_time += duration
            self.total_frames += 1
            self.total_prompts += resp.get("num_prompts", 0)

            if vis is None:
                # Return a black placeholder if image reading fails
                vis = np.zeros((640, 1280, 3), dtype=np.uint8)
            
            return vis
        
        finally:
            if os.path.exists(tmp_vis):
                try: os.remove(tmp_vis)
                except: pass

    def close(self):
        """
        Signals the worker to exit and cleans up resources.
        """
        print("║ [DINOSAM] Closing worker subprocess...")
        try:
            self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except:
            if hasattr(self, 'proc'):
                self.proc.kill()


# =========================================================
# Public APIs
# =========================================================

def _get_worker(cfg_path: str):
    """
    Singleton factory for the persistent worker.
    """
    global _WORKER
    if _WORKER is None:
        _WORKER = _DinoSamWorker(cfg_path)
        atexit.register(_WORKER.close)
    return _WORKER


def run_dinosam(*, cfg_path, image_path):
    """
    High-level API for processing multiple prompts on a single image.
    """
    worker = _get_worker(cfg_path)
    prompts_dict = load_cfg(cfg_path).dinosam_prompt
    return worker.process(image_path, prompts_dict)


def print_dinosam_stats():
    """
    Displays a formatted performance report for the segmentation pipeline.
    """
    global _WORKER
    if _WORKER is not None and _WORKER.total_frames > 0:
        total_f = _WORKER.total_frames
        total_p = _WORKER.total_prompts
        total_t = _WORKER.total_time
        
        # Calculate performance metrics
        avg_frame_fps = total_f / total_t
        avg_mask_fps = total_p / total_t 
        ms_per_mask = (total_t / total_p) * 1000
        
        print(f"╔" + "═"*60)
        print(f"║ [DINOSAM Pipeline Summary]")
        print(f"║ ⚡ Total Frames      : {total_f}")
        print(f"║ ⚡ Total Masks/Prompts: {total_p} (Avg: {total_p/total_f:.1f} per frame)")
        print(f"║ ⚡ Total Time        : {total_t:.2f}s")
        print(f"╠" + "═"*60)
        print(f"║ 🚀 Overall Throughput : {avg_frame_fps:.2f} Frames/sec")
        print(f"║ 🚀 Model Efficiency (FPS)  : {avg_mask_fps:.2f} FPS")
        print(f"║ 🚀 Avg Latency/Mask  : {ms_per_mask:.1f} ms")
        print(f"╚" + "═"*60)


def close_dinosam_worker():
    """Close the persistent worker and release its GPU memory between stages."""
    global _WORKER
    if _WORKER is not None:
        _WORKER.close()
        _WORKER = None

# =========================================================
# Worker process entrypoint (The code below runs in the SUBPROCESS)
# =========================================================

def _worker_loop(cfg_path: str):
    """
    Main loop for the worker subprocess.
    """
    # Defer heavy imports until the subprocess starts
    import torch
    import sys
    import json
    
    def _send(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    try:
        from preprocess.DINOSAM import DINOSAM
        from utils.utils_io import load_cfg
        
        # Clean VRAM cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        cfg = load_cfg(cfg_path)
        dino = DINOSAM(cfg_path)
        
        # Signal the parent process that initialization is finished
        sys.stdout.write("READY\n")
        sys.stdout.flush()
        
    except Exception as e:
        # If initialization fails, send JSON error and exit
        _send({"ok": False, "error": f"Worker init failed: {str(e)}"})
        sys.exit(1)

    # Listen for processing requests from parent via stdin
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "exit":
                break

            vis, p_count = dino.process_and_save(
                image_path=req["image_path"],
                prompts_dict=req["prompts_dict"],
            )

            if vis is None:
                # Fallback to empty image if rendering fails
                vis = np.zeros((640, 1280, 3), dtype=np.uint8)

            cv2.imwrite(req["vis_path"], vis)
            _send({"ok": True, "num_prompts": p_count}) 

        except Exception as e:
            _send({"ok": False, "error": str(e)})

    sys.exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cfg_path", type=str, required=True)
    args = parser.parse_args()

    if args.worker:
        _worker_loop(args.cfg_path)
