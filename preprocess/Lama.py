# -*- coding: utf-8 -*-
# @FileName: Lama.py

"""
====================================================================================================
LaMa Inpainting Persistent Worker Pipeline (Lama.py)
====================================================================================================

Description:
    This script utilizes the Large Mask Inpainting (LaMa) model via ONNX Runtime to 
    seamlessly remove objects and human arms from Project Aria RGB frames. It employs a 
    persistent background worker architecture to avoid the heavy overhead of loading 
    models into VRAM for every frame.

Core Functionalities:
    1.  ONNX CUDA Acceleration: Dynamically links CUDA libraries to ensure optimal GPU 
        execution for the LaMa ONNX model.
    2.  Mask Preprocessing: Automatically dilates input masks to ensure artifact-free 
        boundaries during the inpainting process.
    3.  Multi-Target Inpainting: Sequentially processes different masks (e.g., removing 
        just the arm vs. removing both the arm and the object) and saves distinct outputs.
    4.  Persistent Subprocess: Manages a long-running background worker that communicates 
        via JSON over stdin/stdout, dramatically increasing frame throughput.
    5.  Visualization: Generates split-screen HUD overlays comparing the masked original 
        image with the inpainted result, alongside latency metrics.

Generated Outputs:
    - [dir]/rgb_WoArm.png: Image with the user's arm inpainted (removed).
    - Returns a rendered visualization frame (HUD) back to the main pipeline.

Technical Specifics:
    - Post-processing: Uses Gaussian blur blending between the high-res original image 
      and the upscaled inpainting output to preserve unmasked background sharpness.
====================================================================================================
"""

import os
import sys
import json
import time
import uuid
import atexit
import tempfile
import subprocess
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from utils.utils_vis import draw_glass_rect, draw_status_bar
from utils.utils_vis import C_CYAN, C_GREEN, C_RED, C_GOLD, C_WHITE, C_GRAY
from utils.utils_io import load_cfg

def setup_cuda_paths():
    """
    Dynamically locates and appends standard NVIDIA CUDA libraries to the LD_LIBRARY_PATH.
    This ensures that ONNX Runtime can successfully initialize the CUDAExecutionProvider 
    on Linux environments where paths might not be explicitly set.
    """
    libs_to_find =['nvidia.cublas', 'nvidia.cudnn', 'nvidia.cuda_runtime', 'nvidia.cufft', 'nvidia.curand']
    lib_paths =[]
    for lib_name in libs_to_find:
        try:
            module = __import__(lib_name, fromlist=['lib'])
            lib_dir = os.path.join(module.__path__[0], 'lib')
            if os.path.isdir(lib_dir): lib_paths.append(lib_dir)
        except ImportError: pass

    if sys.platform.startswith('linux'):
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld = ':'.join(lib_paths +[current_ld])
        os.environ['LD_LIBRARY_PATH'] = new_ld

# Execute CUDA path setup immediately upon module import
setup_cuda_paths()


# ==============================================================================
# [Engine] LaMa Inpainting Engine
# ==============================================================================
class LamaEngine:
    """
    Core engine handling ONNX Runtime sessions, pre-processing (dilation & resizing), 
    and post-processing (blending) for the LaMa model.
    """
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.cfg = load_cfg(cfg_path)

        print(f"║ [System] Initializing LaMa Inpainter on GPU...")
        
        # Download or retrieve cached model from HuggingFace Hub
        model_path = hf_hub_download(repo_id=self.cfg.lama_repo_id, filename=self.cfg.lama_filename)
        
        # Configure ONNX Runtime Providers for optimal GPU execution
        providers =[
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        if 'CUDAExecutionProvider' not in self.session.get_providers():
            print("║ [Warning] CUDA not available for LaMa, falling back to CPU!")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Executes the inpainting process on a single image-mask pair.
        
        Args:
            image (np.ndarray): Original BGR image.
            mask (np.ndarray): Grayscale mask indicating areas to remove.
            
        Returns:
            np.ndarray: The inpainted BGR image.
        """
        h, w = image.shape[:2]
        sz = self.cfg.lama_input_size

        # 1. Mask Dilation
        # Expand the mask slightly to ensure object boundaries and shadows are fully covered
        k = self.cfg.lama_mask_dilation
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 2. Preprocess Input Arrays
        # Resize image to network input size and normalize to[0, 1]
        img_res = cv2.resize(image, (sz, sz)).astype(np.float32) / 255.0
        img_res = np.transpose(img_res, (2, 0, 1))[None, ...]
        
        # Resize mask using nearest neighbor to preserve hard edges, format to [0, 1]
        mask_res = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        mask_res = (mask_res > 0.5).astype(np.float32)[None, None, ...]

        # 3. ONNX Inference
        inputs = {
            self.session.get_inputs()[0].name: img_res,
            self.session.get_inputs()[1].name: mask_res
        }
        output = self.session.run(None, inputs)[0]

        # 4. Postprocess & Blending
        output = np.squeeze(output, 0).transpose((1, 2, 0))
        output = np.clip(output * 255 if output.max() <= 1.1 else output, 0, 255).astype(np.uint8)
        
        # Upscale inpainted result back to original resolution
        output_high_res = cv2.resize(output, (w, h))

        # Soft blending: Use Gaussian blur on the mask to seamlessly blend 
        # the inpainted regions into the original high-resolution background.
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0)[..., None]
        final = output_high_res * mask_blur + image * (1 - mask_blur)

        return final.astype(np.uint8)


# ==============================================================================
# [Manager] High-level Wrapper & Visualization
# ==============================================================================
class Lama:
    """Manages file I/O, consecutive mask processing, and visualization rendering."""
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.cfg = load_cfg(cfg_path)
        self.engine = LamaEngine(self.cfg_path)

    def _render_vis(self, orig: np.ndarray, result: np.ndarray, mask: np.ndarray, latency: float, idx: int) -> np.ndarray:
        """Renders the split-screen diagnostic HUD."""
        # Left Panel: Original Image + Mask Overlay
        left_vis = orig.copy()
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_vis[:,:,:2] = 0  # Make mask purely Red
        left_vis = cv2.addWeighted(left_vis, 1.0, mask_vis, 0.5, 0)
        
        # Right Panel: Inpainted Result + HUD Metrics
        right_vis = result.copy()
        draw_glass_rect(right_vis, (10, 10), (350, 180))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(right_vis, "LAMA INPAINTING ENGINE", (20, 30), font, 0.5, C_GOLD, 1, cv2.LINE_AA)
        cv2.putText(right_vis, f"FRAME: {idx:05d}", (20, 60), font, 0.4, C_WHITE, 1, cv2.LINE_AA)
        
        fps = 1.0 / latency if latency > 0 else 0
        cv2.putText(right_vis, f"LATENCY: {latency*1000:.1f}ms", (20, 90), font, 0.4, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(right_vis, f"THROUGHPUT: {fps:.1f} FPS", (20, 110), font, 0.4, C_CYAN, 1, cv2.LINE_AA)
        
        draw_status_bar(right_vis, (20, 140), 300, min(latency/0.2, 1.0), 1.0, f"Load: {latency*1000:.0f}ms", C_GREEN)
        
        return cv2.hconcat([left_vis, right_vis])

    def process_and_save(self, image_path: str, frame_idx: int = 0) -> np.ndarray:
        """
        Loads the image and systematically applies inpainting for standard masks 
        (e.g., removing arm, removing arm+object).
        """
        if not os.path.exists(image_path): 
            return None
        img = cv2.imread(image_path)

        # Only generate rgb_WoArm.png (arm-removed background).
        # rgb_WoArmObj.png is no longer needed — saves ~50% Lama runtime.
        mask_name = "mask_arm"
        mask_path = os.path.join(os.path.dirname(image_path), f"{mask_name}.png")

        if not os.path.exists(mask_path):
            return cv2.hconcat([img, img])

        mask = cv2.imread(mask_path, 0)

        t_start = time.perf_counter()
        res = self.engine.inpaint(img, mask)
        latency = time.perf_counter() - t_start

        save_path = os.path.join(os.path.dirname(image_path), "rgb_WoArm.png")
        cv2.imwrite(save_path, res)

        vis_img = self._render_vis(img, res, mask, latency, frame_idx)
        return vis_img


# ==============================================================================
# [Subprocess Worker] Persistent Worker Management
# ==============================================================================
_LAMA_WORKER = None

class _LamaWorker:
    """Manages the persistent LaMa background process to ensure rapid multi-frame inference."""
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.total_frames = 0
        self.total_time = 0
        
        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        print(f"║ [LAMA] Launching persistent worker...")
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "preprocess.Lama", "--worker", "--cfg_path", cfg_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1, env=env
        )

        # Await initialization readiness from subprocess
        while True:
            line = self.proc.stdout.readline().strip()
            if line == "READY": break
            if line.startswith("{") and not json.loads(line).get("ok", True):
                raise RuntimeError(f"Lama worker failed: {line}")

    def process(self, image_path: str, frame_idx: int) -> np.ndarray:
        """Sends a JSON request to the worker and reads back the rendered visualization."""
        t_start = time.perf_counter()

        tmp_vis = os.path.join(tempfile.gettempdir(), f"lama_vis_{uuid.uuid4().hex}.png")
        req = {"cmd": "process", "image_path": image_path, "frame_idx": frame_idx, "vis_path": tmp_vis}
        
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        
        line = self.proc.stdout.readline()
        if not line: raise RuntimeError("Lama worker died.")
        
        resp = json.loads(line)
        if not resp.get("ok", False): raise RuntimeError(resp.get("error"))
        
        vis = cv2.imread(tmp_vis)
        if os.path.exists(tmp_vis): os.remove(tmp_vis)

        duration = time.perf_counter() - t_start
        self.total_time += duration
        self.total_frames += 1

        return vis if vis is not None else np.zeros((640, 1280, 3), dtype=np.uint8)

    def close(self):
        """Safely terminates the persistent subprocess."""
        try:
            self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except: self.proc.kill()


# ==============================================================================
# [Public Interface] API for Pipeline Integration
# ==============================================================================
def run_lama(image_path: str, cfg_path: str, frame_idx: int = 0) -> np.ndarray:
    """Singleton entry point for triggering the LaMa worker from the main pipeline."""
    global _LAMA_WORKER
    if _LAMA_WORKER is None:
        _LAMA_WORKER = _LamaWorker(cfg_path)
        atexit.register(_LAMA_WORKER.close)

    return _LAMA_WORKER.process(image_path, frame_idx)

def print_lama_stats():
    """Prints cumulative performance and throughput statistics upon pipeline completion."""
    global _LAMA_WORKER
    if _LAMA_WORKER is not None and _LAMA_WORKER.total_frames > 0:
        total_f = _LAMA_WORKER.total_frames
        total_t = _LAMA_WORKER.total_time
        avg_fps = total_f / total_t
        avg_ms = (total_t / total_f) * 1000
        
        print(f"╔" + "═"*60)
        print(f"║ [LAMA Pipeline Summary]")
        print(f"║ ⚡ Processed Frames : {total_f}")
        print(f"║ ⚡ Total Time      : {total_t:.2f}s")
        print(f"║ ⚡ Average Latency : {avg_ms:.1f}ms / frame")
        print(f"║ ⚡ Average FPS     : {avg_fps:.2f} FPS")
        print(f"╚" + "═"*60)
    else:
        print("║[LAMA] No frames were processed.")


def close_lama_worker():
    """Close the persistent worker and release its runtime resources."""
    global _LAMA_WORKER
    if _LAMA_WORKER is not None:
        _LAMA_WORKER.close()
        _LAMA_WORKER = None


# ==============================================================================
# [Worker Executable] Subprocess Loop
# ==============================================================================
def _worker_loop(cfg_path: str):
    """The internal blocking loop executed by the persistent subprocess."""
    def _send(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    try:
        lama_model = Lama(cfg_path)
        sys.stdout.write("READY\n")
        sys.stdout.flush()
    except Exception as e:
        _send({"ok": False, "error": str(e)})
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "exit": break
            
            vis = lama_model.process_and_save(req["image_path"], req["frame_idx"])
            cv2.imwrite(req["vis_path"], vis)
            
            _send({"ok": True})
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
