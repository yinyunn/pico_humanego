# -*- coding: utf-8 -*-
# @FileName: KptsSelector.py

"""
====================================================================================================
Project Aria Keypoints Selection Pipeline (KptsSelector.py)
====================================================================================================

Description:
    This script extracts robust, equidistant keypoints along the contour of segmented masks. 
    It applies morphological operations for noise reduction and leverages contour-based 
    sampling to ensure rotation-invariant point selection for downstream tracking tasks 
    (e.g., CoTracker).

Core Functionalities:
    1.  Mask Preprocessing: Applies morphological closing to fill holes and erosion to remove 
        noise or generate an inner safety margin.
    2.  Contour Extraction: Identifies the external contour of the largest segmented object.
    3.  Equidistant Sampling: Interpolates along the contour arc length to sample a fixed 
        number of equidistant tracking keypoints.
    4.  Visualization: Renders colorful overlays of the selected keypoints onto RGB frames.

Generated Outputs:
    - Visualization Image: Saved to [save_img_path] (e.g., *_vis.png).
    - Returns a list of (x, y) tuples representing the target keypoints.

Technical Specifics:
    - Geometric Stability: The first keypoint is deterministically chosen as the top-most 
      pixel to ensure temporal and spatial consistency.
    - Arc-Length Interpolation: Ensures uniform spacing regardless of the object's shape.
====================================================================================================
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.utils_io import load_cfg


# ==============================================================================
# [Engine] Robust Contour-based Engine
# ==============================================================================
class KeypointsSelectorEngine:
    """
    Engine for processing binary masks and extracting robust equidistant keypoints.
    Provides morphological cleaning and deterministic contour sampling algorithms.
    """

    def __init__(self, cfg_path: str):
        """
        Initializes the KeypointsSelectorEngine with configuration parameters.
        """
        self.cfg = load_cfg(cfg_path)

    def _clamp_odd(self, k: int) -> int:
        """
        Ensures a kernel size is an odd integer, as required by OpenCV morphological operations.

        Args:
            k (int): Input kernel size.

        Returns:
            int: An odd integer kernel size.
        """
        k = int(k)
        return k if (k % 2 == 1) else (k + 1)

    def _morph_kernel(self, k: int) -> np.ndarray:
        """
        Generates an elliptical structuring element for morphological operations.

        Args:
            k (int): Kernel size.

        Returns:
            np.ndarray: The generated structuring element.
        """
        k = self._clamp_odd(k)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def preprocess_mask(self, mask_255: np.ndarray) -> np.ndarray:
        """
        Applies sequential morphological operations to clean the binary mask.
        Typically involves Closing (to fill internal holes) followed by Erosion (to remove boundary noise).

        Args:
            mask_255 (np.ndarray): Input binary mask (values 0 or 255).

        Returns:
            np.ndarray: Cleaned binary mask.
        """
        m = (mask_255 > 127).astype(np.uint8) * 255
        
        if self.cfg.kpts_patch_close:
            k = self._morph_kernel(self.cfg.kpts_close_kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        if self.cfg.kpts_patch_erode:
            k = self._morph_kernel(self.cfg.kpts_erode_kernel)
            m = cv2.erode(m, k, iterations=int(self.cfg.kpts_erode_iters))
        return m

    def select_points(self, mask_raw: np.ndarray) -> List[Tuple[int, int]]:
        """
        Executes robust equidistant contour sampling on the provided mask.
        The algorithm ensures rotation invariance and handles both circular (plates) 
        and elongated objects flawlessly.

        Args:
            mask_raw (np.ndarray): Raw grayscale/binary mask image.

        Returns:
            List[Tuple[int, int]]: A list of (x, y) coordinates for the selected keypoints.
        """
        # 1. Preprocess mask (Close -> Erode)
        mask_pp = self.preprocess_mask(mask_raw)
        
        # 2. Erode to get inner safety margin (Crucial for CoTracker stability)
        if self.cfg.kpts_use_inner_edge:
            k = self._morph_kernel(self.cfg.kpts_edge_erode_kernel)
            mask_eroded = cv2.erode(mask_pp, k, iterations=1)
            # Only use eroded mask if it didn't disappear completely
            if np.count_nonzero(mask_eroded) > self.cfg.kpts_min_edge_pixels:
                mask_pp = mask_eroded

        # 3. Find External Contour
        contours, _ = cv2.findContours(mask_pp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            print("║ [KptsSelector] Warning: No contours found.")
            return[]
            
        # Extract the largest contour
        c = max(contours, key=cv2.contourArea).squeeze(1) # Shape: (N_pixels, 2)
        
        if c.ndim < 2 or c.shape[0] < 10:
            print("║[KptsSelector] Warning: Contour area too small.")
            return[]
            
        # 4. Deterministic Starting Point (Top-most pixel)
        # Sort by Y ascending (primary), X ascending (secondary)
        # This ensures the first point is ALWAYS the geometric top of the object.
        top_idx = np.lexsort((c[:, 0], c[:, 1]))[0]
        c = np.roll(c, -top_idx, axis=0) # Shift array so top_idx is at index 0
        
        # 5. Calculate Arc Lengths
        diffs = np.diff(c, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        dist_close = np.linalg.norm(c[-1] - c[0]) # Connect last point to first
        dists = np.append(dists, dist_close)
        
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        total_len = cum_dists[-1]
        
        # We want to maintain the same number of points as the old banding logic
        N_points = int(self.cfg.kpts_n_bands) * 2 
        
        # 6. Equidistant Interpolation along the contour
        target_dists = np.linspace(0, total_len, N_points, endpoint=False)
        selected_points =[]
        
        for td in target_dists:
            # Find the segment containing the target distance
            idx = np.searchsorted(cum_dists, td) - 1
            idx = np.clip(idx, 0, len(c)-1)
            
            if idx == len(c) - 1:
                p1, p2 = c[idx], c[0]
                d1, d2 = cum_dists[idx], total_len
            else:
                p1, p2 = c[idx], c[idx+1]
                d1, d2 = cum_dists[idx], cum_dists[idx+1]
                
            # Linear interpolation between the two adjacent contour pixels
            if d2 == d1:
                pt = p1
            else:
                alpha = (td - d1) / (d2 - d1)
                pt = p1 + alpha * (p2 - p1)
                
            selected_points.append((int(round(pt[0])), int(round(pt[1]))))
            
        return selected_points


# ==============================================================================
# [Visualization] 
# ==============================================================================
def draw_points(img_rgb: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Visualizes the selected keypoints as colored circles with numeric labels on an RGB image.

    Args:
        img_rgb (np.ndarray): Background RGB image or mask representation.
        points (List[Tuple[int, int]]): List of (x, y) coordinates to draw.

    Returns:
        np.ndarray: The annotated image.
    """
    colors =[
        (52, 152, 219), (46, 204, 113), (231, 76, 60),
        (241, 196, 15), (155, 89, 182), (52, 73, 94),
        (26, 188, 156), (230, 126, 34), (149, 165, 166),
        (192, 57, 43), (39, 174, 96), (41, 128, 185),
    ]
    out = img_rgb.copy()
    for i, (x, y) in enumerate(points):
        color = colors[i % len(colors)]
        c_bgr = (color[2], color[1], color[0]) 
        
        cv2.circle(out, (x, y), 5, c_bgr, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(out, str(i), (x+10, y-10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, c_bgr, 2, cv2.LINE_AA)
    return out


# ==============================================================================
# [Interface] 
# ==============================================================================
def run_kptsselector(cfg_path: str, mask_path: str, save_img_path: str, rgb_path: Optional[str] = None):
    """
    High-level interface to run the keypoints selection pipeline on a given mask file.
    Saves a visualization image and returns the selected points.

    Args:
        cfg_path (str): Path to the configuration YAML/JSON file.
        mask_path (str): Path to the input binary mask image.
        save_img_path (str): Output path to save the visualization image.
        rgb_path (Optional[str]): Path to the original RGB image for visualization background. 
                                  Defaults to None.

    Returns:
        Optional[List[Tuple[int, int]]]: A list of selected keypoint coordinates, or None if failed.
    """
    engine = KeypointsSelectorEngine(cfg_path)

    if not os.path.exists(mask_path):
        print(f"║ [KptsSelector] Error: Mask not found: {mask_path}")
        return None
    
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        print(f"║ [KptsSelector] Error: Failed to read mask: {mask_path}")
        return None

    kpts = engine.select_points(mask_raw)
    
    if not kpts:
        print("║ [KptsSelector] Warning: No keypoints selected.")
        return None

    if rgb_path and os.path.exists(rgb_path):
        viz_bg = cv2.imread(rgb_path)
    else:
        h, w = mask_raw.shape
        viz_bg = np.zeros((h, w, 3), dtype=np.uint8)
        viz_bg[mask_raw > 0] = (50, 50, 50)
    
    viz_img = draw_points(viz_bg, kpts)
    cv2.imwrite(save_img_path, viz_img)
    print(f"║ [KptsSelector] Keypoints Viz saved: {save_img_path}")
    return kpts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--rgb", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess_cfg.json")
    args = parser.parse_args()
    
    run_kptsselector(
        args.cfg_path, 
        args.mask_path, 
        "test_kpts_vis.png", "test_kpts.json", rgb_path=args.rgb
    )