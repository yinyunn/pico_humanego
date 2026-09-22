"""Example adapter for optional reprojection checks.

Copy this file and edit it for your PICO/Aria JSON schema.
The core auditor will NOT guess coordinate frames.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def load_world_points(frame_dir: Path, training_json: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return Nx3 points in the SAME world frame as metadata.c2w.

    Example only. Replace with explicit parsing of your known PICO hand-joint file.
    Return None on frames where points are unavailable.
    """
    # Example pattern:
    # hand = json.loads((frame_dir / "pico_hands.json").read_text())
    # pts = np.asarray(hand["right"]["joints_world"], dtype=float)
    # return pts[:, :3]
    return None


def load_image_points(frame_dir: Path, training_json: Dict[str, Any]) -> Optional[np.ndarray]:
    """Optional Nx2 measured pixels corresponding one-to-one with load_world_points.

    If supplied, the audit reports true mean/P95 reprojection error in pixels.
    """
    return None
