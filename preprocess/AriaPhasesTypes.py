# -*- coding: utf-8 -*-
# @FileName: AriaPhasesTypes.py

"""
====================================================================================================
Project Aria Task Phase Segmentation Data Structures (AriaPhasesTypes.py)
====================================================================================================

Description:
    This module defines the data containers for Project Aria phase segmentation. It categorizes 
    the sequence into different temporal modes (e.g., Moving vs. Static) and detects 
    "stop" events. It stores derived kinematics and handles the automated serialization 
    of phase-related metadata.

Core Components:
    1. AriaPhasesFrame: A dataclass representing the temporal state (mode/stop) and 
       instantaneous kinematics of a single frame.
    2. AriaPhases: A sequence container that manages a list of frames, stores summary 
       statistics, and provides methods for JSON serialization and diagnostic plotting.

Generated Outputs & File Descriptions:
    📁 [mps_path]/aria/
    ├── 📁 all_data/
    │   └── 📁 [00000...idx]/
    │       └── 📄 aria_phases.json        # Per-frame mode, stop status, and speed metrics.
    ├── 📄 aria_phases_results.json        # Sequence summary (durations, phase counts).
    └── 🖼️ aria_phases_analysis.png        # Diagnostic plot visualizing mode transitions.

Technical Specifics:
    - Mode: Numerical encoding of the movement state.
    - Stop: Binary flag indicating a detected cessation of movement.
    - Kinematics: Includes linear speed (m/s) and angular rate (rad/s).
====================================================================================================
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from utils.utils_io import make_json_serializable

from preprocess.AriaPhasesOps import AriaPhasesOps


@dataclass
class AriaPhasesFrame:
    """
    Represents the movement phase and kinematic state of a single time step.

    Attributes:
        idx (int): Global frame index.
        ts (int): Device timestamp in nanoseconds.
        mode (int): Categorical ID representing the current task phase.
        stop (int): Binary flag (1 if stopped, 0 if moving).
        mode_str (str): Human-readable label for the phase mode.
        v (float): Linear speed magnitude (m/s).
        w (float): Angular velocity (rad/s).
        yaw_u_deg (float): Unwrapped continuous yaw orientation in degrees.
    """
    idx: int
    ts: int
    mode: int
    stop: int
    mode_str: str
    v: float
    w: float
    yaw_u_deg: float


@dataclass
class AriaPhases:
    """
    A sequence-level container for AriaPhasesFrame objects.

    Attributes:
        mps_path (str): Root directory for Project Aria MPS data.
        frames (List[AriaPhasesFrame]): Chronological sequence of phase data.
        summary (dict): Aggregated task-level results and statistics.
    """
    mps_path: str
    frames: List[AriaPhasesFrame] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    
    def save_aria_phases_json(self) -> None:
        """
        Serializes phase data into frame-specific and task-summary JSON files.
        Triggers diagnostic plot generation and console reporting.
        """
        all_data_dir = os.path.join(self.mps_path, "preprocess", "all_data")
        os.makedirs(all_data_dir, exist_ok=True)

        # 1. Save per-frame metadata
        for f in self.frames:
            # Ensure frame-specific directory exists
            frame_dir = os.path.join(all_data_dir, f"{f.idx:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            p = os.path.join(frame_dir, "aria_phases.json")
            data = {
                "idx": int(f.idx), 
                "ts": int(f.ts),
                "stop": int(f.stop), 
                "mode": int(f.mode), 
                "mode_str": f.mode_str,
                "linear_speed_mps": float(f.v), 
                "angular_speed_rps": float(f.w),
                "yaw_unwrapped_deg": float(f.yaw_u_deg)
            }
            with open(p, "w") as jf:
                json.dump(data, jf, indent=2)
        
        # 2. Save task-level summary
        summary_path = os.path.join(self.mps_path, "preprocess", "aria_phases_results.json")
        with open(summary_path, "w") as f:
            json.dump(make_json_serializable(self.summary), f, indent=4)
        
        # 3. Generate diagnostic visualization
        analysis_png = os.path.join(self.mps_path, "preprocess", "aria_phases_analysis.png")
        AriaPhasesOps.save_phases_analysis_plot(self, analysis_png)
        print(f"[***] [AriaPhases] Analysis plot saved: {analysis_png}")
        
        # 4. Output terminal summary report
        AriaPhasesOps.pretty_print_report(self.summary)