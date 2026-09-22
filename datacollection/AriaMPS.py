# -*- coding: utf-8 -*-
# @FileName: AriaMPS.py

"""
====================================================================================================
Machine Perception Services (MPS) Processing Pipeline (AriaMPS.py)
====================================================================================================

Description:
    This script provides a high-level wrapper and automation tool for Project Aria MPS CLI. 
    It manages the execution of SLAM and Hand Tracking tasks based on YAML configuration.

Core Functionalities:
    1.  Configuration Integration: Loads credentials, flags, and features from AriaMPS.yaml.
    2.  Feature Selection: Dynamically enables MPS features (SLAM, HAND_TRACKING, etc.).
    3.  Parallel Processing: Orchestrates concurrent MPS tasks and result management.
    4.  Automatic Verification: Validates data integrity based on selected features.

Standardized Output Layout:
    📁 mps_{vrs_name}_vrs/
    ├── 📁 slam/                   <- SLAM trajectory results (if enabled).
    ├── 📁 hand_tracking/          <- Hand pose landmarks (if enabled).
    ├── 📁 else/                   <- Metadata and health logs.
    └── 📄 sample.vrs              <- Source VRS file (renamed).
====================================================================================================
"""

import os
import shutil
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils_io import load_cfg


class AriaMPS:
    """
    Orchestrates Project Aria MPS tasks including execution, re-organization, and validation.
    """

    def __init__(self, cfg_path: str):
        """
        Initializes the processor with parameters from a YAML configuration.
        """
        self.cfg = load_cfg(cfg_path)
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Ensures that authentication parameters and features are present.
        """
        if not self.cfg.username or not self.cfg.password:
            raise ValueError("Missing credentials in AriaMPS.yaml.")
        if not self.cfg.features or len(self.cfg.features) == 0:
            raise ValueError("No features selected in AriaMPS.yaml (e.g., SLAM, HAND_TRACKING).")

    def mps_single_vrs(self, vrs_path: str) -> str:
        """
        Executes MPS for a single VRS file based on selected features.
        """
        vrs_path = os.path.abspath(vrs_path)
        work_dir = os.path.dirname(vrs_path)
        vrs_filename = os.path.basename(vrs_path)
        name = os.path.splitext(vrs_filename)[0]

        target_dir = os.path.join(work_dir, f"mps_{name}_vrs")

        print(f"\n🚀 [MPS START] Features: {self.cfg.features} | Target: {vrs_filename}")

        # Construct the CLI command
        cmd = [
            "aria_mps", "single",
            "--features",
        ]
        cmd.extend(self.cfg.features)  # Add features from YAML
        
        cmd.extend([
            "-i", vrs_filename,
            "-u", self.cfg.username,
            "-p", self.cfg.password,
        ])
        
        if self.cfg.no_ui:
            cmd.append("--no-ui")
        if self.cfg.force:
            cmd.append("--force")

        try:
            subprocess.run(cmd, cwd=work_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ [MPS ERROR] Failed to process {vrs_filename}: {e}")
            return ""

        # Post-processing: Organize results
        # We check if at least one of the expected feature folders exists
        if any(os.path.exists(os.path.join(target_dir, f.lower())) for f in self.cfg.features):
            self._organize_results(work_dir, target_dir, vrs_path, name)
            
            # Automatic integrity check (Feature-Aware)
            if self.mps_check(target_dir):
                print(f"✅ [SUCCESS] Processing and Validation completed: {target_dir}")
                return target_dir
        else:
            print(f"❌ [MPS ERROR] Expected output folders missing for {vrs_filename}.")
        
        return ""

    def _organize_results(self, work_dir: str, target_dir: str, vrs_path: str, name: str) -> None:
        """
        Re-organizes the raw MPS outputs into the standardized internal structure.
        """
        os.makedirs(os.path.join(target_dir, "else"), exist_ok=True)

        # Move source VRS to target directory as 'sample.vrs'
        if os.path.exists(vrs_path):
            shutil.move(vrs_path, os.path.join(target_dir, "sample.vrs"))

        # Manage metadata and log files
        orig_json = os.path.join(work_dir, f"{name}.vrs.json")
        if os.path.exists(orig_json):
            shutil.move(orig_json, os.path.join(target_dir, "else", "sample.vrs.json"))

        health_files = ["vrs_health_check.json", "vrs_health_check_slam.json"]
        for hf in health_files:
            src_hf = os.path.join(target_dir, hf)
            if os.path.exists(src_hf):
                shutil.move(src_hf, os.path.join(target_dir, "else", hf))

    def mps_multi_vrs(self, input_folder: str) -> None:
        """
        Runs parallel MPS tasks for all VRS files in a specified folder.
        """
        input_folder = os.path.abspath(input_folder)
        vrs_files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".vrs")
        ]

        if not vrs_files:
            print(f"⚠️  No VRS files found in {input_folder}")
            return

        print(f"🔍 Found {len(vrs_files)} VRS files. Workers: {self.cfg.workers}")

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            future_to_vrs = {
                executor.submit(self.mps_single_vrs, vrs_file): vrs_file
                for vrs_file in vrs_files
            }

            for future in tqdm(as_completed(future_to_vrs), total=len(vrs_files), desc="Parallel MPS Queue"):
                vrs_name = os.path.basename(future_to_vrs[future])
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ [CRITICAL] Pipeline failed for {vrs_name}: {e}")

    def mps_check(self, mps_path: str) -> bool:
        """
        Validates the existence of critical files based on the enabled features.
        """
        # Core files required for any MPS run
        required_files = [
            "sample.vrs",
            "else/sample.vrs.json",
            "else/vrs_health_check.json",
        ]

        # Feature-specific requirements
        if "SLAM" in self.cfg.features:
            required_files.extend([
                "else/vrs_health_check_slam.json",
                "slam/closed_loop_trajectory.csv",
                "slam/online_calibration.jsonl",
                "slam/open_loop_trajectory.csv",
                "slam/semidense_observations.csv.gz",
                "slam/semidense_points.csv.gz",
                "slam/summary.json",
            ])
        
        if "HAND_TRACKING" in self.cfg.features:
            required_files.extend([
                "hand_tracking/hand_tracking_results.csv",
                "hand_tracking/summary.json",
            ])

        missing = [f for f in required_files if not os.path.exists(os.path.join(mps_path, f))]

        if not missing:
            return True

        print(f"⚠️  [CHECK FAILED] {os.path.basename(mps_path)} is incomplete:")
        for m in missing:
            print(f"   - Missing: {m}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aria MPS Automation Tool")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vrs_file", type=str, help="Process a single VRS file")
    group.add_argument("--vrs_folder", type=str, help="Process all VRS files in a folder")
    
    parser.add_argument("--cfg_path", type=str, default="./cfg/datacollection/AriaMPS.yaml", help="Path to AriaMPS.yaml config file")

    args = parser.parse_args()

    try:
        processor = AriaMPS(args.cfg_path)

        if args.vrs_file:
            processor.mps_single_vrs(args.vrs_file)

        elif args.vrs_folder:
            processor.mps_multi_vrs(args.vrs_folder)

    except Exception as e:
        print(f"❌ [SYSTEM ERROR] {e}")

# ================= Usage Examples =================
# 1) Run single VRS:
# python -m datacollection.AriaMPS --vrs_file "./data/test_0/raw.vrs"
#
# 2) Batch run all VRS under a folder:
# python -m datacollection.AriaMPS --vrs_folder "./data/batch_data/"