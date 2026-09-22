# -*- coding: utf-8 -*-
# @FileName: RobotPreprocess.py

"""
====================================================================================================
Robot Data Preprocessing Orchestrator (RobotPreprocess.py)
====================================================================================================

Description:
    This script serves as the central orchestrator for the robot-collected session preprocessing
    pipeline. It is the robot-domain counterpart of Preprocess.py (which handles Aria human video
    data). Robot sessions originate from RobotTeleop or RobotTeaching collection modes and are
    processed through the same visual pipeline to produce training_data.json.

Core Functionalities:
    1.  Initialization: Reads session_meta.json, enumerates frame directories, and builds the
        complete image list for the session.
    2.  Visual-Spatial Pipeline: Executes deep learning models (DINO, SAM2, CoTracker, LaMa)
        and the DepthLifter module to establish multi-object 6-DOF tracking.
    3.  Consolidation: Generates the final training_data.json via RobotDatasetGen.

Key Differences from Preprocess.py:
    - No VRS/MPS data providers (no Aria glasses data).
    - No AriaPhases (no phase segmentation) -- all frames constitute a single manipulation episode.
    - No AriaCam/AriaHands/AriaSlam extraction stages.
    - Uses DepthLifter instead of CamTriangulator for 2D-to-3D lifting via depth maps.
    - Uses RobotDatasetGen instead of DatasetGen for training data consolidation.
    - Session detection: scans for `teleop_*` or `teaching_*` directories (not `mps_*_vrs`).

Technical Specifics:
    - Recursive Configuration: Loads a master RobotPreprocess.yaml that points to sub-module configs.
    - Decorators: Uses @time_it for precise performance profiling of each pipeline stage.
    - Robustness: Validates session_meta.json existence, cross-checks frame counts against disk,
      and handles missing frames gracefully.
====================================================================================================
"""
import os
import argparse
import shutil
import time
import yaml
from tqdm import tqdm
import json
import re
import traceback

from preprocess.DINOSAMOps import run_dinosam, print_dinosam_stats
from preprocess.KptsSelector import run_kptsselector
from preprocess.CoTrackerOffline import run_cotracker_offline, print_cotracker_offline_stats
from preprocess.DepthLifter import run_depthlifter
from preprocess.Lama import run_lama, print_lama_stats
from preprocess.VisualKpts import run_visualkpts
from preprocess.RobotDatasetGen import run_robot_datasetgen

from utils.utils_media import create_video_from_frames
from utils.utils_math import time_it
from utils.utils_io import load_cfg_dynamic_task


def get_robot_task_list(parent_path, run_range=None):
    """
    Scans the parent directory for folders matching the robot session naming convention:
        teleop_{task}_{NNN}   (teleoperation sessions)
        teaching_{task}_{NNN} (kinesthetic teaching sessions)

    Extracts the numeric index from the folder name, sorts numerically, and optionally
    filters by the provided [start, end] inclusive range.

    Args:
        parent_path (str): Path to the parent directory containing session folders.
        run_range (tuple, optional): (start, end) inclusive index range for filtering.

    Returns:
        list[str]: Sorted list of absolute paths to matching session directories.
    """
    # Match teleop_{anything}_{digits} or teaching_{anything}_{digits}
    pattern = re.compile(r'^(teleop|teaching)_.+_(\d+)$')

    task_folders = []
    if not os.path.exists(parent_path):
        print(f"[Error] Parent path not found: {parent_path}")
        return []

    for item in os.listdir(parent_path):
        full_path = os.path.join(parent_path, item)
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                index = int(match.group(2))
                task_folders.append({
                    "index": index,
                    "path": full_path
                })

    # Sort folders by the numeric index (ascending)
    task_folders.sort(key=lambda x: x["index"])

    # Apply range filtering [start, end] inclusive
    if run_range:
        start, end = run_range
        task_folders = [f for f in task_folders if start <= f["index"] <= end]
        print(f"[Range] Filtering indices from {start} to {end}. Found {len(task_folders)} task(s).")

    return [f["path"] for f in task_folders]


class RobotPreprocess:
    """
    The master orchestrator class for robot-collected session preprocessing.

    Unlike the Aria-based Preprocess class, RobotPreprocess operates on sessions that were
    collected via RobotTeleop or RobotTeaching. There is no VRS/MPS provider, no phase
    segmentation, and no Aria-specific extraction. The full frame sequence is treated as a
    single manipulation episode and processed through the shared visual pipeline (DINO+SAM2,
    KptsSelector, CoTracker, DepthLifter, LaMa, VisualKpts) before final dataset generation
    via RobotDatasetGen.
    """

    def __init__(self, session_path, cfg_path, task, export_video, export_gif, start_from=None):
        """
        Initializes the RobotPreprocess orchestrator.

        Args:
            session_path (str): Absolute path to a single robot session directory
                                (e.g., ./data/serve_bread/teleop_serve_bread_000).
            cfg_path (str): Path to the RobotPreprocess.yaml master configuration file.
            task (str): Task name used to resolve task-specific YAML overrides.
            export_video (bool): Whether to export MP4 visualization videos.
            export_gif (bool): Whether to export GIF animation previews.
            start_from (str): Pipeline step to start from (skip earlier steps).
                              Options: init, dinosam, kptsselector, cotracker, depthlifter, lama, visualkpts, datasetgen
        """
        self.session_path = session_path
        self.cfg_path = cfg_path
        self.task = task
        self.export_video = export_video
        self.export_gif = export_gif
        self.start_from = start_from

        # Recursively loads the master config and automatically resolves all nested sub-configs
        self.cfg = load_cfg_dynamic_task(self.cfg_path, self.session_path, self.task)


    @time_it
    def init_preprocess(self) -> None:
        """
        Initializes the robot session by reading session_meta.json, enumerating all frame
        directories on disk, and building the complete image list.

        Steps:
            1. Validate that session_meta.json exists and load it.
            2. Extract session metadata: fps, width, height, intrinsics (k), n_frames,
               source_type, dual_arm.
            3. Build image_list from all_data/{idx:05d}/rgb.png for all existing frames.
            4. Validate that the actual frame count on disk matches the metadata declaration.
            5. Set the reference frame (first frame) for DINO+SAM2 and KptsSelector.
            6. Since there are no phases, the full sequence is one manipulation episode:
               object_centric_image_list and raw_manip_image_list are both the full image_list.
        """
        # --- 1. Validate session_meta.json ---
        meta_path = os.path.join(self.session_path, "preprocess", "session_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"session_meta.json not found at {meta_path}. "
                f"Ensure this is a valid robot session directory."
            )

        with open(meta_path, 'r') as f:
            self.session_meta = json.load(f)

        # --- 2. Extract metadata fields ---
        self.fps = self.session_meta.get("fps", 30)
        self.frame_width = self.session_meta.get("w", 640)
        self.frame_height = self.session_meta.get("h", 480)
        self.intrinsics = self.session_meta.get("k", None)
        self.n_frames_meta = self.session_meta.get("n_frames", 0)
        self.source_type = self.session_meta.get("source_type", "unknown")
        self.dual_arm = self.session_meta.get("dual_arm", False)

        # Multi-camera support: detect available cameras from session_meta.json
        raw_cameras = self.session_meta.get("cameras", {})
        self.available_cameras = list(raw_cameras.keys()) if raw_cameras else ["cam0"]
        # Ensure cam0 is always present
        if "cam0" not in self.available_cameras:
            self.available_cameras.insert(0, "cam0")

        print(f"[Meta] Session: {os.path.basename(self.session_path)}")
        print(f"[Meta] Source: {self.source_type}, FPS: {self.fps}, "
              f"Resolution: {self.frame_width}x{self.frame_height}, "
              f"Dual-arm: {self.dual_arm}, Declared frames: {self.n_frames_meta}")
        print(f"[Meta] Available cameras: {self.available_cameras} "
              f"(pipeline runs on cam0 only; wrist images used raw)")

        # --- Override DINOSAM arm prompt for robot data ---
        if self.source_type in ("teleop", "teaching"):
            if hasattr(self.cfg, 'DINOSAM') and hasattr(self.cfg.DINOSAM, 'dinosam_prompt'):
                self.cfg.DINOSAM.dinosam_prompt.arm = "robot arms . robot hands ."
                # Re-export DINOSAM.yaml
                dinosam_yaml_path = getattr(self.cfg, 'DINOSAM_path', None)
                if dinosam_yaml_path and os.path.exists(dinosam_yaml_path):
                    with open(dinosam_yaml_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(self.cfg.DINOSAM.to_dict(), f, default_flow_style=False, sort_keys=False)
                # Re-export Preprocess.yaml master snapshot
                preprocess_yaml_path = os.path.join(self.session_path, "preprocess", "cfg", "Preprocess.yaml")
                if os.path.exists(preprocess_yaml_path):
                    with open(preprocess_yaml_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(self.cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
                print(f"[Config] Overrode DINOSAM arm prompt for robot data: 'robot arms . robot hands .'")

        # --- 3. Build image list from disk ---
        all_data_dir = os.path.join(self.session_path, "preprocess", "all_data")
        if not os.path.exists(all_data_dir):
            raise FileNotFoundError(
                f"Frame directory not found: {all_data_dir}. "
                f"Ensure the session has been collected properly."
            )

        self.image_list = []
        # Enumerate sequentially, allowing for gaps in numbering
        frame_dirs = sorted([
            d for d in os.listdir(all_data_dir)
            if os.path.isdir(os.path.join(all_data_dir, d)) and d.isdigit()
        ])

        for frame_dir_name in frame_dirs:
            img_path = os.path.join(all_data_dir, frame_dir_name, "rgb.png")
            if os.path.exists(img_path):
                self.image_list.append(img_path)
            else:
                print(f"[Warning] Missing rgb.png in frame directory: {frame_dir_name}")

        self.num_total_frames = len(self.image_list)

        # --- 4. Validate frame count ---
        if self.n_frames_meta > 0 and self.num_total_frames != self.n_frames_meta:
            print(f"[Warning] Frame count mismatch: metadata declares {self.n_frames_meta} frames, "
                  f"but {self.num_total_frames} frames found on disk.")

        if self.num_total_frames == 0:
            raise RuntimeError(
                f"No frames found in {all_data_dir}. Cannot proceed with preprocessing."
            )

        print(f"[Info] Found {self.num_total_frames} frames on disk.")

        # --- 5. Set reference frame (first frame) ---
        self.object_ref_img_path = self.image_list[0]
        print(f"[Info] Reference frame: {self.object_ref_img_path}")

        # --- 6. Full sequence as single episode (no phase segmentation) ---
        # For CoTracker: both object_centric and raw_manip cover the full sequence
        self.object_centric_image_list = list(self.image_list)
        self.raw_manip_image_list = list(self.image_list)

        print(f"[Info] Full sequence length: {len(self.image_list)} frames "
              f"(no phase segmentation, single manipulation episode)")


    @time_it
    def preprocess_dinosam(self) -> None:
        """
        Executes Grounding DINO and SAM2 for bounding box detection and mask segmentation
        on all frames. Uses the robot arm prompt configuration from the DINOSAM config.
        """

        frames_all_dinosam = []
        dinosam_image_list = self.object_centric_image_list + self.raw_manip_image_list
        # Deduplicate while preserving order (since both lists are identical for robot data)
        seen = set()
        deduped_list = []
        for p in dinosam_image_list:
            if p not in seen:
                seen.add(p)
                deduped_list.append(p)
        dinosam_image_list = deduped_list

        print(f"[Info] DINOSAM Processing {len(dinosam_image_list)} frames...")
        for i, img_path in enumerate(tqdm(dinosam_image_list, desc="DINO+SAM2 Pipeline")):
            if not os.path.exists(img_path):
                continue
            vis_dinosam = run_dinosam(
                cfg_path=self.cfg.DINOSAM_path,
                image_path=img_path,
            )

            if self.export_video:
                frames_all_dinosam.append(vis_dinosam)

        print_dinosam_stats()


    @time_it
    def preprocess_kptsselector(self) -> None:
        """
        Extracts equidistant tracking keypoints from the segmented masks of the reference frame.
        Iterates over all object keys defined in the DINOSAM prompt config (obj_1, obj_2, ...),
        copies the reference mask, and runs KptsSelector to produce per-object keypoint sets.
        Results are saved to kptsselector_results.json.
        """

        ref_frame_dir = os.path.dirname(self.object_ref_img_path)
        obj_keys = [k for k in self.cfg.DINOSAM.dinosam_prompt.keys() if k.startswith("obj")]

        all_kpts_data = {}

        for obj_key in obj_keys:
            src_mask_path = os.path.join(ref_frame_dir, f"mask_{obj_key}.png")
            dst_mask_path = os.path.join(self.session_path, "preprocess", f"dinosam_mask_{obj_key}.png")

            if os.path.exists(src_mask_path):
                shutil.copy2(src_mask_path, dst_mask_path)
                print(f"[Info] Successfully copied {src_mask_path} to {dst_mask_path}")

                kpts = run_kptsselector(
                    cfg_path=self.cfg.KptsSelector_path,
                    mask_path=dst_mask_path,
                    save_img_path=os.path.join(self.session_path, "preprocess", f"kptsselector_vis_{obj_key}.png"),
                    rgb_path=self.object_ref_img_path,
                )

                if kpts:
                    all_kpts_data[obj_key] = kpts
            else:
                print(f"[Warning] Expected mask file NOT found: {src_mask_path}")

        save_json_path = os.path.join(self.session_path, "preprocess", "kptsselector_results.json")
        data = {
            "method": "AUTO_MASK_PCA_EDGE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": all_kpts_data  # {"obj_1": [[u,v]...], "obj_2": [[u,v]...]}
        }
        with open(save_json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[Output] Multi-object Keypoints JSON saved: {save_json_path}")


    @time_it
    def preprocess_cotracker(self) -> None:
        """
        Executes CoTracker3 for dense point tracking across the full frame sequence.
        Since there are no phases in robot data, all frames serve as both the object-centric
        and raw manipulation lists.
        """

        frames_all_cotracker = []
        cotracker_image_list = self.object_centric_image_list + self.raw_manip_image_list
        # Deduplicate while preserving order
        seen = set()
        deduped_list = []
        for p in cotracker_image_list:
            if p not in seen:
                seen.add(p)
                deduped_list.append(p)
        cotracker_image_list = deduped_list

        print(f"[Info] CoTracker Processing {len(cotracker_image_list)} frames...")
        for i, img_path in enumerate(tqdm(cotracker_image_list, desc="CoTracker Pipeline")):
            if not os.path.exists(img_path):
                continue
            vis_cotracker = run_cotracker_offline(
                image_path=img_path,
                cfg_path=self.cfg.CoTracker_path,
                frame_idx=i,
                all_image_paths=cotracker_image_list,
                mps_path=self.session_path,
            )

            if self.export_video:
                frames_all_cotracker.append(vis_cotracker)

        print_cotracker_offline_stats()


    @time_it
    def preprocess_depthlifter(self) -> None:
        """
        Lifts 2D CoTracker tracks to 3D world coordinates using per-frame depth maps.
        This replaces CamTriangulator from the Aria pipeline -- instead of multi-view
        triangulation via SLAM poses, we use monocular/stereo depth to recover 3D positions.
        """

        frames_all_depthlifter = []
        depthlifter_image_list = self.object_centric_image_list
        print(f"[Info] DepthLifter Processing {len(depthlifter_image_list)} frames...")
        for i, img_path in enumerate(tqdm(depthlifter_image_list, desc="DepthLifter Pipeline")):
            if not os.path.exists(img_path):
                continue

            vis_depthlifter = run_depthlifter(
                image_path=img_path,
                cfg_path=self.cfg.DepthLifter_path,
                frame_idx=i,
                all_image_paths=depthlifter_image_list,
                session_path=self.session_path,
            )

            if self.export_video:
                frames_all_depthlifter.append(vis_depthlifter)


    @time_it
    def preprocess_lama(self) -> None:
        """
        Uses LaMa to perform background inpainting, removing the robot arm and manipulated
        objects to produce clean background frames for visual conditioning.
        """

        frames_all_lama = []
        lama_image_list = self.object_centric_image_list + self.raw_manip_image_list
        # Deduplicate while preserving order
        seen = set()
        deduped_list = []
        for p in lama_image_list:
            if p not in seen:
                seen.add(p)
                deduped_list.append(p)
        lama_image_list = deduped_list

        print(f"[Info] Lama Processing {len(lama_image_list)} frames...")
        for i, img_path in enumerate(tqdm(lama_image_list, desc="Lama Pipeline")):
            if not os.path.exists(img_path):
                continue
            vis_lama = run_lama(
                image_path=img_path,
                cfg_path=self.cfg.Lama_path
            )

            if self.export_video:
                frames_all_lama.append(vis_lama)

        print_lama_stats()


    @time_it
    def preprocess_visualkpts(self) -> None:
        """
        Renders aesthetic visual keypoint overlays (object trails and tracking indicators)
        on top of the original frames for visualization and debugging.
        """

        frames_all_visualkpts = []
        visualkpts_image_list = self.object_centric_image_list + self.raw_manip_image_list
        # Deduplicate while preserving order
        seen = set()
        deduped_list = []
        for p in visualkpts_image_list:
            if p not in seen:
                seen.add(p)
                deduped_list.append(p)
        visualkpts_image_list = deduped_list

        print(f"[Info] VisualKpts Processing {len(visualkpts_image_list)} frames...")
        for i, img_path in enumerate(tqdm(visualkpts_image_list, desc="VisualKpts Pipeline")):
            if not os.path.exists(img_path):
                continue
            vis_visualkpts = run_visualkpts(
                image_path=img_path,
                cfg_path=self.cfg.VisualKpts_path,
                frame_idx=i,
                all_image_paths=visualkpts_image_list,
                mps_path=self.session_path,
            )

            if self.export_video:
                frames_all_visualkpts.append(vis_visualkpts)

        if self.export_video:
            create_video_from_frames(
                frames=frames_all_visualkpts,
                save_path=os.path.join(self.session_path, "preprocess", "vis", "visualkpts_vis.mp4"),
                fps=self.fps,
                export_gif=self.export_gif
            )


    @time_it
    def preprocess_datasetgen(self) -> None:
        """
        Consolidates all spatial kinematics, tracked keypoints, and depth-lifted 3D positions
        into the final training_data.json via RobotDatasetGen.
        """

        datasetgen_image_list = self.image_list
        if datasetgen_image_list:
            print(f"[Info] Generating training_data.json for {len(datasetgen_image_list)} frames...")
            ds_stats = run_robot_datasetgen(
                image_list=datasetgen_image_list,
                session_path=self.session_path,
                cfg_path=self.cfg.RobotDatasetGen_path,
            )
        else:
            print("[Skip] Image list is empty. Skipping dataset generation.")


    def _detect_completed_stages(self) -> dict:
        """
        Detect which preprocessing stages have been completed by checking output files.

        Returns:
            dict: {stage_name: bool} indicating completion status.
        """
        pp = os.path.join(self.session_path, "preprocess")

        # --- init: session_meta.json + at least one frame ---
        init_ok = os.path.isfile(os.path.join(pp, "session_meta.json"))

        # --- dinosam: mask files in the first frame directory ---
        first_frame_dir = os.path.join(pp, "all_data", "00000")
        dinosam_ok = False
        if os.path.isdir(first_frame_dir):
            masks = [f for f in os.listdir(first_frame_dir) if f.startswith("mask_obj") and f.endswith(".png")]
            dinosam_ok = len(masks) >= 1

        # --- kptsselector: kptsselector_results.json ---
        kpts_ok = os.path.isfile(os.path.join(pp, "kptsselector_results.json"))

        # --- cotracker: cotracker_results.json ---
        cotracker_ok = os.path.isfile(os.path.join(pp, "cotracker_results.json"))

        # --- depthlifter: depthlifter_results.json ---
        depthlifter_ok = os.path.isfile(os.path.join(pp, "depthlifter_results.json"))

        # --- lama: rgb_WoArm.png in a frame directory ---
        lama_ok = False
        if os.path.isdir(first_frame_dir):
            lama_ok = os.path.isfile(os.path.join(first_frame_dir, "rgb_WoArm.png"))

        # --- visualkpts: vis/visualkpts_vis.mp4 or rgb_WArmObjKpts.png in a frame ---
        visualkpts_ok = (
            os.path.isfile(os.path.join(pp, "vis", "visualkpts_vis.mp4"))
            or (os.path.isdir(first_frame_dir) and
                os.path.isfile(os.path.join(first_frame_dir, "rgb_WArmObjKpts.png")))
        )

        # --- datasetgen: training_data.json in a frame directory ---
        datasetgen_ok = False
        if os.path.isdir(first_frame_dir):
            datasetgen_ok = os.path.isfile(os.path.join(first_frame_dir, "training_data.json"))

        return {
            'init':         init_ok,
            'dinosam':      dinosam_ok,
            'kptsselector': kpts_ok,
            'cotracker':    cotracker_ok,
            'depthlifter':  depthlifter_ok,
            'lama':         lama_ok,
            'visualkpts':   visualkpts_ok,
            'datasetgen':   datasetgen_ok,
        }

    def _find_auto_start(self) -> str:
        """
        Determine the first incomplete stage based on output file checks.

        Respects dependency ordering: if an earlier stage is missing, all
        subsequent stages must be re-run even if their outputs exist
        (because they were produced from stale/incomplete inputs).

        Returns:
            Stage name to start from, or None if all complete.
        """
        completed = self._detect_completed_stages()
        ORDERED_STAGES = ['init', 'dinosam', 'kptsselector', 'cotracker',
                          'depthlifter', 'lama', 'visualkpts', 'datasetgen']

        for stage in ORDERED_STAGES:
            if not completed[stage]:
                return stage
        return None  # All complete

    @time_it
    def run(self) -> None:
        """
        Executes the entire robot preprocessing workflow in sequence.

        Pipeline order:
            1. init_preprocess     -- Load metadata, build image lists
            2. preprocess_dinosam  -- DINO + SAM2 segmentation
            3. preprocess_kptsselector -- Extract object keypoints from reference frame
            4. preprocess_cotracker    -- Dense point tracking across sequence
            5. preprocess_depthlifter  -- Lift 2D tracks to 3D via depth
            6. preprocess_lama         -- Background inpainting
            7. preprocess_visualkpts   -- Aesthetic keypoint overlays
            8. preprocess_datasetgen   -- Final training_data.json generation
        """

        PIPELINE_STEPS = [
            ('init',          self.init_preprocess),
            ('dinosam',       self.preprocess_dinosam),
            ('kptsselector',  self.preprocess_kptsselector),
            ('cotracker',     self.preprocess_cotracker),
            ('depthlifter',   self.preprocess_depthlifter),
            ('lama',          self.preprocess_lama),
            ('visualkpts',    self.preprocess_visualkpts),
            ('datasetgen',    self.preprocess_datasetgen),
        ]
        step_names = [s[0] for s in PIPELINE_STEPS]

        # Determine where to start
        if self.start_from == "auto":
            # Auto-detect: find the first incomplete stage
            auto_start = self._find_auto_start()
            if auto_start is None:
                completed = self._detect_completed_stages()
                status = " | ".join(f"{k}:{'✓' if v else '✗'}" for k, v in completed.items())
                print(f"[Auto] All stages complete, skipping: {status}")
                return
            start_idx = step_names.index(auto_start)
            completed = self._detect_completed_stages()
            status = " | ".join(f"{k}:{'✓' if v else '✗'}" for k, v in completed.items())
            print(f"[Auto] {status}")
            print(f"[Auto] Resuming from: {auto_start} (skipping {start_idx} completed stages)")
        elif self.start_from and self.start_from in step_names:
            start_idx = step_names.index(self.start_from)
            print(f"[Pipeline] Starting from step: {self.start_from} (skipping {start_idx} steps)")
        else:
            start_idx = 0

        # Always run init regardless (loads metadata + image lists)
        if start_idx > 0:
            self.init_preprocess()

        for name, fn in PIPELINE_STEPS[start_idx:]:
            fn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot Data Preprocessing Orchestrator. Processes robot-collected sessions "
                    "(RobotTeleop / RobotTeaching) through the visual pipeline to produce training_data.json."
    )
    # session_path can be a single session folder OR a parent directory containing multiple sessions
    parser.add_argument("--session_path", type=str, required=True,
                        help="Path to a single robot session directory or a parent directory for batch processing")

    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/base/RobotPreprocess.yaml",
                        help="Path to the base RobotPreprocess.yaml")

    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g., serve_bread) to load specific YAML overrides")

    parser.add_argument("--range", type=int, nargs=2, metavar=('START', 'END'),
                        help="Numeric range of session indices to process (e.g., --range 0 5)")

    parser.add_argument("--no-video", action="store_false", dest="export_video", help="Disable MP4 video export")
    parser.add_argument("--no-gif", action="store_false", dest="export_gif", help="Disable GIF animation export")
    parser.add_argument("--start_from", type=str, default="auto",
                        choices=['auto', 'init', 'dinosam', 'kptsselector', 'cotracker', 'depthlifter', 'lama', 'visualkpts', 'datasetgen'],
                        help="'auto' (default): detect completed stages and resume from first incomplete. "
                             "Or specify a stage name to force start from there (e.g., --start_from cotracker). "
                             "'init' forces full reprocessing.")
    args = parser.parse_args()

    # 1. Identify Task Execution Mode
    #    Single session: session_meta.json exists directly in the given path
    #    Batch mode: scan for teleop_*/teaching_* sub-directories
    if os.path.exists(os.path.join(args.session_path, "preprocess", "session_meta.json")):
        final_tasks = [args.session_path]
        print(f"[Mode] Single session detected: {os.path.basename(args.session_path)}")
    else:
        print(f"[Mode] Batch directory detected. Scanning for robot sessions...")
        final_tasks = get_robot_task_list(args.session_path, args.range)

    if not final_tasks:
        print("[Error] No valid robot session folders found. Terminating execution.")
        exit(1)

    # 2. Sequential Pipeline Execution
    print(f"[Batch] Starting sequence for {len(final_tasks)} session(s)...")

    for i, path in enumerate(final_tasks):
        folder_name = os.path.basename(path)
        print(f"\n{'='*100}")
        print(f"[{i+1}/{len(final_tasks)}] Processing: {folder_name}")
        print(f"{'='*100}\n")

        # Reset all singletons between sessions to avoid stale state
        from preprocess.CoTrackerOffline import reset_cotracker_offline
        from preprocess.DepthLifter import reset_depthlifter
        from preprocess.VisualKpts import reset_visualkpts
        reset_cotracker_offline()
        reset_depthlifter()
        reset_visualkpts()

        try:
            preprocess_engine = RobotPreprocess(
                session_path=path,
                cfg_path=args.cfg_path,
                task=args.task,
                export_video=args.export_video,
                export_gif=args.export_gif,
                start_from=args.start_from,
            )
            preprocess_engine.run()

        except Exception as e:
            print(f"\n[Critical Error] Session failed: {folder_name}")
            print(traceback.format_exc())
            print(f"[System] Skipping to next session...\n")
            continue

    print(f"\n[Done] All sessions in the batch have been processed.")


# python -m preprocess.RobotPreprocess --session_path ./data/serve_bread/teleop/teleop_serve_bread_000 --task serve_bread
# python -m preprocess.RobotPreprocess --session_path ./data/serve_bread/teleop/ --task serve_bread --range 0 5