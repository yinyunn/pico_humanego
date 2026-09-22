# -*- coding: utf-8 -*-
# @FileName: run_inference.py
"""
HumanEgo dual-arm real-world inference — reference loop.

    camera ─▶ perception ─▶ ICT + clean image ─▶ policy ─▶ EE trajectory ─▶ robot
       ▲                                                                      │
       └──────────────────────── close the loop ◀────────────────────────────┘

This is a TEMPLATE: it shows the standard structure of a HumanEgo inference
stack. It will not run as-is — you must supply working hardware drivers and a
perception module (the example uses Intel RealSense + Trossen + DINO-SAM/LaMa).
Everything hardware/perception-specific is isolated in the three adapters at the
top and clearly marked with `TODO`; the policy + control logic below is generic.

Pipeline
--------
ONE-TIME (episode start):
    1. estimate object 6DoF poses from a few RGB-D frames  -> object-centric frame
    2. home the arms, open grippers

EVERY STEP (closed loop, ~5-10 Hz):
    3. grab RGB-D
    4. read each arm's EE pose (FK) + gripper state
    5. "latch" grasped objects so their pose tracks the gripper
    6. build the clean, embodiment-agnostic image (inpaint arm + render gripper)
    7. build the ICT from hand + object poses
    8. policy.infer -> future EE trajectory (+ done probability) for both arms
    9. decode trajectory to camera-frame EE targets, execute the first few steps
   10. stop when the policy reports "done"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import yaml

# Make BOTH this folder and the repo root importable regardless of CWD, so the
# flat imports here (interfaces/policy/controller, CamRS/RobotArmTrossen) and the
# package imports inside policy.py (training.*, utils.*) all resolve when you run
# `python inference/run_inference.py cfg/inference/example_dualarm.yaml` from root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interfaces import Camera, Frame, ObjectState, Perception, RobotArm
from policy import ICTPolicy
from controller import TrajectoryController


# =====================================================================
# 1. HARDWARE ADAPTERS  — replace the bodies with your own hardware.
#    These wrap the shipped RealSense / Trossen drivers to satisfy the
#    interfaces in interfaces.py. The mapping is the whole point: copy
#    this pattern for your camera and arm.
# =====================================================================

class RealSenseCamera(Camera):
    """Adapter over the shipped CamRS (Intel RealSense) driver."""
    def __init__(self, cam_cfg_path: str):
        from CamRS import CamRS                       # TODO: your camera SDK
        self.cam = CamRS(cam_cfg_path)

    def get_frame(self) -> Frame:
        d = self.cam.get_rgbd()                       # CamRSData(rgb, depth_m, ...)
        return Frame(rgb=d.rgb, depth_m=d.depth_m, K=self.cam.k_rgb)

    def close(self) -> None:
        self.cam.close()


class TrossenArm(RobotArm):
    """Adapter over the shipped RobotArmTrossen driver."""
    def __init__(self, arm_cfg_path: str):
        from RobotArmTrossen import RobotArmTrossen   # TODO: your robot SDK
        self.arm = RobotArmTrossen(arm_cfg_path)
        self.T_base_in_cam = self.arm.T_base_in_cam   # from hand-eye calibration

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self.arm.get_T_ee_in_cam()
    def move_ee_in_cam(self, T, duration, blocking=False) -> bool:
        return self.arm.move_p_in_cam(self.arm.T_to_p(T), duration=duration, blocking=blocking)
    def get_gripper(self) -> float:
        return 1.0 - self.arm.get_gripper_q()         # driver: 1=open -> ours: 0=open
    def set_gripper(self, value: float, blocking: bool = False) -> None:
        (self.arm.close_gripper if value > 0.5 else self.arm.open_gripper)(blocking=blocking)
    def go_home(self, blocking: bool = True) -> None:
        self.arm.go_home(blocking=blocking)
    def close(self) -> None:
        self.arm.close()


# =====================================================================
# 2. PERCEPTION  — the heaviest part to port. Reference impl uses the
#    shipped preprocess/ engines (DINO-SAM detect+segment, LaMa inpaint).
#    Swap in ANY detector/pose-estimator that returns object poses.
# =====================================================================

class ReferencePerception(Perception):
    """Object 6DoF poses + clean image, mirroring the training preprocessing.

    The exact same detect→segment→lift→pose / inpaint→render pipeline is used
    at training time (see preprocess/), which is why train and test images match.
    """
    def __init__(self, cam: RealSenseCamera, cfg: dict):
        self.cam = cam
        self.prompts = cfg["object_prompts"]          # {"obj1": "a green cup .", ...}
        self.erase_prompt = cfg["erase_prompt"]       # e.g. "a robot arm . a gripper ."
        self.anchor_key = cfg.get("anchor_key", "obj1")
        # Heavy models — load once. (These imports/weights are the engineering cost.)
        # from preprocess.DINOSAM import DINOSAM
        # from preprocess.Lama   import LamaEngine
        # self.detector = DINOSAM(cfg["dinosam_cfg_path"])
        # self.inpainter = LamaEngine(cfg["lama_cfg_path"])
        # self.renderer  = VisualKptsEngine(cfg["visualkpts_cfg_path"])

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        objs: Dict[str, ObjectState] = {}
        for key, prompt in self.prompts.items():
            # (a) detect + segment the object on the first frame
            #     mask = self.detector.process_single(frames[0].rgb, prompt)
            # (b) lift the masked pixels to 3D across frames (robust to depth noise)
            #     pts3d_cam = self.cam.cam.lift3d_for_multi_frames(uv, depths, mask, K)
            # (c) fit a 6DoF pose to the 3D points (PCA / your favorite estimator)
            #     T_in_cam, _ = estimate_frame_pca2(pts3d_cam, is_anchor=(key==self.anchor_key))
            # (d) keypoints in the object's own frame (only needed for PCD features)
            #     kpts_local = (inv(T_in_cam) @ homogeneous(pts3d_cam))
            raise NotImplementedError(
                "Plug in your detector + pose estimator here. See preprocess/ for the "
                "DINO-SAM + depth-lift + PCA reference, or return poses from any source "
                "(AprilTag, FoundationPose, known CAD + ICP, ...)."
            )
        return objs

    def make_clean_image(self, frame, ee_poses_in_cam, grippers) -> np.ndarray:
        # (a) inpaint the real arm out:   mask = detector(rgb, erase_prompt);
        #                                 clean = inpainter.inpaint(rgb, mask)
        # (b) render a virtual gripper at each EE pose onto `clean`, matching the
        #     training visualization:     renderer.process_single_gripper(clean, T_ee, grasp, K)
        # If the model was trained state-only (no image), just return a black frame.
        return frame.rgb.copy()


# =====================================================================
# 3. Small helpers
# =====================================================================

def latch_objects(
    static_objs: Dict[str, ObjectState],
    hands_in_cam: Dict[str, np.ndarray],
    grippers: Dict[str, float],
    latches: dict,
    anchor_key: str,
    grasp_threshold: float = 0.5,
    latch_dist_thresh: float = 0.20,
) -> Dict[str, ObjectState]:
    """Make a grasped object's pose follow the gripper.

    When an arm closes, we find the nearest object to the gripper and — only if it
    is within ``latch_dist_thresh`` metres (i.e. we actually grasped something
    nearby) — lock the relative transform gripper->object. While the arm stays
    closed, that object's pose is driven by the gripper, keeping its ICT token
    consistent with what is physically happening even when vision can't see the
    occluded object.

    This mirrors the training-time data generation (``preprocess/DatasetGen.py``):
      - The nearest search includes ALL objects, the anchor included. In several
        tasks the anchor IS the manipulated object (e.g. the bread in serve_bread),
        so it must be latchable — excluding it would latch the wrong object.
      - ``latch_dist_thresh`` mirrors the 0.20 m gate used during data generation.
        Without it, closing the gripper with no object within reach would wrongly
        latch the nearest distant object and drag its token along the gripper.
    ``anchor_key`` is retained for call-site compatibility; it is no longer used to
    exclude the anchor from latching.
    """
    dynamic = {k: ObjectState(v.T_in_cam.copy(), v.kpts_local) for k, v in static_objs.items()}
    if not static_objs:
        return dynamic
    for side, T_h in hands_in_cam.items():
        if T_h is None:
            continue
        closed = grippers.get(side, 0.0) > grasp_threshold
        if closed and latches.get(side) is None:
            # nearest object to the gripper at the moment of grasp (anchor included)
            best_key, best_dist = None, float("inf")
            for k, v in static_objs.items():
                d = float(np.linalg.norm(v.T_in_cam[:3, 3] - T_h[:3, 3]))
                if d < best_dist:
                    best_dist, best_key = d, k
            # only latch if we actually grasped something nearby (matches training's 0.20 m gate)
            if best_key is not None and best_dist < latch_dist_thresh:
                latches[side] = (best_key, np.linalg.inv(T_h) @ static_objs[best_key].T_in_cam)
            else:
                latches[side] = None
        elif not closed:
            latches[side] = None
        if latches.get(side) is not None:
            key, T_lock = latches[side]
            dynamic[key] = ObjectState(T_h @ T_lock, static_objs[key].kpts_local)
    return dynamic


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =====================================================================
# 4. Main loop
# =====================================================================

def run(cfg_path: str, device: str = "cuda") -> None:
    cfg = load_cfg(cfg_path)
    cfg_sides = cfg["robot"]["sides"]                  # physical arms present, e.g. ["left","right"]
    T_align = np.array(cfg["robot"].get("T_align", np.eye(4).tolist()), dtype=np.float32)
    anchor_key = cfg["perception"].get("anchor_key", "obj1")
    latch_dist_thresh = cfg["perception"].get("latch_dist_thresh", 0.20)  # match DatasetGen's 0.20 m gate
    exec_horizon = cfg["control"].get("exec_horizon", 8)   # steps run before re-planning
    dt = 1.0 / cfg["control"].get("control_hz", 10.0)
    done_threshold = cfg["control"].get("done_threshold", 0.8)

    # ---- build the stack ----
    cam = RealSenseCamera(cfg["camera"]["cfg_path"])
    arms: Dict[str, RobotArm] = {s: TrossenArm(cfg["robot"]["cfg_paths"][s]) for s in cfg_sides}
    policy = ICTPolicy(cfg["policy"], device=device)
    controller = TrajectoryController(arms, cfg["control"])
    perception = ReferencePerception(cam, cfg["perception"])

    # The policy decides how many hands it drives (read from the checkpoint).
    sides = policy.sides                               # e.g. ["left","right"] or ["right"]
    missing = [s for s in sides if s not in arms]
    if missing:
        raise ValueError(f"checkpoint predicts {sides} but no robot arm configured for {missing}. "
                         f"Set robot.sides to include them.")

    # ---- one-time: object-centric setup + homing ----
    print("[run] estimating object poses...")
    n = cfg["perception"].get("n_init_frames", 10)
    static_objs = perception.estimate_objects([cam.get_frame() for _ in range(n)])
    anchor = static_objs.get(anchor_key)               # fixed object-centric reference for the episode
    for arm in arms.values():
        arm.set_gripper(0.0, blocking=True)            # open
        arm.go_home(blocking=True)

    latches: dict = {s: None for s in sides}
    done = {s: False for s in sides}
    print("[run] entering closed-loop control. Ctrl-C to stop.")

    try:
        while not all(done.values()):
            frame = cam.get_frame()

            # read robot state -> HAND-frame poses (EE pose bridged by T_align)
            hands_in_cam = {s: arms[s].get_T_ee_in_cam() @ T_align for s in sides}
            grippers = {s: arms[s].get_gripper() for s in sides}

            # grasped objects (anchor included, e.g. the bread in serve_bread)
            # follow the gripper, gated by latch_dist_thresh
            objs = latch_objects(static_objs, hands_in_cam, grippers, latches,
                                 anchor_key, controller.grasp_threshold,
                                 latch_dist_thresh=latch_dist_thresh)

            # build the policy inputs: clean image, ICT, and (if region-attn) anchor UV
            clean = perception.make_clean_image(frame, hands_in_cam, grippers)
            x_ict, ict_mask = policy.build_ict(hands_in_cam, grippers, objs, anchor_key)
            x_rgb = policy.prepare_image(clean)
            anchor_uv = policy.compute_anchor_uv(anchor, frame.K,
                                                 frame.rgb.shape[1], frame.rgb.shape[0])

            # predict the future trajectory for every hand the policy drives
            traj, done_prob = policy.infer(x_rgb, x_ict, ict_mask, anchor_uv)

            # decode reference-frame predictions -> camera-frame EE targets
            ee_targets: Dict[str, List[np.ndarray]] = {}
            grasp_cmds: Dict[str, np.ndarray] = {}
            for s in sides:
                pos, o6d, grasp = traj[s]
                ee_targets[s] = [policy.decode_ee_in_cam(pos[k], o6d[k], anchor, T_align)
                                 for k in range(len(pos))]
                grasp_cmds[s] = grasp

            # execute the first few steps, then re-plan (receding horizon)
            controller.execute_chunk(ee_targets, grasp_cmds, dt=dt, n_steps=exec_horizon)

            if done_prob > done_threshold:
                print(f"[run] policy reports done (p={done_prob:.2f}).")
                done = {s: True for s in sides}

    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        controller.home()
        for arm in arms.values():
            arm.close()
        cam.close()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "./cfg/inference/example_dualarm.yaml"
    run(cfg_path)
