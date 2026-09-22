# -*- coding: utf-8 -*-
# @FileName: controller.py
"""
TrajectoryController — the execution side of HumanEgo inference (reference template).

The policy predicts a short FUTURE trajectory of end-effector poses; this class
turns that trajectory into smooth, rate-limited robot motion. It is the clean,
synchronous distillation of inference/InferenceController.py (which additionally
runs asynchronously and temporally-ensembles overlapping predictions — see the
notes at the bottom for what we left out and why).

What it does each control step, per arm:
    1. EMA-smooth the target position + Slerp-smooth the rotation toward the goal
    2. clamp the per-step motion (safety cage) so a bad prediction can't lunge
    3. send the EE target to the arm (non-blocking Cartesian servo)
    4. open/close the gripper by thresholding the predicted grasp probability

Everything is in the CAMERA frame; the arm driver converts to its base frame.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from interfaces import RobotArm


def _slerp(R_from: np.ndarray, R_to: np.ndarray, frac: float) -> np.ndarray:
    """Spherical interpolation between two rotation matrices (frac in [0,1])."""
    key = R.from_matrix(np.stack([R_from, R_to]))
    return Slerp([0.0, 1.0], key)([frac]).as_matrix()[0]


class TrajectoryController:
    def __init__(
        self,
        arms: Dict[str, RobotArm],
        cfg: dict,
        advance_simulation: Optional[Callable[[float], None]] = None,
    ):
        """
        Args:
            arms: {"left": RobotArm, "right": RobotArm}  (or a single arm).
            cfg:  the `control` section of the inference config.
        """
        self.arms = arms
        self.alpha_pos = cfg.get("alpha_pos", 0.5)        # EMA: higher = smoother/slower
        self.alpha_rot = cfg.get("alpha_rot", 0.5)
        self.max_pos_step = cfg.get("max_pos_step", 0.05)  # m, per control step (safety cage)
        self.max_rot_step = cfg.get("max_rot_step", None)   # rad, per control step
        self.grasp_threshold = cfg.get("grasp_threshold", 0.5)
        self.safe_z_min = cfg.get("safe_z_min", None)      # optional base-frame Z floor (driver-handled)
        self._last_cmd: Dict[str, np.ndarray] = {s: None for s in arms}
        self.advance_simulation = advance_simulation
        self.realtime_sleep = bool(cfg.get("realtime_sleep", advance_simulation is None))

    def reset(self, seed_from_robot: bool = False) -> None:
        """Reset smoothing, optionally seeding it from measured EE poses.

        Seeding avoids sending the first raw policy prediction without applying
        the per-step displacement clamp.
        """
        if seed_from_robot:
            self._last_cmd = {
                side: arm.get_T_ee_in_cam().copy() for side, arm in self.arms.items()
            }
        else:
            self._last_cmd = {s: None for s in self.arms}

    # ----------------------------------------------------------------
    def _smooth(self, side: str, T_target: np.ndarray) -> np.ndarray:
        """EMA position + Slerp rotation + per-step clamp, relative to last command."""
        prev = self._last_cmd[side]
        if prev is None:
            out = T_target.copy()
        else:
            # position: EMA toward target, then clamp the step length
            p = self.alpha_pos * prev[:3, 3] + (1.0 - self.alpha_pos) * T_target[:3, 3]
            step = p - prev[:3, 3]
            dist = float(np.linalg.norm(step))
            if dist > self.max_pos_step:
                p = prev[:3, 3] + step / dist * self.max_pos_step
            # rotation: Slerp toward target and optionally clamp angular motion.
            rot_fraction = 1.0 - self.alpha_rot
            if self.max_rot_step is not None:
                relative = R.from_matrix(prev[:3, :3].T @ T_target[:3, :3])
                angle = float(relative.magnitude())
                if angle > 1e-9:
                    rot_fraction = min(rot_fraction, float(self.max_rot_step) / angle)
            R_out = _slerp(prev[:3, :3], T_target[:3, :3], rot_fraction)
            out = np.eye(4, dtype=np.float32)
            out[:3, :3], out[:3, 3] = R_out, p
        self._last_cmd[side] = out
        return out

    # ----------------------------------------------------------------
    def execute_chunk(
        self,
        traj_per_arm: Dict[str, List[np.ndarray]],   # {side: [T_ee_in_cam (4,4)] * H}
        grasp_per_arm: Dict[str, np.ndarray],        # {side: grasp_prob (H,)}
        dt: float,
        n_steps: int,
    ) -> None:
        """Execute the first `n_steps` of the predicted trajectory (receding horizon).

        Only a few of the H predicted steps are run before the loop re-plans on a
        fresh observation — this is what keeps the policy reactive (closed-loop).
        """
        for k in range(n_steps):
            for side, arm in self.arms.items():
                if side not in traj_per_arm or k >= len(traj_per_arm[side]):
                    continue  # this arm isn't driven by the policy this episode
                T_cmd = self._smooth(side, traj_per_arm[side][k])
                ok = arm.move_ee_in_cam(T_cmd, duration=dt, blocking=False)
                if not ok:
                    # IK / reachability failure: skip this step rather than fight it.
                    # (Production code optionally nudges the EE up in Z to escape singularities.)
                    self._last_cmd[side] = arm.get_T_ee_in_cam()
                grasp = float(grasp_per_arm[side][k])
                arm.set_gripper(1.0 if grasp > self.grasp_threshold else 0.0)
            if self.advance_simulation is not None:
                self.advance_simulation(dt)
            if self.realtime_sleep:
                time.sleep(dt)

    # ----------------------------------------------------------------
    def home(self) -> None:
        for arm in self.arms.values():
            arm.set_gripper(0.0, blocking=True)  # open
            arm.go_home(blocking=True)


# ─────────────────────────────────────────────────────────────────────────────
# Left out vs. the production inference/InferenceController.py (add if you need it):
#   * Async worker thread running at a fixed control rate, decoupled from the
#     (slower) policy inference rate — gives smoother motion at high control Hz.
#   * Temporal ensembling: average several overlapping predictions per timestep
#     (weighted) before executing — markedly reduces jitter.
#   * Grasp latching: once the gripper closes on an object, keep it closed.
#   * Forced post-grasp lift, draggable/manual override, richer safety limits.
# The conceptual core — smooth + rate-limit + servo EE + threshold grasp — is here.
# ─────────────────────────────────────────────────────────────────────────────
