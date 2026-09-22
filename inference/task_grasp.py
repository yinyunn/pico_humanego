"""Physical grasp assist for the MuJoCo validation scene.

The policy decides *when* to grasp, but this small controller owns the final
approach and closure.  It deliberately uses the configured physical
``grasp_site`` and the live object geom, so a policy hand-pose error cannot be
turned into a fake attachment by the task-level placement controller.
"""

from __future__ import annotations

import numpy as np
import mujoco

from ik_data import solve_ik_dls
from mujoco_backend import MujocoBackend


class MujocoGraspAssist:
    """Approach the object along the current jaw normal and verify both jaws."""

    def __init__(self, backend: MujocoBackend, arm, cfg: dict) -> None:
        self.backend = backend
        self.arm = arm
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.start_distance = float(self.cfg.get("start_distance", 0.12))
        self.pregrasp_clearance = float(self.cfg.get("pregrasp_clearance", 0.045))
        self.waypoint_tolerance = float(self.cfg.get("waypoint_tolerance", 0.010))
        self.approach_tolerance = float(self.cfg.get("approach_tolerance", 0.018))
        self.close_timeout = float(self.cfg.get("close_timeout", 0.80))
        self.max_joint_step = float(
            self.cfg.get("max_joint_step", arm.max_command_joint_step)
        )
        self.ik_cfg = self.cfg.get("ik", {})
        self.object_geom = str(
            self.cfg.get("object_geom", "croissant_geom")
        )
        self.object_geom_id = backend.name2id(
            mujoco.mjtObj.mjOBJ_GEOM, self.object_geom
        )
        self.site_id = backend.name2id(mujoco.mjtObj.mjOBJ_SITE, arm.site_name)

        self.phase = "idle"
        self.active = False
        self.failed = False
        self.started_at = -np.inf
        self.close_started = -np.inf
        self.approach_axis_world = np.zeros(3, dtype=np.float64)
        self.grasp_orientation = np.zeros(4, dtype=np.float64)
        self.pregrasp_target = np.full(3, np.nan, dtype=np.float64)
        self.approach_target = np.full(3, np.nan, dtype=np.float64)
        self.last_error = float("nan")
        self.last_site = np.full(3, np.nan, dtype=np.float64)
        self.last_object = np.full(3, np.nan, dtype=np.float64)
        self.last_ik = None
        self.last_command_qpos = None
        self._old_auto_align = bool(getattr(arm, "grasp_auto_align", False))
        self._old_allow_single = bool(
            getattr(arm, "grasp_allow_single_contact", True)
        )

    def _restore_arm_policy(self) -> None:
        self.arm.grasp_auto_align = self._old_auto_align
        self.arm.grasp_allow_single_contact = self._old_allow_single

    def start(self) -> bool:
        """Take ownership after policy intent, before any physical contact."""
        if not self.enabled or self.active:
            return self.active
        with self.backend.lock:
            site = self.backend.data.site_xpos[self.site_id].copy()
            obj = self.backend.data.geom_xpos[self.object_geom_id].copy()
            rotation = self.backend.data.site_xmat[self.site_id].reshape(3, 3).copy()
            axis = rotation[:, 2].copy()
            direction = obj - site
            if np.linalg.norm(axis) < 1e-8 or np.linalg.norm(direction) < 1e-8:
                return False
            axis /= np.linalg.norm(axis)
            # The approach axis is oriented from the tool toward the object.
            # This makes the pregrasp waypoint a retreat, rather than a push.
            if float(np.dot(axis, direction)) < 0.0:
                axis *= -1.0
            self.approach_axis_world = axis
            mujoco.mju_mat2Quat(self.grasp_orientation, rotation.reshape(-1))
            self.approach_target = obj.copy()
            self.pregrasp_target = obj - axis * self.pregrasp_clearance
            self.pregrasp_target[2] = max(
                self.pregrasp_target[2], float(self.arm.safe_world_min[2])
            )
            self.started_at = float(self.backend.data.time)

        # Disable the old single-contact alignment state machine while this
        # controller owns the final approach.  A one-sided contact is still
        # rejected by the bilateral contact check below, but it must not cause
        # an unrelated lateral waypoint to move the object.
        self.arm.grasp_auto_align = False
        self.arm.grasp_allow_single_contact = True
        self.arm.set_gripper(0.0, blocking=False)
        self.phase = "pregrasp"
        self.active = True
        self.failed = False
        return True

    def _command_site(self, target: np.ndarray) -> bool:
        with self.backend.lock:
            current_qpos = self.backend.data.qpos[self.arm.qpos_ids].copy()
            self.last_ik = solve_ik_dls(
                self.backend.model,
                self.backend.data.qpos.copy(),
                self.arm.site_name,
                target,
                self.grasp_orientation,
                self.arm.joint_names,
                max_iter=int(self.ik_cfg.get("max_iter", 180)),
                position_tolerance=float(
                    self.ik_cfg.get("position_tolerance", 0.004)
                ),
                rotation_tolerance=float(
                    self.ik_cfg.get("rotation_tolerance", 0.20)
                ),
                damping=float(self.ik_cfg.get("damping", 0.05)),
                rotation_weight=0.0,
                max_joint_step=float(self.ik_cfg.get("max_joint_step", 0.12)),
                position_only=True,
            )
            self.last_error = float(self.last_ik.position_error)
            if not self.last_ik.success:
                return False
            solution = self.last_ik.qpos[self.arm.qpos_ids].copy()
            delta = solution - current_qpos
            delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
            command = current_qpos + delta
            ranges = self.backend.model.jnt_range[self.arm.joint_ids]
            limited = self.backend.model.jnt_limited[self.arm.joint_ids].astype(bool)
            command[limited] = np.clip(command[limited], ranges[limited, 0], ranges[limited, 1])
            self.backend.data.qpos[self.arm.qpos_ids] = command
            self.backend.data.qvel[self.arm.dof_ids] = 0.0
            mujoco.mj_forward(self.backend.model, self.backend.data)
            self.backend.data.ctrl[self.arm.actuator_ids] = command
            self.last_command_qpos = command.copy()
        return True

    def _measure(self) -> tuple[float, tuple[bool, bool]]:
        with self.backend.lock:
            mujoco.mj_forward(self.backend.model, self.backend.data)
            self.last_site = self.backend.data.site_xpos[self.site_id].copy()
            self.last_object = self.backend.data.geom_xpos[self.object_geom_id].copy()
            distance = float(np.linalg.norm(self.last_site - self.last_object))
            contacts = self.arm._grasp_contacts_locked()
        return distance, contacts

    def step(self, duration: float) -> str:
        """Advance one assist tick and return the physical grasp phase."""
        if not self.active:
            return "failed" if self.failed else "idle"

        if self.phase in ("pregrasp", "approach"):
            if self.phase == "pregrasp":
                target = self.pregrasp_target
            else:
                # Track the live object only during the final approach.  This
                # keeps the target valid if the object moved by a tiny amount
                # during the pregrasp motion, without following it after a
                # contact has been detected.
                with self.backend.lock:
                    target = self.backend.data.geom_xpos[self.object_geom_id].copy()
                self.approach_target = target.copy()
            if not self._command_site(target):
                self.phase = "failed"
                self.failed = True
                self.active = False
                self._restore_arm_policy()
                return self.phase
            self.arm.set_gripper(0.0, blocking=False)
            self.backend.step_for(duration)
            distance, contacts = self._measure()
            if self.phase == "pregrasp" and self.last_error <= self.waypoint_tolerance:
                self.phase = "approach"
                print(
                    f"[grasp-assist] pregrasp reached site={np.round(self.last_site, 3)} "
                    f"object={np.round(self.last_object, 3)}"
                )
            elif self.phase == "approach" and (
                distance <= self.approach_tolerance or any(contacts)
            ):
                # A contact can be reported before the site reaches the
                # center-distance threshold.  Continuing to chase the live
                # object in that state pushes the free body away with the
                # first finger.  Freeze Cartesian motion immediately and let
                # the validated gripper state machine close in place.
                self.phase = "closing"
                self.close_started = float(self.backend.data.time)
                self.arm.set_gripper(1.0, blocking=False)
                print(
                    f"[grasp-assist] closing distance={distance:.4f} "
                    f"contacts={int(contacts[0])}{int(contacts[1])}"
                )
            return self.phase

        if self.phase == "closing":
            self.arm.set_gripper(1.0, blocking=False)
            self.backend.step_for(duration)
            distance, contacts = self._measure()
            diagnostics = self.arm.get_grasp_diagnostics()
            if (
                diagnostics.get("state") == "grasped"
                and contacts == (True, True)
                and diagnostics.get("contact_stable_time", 0.0)
                >= float(getattr(self.arm, "grasp_contact_stable_time", 0.15))
            ):
                self.phase = "grasped"
                self.active = False
                print(
                    f"[grasp-assist] bilateral grasp confirmed distance={distance:.4f} "
                    f"contacts={int(contacts[0])}{int(contacts[1])}"
                )
                return self.phase
            if float(self.backend.data.time) - self.close_started >= self.close_timeout:
                self.phase = "failed"
                self.failed = True
                self.active = False
                self._restore_arm_policy()
                print(
                    f"[grasp-assist] failed: bilateral contact not reached "
                    f"distance={distance:.4f} contacts={int(contacts[0])}{int(contacts[1])}"
                )
                return self.phase
            return self.phase

        return self.phase
