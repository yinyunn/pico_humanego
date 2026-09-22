"""Task-level object placement controller for the MuJoCo validation scene."""

from __future__ import annotations

import numpy as np
import mujoco

from ik_data import solve_ik_dls
from mujoco_backend import MujocoBackend


class MujocoTaskPlacement:
    """Place the currently grasped free body on a configured target body.

    The policy is deliberately not called while this controller is active.
    The object is held by the simulator-side grasp attachment and the arm is
    driven through three Cartesian object waypoints: lift, transfer, place.
    """

    def __init__(self, backend: MujocoBackend, arm, cfg: dict) -> None:
        self.backend = backend
        self.arm = arm
        self.cfg = cfg or {}
        self.target_body = str(self.cfg.get("target_body", "plate"))
        self.object_geom = str(self.cfg.get("object_geom", "croissant_geom"))
        self.placement_clearance = float(self.cfg.get("placement_clearance", 0.006))
        self.carry_height = float(self.cfg.get("carry_height", 1.02))
        self.waypoint_tolerance = float(self.cfg.get("waypoint_tolerance", 0.012))
        self.release_settle_time = float(self.cfg.get("release_settle_time", 0.35))
        self.max_joint_step = float(
            self.cfg.get("max_joint_step", arm.max_command_joint_step)
        )
        self.ik_cfg = self.cfg.get("ik", {})
        self.active = False
        self.ready_to_release = False
        self.released = False
        self.release_elapsed = 0.0
        self.phase = "idle"
        self.phase_index = 0
        self.waypoints_object_world: list[np.ndarray] = []
        self.attachment_offset_world = np.zeros(3, dtype=np.float64)
        self.last_error = float("nan")
        self.last_ik = None
        self.last_object_position = np.full(3, np.nan, dtype=np.float64)
        self.last_site_position = np.full(3, np.nan, dtype=np.float64)
        self.last_target_site = np.full(3, np.nan, dtype=np.float64)
        self.last_command_qpos = None

        self.object_geom_id = backend.name2id(
            mujoco.mjtObj.mjOBJ_GEOM, self.object_geom
        )

    def start(self) -> bool:
        """Start placement after the simulator sees a bilateral grasp.

        In pure-policy grasp mode the diagnostic state intentionally remains
        ``closing`` because the grasp state machine is disabled.  Bilateral
        finger/object contacts are the physical fact we need here, so do not
        require the policy-independent ``grasped`` state before latching the
        task attachment.
        """
        if self.active:
            return True
        diagnostics = self.arm.get_grasp_diagnostics()
        finger_contacts = tuple(diagnostics.get("finger_contacts", ()))
        if finger_contacts != (True, True):
            return False
        if not self.arm.activate_task_attachment():
            self.last_error = float("nan")
            return False

        with self.backend.lock:
            object_position = self.backend.data.geom_xpos[self.object_geom_id].copy()
            self.last_object_position = object_position.copy()
            plate_body_id = self.backend.name2id(
                mujoco.mjtObj.mjOBJ_BODY, self.target_body
            )
            plate_position = self.backend.data.xpos[plate_body_id].copy()
            plate_geom_id = self.backend.name2id(
                mujoco.mjtObj.mjOBJ_GEOM, f"{self.target_body}_geom"
            )
            plate_half_height = float(self.backend.model.geom_size[plate_geom_id, 1])
            object_half_height = float(
                self.backend.model.geom_size[self.object_geom_id, 2]
            )
            place_z = (
                plate_position[2]
                + plate_half_height
                + object_half_height
                + self.placement_clearance
            )
            carry_z = max(
                self.carry_height,
                float(object_position[2]) + 0.10,
                float(place_z) + 0.10,
            )
            self.waypoints_object_world = [
                np.array([object_position[0], object_position[1], carry_z]),
                np.array([plate_position[0], plate_position[1], carry_z]),
                np.array([plate_position[0], plate_position[1], place_z]),
            ]
            self.attachment_offset_world = (
                self.arm._grasp_attachment_offset_world.copy()
            )

        self.active = True
        self.phase_index = 0
        self.phase = "lift"
        self.ready_to_release = False
        self.released = False
        self.release_elapsed = 0.0
        self.arm.set_gripper(1.0, blocking=False)
        return True

    def _solve_current_waypoint(self) -> bool:
        target_object = self.waypoints_object_world[self.phase_index]
        target_site = target_object - self.attachment_offset_world
        self.last_target_site = target_site.copy()
        with self.backend.lock:
            site_id = self.backend.name2id(
                mujoco.mjtObj.mjOBJ_SITE, self.arm.site_name
            )
            current_site = self.backend.data.site_xpos[site_id].copy()
            current_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(
                current_quat,
                self.backend.data.site_xmat[site_id].reshape(-1),
            )
            self.last_ik = solve_ik_dls(
                self.backend.model,
                self.backend.data.qpos.copy(),
                self.arm.site_name,
                target_site,
                current_quat,
                self.arm.joint_names,
                max_iter=int(self.ik_cfg.get("max_iter", 180)),
                position_tolerance=float(
                    self.ik_cfg.get("position_tolerance", 0.006)
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
            current_qpos = self.backend.data.qpos[self.arm.qpos_ids].copy()
            delta = self.last_ik.qpos[self.arm.qpos_ids] - current_qpos
            if bool(self.cfg.get("direct_solution", True)):
                command = self.last_ik.qpos[self.arm.qpos_ids].copy()
            else:
                command = current_qpos + np.clip(
                    delta, -self.max_joint_step, self.max_joint_step
                )
            joint_ranges = self.backend.model.jnt_range[self.arm.joint_ids]
            limited = self.backend.model.jnt_limited[self.arm.joint_ids].astype(bool)
            command[limited] = np.clip(
                command[limited], joint_ranges[limited, 0], joint_ranges[limited, 1]
            )
            # The placement controller is a simulator-side task controller.
            # Apply the bounded, validated IK increment directly so gravity and
            # actuator lag cannot pull the arm away from a transport waypoint
            # between two control ticks.  Keep the actuator targets in sync for
            # the subsequent physics step.
            self.backend.data.qpos[self.arm.qpos_ids] = command
            self.backend.data.qvel[self.arm.dof_ids] = 0.0
            mujoco.mj_forward(self.backend.model, self.backend.data)
            self.backend.data.ctrl[self.arm.actuator_ids] = command
            self.last_command_qpos = command.copy()
        return True

    def step(self, duration: float) -> str:
        """Advance placement by one control period."""
        if not self.active:
            return "idle"
        if self.ready_to_release:
            return "ready_to_release"
        if not self._solve_current_waypoint():
            self.backend.step_for(duration)
            return "ik_failed"
        self.backend.step_for(duration)
        with self.backend.lock:
            # Position actuators and gravity can move the arm noticeably during
            # the physics interval.  This task-level controller owns the pose,
            # so re-apply the validated command at the end of the interval
            # before measuring the transported object.
            if self.last_command_qpos is not None:
                self.backend.data.qpos[self.arm.qpos_ids] = self.last_command_qpos
                self.backend.data.qvel[self.arm.dof_ids] = 0.0
                mujoco.mj_forward(self.backend.model, self.backend.data)
                if self.backend.step_hook is not None:
                    self.backend.step_hook()
            object_position = self.backend.data.geom_xpos[self.object_geom_id].copy()
            self.last_object_position = object_position.copy()
            site_id = self.backend.name2id(
                mujoco.mjtObj.mjOBJ_SITE, self.arm.site_name
            )
            self.last_site_position = self.backend.data.site_xpos[site_id].copy()
        self.last_error = float(
            np.linalg.norm(object_position - self.waypoints_object_world[self.phase_index])
        )
        if self.last_error <= self.waypoint_tolerance:
            self.phase_index += 1
            if self.phase_index >= len(self.waypoints_object_world):
                self.ready_to_release = True
                self.phase = "ready_to_release"
            else:
                self.phase = ("transfer", "place")[self.phase_index - 1]
        return self.phase

    def release_step(self, duration: float) -> bool:
        """Open the gripper and wait for the free body to settle."""
        if not self.ready_to_release:
            return False
        if not self.released:
            self.arm.set_gripper(0.0, blocking=False)
            self.released = True
        self.backend.step_for(duration)
        self.release_elapsed += float(duration)
        return self.release_elapsed >= self.release_settle_time
