"""HumanEgo Camera, Perception and RobotArm adapters backed by MuJoCo."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _WORKSPACE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ik_data import IKResult, solve_ik_dls
from interfaces import Camera, Frame, ObjectState, Perception, RobotArm
from mujoco_backend import MujocoBackend
from preprocess.VisualKpts import VisualKptsEngine


def _as_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    return transform


class MujocoCamera(Camera):
    def __init__(self, backend: MujocoBackend) -> None:
        self.backend = backend

    def get_frame(self) -> Frame:
        if not self.backend.render_raw_frame:
            return Frame(
                rgb=np.zeros((self.backend.height, self.backend.width, 3), dtype=np.uint8),
                depth_m=np.zeros((self.backend.height, self.backend.width), dtype=np.float32),
                K=self.backend.camera_intrinsics.astype(np.float32),
            )
        rgb = self.backend.render_rgb(hide_robot=False)
        depth = self.backend.render_depth(hide_robot=False)
        return Frame(
            rgb=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth_m=depth,
            K=self.backend.camera_intrinsics.astype(np.float32),
        )

    def close(self) -> None:
        # The runner owns the shared backend and closes it after all adapters.
        pass


class MujocoPerception(Perception):
    """Read object truth and render the training-time visual abstraction."""

    def __init__(self, backend: MujocoBackend, cfg: dict) -> None:
        self.backend = backend
        self.anchor_key = cfg.get("anchor_key", "obj1")
        self.freeze_anchor_after_grasp = bool(
            cfg.get("freeze_anchor_after_grasp", False)
        )
        self._anchor_pose_initial: np.ndarray | None = None
        self._anchor_pose_frozen = False
        self.object_specs = cfg["objects"]
        self.renderer = VisualKptsEngine(cfg["visualkpts_cfg_path"])

    @staticmethod
    def _make_local_keypoints(spec: dict) -> np.ndarray:
        count = int(spec.get("keypoint_count", 24))
        shape = spec.get("keypoint_shape", "ellipse")
        if "keypoints_local" in spec:
            return np.asarray(spec["keypoints_local"], dtype=np.float32).reshape(-1, 3)
        if shape not in ("ellipse", "circle"):
            raise ValueError(f"Unsupported simulated keypoint shape: {shape}")
        radii = np.asarray(spec.get("keypoint_radii", [0.05, 0.03]), dtype=np.float32)
        if radii.shape != (2,):
            raise ValueError("keypoint_radii must contain [radius_x, radius_y]")
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        points = np.zeros((count, 3), dtype=np.float32)
        points[:, 0] = radii[0] * np.cos(angles)
        points[:, 1] = radii[1] * np.sin(angles)
        points[:, 2] = float(spec.get("keypoint_z", 0.0))
        return points

    def get_objects(self) -> Dict[str, ObjectState]:
        objects: Dict[str, ObjectState] = {}
        for key, spec in self.object_specs.items():
            T_body_in_world = self.backend.body_pose_world(spec["body"])
            T_object_in_body = _as_transform(spec.get("T_object_in_body", np.eye(4)))
            T_object_in_cam = self.backend.pose_world_to_camera(
                T_body_in_world @ T_object_in_body
            )
            if key == self.anchor_key and self._anchor_pose_initial is None:
                self._anchor_pose_initial = T_object_in_cam.copy()
            if (
                key == self.anchor_key
                and self.freeze_anchor_after_grasp
                and self._anchor_pose_frozen
                and self._anchor_pose_initial is not None
            ):
                T_object_in_cam = self._anchor_pose_initial.copy()
            objects[key] = ObjectState(
                T_in_cam=T_object_in_cam.astype(np.float32),
                kpts_local=self._make_local_keypoints(spec),
            )
        return objects

    def set_anchor_frozen(self, frozen: bool) -> None:
        """Match the released checkpoint's static-anchor preprocessing.

        The serve_bread DatasetGen selected the manipulated bread as anchor and
        skipped it in its kinematic latch state machine.  Keep the physical
        body dynamic, but freeze only the policy-side anchor token/keypoints.
        """
        if self.freeze_anchor_after_grasp:
            self._anchor_pose_frozen = bool(frozen)

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        del frames
        return self.get_objects()

    @staticmethod
    def _project_keypoints(
        state: ObjectState, intrinsics: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(state.kpts_local, dtype=np.float64)
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        homogeneous = np.column_stack((points, np.ones(len(points))))
        points_camera = (state.T_in_cam @ homogeneous.T).T[:, :3]
        valid = points_camera[:, 2] > 1e-4
        projected = np.full((len(points), 2), np.nan, dtype=np.float32)
        uvw = (intrinsics @ points_camera[valid].T).T
        projected[valid] = uvw[:, :2] / uvw[:, 2:3]
        return projected

    def make_clean_image(
        self,
        frame: Frame,
        ee_poses_in_cam: Dict[str, np.ndarray],
        grippers: Dict[str, float],
    ) -> np.ndarray:
        del frame
        clean_rgb = self.backend.render_rgb(hide_robot=True)
        clean = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
        intrinsics = self.backend.camera_intrinsics

        # Match preprocessing order: virtual hand first, object keypoints second.
        # HumanEgo's hand frame is (+X jaw, +Y approach, +Z normal), while
        # process_single_gripper's local wireframe uses (+X jaw, +Z approach)
        # and draws the wrist along -Z.  Convert only the visualization pose;
        # ICT continues to receive the original model hand frame.
        hand_to_visual = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        for side in sorted(ee_poses_in_cam):
            pose = ee_poses_in_cam[side]
            if pose is not None:
                visual_pose = np.asarray(pose, dtype=np.float64).copy()
                visual_pose[:3, :3] = visual_pose[:3, :3] @ hand_to_visual
                clean = self.renderer.process_single_gripper(
                    clean,
                    visual_pose,
                    bool(grippers.get(side, 0.0) > 0.5),
                    intrinsics,
                )

        self.renderer.reset_obj_counter()
        objects = self.get_objects()
        for key in self.object_specs:
            state = objects[key]
            points_2d = self._project_keypoints(state, intrinsics)
            clean = self.renderer.process_single_obj(clean, points_2d)
        return clean

    def task_success(
        self,
        gripper_closed: float,
        *,
        radius: float = 0.10,
        max_height: float = 0.10,
    ) -> bool:
        bread = self.backend.body_pose_world(self.object_specs[self.anchor_key]["body"])
        target_keys = [key for key in self.object_specs if key != self.anchor_key]
        if not target_keys:
            return False
        plate = self.backend.body_pose_world(self.object_specs[target_keys[0]]["body"])
        horizontal_distance = float(np.linalg.norm(bread[:2, 3] - plate[:2, 3]))
        height = float(bread[2, 3] - plate[2, 3])
        return horizontal_distance <= radius and height <= max_height and gripper_closed < 0.5


class MujocoRobotArm(RobotArm):
    def __init__(self, backend: MujocoBackend, cfg: dict) -> None:
        self.backend = backend
        self.site_name = cfg.get("ee_site", "ee_site")
        self.base_body = cfg.get("base_body", "piper_mount")
        self.joint_names = tuple(cfg.get("joint_names", [f"joint{i}" for i in range(1, 7)]))
        self.actuator_names = tuple(
            cfg.get("actuator_names", [f"a_joint{i}" for i in range(1, 7)])
        )
        if len(self.joint_names) != len(self.actuator_names):
            raise ValueError("joint_names and actuator_names must have equal length")

        model = backend.model
        self.joint_ids = np.asarray(
            [backend.name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        )
        self.qpos_ids = np.asarray([model.jnt_qposadr[jid] for jid in self.joint_ids])
        self.dof_ids = np.asarray([model.jnt_dofadr[jid] for jid in self.joint_ids])
        self.actuator_ids = np.asarray(
            [backend.name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names]
        )

        self.gripper_joint_id = backend.name2id(
            mujoco.mjtObj.mjOBJ_JOINT, cfg.get("gripper_joint", "gripper")
        )
        self.gripper_qpos_id = int(model.jnt_qposadr[self.gripper_joint_id])
        self.gripper_actuator_id = backend.name2id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.get("gripper_actuator", "a_gripper")
        )
        self.gripper_open_qpos = float(cfg.get("gripper_open_qpos", 0.1))
        self.gripper_closed_qpos = float(cfg.get("gripper_closed_qpos", 0.0))
        self.home_qpos = np.asarray(cfg["home_qpos"], dtype=np.float64)
        if self.home_qpos.shape != (len(self.joint_names),):
            raise ValueError("home_qpos must contain one value for each arm joint")
        self.home_duration = float(cfg.get("home_duration", 2.0))
        self.teleport_home = bool(cfg.get("teleport_home", True))
        self.safe_world_min = np.asarray(
            cfg.get("workspace_min_world", [-0.65, -0.45, 0.81]), dtype=np.float64
        )
        self.safe_world_max = np.asarray(
            cfg.get("workspace_max_world", [0.65, 0.45, 1.45]), dtype=np.float64
        )
        self.ik_cfg = cfg.get("ik", {})
        self.T_align = _as_transform(cfg.get("T_align", np.eye(4)))
        # A 6-DoF Piper cannot realize every human-hand wrist orientation at
        # the calibrated approach pose.  When enabled, retain the policy's
        # Cartesian target position exactly and use orientation only for the
        # diagnostic pose; no simulator-side target is introduced.
        self.ik_position_only = bool(self.ik_cfg.get("position_only", False))
        self.max_solution_joint_delta = float(
            cfg.get("max_solution_joint_delta", 1.0)
        )
        self.max_command_joint_step = float(cfg.get("max_command_joint_step", 0.12))
        self.direct_ik = bool(self.ik_cfg.get("direct_qpos", False))
        self.ik_accept_position_error = float(
            cfg.get("ik_accept_position_error", 0.015)
        )
        self.ik_accept_rotation_error = float(
            cfg.get("ik_accept_rotation_error", 0.10)
        )
        self.last_ik: IKResult | None = None
        self.ik_failure_count = 0
        self.last_rejection_reason = ""

        grasp_cfg = cfg.get("grasp_validation", {})
        self.grasp_validation_enabled = bool(grasp_cfg.get("enabled", False))
        # When false, contact data remains available for diagnostics, but it
        # never overrides policy Cartesian or gripper commands.
        self.grasp_enforce_control = bool(
            grasp_cfg.get("enforce_control", True)
        )
        self.grasp_max_center_distance = float(
            grasp_cfg.get("max_center_distance", 0.06)
        )
        self.grasp_allow_single_contact = bool(
            grasp_cfg.get("allow_single_contact_to_close", True)
        )
        self.grasp_allow_single_contact_after_alignment = bool(
            grasp_cfg.get("allow_single_contact_after_alignment", False)
        )
        self.grasp_auto_align = bool(
            grasp_cfg.get("auto_align_single_contact", False)
        )
        self.grasp_prealign_distance = float(
            grasp_cfg.get("prealign_distance", 0.0)
        )
        self.grasp_prealign_max_step = float(
            grasp_cfg.get("prealign_max_step", 0.04)
        )
        self.grasp_prealign_accept_position_error = float(
            grasp_cfg.get("prealign_accept_position_error", 0.03)
        )
        self.grasp_prealign_accept_rotation_error = float(
            grasp_cfg.get("prealign_accept_rotation_error", 0.12)
        )
        self.grasp_prealign_max_jaw_vertical = float(
            grasp_cfg.get("prealign_max_jaw_vertical", 0.25)
        )
        self.grasp_prealign_min_approach_vertical = float(
            grasp_cfg.get("prealign_min_approach_vertical", 0.85)
        )
        self.grasp_alignment_tolerance = float(
            grasp_cfg.get("alignment_tolerance", 0.008)
        )
        self.grasp_alignment_min_world_z = float(
            grasp_cfg.get("alignment_min_world_z", self.safe_world_min[2])
        )
        self.grasp_alignment_max_step = float(
            grasp_cfg.get("alignment_max_step", 0.003)
        )
        self.grasp_alignment_damping = float(
            grasp_cfg.get("alignment_damping", 0.03)
        )
        self.grasp_alignment_rotation_weight = float(
            grasp_cfg.get("alignment_rotation_weight", 0.3)
        )
        self.grasp_alignment_max_joint_step = float(
            grasp_cfg.get("alignment_max_joint_step", 0.06)
        )
        self.grasp_alignment_timeout = float(
            grasp_cfg.get("alignment_timeout", 3.0)
        )
        self.grasp_alignment_retreat_distance = float(
            grasp_cfg.get("alignment_retreat_distance", 0.035)
        )
        self.grasp_alignment_waypoint_tolerance = float(
            grasp_cfg.get("alignment_waypoint_tolerance", 0.003)
        )
        self.grasp_alignment_center_tolerance = float(
            grasp_cfg.get("alignment_center_tolerance", 0.02)
        )
        self.grasp_alignment_min_clearance = float(
            grasp_cfg.get("alignment_min_clearance", 0.005)
        )
        self.grasp_close_timeout = float(grasp_cfg.get("close_timeout", 0.5))
        self.grasp_retry_cooldown = float(grasp_cfg.get("retry_cooldown", 0.3))
        self.grasp_contact_stable_time = float(
            grasp_cfg.get("stable_contact_time", 0.15)
        )
        self.grasp_object_geom_id: int | None = None
        self.grasp_object_joint_id: int | None = None
        self.grasp_object_qpos_ids: np.ndarray | None = None
        self.grasp_finger_geom_ids: tuple[int, ...] = ()
        self.grasp_site_id: int | None = None
        if self.grasp_validation_enabled:
            self.grasp_object_geom_id = backend.name2id(
                mujoco.mjtObj.mjOBJ_GEOM,
                grasp_cfg.get("object_geom", "croissant_geom"),
            )
            self.grasp_object_joint_id = backend.name2id(
                mujoco.mjtObj.mjOBJ_JOINT,
                grasp_cfg.get("object_joint", "croissant_joint"),
            )
            object_qpos_start = int(model.jnt_qposadr[self.grasp_object_joint_id])
            self.grasp_object_qpos_ids = np.arange(
                object_qpos_start, object_qpos_start + 7, dtype=np.int32
            )
            finger_names = tuple(
                grasp_cfg.get(
                    "finger_geoms",
                    ["piper_finger1_geom", "piper_finger2_geom"],
                )
            )
            if len(finger_names) != 2:
                raise ValueError("grasp_validation.finger_geoms must contain two names")
            self.grasp_finger_geom_ids = tuple(
                backend.name2id(mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in finger_names
            )
            self.grasp_site_id = backend.name2id(
                mujoco.mjtObj.mjOBJ_SITE,
                grasp_cfg.get("grasp_site", "grasp_site"),
            )
            backend.step_hook = self._step_hook
        self._grasp_state = "open"
        self._grasp_requested_closed = False
        self._grasp_close_started = -np.inf
        self._grasp_retry_after = -np.inf
        self._grasp_blocked_by_distance = False
        self._grasp_alignment_active = False
        self._grasp_alignment_deadline = -np.inf
        self._grasp_alignment_phase = "idle"
        self._grasp_alignment_waypoints: tuple[np.ndarray, ...] = ()
        self._grasp_alignment_waypoint_error = float("nan")
        self._grasp_alignment_joint_step = 0.0
        self._grasp_alignment_ik_success = False
        self._grasp_alignment_ik_position_error = float("nan")
        self._grasp_alignment_ik_rotation_error = float("nan")
        self._grasp_alignment_solution_world = np.full(3, np.nan, dtype=np.float64)
        self._grasp_attachment_active = False
        self._grasp_attachment_offset_world = np.zeros(3, dtype=np.float64)
        self._grasp_attachment_rotation = np.eye(3, dtype=np.float64)
        self._grasp_object_frame_in_body = np.eye(4, dtype=np.float64)
        self._grasp_alignment_target_quaternion = np.full(4, np.nan)
        self._grasp_suggested_align_translation = np.full(3, np.nan)
        self._grasp_alignment_start_axes_world = np.full((2, 3), np.nan)
        self._grasp_suggested_ee_site_quaternion = np.full(4, np.nan)
        self._grasp_prealign_phase = "idle"
        self._grasp_contact_since: float | None = None

        self.T_base_in_cam = self.backend.pose_world_to_camera(
            self.backend.body_pose_world(self.base_body)
        ).astype(np.float32)

    def _step_hook(self) -> None:
        if not self._grasp_attachment_active or self.grasp_object_qpos_ids is None:
            return
        site_position = self.backend.data.site_xpos[int(self.grasp_site_id)].copy()
        site_rotation = self.backend.data.site_xmat[int(self.grasp_site_id)].reshape(3, 3)
        hand_position = site_position + site_rotation @ self.T_align[:3, 3]
        hand_rotation = site_rotation @ self.T_align[:3, :3]
        object_frame_position = hand_position + self._grasp_attachment_offset_world
        object_frame_rotation = hand_rotation @ self._grasp_attachment_rotation
        object_in_body = self._grasp_object_frame_in_body
        body_rotation = object_frame_rotation @ object_in_body[:3, :3].T
        body_position = object_frame_position - body_rotation @ object_in_body[:3, 3]
        qpos_ids = self.grasp_object_qpos_ids
        self.backend.data.qpos[qpos_ids[:3]] = body_position
        object_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(object_quaternion, body_rotation.reshape(-1))
        self.backend.data.qpos[qpos_ids[3:7]] = object_quaternion
        dof_start = int(self.backend.model.jnt_dofadr[self.grasp_object_joint_id])
        self.backend.data.qvel[dof_start : dof_start + 6] = 0.0
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def activate_task_attachment(self, object_pose_world: np.ndarray | None = None) -> bool:
        """Latch the currently validated grasp for the task-level controller.

        The policy is responsible for reaching and closing the gripper.  Once
        that grasp has been validated, the placement controller may explicitly
        take ownership of object transport without changing the policy's
        camera/action calibration.
        """
        if not self.grasp_validation_enabled or self.grasp_object_geom_id is None:
            return False
        with self.backend.lock:
            if not all(self._grasp_contacts_locked()):
                return False
            site_id = int(self.grasp_site_id)
            object_id = int(self.grasp_object_geom_id)
            site_position = self.backend.data.site_xpos[site_id].copy()
            site_rotation = self.backend.data.site_xmat[site_id].reshape(3, 3)
            hand_position = site_position + site_rotation @ self.T_align[:3, 3]
            hand_rotation = site_rotation @ self.T_align[:3, :3]
            body_id = int(self.backend.model.geom_bodyid[object_id])
            body_pose = np.eye(4, dtype=np.float64)
            body_pose[:3, :3] = self.backend.data.xmat[body_id].reshape(3, 3)
            body_pose[:3, 3] = self.backend.data.xpos[body_id]
            if object_pose_world is None:
                object_pose_world = body_pose.copy()
                object_pose_world[:3, :3] = self.backend.data.geom_xmat[object_id].reshape(3, 3)
                object_pose_world[:3, 3] = self.backend.data.geom_xpos[object_id]
            object_pose_world = _as_transform(object_pose_world)
            self._grasp_object_frame_in_body = np.linalg.inv(body_pose) @ object_pose_world
            self._grasp_attachment_offset_world = (
                object_pose_world[:3, 3]
                - hand_position
            )
            self._grasp_attachment_rotation = (
                hand_rotation.T @ object_pose_world[:3, :3]
            )
            self._grasp_attachment_active = True
            self._grasp_requested_closed = True
            self._grasp_state = "grasped"
            self._grasp_blocked_by_distance = False
            self.backend.data.ctrl[self.gripper_actuator_id] = self.gripper_closed_qpos
            return True

    def _grasp_contacts_locked(self) -> tuple[bool, bool]:
        """Return whether each finger is contacting the configured object."""
        if not self.grasp_validation_enabled:
            return False, False
        contacts = [False, False]
        object_id = int(self.grasp_object_geom_id)
        for contact_index in range(self.backend.data.ncon):
            contact = self.backend.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            for finger_index, finger_id in enumerate(self.grasp_finger_geom_ids):
                if (geom1 == finger_id and geom2 == object_id) or (
                    geom2 == finger_id and geom1 == object_id
                ):
                    contacts[finger_index] = True
        return bool(contacts[0]), bool(contacts[1])

    def _grasp_center_distance_locked(self) -> float:
        if not self.grasp_validation_enabled:
            return float("nan")
        object_position = self.backend.data.geom_xpos[int(self.grasp_object_geom_id)]
        grasp_position = self.backend.data.site_xpos[int(self.grasp_site_id)]
        return float(np.linalg.norm(grasp_position - object_position))

    def _jaw_geometry_locked(self) -> tuple[float, float, np.ndarray]:
        """Return object errors and the world-space finger-closing axis."""
        if not self.grasp_validation_enabled:
            return float("nan"), float("nan"), np.zeros(3, dtype=np.float64)
        finger_positions = np.asarray(
            [
                self.backend.data.geom_xpos[geom_id]
                for geom_id in self.grasp_finger_geom_ids
            ],
            dtype=np.float64,
        )
        jaw_vector = finger_positions[1] - finger_positions[0]
        jaw_norm = float(np.linalg.norm(jaw_vector))
        if jaw_norm < 1e-9:
            return float("nan"), float("nan"), np.zeros(3, dtype=np.float64)
        jaw_axis = jaw_vector / jaw_norm
        jaw_midpoint = finger_positions.mean(axis=0)
        object_position = self.backend.data.geom_xpos[int(self.grasp_object_geom_id)]
        offset = object_position - jaw_midpoint
        lateral = float(np.dot(offset, jaw_axis))
        orthogonal = float(np.linalg.norm(offset - lateral * jaw_axis))
        return lateral, orthogonal, jaw_axis

    def _jaw_alignment_locked(self) -> tuple[float, float]:
        lateral, orthogonal, _ = self._jaw_geometry_locked()
        return lateral, orthogonal

    def _refresh_grasp_state_locked(self) -> tuple[bool, bool]:
        contacts = self._grasp_contacts_locked()
        if not self.grasp_validation_enabled or not self._grasp_requested_closed:
            return contacts
        if not self.grasp_enforce_control:
            # Pure policy evaluation: do not run the timeout/retry state
            # machine.  Contact truth is sampled by diagnostics only.
            return contacts
        if all(contacts):
            if self._grasp_state == "closing":
                if self._grasp_contact_since is None:
                    self._grasp_contact_since = float(self.backend.data.time)
                stable_for = (
                    float(self.backend.data.time) - self._grasp_contact_since
                )
                if stable_for >= self.grasp_contact_stable_time:
                    self._grasp_state = "grasped"
                    self._grasp_blocked_by_distance = False
                    self._grasp_alignment_active = False
                    self._grasp_alignment_deadline = -np.inf
                    self._grasp_alignment_phase = "idle"
                    self._grasp_alignment_waypoints = ()
                    self._grasp_alignment_waypoint_error = float("nan")
                    self._grasp_alignment_joint_step = 0.0
            elif self._grasp_state == "grasped":
                self._grasp_blocked_by_distance = False
        else:
            self._grasp_contact_since = None
            if self._grasp_state == "grasped" and not self._grasp_attachment_active:
                self._grasp_state = "closing"
                self._grasp_blocked_by_distance = True

        if (
            self._grasp_state == "closing"
            and self.backend.data.time - self._grasp_close_started
            >= self.grasp_close_timeout
        ):
            # A one-sided or empty closure is not a grasp. Reopen and let the
            # policy adjust the pose before another distance-gated attempt.
            self.backend.data.ctrl[self.gripper_actuator_id] = self.gripper_open_qpos
            self._grasp_state = "retry_wait"
            self._grasp_retry_after = (
                float(self.backend.data.time) + self.grasp_retry_cooldown
            )
        return contacts

    def _start_grasp_alignment_locked(self) -> None:
        """Plan a collision-free retreat/shift/re-approach maneuver."""
        grasp_position = self.backend.data.site_xpos[int(self.grasp_site_id)].copy()
        grasp_rotation = self.backend.data.site_xmat[int(self.grasp_site_id)].reshape(
            3, 3
        )
        object_position = self.backend.data.geom_xpos[int(self.grasp_object_geom_id)]
        lateral_error, _, jaw_axis = self._jaw_geometry_locked()
        horizontal_jaw_axis = jaw_axis.copy()
        horizontal_jaw_axis[2] = 0.0
        horizontal_jaw_norm = float(np.linalg.norm(horizontal_jaw_axis))
        if horizontal_jaw_norm > 1e-9:
            horizontal_jaw_axis /= horizontal_jaw_norm
        else:
            horizontal_jaw_axis = jaw_axis.copy()

        # T_hand = T_ee @ T_align.  For the same predicted hand pose, a local
        # T_align translation t moves the decoded EE by -R_ee @ t.  Therefore
        # this is the translation that would have centered the jaw before the
        # first side contact; it is useful as an embodiment calibration signal.
        ee_site_id = self.backend.name2id(mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        ee_rotation = self.backend.data.site_xmat[ee_site_id].reshape(3, 3)
        desired_world_shift = lateral_error * horizontal_jaw_axis
        self._grasp_suggested_align_translation = (
            -ee_rotation.T @ desired_world_shift
        )
        self._grasp_alignment_start_axes_world = np.stack(
            (jaw_axis, grasp_rotation[:, 2])
        )
        mujoco.mju_mat2Quat(
            self._grasp_alignment_target_quaternion,
            grasp_rotation.reshape(-1),
        )
        desired_jaw_axis = jaw_axis.copy()
        desired_jaw_axis[2] = 0.0
        desired_jaw_norm = float(np.linalg.norm(desired_jaw_axis))
        if desired_jaw_norm > 1e-9:
            desired_jaw_axis /= desired_jaw_norm
            desired_tool_z = np.array([0.0, 0.0, 1.0])
            desired_tool_x = np.cross(desired_jaw_axis, desired_tool_z)
            desired_body_rotation = np.column_stack(
                (desired_tool_x, desired_jaw_axis, desired_tool_z)
            )
            body_correction = ee_rotation.T @ desired_body_rotation
            # R_site = R_body @ R_site_local and R_body_desired =
            # R_site_target @ body_correction, hence R_site_local=Q^T.
            suggested_site_rotation = body_correction.T
            mujoco.mju_mat2Quat(
                self._grasp_suggested_ee_site_quaternion,
                suggested_site_rotation.reshape(-1),
            )

        # The gripper's local Z is its approach axis. Pick its sign from the
        # object direction so the first waypoint always moves away from contact.
        retreat_axis = grasp_rotation[:, 2].copy()
        if float(np.dot(retreat_axis, object_position - grasp_position)) > 0.0:
            retreat_axis *= -1.0
        retreat = self.grasp_alignment_retreat_distance * retreat_axis
        # The object lies on the tabletop.  A jaw-axis correction is intended
        # to be horizontal; retaining a small wrist-induced Z component can
        # drive the free body into or off the table during re-alignment.
        lateral_shift = lateral_error * horizontal_jaw_axis
        self._grasp_alignment_waypoints = (
            grasp_position + retreat,
            grasp_position + retreat + lateral_shift,
            grasp_position + lateral_shift,
        )
        self._grasp_alignment_phase = "retracting"
        self._grasp_alignment_active = True
        self._grasp_alignment_waypoint_error = self.grasp_alignment_retreat_distance
        self._grasp_alignment_joint_step = 0.0
        self._grasp_alignment_deadline = (
            float(self.backend.data.time) + self.grasp_alignment_timeout
        )

    def _reset_grasp_alignment_locked(self) -> None:
        self._grasp_alignment_active = False
        self._grasp_attachment_active = False
        self._grasp_attachment_rotation = np.eye(3, dtype=np.float64)
        self._grasp_object_frame_in_body = np.eye(4, dtype=np.float64)
        self._grasp_alignment_deadline = -np.inf
        self._grasp_alignment_phase = "idle"
        self._grasp_alignment_waypoints = ()
        self._grasp_alignment_waypoint_error = float("nan")
        self._grasp_alignment_joint_step = 0.0
        self._grasp_contact_since = None

    def _apply_grasp_alignment_step_locked(self) -> bool:
        """Move the open jaw toward the object using position-only DLS.

        This deliberately bypasses the normal 6DoF IK path: the corrective
        translation can be smaller than its acceptance tolerance, while an
        unnecessary orientation solve can reject an otherwise safe lateral
        adjustment.
        """
        phase_indices = {"retracting": 0, "shifting": 1, "approaching": 2}
        waypoint_index = phase_indices.get(self._grasp_alignment_phase)
        if waypoint_index is None or len(self._grasp_alignment_waypoints) != 3:
            return False
        current_position = self.backend.data.site_xpos[int(self.grasp_site_id)]
        waypoint = self._grasp_alignment_waypoints[waypoint_index]

        # During the lateral shift the croissant is a free body and may move
        # slightly after one finger touches it.  Recompute the correction from
        # the current jaw/object geometry instead of chasing the stale shift
        # waypoint created when alignment started.  Retraction and final
        # approach remain explicit fixed waypoints.
        if self._grasp_alignment_phase == "shifting":
            lateral_error, _, jaw_axis = self._jaw_geometry_locked()
            if np.isfinite(lateral_error) and np.linalg.norm(jaw_axis) > 1e-9:
                # Stop the lateral correction as soon as the jaw is centered.
                # Continuing to use the signed error for another control tick
                # can push a free object past the center and into the table.
                if abs(lateral_error) <= self.grasp_alignment_tolerance:
                    self._grasp_alignment_phase = "approaching"
                    waypoint = self.backend.data.geom_xpos[
                        int(self.grasp_object_geom_id)
                    ].copy()
                else:
                    shift_axis = jaw_axis.copy()
                    shift_axis[2] = 0.0
                    shift_norm = float(np.linalg.norm(shift_axis))
                    if shift_norm > 1e-9:
                        shift_axis /= shift_norm
                    else:
                        shift_axis = jaw_axis
                    waypoint = current_position + lateral_error * shift_axis
        elif self._grasp_alignment_phase == "approaching":
            # Once the jaw is laterally centered, approach the object's current
            # center rather than the pre-contact waypoint.  The object is a
            # free body, so the original approach target can be stale after a
            # one-sided contact/retreat cycle.
            waypoint = self.backend.data.geom_xpos[
                int(self.grasp_object_geom_id)
            ].copy()

        # Alignment uses a dedicated position-only IK path, so enforce the
        # same workspace cage here as the normal Cartesian path.  In
        # particular, never let the fingertip midpoint enter the tabletop.
        waypoint = np.maximum(waypoint, self.safe_world_min)
        waypoint = np.minimum(waypoint, self.safe_world_max)
        waypoint[2] = max(waypoint[2], self.grasp_alignment_min_world_z)
        position_delta = waypoint - current_position
        distance = float(np.linalg.norm(position_delta))
        self._grasp_alignment_waypoint_error = distance
        contact_cleared = (
            self._grasp_alignment_phase == "retracting"
            and not any(self._grasp_contacts_locked())
            and distance >= self.grasp_alignment_min_clearance
        )
        if distance <= self.grasp_alignment_waypoint_tolerance or contact_cleared:
            if self._grasp_alignment_phase == "retracting":
                # A large fixed retreat can be unreachable near a joint limit.
                # Once contact has cleared, retain the achieved clearance and
                # rebase only the lateral waypoint on the measured position.
                lateral_shift = (
                    self._grasp_alignment_waypoints[1]
                    - self._grasp_alignment_waypoints[0]
                )
                approach_target = self._grasp_alignment_waypoints[2]
                self._grasp_alignment_waypoints = (
                    current_position.copy(),
                    current_position + lateral_shift,
                    approach_target,
                )
                self._grasp_alignment_phase = "shifting"
            elif self._grasp_alignment_phase == "shifting":
                self._grasp_alignment_phase = "approaching"
            else:
                self._grasp_alignment_phase = "ready"
                self.backend.data.ctrl[self.actuator_ids] = self.backend.data.qpos[
                    self.qpos_ids
                ]
                return False
            waypoint_index += 1
            waypoint = self._grasp_alignment_waypoints[waypoint_index]
            position_delta = waypoint - current_position
            distance = float(np.linalg.norm(position_delta))
            self._grasp_alignment_waypoint_error = distance
        if distance > self.grasp_alignment_max_step:
            position_delta *= self.grasp_alignment_max_step / distance

        # Use the bounded incremental target for IK.  Previously the bounded
        # delta was computed only for diagnostics, while IK still received the
        # full waypoint; that could overshoot the lateral alignment target.
        ik_target = current_position + position_delta
        if self._grasp_alignment_phase == "approaching":
            # The final approach is intentionally solved against the current
            # object center.  Unlike the lateral shift, it must close the
            # whole remaining orthogonal gap in one position-only solve.
            ik_target = waypoint.copy()
        alignment_target_quaternion = self._grasp_alignment_target_quaternion
        alignment_rotation_weight = self.grasp_alignment_rotation_weight
        if self._grasp_alignment_phase == "approaching":
            # The approach phase should close the positional gap to the object;
            # re-solving the original contact orientation here can trade away
            # the position correction near a Piper singularity.  Keep the
            # orientation currently held by the site and use position-only IK.
            current_quaternion = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(
                current_quaternion,
                self.backend.data.site_xmat[int(self.grasp_site_id)].reshape(-1),
            )
            alignment_target_quaternion = current_quaternion
            alignment_rotation_weight = 0.0

        grasp_site_name = mujoco.mj_id2name(
            self.backend.model,
            mujoco.mjtObj.mjOBJ_SITE,
            int(self.grasp_site_id),
        )
        alignment_ik = solve_ik_dls(
            self.backend.model,
            self.backend.data.qpos.copy(),
            grasp_site_name,
            ik_target,
            alignment_target_quaternion,
            self.joint_names,
            max_iter=80,
            position_tolerance=self.grasp_alignment_waypoint_tolerance * 0.5,
            rotation_tolerance=0.10,
            damping=self.grasp_alignment_damping,
            rotation_weight=alignment_rotation_weight,
            max_joint_step=self.grasp_alignment_max_joint_step,
        )
        self._grasp_alignment_ik_success = bool(alignment_ik.success)
        self._grasp_alignment_ik_position_error = float(alignment_ik.position_error)
        self._grasp_alignment_ik_rotation_error = float(alignment_ik.rotation_error)
        preview_data = mujoco.MjData(self.backend.model)
        preview_data.qpos[:] = alignment_ik.qpos
        preview_data.qvel[:] = 0.0
        mujoco.mj_forward(self.backend.model, preview_data)
        self._grasp_alignment_solution_world = preview_data.site_xpos[
            int(self.grasp_site_id)
        ].copy()
        current_qpos = self.backend.data.qpos[self.qpos_ids].copy()
        if not np.isfinite(alignment_ik.qpos).all():
            self.last_rejection_reason = "grasp alignment IK returned invalid joints"
            self.backend.data.ctrl[self.actuator_ids] = current_qpos
            self._reset_grasp_alignment_locked()
            self._grasp_state = "retry_wait"
            self._grasp_retry_after = float(self.backend.data.time) + self.grasp_retry_cooldown
            return False

        if not alignment_ik.success:
            self.last_rejection_reason = (
                "grasp alignment IK did not converge: "
                f"pos={alignment_ik.position_error:.4f} "
                f"rot={alignment_ik.rotation_error:.4f}"
            )
            self.backend.data.ctrl[self.actuator_ids] = current_qpos
            self._grasp_alignment_joint_step = 0.0
            self._reset_grasp_alignment_locked()
            self._grasp_state = "retry_wait"
            self._grasp_retry_after = float(self.backend.data.time) + self.grasp_retry_cooldown
            return False

        reached_approach_target = (
            self._grasp_alignment_phase == "approaching"
            and alignment_ik.position_error
            <= self.grasp_alignment_waypoint_tolerance
            and self._grasp_center_distance_locked()
            <= self.grasp_alignment_center_tolerance
        )

        if self._grasp_alignment_phase == "approaching":
            # Once the jaw is centered, the full position-only solution is the
            # desired final approach.  Keeping the per-tick joint clip here
            # leaves a large orthogonal gap at postures where the actuator
            # path is poorly conditioned.
            commanded_qpos = alignment_ik.qpos[self.qpos_ids].copy()
            command_delta = commanded_qpos - current_qpos
        else:
            solution_delta = alignment_ik.qpos[self.qpos_ids] - current_qpos
            command_delta = np.clip(
                solution_delta,
                -self.grasp_alignment_max_joint_step,
                self.grasp_alignment_max_joint_step,
            )
            commanded_qpos = current_qpos + command_delta
        self._grasp_alignment_joint_step = float(np.max(np.abs(command_delta)))
        joint_ranges = self.backend.model.jnt_range[self.joint_ids]
        joint_limited = self.backend.model.jnt_limited[self.joint_ids].astype(bool)
        commanded_qpos[joint_limited] = np.clip(
            commanded_qpos[joint_limited],
            joint_ranges[joint_limited, 0],
            joint_ranges[joint_limited, 1],
        )
        # Alignment is a simulator-side recovery maneuver.  The position
        # actuators can lag or move through a collision-constrained posture in
        # the opposite direction, even after the DLS solution has converged.
        # Apply this bounded, already-validated joint increment directly so
        # the corrective path follows the same IK geometry it was solved for.
        self.backend.data.qpos[self.qpos_ids] = commanded_qpos
        self.backend.data.qvel[self.dof_ids] = 0.0
        mujoco.mj_forward(self.backend.model, self.backend.data)
        self.backend.data.ctrl[self.actuator_ids] = commanded_qpos
        self.last_rejection_reason = ""
        if reached_approach_target:
            self._grasp_alignment_active = False
            self._grasp_alignment_phase = "idle"
            self._grasp_alignment_waypoints = ()
            self._grasp_alignment_waypoint_error = 0.0
            # Reaching the alignment waypoint only means that the jaw is
            # positioned for closure.  Physical grasp success is decided by
            # _refresh_grasp_state_locked after both finger contacts appear.
            self._grasp_state = "closing"
            self._grasp_close_started = float(self.backend.data.time)
            self._grasp_blocked_by_distance = False
        return True

    def get_grasp_diagnostics(self) -> dict:
        """Expose simulation-only grasp truth for logging and verification."""
        with self.backend.lock:
            contacts = self._refresh_grasp_state_locked()
            lateral_error, orthogonal_error = self._jaw_alignment_locked()
            diagnostic_site_id = (
                int(self.grasp_site_id)
                if self.grasp_site_id is not None
                else self.backend.name2id(mujoco.mjtObj.mjOBJ_SITE, self.site_name)
            )
            alignment_waypoint = np.full(3, np.nan, dtype=np.float64)
            if self._grasp_alignment_active and self._grasp_alignment_waypoints:
                phase_indices = {"retracting": 0, "shifting": 1, "approaching": 2}
                waypoint_index = phase_indices.get(self._grasp_alignment_phase)
                if waypoint_index is not None:
                    alignment_waypoint = self._grasp_alignment_waypoints[waypoint_index].copy()
            return {
                "state": self._grasp_state,
                "requested_closed": self._grasp_requested_closed,
                "center_distance": self._grasp_center_distance_locked(),
                "finger_contacts": contacts,
                "contact_stable_time": (
                    float(self.backend.data.time) - self._grasp_contact_since
                    if self._grasp_contact_since is not None
                    else 0.0
                ),
                "jaw_lateral_error": lateral_error,
                "jaw_orthogonal_error": orthogonal_error,
                "jaw_axis_world": (
                    self._jaw_geometry_locked()[2].copy()
                    if self.grasp_validation_enabled
                    else np.zeros(3, dtype=np.float64)
                ),
                "grasp_approach_axis_world": (
                    self.backend.data.site_xmat[int(self.grasp_site_id)]
                    .reshape(3, 3)[:, 2]
                    .copy()
                    if self.grasp_validation_enabled
                    else np.zeros(3, dtype=np.float64)
                ),
                "alignment_phase": self._grasp_alignment_phase,
                "alignment_waypoint_error": self._grasp_alignment_waypoint_error,
                "alignment_joint_step": self._grasp_alignment_joint_step,
                "alignment_ik_success": self._grasp_alignment_ik_success,
                "alignment_ik_position_error": self._grasp_alignment_ik_position_error,
                "alignment_ik_rotation_error": self._grasp_alignment_ik_rotation_error,
                "grasp_site_world": self.backend.data.site_xpos[diagnostic_site_id].copy(),
                "grasp_object_world": (
                    self.backend.data.geom_xpos[int(self.grasp_object_geom_id)].copy()
                    if self.grasp_object_geom_id is not None
                    else np.full(3, np.nan, dtype=np.float64)
                ),
                "attachment_active": self._grasp_attachment_active,
                "attachment_offset_world": self._grasp_attachment_offset_world.copy(),
                "alignment_waypoint_world": alignment_waypoint,
                "alignment_solution_world": self._grasp_alignment_solution_world.copy(),
                "arm_ctrl_gap": float(
                    np.max(
                        np.abs(
                            self.backend.data.ctrl[self.actuator_ids]
                            - self.backend.data.qpos[self.qpos_ids]
                        )
                    )
                ),
                "prealign_phase": self._grasp_prealign_phase,
                "suggested_align_translation": (
                    self._grasp_suggested_align_translation.copy()
                ),
                "alignment_start_axes_world": (
                    self._grasp_alignment_start_axes_world.copy()
                ),
                "suggested_ee_site_quaternion": (
                    self._grasp_suggested_ee_site_quaternion.copy()
                ),
                "blocked_by_distance": self._grasp_blocked_by_distance,
                "motion_held": (
                    self.grasp_validation_enabled
                    and self.grasp_enforce_control
                    and self._grasp_requested_closed
                    and (
                        self._grasp_state
                        in ("closing", "retry_wait", "alignment_required")
                        or any(contacts)
                    )
                ),
            }

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self.backend.pose_world_to_camera(
            self.backend.site_pose_world(self.site_name)
        ).astype(np.float32)

    def move_ee_in_cam(
        self, T_ee_in_cam: np.ndarray, duration: float, blocking: bool = False
    ) -> bool:
        with self.backend.lock:
            contacts = self._refresh_grasp_state_locked()
            if (
                self.grasp_validation_enabled
                and self.grasp_enforce_control
                and self.grasp_auto_align
                and self._grasp_requested_closed
                and self._grasp_alignment_active
            ):
                self._grasp_state = "aligning"
                return self._apply_grasp_alignment_step_locked()
            if (
                self.grasp_validation_enabled
                and self.grasp_enforce_control
                and self._grasp_requested_closed
                and (
                    self._grasp_state
                    in ("closing", "retry_wait", "alignment_required")
                    or (any(contacts) and not self._grasp_attachment_active)
                )
            ):
                # Keep the fingertips stationary while the jaw closes. Moving
                # before bilateral contact is what causes a one-sided drag.
                current_arm_qpos = self.backend.data.qpos[self.qpos_ids]
                for actuator_id, command in zip(
                    self.actuator_ids, current_arm_qpos
                ):
                    self.backend.data.ctrl[actuator_id] = command
                return True

        target_camera = np.asarray(T_ee_in_cam, dtype=np.float64)
        if target_camera.shape != (4, 4) or not np.isfinite(target_camera).all():
            self.ik_failure_count += 1
            self.last_rejection_reason = "invalid target transform"
            return False
        target_world = self.backend.pose_camera_to_world(target_camera)
        target_position = target_world[:3, 3]
        with self.backend.lock:
            self._grasp_prealign_phase = "idle"
            contacts = self._grasp_contacts_locked()
            should_prealign = (
                self.grasp_validation_enabled
                and self.grasp_enforce_control
                and self.grasp_auto_align
                and self._grasp_requested_closed
                and not self._grasp_alignment_active
                and not any(contacts)
                and self._grasp_center_distance_locked()
                <= self.grasp_prealign_distance
            )
            if should_prealign:
                lateral_error, _, jaw_axis = self._jaw_geometry_locked()
                if np.isfinite(lateral_error):
                    ee_site_id = self.backend.name2id(
                        mujoco.mjtObj.mjOBJ_SITE, self.site_name
                    )
                    current_position = self.backend.data.site_xpos[ee_site_id]
                    approach_axis = self.backend.data.site_xmat[
                        int(self.grasp_site_id)
                    ].reshape(3, 3)[:, 2]
                    orientation_ready = (
                        abs(float(jaw_axis[2]))
                        <= self.grasp_prealign_max_jaw_vertical
                        and abs(float(approach_axis[2]))
                        >= self.grasp_prealign_min_approach_vertical
                    )
                    if not orientation_ready:
                        self._grasp_prealign_phase = "orienting"
                        target_position = current_position.copy()
                        target_world[:3, 3] = target_position
                    elif abs(lateral_error) > self.grasp_alignment_tolerance:
                        self._grasp_prealign_phase = "centering"
                        desired_lateral = float(
                            np.clip(
                                lateral_error,
                                -self.grasp_prealign_max_step,
                                self.grasp_prealign_max_step,
                            )
                        )
                        # Freeze approach and wrist rotation until the open jaw
                        # is centered. Otherwise the policy reaches the object
                        # before the lateral correction can take effect.
                        centering_axis = jaw_axis.copy()
                        centering_axis[2] = 0.0
                        centering_norm = float(np.linalg.norm(centering_axis))
                        if centering_norm > 1e-9:
                            centering_axis /= centering_norm
                        target_position = (
                            current_position + desired_lateral * centering_axis
                        )
                        target_world[:3, 3] = target_position
                        target_world[:3, :3] = self.backend.data.site_xmat[
                            ee_site_id
                        ].reshape(3, 3)
                    else:
                        self._grasp_prealign_phase = "aligned"
        # The policy was trained on a hand/control point above the physical
        # finger mesh.  Keep the finger mesh clear of the tabletop by lifting
        # only an under-height Z target; retain hard rejection for lateral or
        # upper workspace violations.
        if target_position[2] < self.safe_world_min[2]:
            target_position = target_position.copy()
            target_position[2] = self.safe_world_min[2]
            target_world[:3, 3] = target_position
        lateral_or_upper_violation = (
            np.any(target_position[:2] < self.safe_world_min[:2])
            or np.any(target_position[:2] > self.safe_world_max[:2])
            or target_position[2] > self.safe_world_max[2]
        )
        if lateral_or_upper_violation:
            self.ik_failure_count += 1
            self.last_rejection_reason = (
                f"target outside world workspace: {np.round(target_position, 3)}"
            )
            return False

        target_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quaternion, target_world[:3, :3].reshape(-1))
        with self.backend.lock:
            self.last_ik = solve_ik_dls(
                self.backend.model,
                self.backend.data.qpos.copy(),
                self.site_name,
                target_position,
                target_quaternion,
                self.joint_names,
                max_iter=int(self.ik_cfg.get("max_iter", 120)),
                position_tolerance=float(self.ik_cfg.get("position_tolerance", 1e-3)),
                rotation_tolerance=float(self.ik_cfg.get("rotation_tolerance", 2e-2)),
                damping=float(self.ik_cfg.get("damping", 5e-2)),
                rotation_weight=float(self.ik_cfg.get("rotation_weight", 0.35)),
                max_joint_step=float(self.ik_cfg.get("max_joint_step", 0.12)),
                position_only=self.ik_position_only,
            )
            accepted_position_error = (
                self.grasp_prealign_accept_position_error
                if should_prealign
                else self.ik_accept_position_error
            )
            accepted_rotation_error = (
                self.grasp_prealign_accept_rotation_error
                if should_prealign
                else self.ik_accept_rotation_error
            )
            approximately_reached = (
                self.last_ik.position_error <= accepted_position_error
                and self.last_ik.rotation_error <= accepted_rotation_error
            )
            if not self.last_ik.success and not approximately_reached:
                self.ik_failure_count += 1
                self.last_rejection_reason = (
                    f"IK did not converge: pos={self.last_ik.position_error:.4f} "
                    f"rot={self.last_ik.rotation_error:.4f}"
                )
                return False
            current_arm_qpos = self.backend.data.qpos[self.qpos_ids].copy()
            solution_delta = self.last_ik.qpos[self.qpos_ids] - current_arm_qpos
            if float(np.max(np.abs(solution_delta))) > self.max_solution_joint_delta:
                self.ik_failure_count += 1
                self.last_rejection_reason = (
                    "IK solution is on a distant joint branch: "
                    f"max_delta={np.max(np.abs(solution_delta)):.3f}"
                )
                return False
            commanded_qpos = current_arm_qpos + np.clip(
                solution_delta,
                -self.max_command_joint_step,
                self.max_command_joint_step,
            )
            for actuator_id, command in zip(self.actuator_ids, commanded_qpos):
                self.backend.data.ctrl[actuator_id] = command
            if self.direct_ik:
                # Simulation-only position servo: the target still originates
                # from the policy and is solved through named-joint IK; this
                # removes low-level PID/gravity lag from calibration tests.
                self.backend.data.qpos[self.qpos_ids] = commanded_qpos
                self.backend.data.qvel[self.dof_ids] = 0.0
                mujoco.mj_forward(self.backend.model, self.backend.data)
            self.last_rejection_reason = ""
        if blocking:
            self.backend.step_for(duration)
        return True

    def get_gripper(self) -> float:
        with self.backend.lock:
            self._refresh_grasp_state_locked()
            current = float(self.backend.data.qpos[self.gripper_qpos_id])
        denominator = self.gripper_open_qpos - self.gripper_closed_qpos
        if abs(denominator) < 1e-9:
            return 0.0
        return float(np.clip((self.gripper_open_qpos - current) / denominator, 0.0, 1.0))

    def set_gripper(self, value: float, blocking: bool = False) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        with self.backend.lock:
            now = float(self.backend.data.time)
            policy_requests_closed = value > 0.5
            alignment_latched = (
                self.grasp_enforce_control
                and
                self._grasp_state in ("aligning", "closing")
                and now < self._grasp_alignment_deadline
            )
            requested_closed = policy_requests_closed or alignment_latched
            self._grasp_requested_closed = requested_closed
            contacts = self._refresh_grasp_state_locked()

            if self._grasp_alignment_active and now >= self._grasp_alignment_deadline:
                self._reset_grasp_alignment_locked()
                self._grasp_state = "retry_wait"
                self._grasp_retry_after = now + self.grasp_retry_cooldown

            if not requested_closed:
                self._grasp_state = "open"
                self._grasp_blocked_by_distance = False
                self._reset_grasp_alignment_locked()
                target = self.gripper_open_qpos
            elif not self.grasp_enforce_control:
                # Pure policy mode: the scalar gripper prediction is the only
                # source of the close command.  Contact truth is diagnostic.
                self._grasp_state = "closing"
                self._grasp_blocked_by_distance = False
                target = self.gripper_closed_qpos
            elif not self.grasp_validation_enabled:
                self._grasp_state = "closing"
                target = self.gripper_closed_qpos
            elif self._grasp_state == "grasped":
                target = self.gripper_closed_qpos
            elif self._grasp_state == "closing":
                target = self.gripper_closed_qpos
            elif self.backend.data.time < self._grasp_retry_after:
                self._grasp_state = "retry_wait"
                target = self.gripper_open_qpos
            elif self._grasp_alignment_active:
                lateral_error, _ = self._jaw_alignment_locked()
                if (
                    self._grasp_alignment_phase == "approaching"
                    and self._grasp_alignment_ik_success
                    and self._grasp_alignment_ik_position_error
                    <= self.grasp_alignment_waypoint_tolerance
                ):
                    self._grasp_alignment_active = False
                    self._grasp_alignment_phase = "idle"
                    self._grasp_alignment_waypoints = ()
                    self._grasp_alignment_waypoint_error = 0.0
                    self._grasp_state = "closing"
                    self._grasp_close_started = now
                    self._grasp_blocked_by_distance = False
                    target = self.gripper_closed_qpos
                elif (
                    self._grasp_alignment_phase == "shifting"
                    and np.isfinite(lateral_error)
                    and abs(lateral_error) <= self.grasp_alignment_tolerance
                ):
                    self._grasp_alignment_phase = "approaching"
                    self._grasp_state = "aligning"
                    self._grasp_blocked_by_distance = True
                    target = self.gripper_open_qpos
                elif (
                    self._grasp_alignment_phase == "ready"
                    and abs(lateral_error) <= self.grasp_alignment_tolerance
                ):
                    self._grasp_alignment_active = False
                    self._grasp_alignment_phase = "idle"
                    self._grasp_alignment_waypoints = ()
                    self._grasp_state = "closing"
                    self._grasp_close_started = now
                    self._grasp_alignment_deadline = now + self.grasp_close_timeout
                    self._grasp_blocked_by_distance = False
                    target = self.gripper_closed_qpos
                else:
                    self._grasp_state = "aligning"
                    self._grasp_blocked_by_distance = True
                    target = self.gripper_open_qpos
            elif any(contacts) and not self.grasp_allow_single_contact:
                if self.grasp_auto_align:
                    self._start_grasp_alignment_locked()
                self._grasp_state = (
                    "aligning" if self._grasp_alignment_active else "alignment_required"
                )
                self._grasp_blocked_by_distance = True
                target = self.gripper_open_qpos
            else:
                distance = self._grasp_center_distance_locked()
                close_is_near = distance <= self.grasp_max_center_distance
                close_has_contact = self.grasp_allow_single_contact and any(contacts)
                if close_is_near or close_has_contact:
                    self._grasp_state = "closing"
                    self._grasp_close_started = float(self.backend.data.time)
                    self._grasp_blocked_by_distance = False
                    target = self.gripper_closed_qpos
                else:
                    self._grasp_state = "waiting_near_object"
                    self._grasp_blocked_by_distance = True
                    target = self.gripper_open_qpos
            self.backend.data.ctrl[self.gripper_actuator_id] = target
        if blocking:
            self.backend.step_for(0.5)
            with self.backend.lock:
                self._refresh_grasp_state_locked()

    def go_home(self, blocking: bool = True) -> None:
        with self.backend.lock:
            self._grasp_state = "open"
            self._grasp_requested_closed = False
            self._grasp_blocked_by_distance = False
            self._grasp_close_started = -np.inf
            self._grasp_retry_after = -np.inf
            self._reset_grasp_alignment_locked()
            self.backend.data.ctrl[self.actuator_ids] = self.home_qpos
            if self.teleport_home:
                self.backend.data.qpos[self.qpos_ids] = self.home_qpos
                self.backend.data.qvel[self.dof_ids] = 0.0
                mujoco.mj_forward(self.backend.model, self.backend.data)
        if blocking:
            self.backend.step_for(self.home_duration)

    def close(self) -> None:
        pass
