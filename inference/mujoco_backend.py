"""Shared MuJoCo runtime and coordinate conversions for HumanEgo simulation."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


_R_CV_FROM_MJ_CAMERA = np.diag([1.0, -1.0, -1.0])


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


class MujocoBackend:
    """Own the one model/data pair shared by camera, perception and robot."""

    def __init__(
        self,
        xml_path: str,
        *,
        camera_name: str = "ego_cam",
        width: int = 720,
        height: int = 480,
        robot_geom_group: int = 1,
        show_viewer: bool = False,
        render_raw_frame: bool = False,
    ) -> None:
        self.xml_path = str(Path(xml_path).expanduser().resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.robot_geom_group = int(robot_geom_group)
        # The MuJoCo inference path builds the training-style clean image
        # itself.  A raw RGB-D frame is therefore optional and can be skipped
        # to avoid two extra renderer passes per policy cycle.
        self.render_raw_frame = bool(render_raw_frame)
        self.lock = threading.RLock()
        self.step_hook = None

        self.camera_id = self.name2id(mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.clean_scene_option = mujoco.MjvOption()
        if not 0 <= self.robot_geom_group < len(self.clean_scene_option.geomgroup):
            raise ValueError(f"robot geom group must be in [0, 5], got {robot_geom_group}")
        self.clean_scene_option.geomgroup[self.robot_geom_group] = 0

        self.viewer = None
        if show_viewer:
            from mujoco import viewer as mujoco_viewer

            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        self.hold_actuators_at_current_position()

    def name2id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object does not exist: {name}")
        return int(object_id)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def camera_intrinsics(self) -> np.ndarray:
        vertical_fov = math.radians(float(self.model.cam_fovy[self.camera_id]))
        focal = 0.5 * self.height / math.tan(0.5 * vertical_fov)
        return np.array(
            [
                [focal, 0.0, 0.5 * self.width],
                [0.0, focal, 0.5 * self.height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def hold_actuators_at_current_position(self) -> None:
        """Initialize joint position actuators from current qpos."""
        with self.lock:
            for actuator_id in range(self.model.nu):
                joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                if joint_id >= 0:
                    qpos_id = int(self.model.jnt_qposadr[joint_id])
                    value = float(self.data.qpos[qpos_id])
                    low, high = self.model.actuator_ctrlrange[actuator_id]
                    self.data.ctrl[actuator_id] = np.clip(value, low, high)

    def reset(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.hold_actuators_at_current_position()

    def step_for(self, duration: float) -> None:
        steps = max(1, int(round(float(duration) / self.timestep)))
        with self.lock:
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
                if self.step_hook is not None:
                    self.step_hook()
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def forward(self) -> None:
        with self.lock:
            mujoco.mj_forward(self.model, self.data)

    def render_rgb(self, *, hide_robot: bool = False) -> np.ndarray:
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            option = self.clean_scene_option if hide_robot else None
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(
                self.data, camera=self.camera_name, scene_option=option
            )
            return self.renderer.render().copy()

    def render_depth(self, *, hide_robot: bool = False) -> np.ndarray:
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            option = self.clean_scene_option if hide_robot else None
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(
                self.data, camera=self.camera_name, scene_option=option
            )
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            return depth.astype(np.float32, copy=False)

    def T_camera_cv_in_world(self) -> np.ndarray:
        """Pose of the OpenCV optical camera frame in MuJoCo world."""
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            rotation_world_from_mj_camera = self.data.cam_xmat[self.camera_id].reshape(3, 3)
            rotation_world_from_cv_camera = (
                rotation_world_from_mj_camera @ _R_CV_FROM_MJ_CAMERA
            )
            return _pose(rotation_world_from_cv_camera, self.data.cam_xpos[self.camera_id])

    def T_world_in_camera_cv(self) -> np.ndarray:
        return np.linalg.inv(self.T_camera_cv_in_world())

    def body_pose_world(self, body_name: str) -> np.ndarray:
        body_id = self.name2id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            return _pose(
                self.data.xmat[body_id].reshape(3, 3), self.data.xpos[body_id]
            )

    def site_pose_world(self, site_name: str) -> np.ndarray:
        site_id = self.name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            return _pose(
                self.data.site_xmat[site_id].reshape(3, 3),
                self.data.site_xpos[site_id],
            )

    def pose_world_to_camera(self, T_in_world: np.ndarray) -> np.ndarray:
        return self.T_world_in_camera_cv() @ np.asarray(T_in_world, dtype=np.float64)

    def pose_camera_to_world(self, T_in_camera: np.ndarray) -> np.ndarray:
        return self.T_camera_cv_in_world() @ np.asarray(T_in_camera, dtype=np.float64)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def wait_for_viewer(self) -> None:
        """Keep a passive viewer open until the user closes it."""
        if self.viewer is None:
            return
        print("[sim] viewer is paused at the final state; close the window or press Ctrl+C to exit.")
        try:
            while self.viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[sim] viewer interrupted.")
