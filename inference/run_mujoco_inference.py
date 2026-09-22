"""Run a released HumanEgo policy in the Piper MuJoCo serve_bread scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml

_INFERENCE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _INFERENCE_DIR.parent
_WORKSPACE_ROOT = _REPO_ROOT.parent
for _path in (_INFERENCE_DIR, _REPO_ROOT, _WORKSPACE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from controller import TrajectoryController
from interfaces import RobotArm
from mujoco_adapters import MujocoCamera, MujocoPerception, MujocoRobotArm
from mujoco_backend import MujocoBackend
from policy import ICTPolicy
from task_grasp import MujocoGraspAssist
from task_placement import MujocoTaskPlacement


def _load_config(path: str) -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as stream:
        return yaml.safe_load(stream), config_path.parent


def _resolve_path(config_dir: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _project(point_camera: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    if point_camera[2] <= 1e-4:
        return None
    uvw = K @ point_camera
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def _draw_pose_axes(
    image: np.ndarray,
    T_pose_in_cam: np.ndarray,
    K: np.ndarray,
    label: str,
    axis_length: float = 0.06,
) -> np.ndarray:
    """Draw one right-handed pose frame in the OpenCV camera image.

    Axis colors follow the usual convention: X=red, Y=green, Z=blue.
    ``T_pose_in_cam`` is a pose of the frame in the OpenCV optical frame.
    """
    canvas = image
    T = np.asarray(T_pose_in_cam, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return canvas

    origin = _project(T[:3, 3], K)
    if origin is None:
        return canvas

    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR: X/Y/Z
    axis_names = ("X", "Y", "Z")
    cv2.circle(canvas, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (origin[0] + 5, origin[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for axis_index, (axis_name, color) in enumerate(zip(axis_names, axis_colors)):
        endpoint_3d = T[:3, 3] + axis_length * T[:3, axis_index]
        endpoint = _project(endpoint_3d, K)
        if endpoint is None:
            continue
        cv2.arrowedLine(canvas, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(
            canvas,
            f"{label}-{axis_name}",
            (endpoint[0] + 3, endpoint[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_camera_frame_legend(image: np.ndarray) -> np.ndarray:
    """Draw a fixed camera-frame triad because the camera origin is off-screen."""
    canvas = image
    origin = (42, 52)
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR: X/Y/Z
    axis_names = ("X right", "Y down", "Z forward")
    cv2.putText(
        canvas,
        "CAM (OpenCV)",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, (color, name) in enumerate(zip(axis_colors, axis_names)):
        end = (origin[0] + (28 if index == 0 else 0), origin[1] + (28 if index == 1 else 0))
        if index == 2:
            end = (origin[0] + 20, origin[1] - 20)
        cv2.arrowedLine(canvas, origin, end, color, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(
            canvas,
            name,
            (origin[0] + 34, origin[1] + 4 + index * 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_coordinate_frames(
    image: np.ndarray,
    K: np.ndarray,
    frames: List[tuple[str, np.ndarray, float]],
) -> np.ndarray:
    """Overlay camera, robot, hand, grasp and object coordinate frames."""
    canvas = _draw_camera_frame_legend(image.copy())
    for label, pose, axis_length in frames:
        canvas = _draw_pose_axes(canvas, pose, K, label, axis_length)
    return canvas


def _draw_alignment_metrics(
    image: np.ndarray,
    T_ee_in_cam: np.ndarray,
    T_grasp_in_cam: np.ndarray,
    T_hand_in_cam: np.ndarray,
    grasp_diagnostics: dict,
    backend: MujocoBackend,
) -> np.ndarray:
    """Draw frame-to-gripper alignment diagnostics without affecting control."""
    R_camera_from_world = backend.T_world_in_camera_cv()[:3, :3]
    jaw_axis_camera = R_camera_from_world @ np.asarray(
        grasp_diagnostics.get("jaw_axis_world", np.zeros(3)), dtype=np.float64
    )
    approach_axis_camera = R_camera_from_world @ np.asarray(
        grasp_diagnostics.get("grasp_approach_axis_world", np.zeros(3)),
        dtype=np.float64,
    )
    hand_x = T_hand_in_cam[:3, 0]
    hand_y = T_hand_in_cam[:3, 1]
    jaw_norm = np.linalg.norm(jaw_axis_camera)
    approach_norm = np.linalg.norm(approach_axis_camera)
    hand_x_dot_jaw = (
        float(np.dot(hand_x, jaw_axis_camera) / jaw_norm)
        if jaw_norm > 1e-8
        else float("nan")
    )
    hand_y_dot_approach = (
        float(np.dot(hand_y, approach_axis_camera) / approach_norm)
        if approach_norm > 1e-8
        else float("nan")
    )
    ee_grasp_offset = float(
        np.linalg.norm(T_grasp_in_cam[:3, 3] - T_ee_in_cam[:3, 3])
    )
    hand_grasp_offset = float(
        np.linalg.norm(T_grasp_in_cam[:3, 3] - T_hand_in_cam[:3, 3])
    )
    lines = (
        f"align: Hx.jaw={hand_x_dot_jaw:+.3f} "
        f"Hy.approach={hand_y_dot_approach:+.3f}",
        f"origin: EE-GRASP={ee_grasp_offset:.3f}m "
        f"HAND-GRASP={hand_grasp_offset:.3f}m",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, image.shape[0] - 28 + index * 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def _alignment_metric_text(
    grasp_diagnostics: dict,
    T_ee_in_cam: np.ndarray,
    T_align: np.ndarray,
    backend: MujocoBackend,
) -> str:
    """Compact alignment metrics for the periodic terminal log."""
    T_hand_in_cam = T_ee_in_cam @ T_align
    R_camera_from_world = backend.T_world_in_camera_cv()[:3, :3]
    jaw_axis_camera = R_camera_from_world @ np.asarray(
        grasp_diagnostics.get("jaw_axis_world", np.zeros(3)), dtype=np.float64
    )
    approach_axis_camera = R_camera_from_world @ np.asarray(
        grasp_diagnostics.get("grasp_approach_axis_world", np.zeros(3)),
        dtype=np.float64,
    )
    jaw_norm = np.linalg.norm(jaw_axis_camera)
    approach_norm = np.linalg.norm(approach_axis_camera)
    x_dot = (
        float(np.dot(T_hand_in_cam[:3, 0], jaw_axis_camera) / jaw_norm)
        if jaw_norm > 1e-8
        else float("nan")
    )
    y_dot = (
        float(np.dot(T_hand_in_cam[:3, 1], approach_axis_camera) / approach_norm)
        if approach_norm > 1e-8
        else float("nan")
    )
    return f"HxJaw={x_dot:+.2f} HyApp={y_dot:+.2f} off={np.linalg.norm(T_hand_in_cam[:3, 3] - T_ee_in_cam[:3, 3]):.3f}m"


def _draw_predicted_trajectory(
    image: np.ndarray,
    policy: ICTPolicy,
    trajectory,
    anchor,
    T_align: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    points = []
    positions, rotations, _ = trajectory
    for index in range(len(positions)):
        T_ee = policy.decode_ee_in_cam(
            positions[index], rotations[index], anchor, T_align
        )
        uv = _project(T_ee[:3, 3], K)
        if uv is not None:
            points.append(uv)
    if len(points) >= 2:
        cv2.polylines(
            canvas, [np.asarray(points, dtype=np.int32)], False, (255, 0, 255), 2
        )
    for index, point in enumerate(points[::5]):
        cv2.circle(canvas, point, 4, (255, 255, 255), -1)
        cv2.putText(
            canvas,
            str(index * 5),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run(
    config_path: str,
    *,
    dry_run: bool = False,
    show_viewer: bool | None = None,
    max_cycles_override: int | None = None,
) -> bool:
    cfg, config_dir = _load_config(config_path)
    cfg["simulation"]["xml_path"] = _resolve_path(
        config_dir, cfg["simulation"]["xml_path"]
    )
    cfg["policy"]["ckpt"] = _resolve_path(config_dir, cfg["policy"]["ckpt"])
    cfg["perception"]["visualkpts_cfg_path"] = _resolve_path(
        config_dir, cfg["perception"]["visualkpts_cfg_path"]
    )
    output_dir = Path(
        _resolve_path(config_dir, cfg["runtime"].get("output_dir", "./mujoco_output"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_viewer is not None:
        cfg["simulation"]["show_viewer"] = show_viewer
    requested_device = cfg["runtime"].get("device", "auto")
    seed = int(cfg["runtime"].get("seed", 7))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = (
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    execute_actions = bool(cfg["runtime"].get("execute_actions", True)) and not dry_run
    max_cycles = int(
        max_cycles_override
        if max_cycles_override is not None
        else cfg["runtime"].get("max_cycles", 250)
    )

    backend = MujocoBackend(**cfg["simulation"])
    camera = MujocoCamera(backend)
    arm = MujocoRobotArm(backend, cfg["robot"])
    arms: Dict[str, RobotArm] = {"right": arm}
    perception = MujocoPerception(backend, cfg["perception"])
    policy = ICTPolicy(cfg["policy"], device=device)
    if policy.sides != ["right"]:
        raise ValueError(
            f"The scene contains one right arm, but checkpoint predicts {policy.sides}"
        )
    controller = TrajectoryController(
        arms, cfg["control"], advance_simulation=backend.step_for
    )
    placement = MujocoTaskPlacement(backend, arm, cfg.get("placement", {}))
    grasp_assist = MujocoGraspAssist(backend, arm, cfg.get("grasp_assist", {}))

    T_align = np.asarray(cfg["robot"].get("T_align", np.eye(4)), dtype=np.float32)
    control_dt = 1.0 / float(cfg["control"].get("control_hz", 10.0))
    exec_horizon = int(cfg["control"].get("exec_horizon", 2))
    done_threshold = float(cfg["control"].get("done_threshold", 0.8))
    debug_every = int(cfg["runtime"].get("debug_every", 10))
    min_done_cycle = int(cfg["runtime"].get("min_cycles_before_done", 5))
    realtime_sleep = bool(cfg["control"].get("realtime_sleep", False))
    task_mode = str(cfg["runtime"].get("task_mode", "hybrid")).lower()
    if task_mode not in ("policy", "hybrid"):
        raise ValueError("runtime.task_mode must be 'policy' or 'hybrid'")
    success = False
    placement_mode = "policy"
    placement_last_report = None
    grasp_assist_mode = "policy"
    model_attachment_enabled = bool(
        cfg["robot"].get("grasp_validation", {}).get(
            "attach_on_bilateral_contact", False
        )
    )

    print(
        f"[sim] device={device} seed={seed} execute_actions={execute_actions} "
        f"physics_dt={backend.timestep:.4f}s control_dt={control_dt:.3f}s"
    )
    try:
        arm.set_gripper(0.0, blocking=True)
        arm.go_home(blocking=True)
        controller.reset(seed_from_robot=True)

        for cycle in range(max_cycles):
            cycle_started = time.perf_counter()
            timing = {}
            if model_attachment_enabled:
                perception.set_anchor_frozen(
                    bool(arm.get_grasp_diagnostics().get("attachment_active", False))
                )
            frame = camera.get_frame()
            timing["capture"] = time.perf_counter() - cycle_started
            objects = perception.get_objects()
            anchor = objects[cfg["perception"].get("anchor_key", "obj1")]
            T_ee_in_cam = arm.get_T_ee_in_cam()
            hands_in_cam = {"right": T_ee_in_cam @ T_align}
            grippers = {"right": arm.get_gripper()}
            timing["state"] = time.perf_counter() - cycle_started - timing["capture"]

            if task_mode == "hybrid" and placement_mode == "policy" and bool(
                cfg.get("placement", {}).get("enabled", True)
            ):
                grasp_diagnostics = arm.get_grasp_diagnostics()
                finger_contacts = tuple(grasp_diagnostics.get("finger_contacts", ()))
                if (
                    grasp_diagnostics.get("requested_closed", False)
                    and finger_contacts == (True, True)
                    and placement.start()
                ):
                    placement_mode = "placement"
                    print(
                        "[sim] grasp verified; switching to task placement "
                        "controller."
                    )

            # HumanEgo remains the owner of the complete action trajectory.
            # The attachment is only a MuJoCo grasp fact: once both finger
            # geoms touch the object while the policy requests closure, keep
            # the free body rigidly coupled to the TCP and continue executing
            # the model's future transport/release predictions below.
            if task_mode == "policy" and model_attachment_enabled and execute_actions:
                grasp_diagnostics = arm.get_grasp_diagnostics()
                if (
                    grasp_diagnostics.get("requested_closed", False)
                    and tuple(grasp_diagnostics.get("finger_contacts", ()))
                    == (True, True)
                    and not grasp_diagnostics.get("attachment_active", False)
                    and arm.activate_task_attachment(
                        backend.pose_camera_to_world(
                            objects[cfg["perception"].get("anchor_key", "obj1")].T_in_cam
                        )
                    )
                ):
                    print(
                        "[sim] bilateral grasp verified; keeping the model "
                        "trajectory active with simulator grasp attachment."
                    )

            if placement_mode == "placement":
                placement_phase = placement.step(control_dt)
                if placement_phase != placement_last_report or cycle % 10 == 0:
                    ik_detail = ""
                    if placement.last_ik is not None:
                        ik_detail = (
                            f" ik={int(placement.last_ik.success)}"
                            f" ik_err={placement.last_ik.position_error:.4f}"
                        )
                    target_index = min(
                        placement.phase_index,
                        len(placement.waypoints_object_world) - 1,
                    )
                    print(
                        f"[placement] phase={placement_phase} "
                        f"object_error={placement.last_error:.4f} "
                        f"object={np.round(placement.last_object_position, 3)} "
                        f"target={np.round(placement.waypoints_object_world[target_index], 3)}"
                        f" site={np.round(placement.last_site_position, 3)}"
                        f" site_target={np.round(placement.last_target_site, 3)}"
                        f"{ik_detail}"
                    )
                    placement_last_report = placement_phase
                if placement_phase == "ik_failed":
                    print(
                        f"[placement] IK failed phase={placement.phase} "
                        f"position_error={placement.last_error:.4f}"
                    )
                elif placement.ready_to_release:
                    placement_mode = "release"
                    print(
                        "[placement] object center reached the plate center; "
                        "opening gripper."
                    )
                if placement_mode == "placement":
                    if realtime_sleep:
                        time.sleep(
                            max(0.0, control_dt - (time.perf_counter() - cycle_started))
                        )
                    continue

            if placement_mode == "release":
                settled = placement.release_step(control_dt)
                success = perception.task_success(
                    arm.get_gripper(),
                    radius=float(cfg["runtime"].get("success_radius", 0.10)),
                    max_height=float(cfg["runtime"].get("success_max_height", 0.10)),
                )
                if success:
                    print(f"[sim] task success after task placement at cycle {cycle}.")
                    break
                if settled:
                    print(
                        "[placement] release settled but task success condition "
                        "is false."
                    )
                    break
                if realtime_sleep:
                    time.sleep(
                        max(0.0, control_dt - (time.perf_counter() - cycle_started))
                    )
                continue

            clean = perception.make_clean_image(frame, hands_in_cam, grippers)
            x_ict, ict_mask = policy.build_ict(
                hands_in_cam, grippers, objects, cfg["perception"].get("anchor_key", "obj1")
            )
            x_rgb = policy.prepare_image(clean)
            anchor_uv = policy.compute_anchor_uv(
                anchor, frame.K, frame.rgb.shape[1], frame.rgb.shape[0]
            )
            timing["input"] = time.perf_counter() - cycle_started - sum(
                timing.values()
            )
            trajectory, done_probability = policy.infer(
                x_rgb, x_ict, ict_mask, anchor_uv
            )
            timing["infer"] = time.perf_counter() - cycle_started - sum(
                timing.values()
            )

            positions, rotations, grasps = trajectory["right"]
            targets: List[np.ndarray] = [
                policy.decode_ee_in_cam(
                    positions[index], rotations[index], anchor, T_align
                )
                for index in range(len(positions))
            ]

            # The checkpoint supplies the grasp intent.  Once it is near the
            # object, let the physical grasp controller own the final approach
            # and closure; do not let a policy waypoint or a single contact
            # create a task-level attachment.
            if (
                task_mode == "hybrid"
                and
                execute_actions
                and grasp_assist_mode == "policy"
                and float(grasps.reshape(-1)[0])
                > float(cfg["control"].get("grasp_threshold", 0.5))
            ):
                grasp_diagnostics = arm.get_grasp_diagnostics()
                if (
                    grasp_diagnostics.get("state") not in ("grasped", "closing")
                    and grasp_diagnostics.get("center_distance", np.inf)
                    <= grasp_assist.start_distance
                    and grasp_assist.start()
                ):
                    grasp_assist_mode = "assist"
                    print(
                        "[sim] policy grasp intent is near the object; switching "
                        "to physical grasp assist."
                    )

            if task_mode == "hybrid" and execute_actions and grasp_assist_mode == "assist":
                grasp_phase = grasp_assist.step(control_dt)
                if grasp_phase == "grasped":
                    grasp_assist_mode = "grasped"
                elif grasp_phase == "failed":
                    grasp_assist_mode = "failed"
                    print("[sim] physical grasp assist failed; stopping before placement.")
                    break
                if cycle % debug_every == 0 or grasp_phase in ("pregrasp", "approach", "closing"):
                    grasp_diagnostics = arm.get_grasp_diagnostics()
                    print(
                        f"[grasp-assist] cycle={cycle:03d} phase={grasp_phase} "
                        f"site={np.round(grasp_assist.last_site, 3)} "
                        f"object={np.round(grasp_assist.last_object, 3)} "
                        f"distance={grasp_diagnostics['center_distance']:.4f} "
                        f"contacts={int(grasp_diagnostics['finger_contacts'][0])}"
                        f"{int(grasp_diagnostics['finger_contacts'][1])}"
                    )
                if realtime_sleep:
                    time.sleep(
                        max(0.0, control_dt - (time.perf_counter() - cycle_started))
                    )
                # A confirmed grasp is handed to placement at the next loop
                # top, where its bilateral-contact gate is checked again.
                continue

            if debug_every > 0 and cycle % debug_every == 0:
                grasp_diagnostics = arm.get_grasp_diagnostics()
                diagnostic = _draw_predicted_trajectory(
                    clean, policy, trajectory["right"], anchor, T_align, frame.K
                )
                if bool(cfg["runtime"].get("visualize_coordinate_frames", True)):
                    grasp_site_name = cfg["robot"].get("grasp_validation", {}).get(
                        "grasp_site", "grasp_site"
                    )
                    T_grasp_in_cam = backend.pose_world_to_camera(
                        backend.site_pose_world(grasp_site_name)
                    )
                    T_target_hand_in_cam = targets[0] @ T_align
                    frame_specs: List[tuple[str, np.ndarray, float]] = [
                        ("EE", T_ee_in_cam, 0.055),
                        ("GRASP", T_grasp_in_cam, 0.045),
                        ("HAND", hands_in_cam["right"], 0.065),
                        ("HAND*", T_target_hand_in_cam, 0.065),
                    ]
                    for object_key, object_state in objects.items():
                        frame_specs.append(
                            (f"{object_key}", object_state.T_in_cam, 0.045)
                        )
                    diagnostic = _draw_coordinate_frames(
                        diagnostic, frame.K, frame_specs
                    )
                    diagnostic = _draw_alignment_metrics(
                        diagnostic,
                        T_ee_in_cam,
                        T_grasp_in_cam,
                        hands_in_cam["right"],
                        grasp_diagnostics,
                        backend,
                    )
                cv2.imwrite(str(output_dir / f"cycle_{cycle:04d}.png"), diagnostic)
                # Compare like-for-like policy hand poses. ``targets`` are Piper
                # root commands; multiplying by T_align recovers the fingertip
                # midpoint pose used in the ICT and HumanEgo action space.
                target0 = (targets[0] @ T_align)[:3, 3]
                current = hands_in_cam["right"][:3, 3]
                # All quantities below are converted to MuJoCo world coordinates.
                # This is intentionally separate from the policy/camera debug
                # values above: it lets us tell a transport-target error from an
                # attachment or success-condition error.
                plate_world = backend.body_pose_world(
                    perception.object_specs[
                        next(key for key in perception.object_specs if key != perception.anchor_key)
                    ]["body"]
                )[:3, 3]
                object_world = backend.body_pose_world(
                    perception.object_specs[perception.anchor_key]["body"]
                )[:3, 3]
                target_site_world = backend.pose_camera_to_world(targets[0])[:3, 3]
                target_hand_world = backend.pose_camera_to_world(
                    targets[0] @ T_align
                )[:3, 3]
                attachment_offset_world = np.asarray(
                    grasp_diagnostics.get("attachment_offset_world", np.zeros(3)),
                    dtype=np.float64,
                )
                commanded_site_world = target_site_world.copy()
                if bool(grasp_diagnostics.get("attachment_active", False)):
                    commanded_site_world -= attachment_offset_world
                target_plate_distance = float(
                    np.linalg.norm(target_site_world[:2] - plate_world[:2])
                )
                commanded_plate_distance = float(
                    np.linalg.norm(commanded_site_world[:2] - plate_world[:2])
                )
                target_hand_plate_distance = float(
                    np.linalg.norm(target_hand_world[:2] - plate_world[:2])
                )
                object_plate_distance = float(
                    np.linalg.norm(object_world[:2] - plate_world[:2])
                )
                finger_contacts = grasp_diagnostics["finger_contacts"]
                alignment_detail = ""
                if (
                    grasp_diagnostics["alignment_phase"] != "idle"
                    or grasp_diagnostics["prealign_phase"] != "idle"
                ):
                    alignment_detail = (
                        f" align={grasp_diagnostics['alignment_phase']}"
                        f" prealign={grasp_diagnostics['prealign_phase']}"
                        f" wp_err={grasp_diagnostics['alignment_waypoint_error']:.3f}"
                        f" joint_step={grasp_diagnostics['alignment_joint_step']:.4f}"
                        f" ik={int(grasp_diagnostics['alignment_ik_success'])}"
                        f" ik_pos={grasp_diagnostics['alignment_ik_position_error']:.3f}"
                        f" site={np.round(grasp_diagnostics['grasp_site_world'], 3)}"
                        f" sol={np.round(grasp_diagnostics['alignment_solution_world'], 3)}"
                        f" obj={np.round(grasp_diagnostics['grasp_object_world'], 3)}"
                        f" wp_z={grasp_diagnostics['alignment_waypoint_world'][2]:.3f}"
                        f" ctrl_gap={grasp_diagnostics['arm_ctrl_gap']:.3f}"
                    )
                grasp_preview = np.asarray(
                    grasps[: min(exec_horizon, len(grasps))]
                ).reshape(-1)
                anchor_pose = objects[cfg["perception"].get("anchor_key", "obj1")].T_in_cam
                other_key = next(
                    key for key in objects
                    if key != cfg["perception"].get("anchor_key", "obj1")
                )
                plate_in_anchor = (
                    np.linalg.inv(anchor_pose) @ objects[other_key].T_in_cam
                )[:3, 3]
                hand_in_anchor = (
                    np.linalg.inv(anchor_pose) @ hands_in_cam["right"]
                )[:3, 3]
                ict_values = x_ict[0, ict_mask[0]].detach().float()
                ict_summary = (
                    f"ict_absmax={float(ict_values.abs().max()):.2f}"
                    f" plateA={np.round(plate_in_anchor, 3)}"
                    f" handA={np.round(hand_in_anchor, 3)}"
                )
                print(
                    f"[sim] cycle={cycle:03d} done={done_probability:.3f} "
                    f"grasp={float(grasps.reshape(-1)[0]):.3f} "
                    f"gseq={np.round(grasp_preview, 2)} "
                    f"current={np.round(current, 3)} target0={np.round(target0, 3)} "
                    f"physical={grasp_diagnostics['state']} "
                    f"ee_obj={grasp_diagnostics['center_distance']:.3f} "
                    f"contacts={int(finger_contacts[0])}{int(finger_contacts[1])} "
                    f"jaw_lat={grasp_diagnostics['jaw_lateral_error']:+.3f}"
                    f" objw={np.round(grasp_diagnostics['grasp_object_world'], 3)}"
                    f" platew={np.round(plate_world, 3)}"
                    f" tgtw={np.round(target_site_world, 3)}"
                    f" d_tgt_plate={target_plate_distance:.3f}"
                    f" d_cmd_plate={commanded_plate_distance:.3f}"
                    f" d_hand_plate={target_hand_plate_distance:.3f}"
                    f" d_obj_plate={object_plate_distance:.3f}"
                    f" attach={int(bool(grasp_diagnostics.get('attachment_active', False)))}"
                    f" offw={np.round(attachment_offset_world, 3)}"
                    f" {ict_summary}"
                    f" t={timing}"
                    f" align_x={_alignment_metric_text(grasp_diagnostics, T_ee_in_cam, T_align, backend)}"
                    f"{alignment_detail}"
                )

            if execute_actions:
                attachment_before = False
                if model_attachment_enabled:
                    attachment_before = bool(
                        arm.get_grasp_diagnostics().get("attachment_active", False)
                    )
                failures_before = arm.ik_failure_count
                controller.execute_chunk(
                    {"right": targets},
                    {"right": grasps.reshape(-1)},
                    dt=control_dt,
                    n_steps=min(exec_horizon, len(targets)),
                )
                if arm.ik_failure_count > failures_before:
                    print(f"[sim] command rejected: {arm.last_rejection_reason}")
                if attachment_before and model_attachment_enabled:
                    attachment_after = bool(
                        arm.get_grasp_diagnostics().get("attachment_active", False)
                    )
                    if not attachment_after:
                        print(
                            "[sim] model grasp output opened the gripper; "
                            "released simulator attachment."
                        )
            else:
                backend.step_for(control_dt)
            timing["execute"] = time.perf_counter() - cycle_started - sum(
                timing.values()
            )
            if debug_every > 0 and cycle % debug_every == 0:
                print(f"[profile] cycle={cycle:03d} seconds={timing}")

            success = perception.task_success(
                arm.get_gripper(),
                radius=float(cfg["runtime"].get("success_radius", 0.10)),
                max_height=float(cfg["runtime"].get("success_max_height", 0.10)),
            )
            if success:
                print(f"[sim] task success at cycle {cycle}.")
                break
            if (
                not execute_actions
                and cycle >= min_done_cycle
                and done_probability > done_threshold
            ):
                print(f"[sim] policy done probability reached {done_probability:.3f}.")
                break
    except KeyboardInterrupt:
        print("[sim] interrupted.")
    finally:
        print(f"[sim] finished success={success} ik_failures={arm.ik_failure_count}")
        if bool(cfg["simulation"].get("show_viewer", False)):
            backend.wait_for_viewer()
        if cfg["runtime"].get("home_on_exit", True):
            controller.home()
        arm.close()
        camera.close()
        backend.close()
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_REPO_ROOT / "cfg/inference/mujoco_serve_bread.yaml"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="infer and save trajectories without moving"
    )
    parser.add_argument("--viewer", action="store_true", help="open the passive MuJoCo viewer")
    parser.add_argument("--cycles", type=int, default=None)
    args = parser.parse_args()
    run(
        args.config,
        dry_run=args.dry_run,
        show_viewer=True if args.viewer else None,
        max_cycles_override=args.cycles,
    )


if __name__ == "__main__":
    main()
