"""Same-exposure stereo geometry helpers; no tracking or camera calibration fitting.

All pixel coordinates are in the single-eye image, never in the SBS canvas.
The caller must supply projection matrices for the actual image domain.
"""

from itertools import product

import cv2
import numpy as np


MP_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_FRAME_JOINTS = (0, 2, 4, 5, 8, 17)


def camera_to_head_pair(calibration):
    """Convert image-camera points to head coordinates with confirmed semantics.

    ``E`` maps raw-camera coordinates to head coordinates and ``D`` maps raw
    camera coordinates to image-camera coordinates. Therefore:
    ``p_head = E @ inv(D) @ p_image``. D is a basis conversion; it contains no
    information about either camera's installed position.
    """
    image_to_raw = np.eye(4)
    image_to_raw[:3, :3] = calibration.D.T
    return tuple(e @ image_to_raw for e in (calibration.E_left, calibration.E_right))


def stereo_matrices(K_left, K_right, left_to_right):
    T = np.asarray(left_to_right, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Invalid stereo transform")
    R, t = T[:3, :3], T[:3, 3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError("Stereo rotation is not a proper rotation")
    if not np.allclose(T[3], [0, 0, 0, 1]) or np.linalg.norm(t) < 1e-6:
        raise ValueError("Stereo baseline is missing or degenerate")
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K_right).T @ tx @ R @ np.linalg.inv(K_left)
    return np.asarray(K_left) @ np.eye(4)[:3], np.asarray(K_right) @ T[:3], F


def image_camera_projection(K, raw_camera_to_head, raw_to_image):
    """Build ``P = K @ D @ inv(E)`` for a point expressed in head frame."""
    E = np.asarray(raw_camera_to_head, dtype=np.float64)
    D = np.asarray(raw_to_image, dtype=np.float64)
    if E.shape != (4, 4) or D.shape != (3, 3):
        raise ValueError("Expected E=4x4 and D=3x3")
    head_to_raw = np.linalg.inv(E)
    raw_to_image_4 = np.eye(4)
    raw_to_image_4[:3, :3] = D
    head_to_image = raw_to_image_4 @ head_to_raw
    return np.asarray(K, dtype=np.float64) @ head_to_image[:3], head_to_image


def triangulate_joints_in_image_cameras(
    left_uv, right_uv, K_left, K_right,
    E_left_raw_to_head, E_right_raw_to_head, raw_to_image, **quality,
):
    """Triangulate with explicit ``K D inv(E)`` projection matrices.

    The homogeneous solution is in head coordinates. Per-eye XYZ is then
    expressed in each OpenCV/image-camera coordinate system. No raw-camera XYZ
    is passed to OpenCV projection or returned as the hand reconstruction.
    """
    a, b = np.asarray(left_uv, float), np.asarray(right_uv, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite([a, b]).all():
        raise ValueError("Expected finite matching Nx2 single-eye pixels")
    P_left, head_to_left = image_camera_projection(K_left, E_left_raw_to_head, raw_to_image)
    P_right, head_to_right = image_camera_projection(K_right, E_right_raw_to_head, raw_to_image)
    X = cv2.triangulatePoints(P_left, P_right, a.T, b.T).T
    finite = np.abs(X[:, 3]) > 1e-10
    xyz_head = np.full((len(a), 3), np.nan)
    xyz_head[finite] = X[finite, :3] / X[finite, 3:4]

    def transform(points, T):
        return points @ T[:3, :3].T + T[:3, 3]

    xyz_left = transform(xyz_head, head_to_left)
    xyz_right = transform(xyz_head, head_to_right)
    left_image_to_head = np.linalg.inv(head_to_left)
    left_to_right = head_to_right @ left_image_to_head
    _, _, F = stereo_matrices(K_left, K_right, left_to_right)
    result = {
        "xyz_head": xyz_head,
        "xyz_left_image_camera": xyz_left,
        "xyz_right_image_camera": xyz_right,
        "epipolar_error_px": epipolar_errors(a, b, F),
    }
    for eye, P in (("left", P_left), ("right", P_right)):
        projected = np.column_stack([xyz_head, np.ones(len(xyz_head))]) @ P.T
        uv = np.full((len(projected), 2), np.nan)
        ok = np.abs(projected[:, 2]) > 1e-10
        uv[ok] = projected[ok, :2] / projected[ok, 2:3]
        result[eye + "_reprojected_uv"] = uv
        result[eye + "_reprojection_error_px"] = np.linalg.norm(
            uv - (a if eye == "left" else b), axis=1
        )
    rays_left = np.column_stack([a, np.ones(len(a))]) @ np.linalg.inv(K_left).T
    rays_right = np.column_stack([b, np.ones(len(b))]) @ np.linalg.inv(K_right).T @ left_to_right[:3, :3]
    cosine = np.sum(rays_left * rays_right, axis=1) / (
        np.linalg.norm(rays_left, axis=1) * np.linalg.norm(rays_right, axis=1)
    )
    result["ray_angle_deg"] = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    reprojection_px = float(quality.get("reprojection_px", 4.0))
    epipolar_px = float(quality.get("epipolar_px", 6.0))
    min_depth_m = float(quality.get("min_depth_m", 0.10))
    max_depth_m = float(quality.get("max_depth_m", 2.0))
    min_ray_angle_deg = float(quality.get("min_ray_angle_deg", 0.5))
    result["valid"] = (
        finite & np.isfinite(xyz_head).all(axis=1)
        & (xyz_left[:, 2] >= min_depth_m) & (xyz_left[:, 2] <= max_depth_m)
        & (xyz_right[:, 2] >= min_depth_m) & (xyz_right[:, 2] <= max_depth_m)
        & (result["left_reprojection_error_px"] <= reprojection_px)
        & (result["right_reprojection_error_px"] <= reprojection_px)
        & (result["epipolar_error_px"] <= epipolar_px)
        & (result["ray_angle_deg"] >= min_ray_angle_deg)
    )
    return result


def epipolar_errors(left_uv, right_uv, F):
    left = np.column_stack([left_uv, np.ones(len(left_uv))])
    right = np.column_stack([right_uv, np.ones(len(right_uv))])
    lines_r, lines_l = left @ F.T, right @ F
    residual = np.abs(np.sum(right * lines_r, axis=1))
    return 0.5 * residual * (1 / np.maximum(np.linalg.norm(lines_r[:, :2], axis=1), 1e-12)
                             + 1 / np.maximum(np.linalg.norm(lines_l[:, :2], axis=1), 1e-12))


def triangulate_joints(left_uv, right_uv, K_left, K_right, left_to_right, *,
                       reprojection_px=4.0, epipolar_px=6.0, min_depth_m=0.10,
                       max_depth_m=2.0, min_ray_angle_deg=0.5):
    """Triangulate one same-time pair; retain rejected raw estimates for QA."""
    a, b = np.asarray(left_uv, float), np.asarray(right_uv, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite([a, b]).all():
        raise ValueError("Expected finite matching Nx2 single-eye pixels")
    P1, P2, F = stereo_matrices(K_left, K_right, left_to_right)
    X = cv2.triangulatePoints(P1, P2, a.T, b.T).T
    finite = np.abs(X[:, 3]) > 1e-10
    xyz = np.full((len(a), 3), np.nan)
    xyz[finite] = X[finite, :3] / X[finite, 3:4]
    right_xyz = xyz @ left_to_right[:3, :3].T + left_to_right[:3, 3]

    def project(points, K):
        proj = points @ np.asarray(K).T
        uv = np.full((len(points), 2), np.nan)
        ok = np.abs(proj[:, 2]) > 1e-10
        uv[ok] = proj[ok, :2] / proj[ok, 2:3]
        return uv

    uv_l, uv_r = project(xyz, K_left), project(right_xyz, K_right)
    err_l, err_r = np.linalg.norm(uv_l - a, axis=1), np.linalg.norm(uv_r - b, axis=1)
    epi = epipolar_errors(a, b, F)
    rays_l = np.column_stack([a, np.ones(len(a))]) @ np.linalg.inv(K_left).T
    rays_r = np.column_stack([b, np.ones(len(b))]) @ np.linalg.inv(K_right).T @ left_to_right[:3, :3]
    cos = np.sum(rays_l * rays_r, axis=1) / (np.linalg.norm(rays_l, axis=1) * np.linalg.norm(rays_r, axis=1))
    angle = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
    valid = (finite & np.isfinite(xyz).all(axis=1)
             & (xyz[:, 2] >= min_depth_m) & (xyz[:, 2] <= max_depth_m)
             & (right_xyz[:, 2] >= min_depth_m) & (right_xyz[:, 2] <= max_depth_m)
             & (err_l <= reprojection_px) & (err_r <= reprojection_px)
             & (epi <= epipolar_px) & (angle >= min_ray_angle_deg))
    return {"xyz_left_image_camera_relative": xyz,
            "xyz_right_image_camera_relative": right_xyz, "valid": valid,
            "left_reprojection_error_px": err_l, "right_reprojection_error_px": err_r,
            "epipolar_error_px": epi, "ray_angle_deg": angle,
            "left_reprojected_uv": uv_l, "right_reprojected_uv": uv_r}


def hand_shape_valid(xyz, valid, *, min_palm_m=0.035, max_palm_m=0.13,
                     min_bone_m=0.004, max_bone_m=0.12):
    """Conservative whole-hand gate; never invent a pose from fallback axes."""
    if len(xyz) != 21 or not np.all(valid):
        return False
    palm = np.linalg.norm(xyz[5] - xyz[17])
    bones = [np.linalg.norm(xyz[a] - xyz[b]) for a, b in MP_BONES]
    # The existing midpoint frame uses MP2 (thumb MCP), MP5 (index MCP), MP0.
    axis = xyz[5] - xyz[2]
    forward = (xyz[5] + xyz[2]) / 2 - xyz[0]
    sine = np.linalg.norm(np.cross(axis, forward)) / max(np.linalg.norm(axis) * np.linalg.norm(forward), 1e-12)
    return bool(min_palm_m <= palm <= max_palm_m and min(bones) >= min_bone_m
                and max(bones) <= max_bone_m and sine > 0.1)


def hand_frame_valid(xyz, valid, *, min_valid_ratio=0.65,
                     min_palm_m=0.035, max_palm_m=0.13):
    """Gate the six joints needed by the existing midpoint hand frame."""
    xyz, valid = np.asarray(xyz), np.asarray(valid, dtype=bool)
    if xyz.shape != (21, 3) or valid.shape != (21,):
        return False
    if not valid[list(HAND_FRAME_JOINTS)].all() or np.mean(valid) < min_valid_ratio:
        return False
    palm = np.linalg.norm(xyz[5] - xyz[17])
    x = xyz[5] - xyz[2]
    y = (xyz[5] + xyz[2]) / 2 - xyz[0]
    sine = np.linalg.norm(np.cross(x, y)) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)
    return bool(min_palm_m <= palm <= max_palm_m and sine > 0.1)


def interpolate_short_gaps(points, valid, max_gap=5):
    """Linearly fill bounded per-joint gaps without extrapolating long gaps."""
    values = np.asarray(points, dtype=float).copy()
    mask = np.asarray(valid, dtype=bool).copy()
    if values.ndim != 3 or values.shape[1:] != (21, 3) or mask.shape != values.shape[:2]:
        raise ValueError("Expected points Nx21x3 and valid Nx21")
    values[~mask] = np.nan
    for joint in range(21):
        index = 0
        while index < len(values):
            if mask[index, joint]:
                index += 1
                continue
            start = index
            while index < len(values) and not mask[index, joint]:
                index += 1
            end = index
            gap = end - start
            if start > 0 and end < len(values) and gap <= max_gap:
                for offset, frame in enumerate(range(start, end), 1):
                    weight = offset / (gap + 1)
                    values[frame, joint] = ((1 - weight) * values[start - 1, joint]
                                            + weight * values[end, joint])
                    mask[frame, joint] = True
    return values, mask


def smooth_valid_points(points, valid, alpha=0.55):
    """Forward/backward EMA on valid runs; symmetric blend avoids frame lag."""
    values = np.asarray(points, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    result = values.copy()
    for joint in range(21):
        index = 0
        while index < len(values):
            while index < len(values) and not mask[index, joint]:
                index += 1
            start = index
            while index < len(values) and mask[index, joint]:
                index += 1
            end = index
            if end <= start:
                continue
            segment = values[start:end, joint]
            forward = segment.copy()
            backward = segment.copy()
            for i in range(1, len(segment)):
                forward[i] = alpha * segment[i] + (1 - alpha) * forward[i - 1]
            for i in range(len(segment) - 2, -1, -1):
                backward[i] = alpha * segment[i] + (1 - alpha) * backward[i + 1]
            result[start:end, joint] = 0.5 * (forward + backward)
    result[~mask] = np.nan
    return result


def match_hands(left, right, K, left_to_right=None, *, camera_geometry=None,
                confidence_min=0.5, ambiguity_margin_px=2.0,
                require_full_hand=True, min_valid_ratio=0.65,
                **triangulation_options):
    """Unique handedness + epipolar/wrist matches. No list-index correspondence.

    An ambiguous pair or conflicting handedness is rejected, not guessed from
    native joints. Scores are handedness classification scores, not joint scores.
    """
    candidates = []
    for li, ri in product(range(len(left)), range(len(right))):
        l, r = left[li], right[ri]
        if l["hand_side"] != r["hand_side"] or min(l["score"], r["score"]) < confidence_min:
            continue
        uv_l = np.array([[j["u_eye"], j["v_eye"]] for j in l["joints"]])
        uv_r = np.array([[j["u_eye"], j["v_eye"]] for j in r["joints"]])
        if uv_l.shape != (21, 2) or uv_r.shape != (21, 2):
            continue
        if camera_geometry is None:
            if left_to_right is None:
                raise ValueError("Provide image-camera relative pose or explicit camera geometry")
            tri = triangulate_joints(uv_l, uv_r, K, K, left_to_right, **triangulation_options)
            xyz = tri["xyz_left_image_camera_relative"]
        else:
            tri = triangulate_joints_in_image_cameras(
                uv_l, uv_r, K, K,
                camera_geometry["E_left_raw_to_head"],
                camera_geometry["E_right_raw_to_head"],
                camera_geometry["raw_to_image"],
                **triangulation_options,
            )
            xyz = tri["xyz_left_image_camera"]
        cost = float(np.median(tri["epipolar_error_px"]) + tri["epipolar_error_px"][0])
        candidates.append({"left_id": li, "right_id": ri, "side": l["hand_side"],
                           "cost_px": cost, "handedness_score": min(l["score"], r["score"]),
                           "triangulation": tri, "left_uv": uv_l, "right_uv": uv_r})
    accepted = []
    for c in sorted(candidates, key=lambda x: x["cost_px"]):
        competitors = [v for v in candidates if v is not c and
                       (v["left_id"] == c["left_id"] or v["right_id"] == c["right_id"] or v["side"] == c["side"])]
        unique = all(c["cost_px"] + ambiguity_margin_px < v["cost_px"] for v in competitors)
        tri = c["triangulation"]
        quality_ok = (hand_shape_valid(xyz, tri["valid"])
                      if require_full_hand else
                      hand_frame_valid(xyz, tri["valid"], min_valid_ratio=min_valid_ratio))
        c["accepted"] = unique and quality_ok
        c["rejection"] = None if c["accepted"] else ("ambiguous_correspondence" if not unique else "joint_or_hand_quality")
        if c["accepted"]:
            accepted.append(c)
    return accepted, candidates
