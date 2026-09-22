"""
Shared Interaction-Centric Token (ICT) building utilities for baselines.

Replicates the FlowMatchingDataloader._build_ict() logic in a standalone
function so all baselines can share it without importing the full dataloader.

ICT format (single hand, object_centric):
    [TypeID(1), pose_in_ref(9), hand_in_entity(9), flag(1)] = 20 dims
where:
    pose_in_ref(9) = [normalized_pos(3), normalized_o6d(6)]
    hand_in_entity(9) = relative hand pose in entity frame

Constants:
    TYPE_PAD = 0, TYPE_HAND_L = 1, TYPE_HAND_R = 2,
    TYPE_OBJ_ANCHOR = 3, TYPE_OBJ_OTHER = 4
"""

import numpy as np
from utils.utils_math import rotmat_to_o6d, normalize_o6d, normalize_pos

# Token type IDs
TYPE_PAD = 0.0
TYPE_HAND_L = 1.0
TYPE_HAND_R = 2.0
TYPE_OBJ_ANCHOR = 3.0
TYPE_OBJ_OTHER = 4.0

# Default token dims for single hand
ICT_DIM_SINGLE_HAND = 20   # TypeID(1) + pose(9) + hand_in_entity(9) + flag(1)
MAX_ICT_DEFAULT = 8

_HAND_KEYS = {
    "aria_mps": "hands",
    "mediapipe": "hands_mediapipe",
    "wilor": "hands_wilor",
    "hamer": "hands_hamer",
}


def _encode_geometry(T_matrix, pos_mean, pos_std):
    """Encode 4x4 transform as [normalized_pos(3), normalized_o6d(6)] = 9D."""
    pos = normalize_pos(T_matrix[:3, 3].astype(np.float32), pos_mean, pos_std)
    o6d = normalize_o6d(rotmat_to_o6d(T_matrix[:3, :3].astype(np.float32)))
    return np.concatenate([pos, o6d]).astype(np.float32)


def build_ict(
    d: dict,
    pos_mean: np.ndarray,
    pos_std: np.ndarray,
    single_hand_side: str = 'right',
    max_ict: int = MAX_ICT_DEFAULT,
    ict_dim: int = ICT_DIM_SINGLE_HAND,
    hand_tracking_method: str = 'aria_mps',
    frame_mode: str = 'camera_frame',
):
    """
    Build ICTs from a training_data.json dict.

    Returns:
        x_ict: (max_ict, ict_dim) float32
        ict_mask: (max_ict,) bool
    """
    hand_entity_key = _HAND_KEYS.get(hand_tracking_method, "hands")

    # Compute reference frame transform
    w_trans = d["metadata"]["world_transforms"]
    if frame_mode == 'camera_frame':
        T_ref_w = np.array(w_trans["cam0"], dtype=np.float32)
    elif frame_mode == 'anchor_frame':
        T_ref_w = np.array(w_trans["virtual_static_anchor"], dtype=np.float32)
    else:
        raise ValueError(f"Unknown frame_mode: {frame_mode}")
    T_w2ref = np.linalg.inv(T_ref_w).astype(np.float64)

    ents = d.get("entities", {})
    hands = ents.get(hand_entity_key, {})
    if hands is None:
        hands = {}
    objs = ents.get("objects", {})
    anchor_key = d["metadata"].get("anchor_key", "obj1")

    # Get hand transforms in world
    T_hR_w = np.array(hands["right"]["T_hand_to_world"], dtype=np.float64) if "right" in hands else None
    T_hL_w = np.array(hands["left"]["T_hand_to_world"], dtype=np.float64) if "left" in hands else None

    # For single-hand mode
    T_h_w = T_hR_w if single_hand_side == "right" else T_hL_w

    def calc_hand_relation(T_ent_w):
        """Compute relative hand-in-entity encoding (9D for single hand)."""
        T_w2ent = np.linalg.inv(T_ent_w)
        if T_h_w is not None:
            return _encode_geometry(T_w2ent @ T_h_w, pos_mean, pos_std)
        else:
            return np.zeros(9, dtype=np.float32)

    tokens = []

    # 1. Hand token
    hand_sides = [single_hand_side]
    for side in hand_sides:
        if side in hands:
            T_hs_w = np.array(hands[side]["T_hand_to_world"], dtype=np.float64)
            grasp = float(hands[side]["grasp"])
            type_id = TYPE_HAND_L if side == "left" else TYPE_HAND_R

            pose_in_ref = _encode_geometry(T_w2ref @ T_hs_w, pos_mean, pos_std)
            hand_in_hand = calc_hand_relation(T_hs_w)
            tok = np.concatenate([[type_id], pose_in_ref, hand_in_hand, [grasp]])
            tokens.append(tok.astype(np.float32))
        else:
            tokens.append(np.zeros(ict_dim, dtype=np.float32))

    # 2. Anchor object token
    if anchor_key in objs:
        T_anc_w = np.array(objs[anchor_key]["T_obj_to_world"], dtype=np.float64)
        pose_in_ref = _encode_geometry(T_w2ref @ T_anc_w, pos_mean, pos_std)
        hand_in_anc = calc_hand_relation(T_anc_w)
        tok = np.concatenate([[TYPE_OBJ_ANCHOR], pose_in_ref, hand_in_anc, [-1.0]])
        tokens.append(tok.astype(np.float32))

    # 3. Other object tokens
    for k, v in objs.items():
        if k == anchor_key:
            continue
        T_obj_w = np.array(v["T_obj_to_world"], dtype=np.float64)
        pose_in_ref = _encode_geometry(T_w2ref @ T_obj_w, pos_mean, pos_std)
        hand_in_obj = calc_hand_relation(T_obj_w)
        tok = np.concatenate([[TYPE_OBJ_OTHER], pose_in_ref, hand_in_obj, [-1.0]])
        tokens.append(tok.astype(np.float32))

    # 4. Pad and mask
    x_ict = np.zeros((max_ict, ict_dim), dtype=np.float32)
    ict_mask = np.zeros(max_ict, dtype=bool)
    n_tok = min(len(tokens), max_ict)
    for i in range(n_tok):
        if tokens[i][0] != TYPE_PAD:
            x_ict[i] = tokens[i]
            ict_mask[i] = True

    return x_ict, ict_mask


def build_ict_from_poses(
    hand_T_cam: np.ndarray,
    grasp: float,
    obj_T_cam_list: list,
    pos_mean: np.ndarray,
    pos_std: np.ndarray,
    single_hand_side: str = 'right',
    max_ict: int = MAX_ICT_DEFAULT,
    ict_dim: int = ICT_DIM_SINGLE_HAND,
    anchor_idx: int = 0,
):
    """
    Build ICTs from explicit poses (for real-world inference).

    All poses should already be in the reference frame (cam0).

    Args:
        hand_T_cam: (4,4) hand pose in camera frame
        grasp: float grasp value
        obj_T_cam_list: list of (4,4) object poses in camera frame
        pos_mean, pos_std: normalization stats
        single_hand_side: 'left' or 'right'
        max_ict: max tokens
        ict_dim: token dimension
        anchor_idx: which object in list is the anchor (default 0)

    Returns:
        x_ict: (max_ict, ict_dim) float32
        ict_mask: (max_ict,) bool
    """
    # In camera frame, T_w2ref = I (poses are already in cam)
    # So pose_in_ref is just the cam-frame encoding
    type_id = TYPE_HAND_R if single_hand_side == "right" else TYPE_HAND_L

    def calc_hand_relation(T_ent_cam):
        T_cam2ent = np.linalg.inv(T_ent_cam)
        return _encode_geometry(T_cam2ent @ hand_T_cam, pos_mean, pos_std)

    tokens = []

    # 1. Hand token
    pose_in_ref = _encode_geometry(hand_T_cam, pos_mean, pos_std)
    hand_in_hand = calc_hand_relation(hand_T_cam)
    tok = np.concatenate([[type_id], pose_in_ref, hand_in_hand, [grasp]])
    tokens.append(tok.astype(np.float32))

    # 2. Object tokens
    for i, T_obj_cam in enumerate(obj_T_cam_list):
        if i == anchor_idx:
            obj_type = TYPE_OBJ_ANCHOR
        else:
            obj_type = TYPE_OBJ_OTHER
        pose_in_ref = _encode_geometry(T_obj_cam, pos_mean, pos_std)
        hand_in_obj = calc_hand_relation(T_obj_cam)
        tok = np.concatenate([[obj_type], pose_in_ref, hand_in_obj, [-1.0]])
        tokens.append(tok.astype(np.float32))

    # 3. Pad and mask
    x_ict = np.zeros((max_ict, ict_dim), dtype=np.float32)
    ict_mask = np.zeros(max_ict, dtype=bool)
    n_tok = min(len(tokens), max_ict)
    for i in range(n_tok):
        if tokens[i][0] != TYPE_PAD:
            x_ict[i] = tokens[i]
            ict_mask[i] = True

    return x_ict, ict_mask
