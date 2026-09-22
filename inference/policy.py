# -*- coding: utf-8 -*-
# @FileName: policy.py
"""
ICTPolicy — the model-side of HumanEgo inference (reference template).

Wraps the trained FlowMatchingModel and exposes exactly what the robot loop needs:

    (load)         reconstruct the model from a checkpoint + training config
    build_ict      turn current hand & object poses into Interaction-Centric
                   Tokens (ICT) — the embodiment/viewpoint-invariant state
    prepare_image  turn a clean RGB into the normalized network tensor
    compute_anchor_uv  (only if the model uses region attention)
    infer          flow-matching ODE solve -> future hand trajectory + done
    decode_ee_in_cam   map a prediction back to an EE target in the camera frame

This is a cleaned distillation of inference/InferencePolicy.py. It still reads
every architecture flag from the checkpoint's training config so it loads the
real released checkpoints; it just drops the legacy-config auto-migration and
the checkpoint-key auto-detection that the production loader adds for robustness.

It mirrors the ICT construction in training/FlowMatchingDataloader._build_ict so
train- and test-time tokens are identical — that matching is what makes the
policy transfer. Everything here is hardware-agnostic: poses come in as 4x4
matrices in the CAMERA frame, trajectories come out as EE targets in that frame.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Real, shipped modules (present in this repo) — this is "how to load the model".
from training.FlowMatchingModel import FlowMatchingModel
from utils.ict import TYPE_HAND_L, TYPE_HAND_R, TYPE_OBJ_ANCHOR, TYPE_OBJ_OTHER
from utils.utils_math import (
    rotmat_to_o6d,
    o6d_to_rotmat,
    normalize_o6d,
    normalize_pos,
    unnormalize_pos,
)

from interfaces import ObjectState


def _inv(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T)


class ICTPolicy:
    def __init__(self, cfg: dict, device: str = "cuda"):
        """
        Args:
            cfg: the `policy` section of the inference config (see
                 cfg/inference/example_dualarm.yaml). Must contain `ckpt`.
            device: 'cuda' or 'cpu'.

        Loads `config.json` + `dataset_stats.json` from next to the checkpoint
        (the trainer writes both there): the first carries every architecture
        flag, the second the position normalization. We build the model to match
        and load the weights with strict=True.
        """
        self.device = device
        self.cfg = cfg
        ckpt_path = cfg["ckpt"]
        ckpt_dir = os.path.dirname(ckpt_path)

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            tcfg = json.load(f)
        with open(os.path.join(ckpt_dir, "dataset_stats.json")) as f:
            stats = json.load(f)
        self.pos_mean = np.array(stats["pos"]["mean"], dtype=np.float32)
        self.pos_std = np.array(stats["pos"]["std"], dtype=np.float32)

        # ---- I/O contract (read straight from the training config) ----
        self.single_hand = tcfg.get("single_hand", False)         # template default: dual-arm
        self.single_hand_side = tcfg.get("single_hand_side", "right")
        self.centric_mode = tcfg.get("centric_mode", "object_centric")
        self.frame_mode = tcfg.get("frame_mode", "anchor_frame")  # 'anchor_frame' | 'camera_frame'
        self.action_mode = tcfg.get("action_mode", "absolute")
        self.pred_horizon = tcfg.get("pred_horizon", 50)
        self.max_ict = tcfg.get("max_ict", 8)
        self.img_size = tuple(tcfg.get("image_size", [240, 320]))  # (H, W)
        self.use_object_tokens = self.centric_mode != "ego_centric"

        self.sides = [self.single_hand_side] if self.single_hand else ["left", "right"]
        self.num_hands = len(self.sides)
        self.ict_dim = 20 if self.single_hand else 29              # [type1 + pose9 + hand_rel(9|18) + flag1]
        self.base_action_dim = 10 * self.num_hands                 # per hand: pos3 + o6d6 + grasp1
        self.num_inference_steps = cfg.get("num_inference_steps",
                                           tcfg.get("num_inference_steps", 10))

        # ---- architecture flags (these CHANGE which layers exist -> must match the ckpt) ----
        self.use_pcd_features = tcfg.get("use_pcd_features", False)
        self.use_aux_obj_dynamics = tcfg.get("use_aux_obj_dynamics", False)
        self.use_aux_visual_foresight = tcfg.get("use_aux_visual_foresight", False)
        self.use_region_attn = tcfg.get("use_region_attn", False)
        self.use_done_in_flow = tcfg.get("use_done_in_flow", False)

        # action layout = [hands(base) | object-dynamics(9, optional) | done(1, optional)]
        self.obj_action_dim = 9 if self.use_aux_obj_dynamics else 0
        self.action_dim = self.base_action_dim + self.obj_action_dim + (1 if self.use_done_in_flow else 0)

        if self.action_mode != "absolute":
            print("[ICTPolicy] NOTE: template decodes 'absolute' actions only; for 'delta' "
                  "compose each prediction with the EE pose at re-anchor time (InferencePolicy.py).")
        if self.use_pcd_features:
            print("[ICTPolicy] NOTE: checkpoint uses PCD features but this template feeds none "
                  "(model tolerates it / runs degraded). See InferencePolicy.prepare_pcd_input.")

        # ---- build to match, then load weights ----
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        self.model = FlowMatchingModel(
            single_hand=self.single_hand,
            pred_horizon=self.pred_horizon,
            max_ict=self.max_ict,
            img_size=self.img_size,
            patch_size=tcfg.get("patch_size", 16),
            vision_embed_dim=tcfg.get("vision_embed_dim", 384),
            num_decoder_layers=tcfg.get("num_decoder_layers", 6),
            num_heads=tcfg.get("num_heads", 8),
            mlp_ratio=tcfg.get("mlp_ratio", 4.0),
            dropout=tcfg.get("dropout", 0.05),
            use_pcd_features=self.use_pcd_features,
            use_aux_obj_dynamics=self.use_aux_obj_dynamics,
            use_aux_visual_foresight=self.use_aux_visual_foresight,
            use_aux_temporal_contrastive=tcfg.get("use_aux_temporal_contrastive", False),
            use_region_attn=self.use_region_attn,
            use_pre_norm=tcfg.get("use_pre_norm", True),
            use_ctx_norm=tcfg.get("use_ctx_norm", True),
            use_done_in_flow=self.use_done_in_flow,
        ).to(device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # Fixed noise -> deterministic action per call (resample for a stochastic policy).
        self.noise = torch.randn(1, self.pred_horizon, self.action_dim, device=device)
        print(f"[ICTPolicy] loaded {ckpt_path} | hands={self.num_hands} ict_dim={self.ict_dim} "
              f"frame={self.frame_mode} region_attn={self.use_region_attn} steps={self.num_inference_steps}")

    # ================================================================
    # Geometry helpers
    # ================================================================
    def _encode(self, T: np.ndarray) -> np.ndarray:
        """4x4 pose -> normalized 9D [pos(3), o6d(6)] (same encoding as training)."""
        pos = normalize_pos(T[:3, 3].astype(np.float32), self.pos_mean, self.pos_std)
        o6d = normalize_o6d(rotmat_to_o6d(T[:3, :3].astype(np.float32)))
        return np.concatenate([pos, o6d]).astype(np.float32)

    def _T_cam_in_ref(self, anchor: Optional[ObjectState]) -> np.ndarray:
        """Camera->reference transform. Anchor frame = the anchor object's frame."""
        if self.frame_mode == "anchor_frame" and anchor is not None:
            return _inv(anchor.T_in_cam)
        return np.eye(4, dtype=np.float32)  # camera_frame: reference IS the camera

    # ================================================================
    # Visual input
    # ================================================================
    def prepare_image(self, rgb_bgr: np.ndarray) -> torch.Tensor:
        """Clean BGR image -> (1, 3, H, W) float tensor in [0,1], RGB order."""
        h, w = self.img_size
        img = cv2.resize(rgb_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # training used RGB
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return x.unsqueeze(0).to(self.device)

    def compute_anchor_uv(self, anchor: Optional[ObjectState], K: np.ndarray,
                          img_w: int, img_h: int) -> Optional[torch.Tensor]:
        """Project the anchor center to normalized [0,1] UV (only if region attention)."""
        if not self.use_region_attn or anchor is None:
            return None
        p = anchor.T_in_cam[:3, 3]
        if p[2] < 1e-4:
            uv = np.array([0.5, 0.5], dtype=np.float32)
        else:
            u = float(np.clip((K[0, 0] * p[0] / p[2] + K[0, 2]) / img_w, 0.0, 1.0))
            v = float(np.clip((K[1, 1] * p[1] / p[2] + K[1, 2]) / img_h, 0.0, 1.0))
            uv = np.array([u, v], dtype=np.float32)
        return torch.from_numpy(uv).unsqueeze(0).to(self.device)

    # ================================================================
    # ICT construction  (must match training/FlowMatchingDataloader._build_ict)
    # ================================================================
    def build_ict(
        self,
        ee_poses_in_cam: Dict[str, np.ndarray],   # {"left": T(4,4), "right": T(4,4)} HAND-frame poses
        grippers: Dict[str, float],               # {"left": 0..1, "right": 0..1}
        obj_states: Dict[str, ObjectState],       # {"obj1": ObjectState, ...}, obj1 == anchor
        anchor_key: str = "obj1",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the (1, max_ict, ict_dim) ICT sequence + (1, max_ict) validity mask.

        Token order MUST match training: hand(s) first, then the anchor object,
        then the rest. Each token is
            [type_id, pose_in_ref(9), hand-in-entity(9 single / 18 dual), flag]
        where hand-in-entity encodes where each hand sits relative to that entity
        — that relative encoding is what makes the representation interaction-centric.
        """
        anchor = obj_states.get(anchor_key)
        T_cam_in_ref = self._T_cam_in_ref(anchor)

        hands_in_ref: Dict[str, np.ndarray] = {}
        for side, T_h_cam in ee_poses_in_cam.items():
            if T_h_cam is not None:
                hands_in_ref[side] = T_cam_in_ref @ T_h_cam

        def hand_in_entity(T_ent_in_ref: np.ndarray) -> np.ndarray:
            """Pose of each hand expressed in the entity frame (9D single / 18D dual)."""
            T_ref_in_ent = _inv(T_ent_in_ref)
            rels = []
            for s in self.sides:
                T_h = hands_in_ref.get(s)
                rels.append(self._encode(T_ref_in_ent @ T_h) if T_h is not None
                            else np.zeros(9, dtype=np.float32))
            return np.concatenate(rels)

        tokens: List[np.ndarray] = []

        # 1) hand token(s)
        for s in self.sides:
            T_h = hands_in_ref.get(s)
            if T_h is None:
                tokens.append(np.zeros(self.ict_dim, dtype=np.float32))
                continue
            type_id = TYPE_HAND_L if s == "left" else TYPE_HAND_R
            rel = hand_in_entity(T_h) if self.use_object_tokens \
                else np.zeros(9 * self.num_hands, dtype=np.float32)
            tok = np.concatenate([[type_id], self._encode(T_h), rel, [float(grippers.get(s, 0.0))]])
            tokens.append(tok.astype(np.float32))

        # 2) object tokens (anchor first, then others) — object-centric only
        if self.use_object_tokens:
            ordered = [anchor_key] + [k for k in obj_states if k != anchor_key]
            for k in ordered:
                st = obj_states.get(k)
                if st is None:
                    continue
                T_obj_in_ref = T_cam_in_ref @ st.T_in_cam
                type_id = TYPE_OBJ_ANCHOR if k == anchor_key else TYPE_OBJ_OTHER
                tok = np.concatenate(
                    [[type_id], self._encode(T_obj_in_ref), hand_in_entity(T_obj_in_ref), [-1.0]]
                )
                tokens.append(tok.astype(np.float32))

        # 3) pad / mask to max_ict
        x = np.zeros((self.max_ict, self.ict_dim), dtype=np.float32)
        mask = np.zeros((self.max_ict,), dtype=bool)
        for i, tok in enumerate(tokens[: self.max_ict]):
            if tok[0] == 0.0:   # TYPE_PAD -> missing entity, leave masked
                continue
            x[i], mask[i] = tok, True

        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return x, mask

    # ================================================================
    # Flow-matching inference
    # ================================================================
    @torch.no_grad()
    def infer(
        self,
        x_rgb: torch.Tensor,
        x_ict: torch.Tensor,
        ict_mask: torch.Tensor,
        anchor_uv: Optional[torch.Tensor] = None,   # required iff use_region_attn
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
        """Solve the flow ODE and unpack the future hand trajectory.

        Returns:
            traj: {side: (pos(K,3), o6d(K,6), grasp(K,))} in the REFERENCE frame.
            done_prob: scalar in [0,1] — policy's "task finished" estimate.
        """
        x_t = self.noise.clone()
        dt = 1.0 / self.num_inference_steps
        out = None
        for i in range(self.num_inference_steps):                  # forward Euler ODE integration
            t = torch.full((1, 1), i * dt, device=self.device)
            out = self.model(x_rgb=x_rgb, x_ict=x_ict, ict_mask=ict_mask,
                             x_t=x_t, t=t, anchor_uv=anchor_uv)
            x_t = x_t + out["v_pred"] * dt

        a = x_t[0].cpu().numpy()  # (K, action_dim)

        # done: either the last flow dim, or a dedicated BCE head
        if self.use_done_in_flow:
            done_prob = float(1.0 / (1.0 + np.exp(-a[:, -1].mean())))
            a = a[:, :-1]
        else:
            done_prob = float(1.0 / (1.0 + np.exp(-out["done_logit"][0, 0].cpu().item())))

        def sig(z):
            return 1.0 / (1.0 + np.exp(-z))

        # hand sub-vector is always the FIRST base_action_dim entries (object-dynamics,
        # if present, sit after it and are ignored here).
        traj: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if self.single_hand:
            traj[self.single_hand_side] = (a[:, 0:3], a[:, 3:9], sig(a[:, 9:10]))
        else:
            # dual-hand layout (grouped, NOT interleaved):
            #   [L_pos 0:3 | R_pos 3:6 | L_o6d 6:12 | R_o6d 12:18 | L_g 18 | R_g 19]
            traj["left"] = (a[:, 0:3], a[:, 6:12], sig(a[:, 18:19]))
            traj["right"] = (a[:, 3:6], a[:, 12:18], sig(a[:, 19:20]))
        return traj, done_prob

    # ================================================================
    # Decode one predicted step -> EE command in the camera frame
    # ================================================================
    def decode_ee_in_cam(
        self,
        pos: np.ndarray,                 # (3,) predicted normalized position (reference frame)
        o6d: np.ndarray,                 # (6,) predicted 6D rotation       (reference frame)
        anchor: Optional[ObjectState],   # anchor object (defines the reference frame)
        T_align: np.ndarray,             # (4,4) HAND-frame -> EE-frame bridge (see run_inference.py)
    ) -> np.ndarray:
        """Map one (pos, o6d) prediction back to a 4x4 EE target in the camera frame.

        Chain:  pred (hand pose in REF)  --T_ref_in_cam-->  hand pose in CAM
                                          --inv(T_align)-->  EE pose in CAM
        """
        T_hand_in_ref = np.eye(4, dtype=np.float32)
        T_hand_in_ref[:3, :3] = o6d_to_rotmat(normalize_o6d(o6d))
        T_hand_in_ref[:3, 3] = unnormalize_pos(pos, self.pos_mean, self.pos_std)

        T_ref_in_cam = anchor.T_in_cam if (self.frame_mode == "anchor_frame" and anchor is not None) \
            else np.eye(4, dtype=np.float32)
        T_hand_in_cam = T_ref_in_cam @ T_hand_in_ref
        return T_hand_in_cam @ _inv(T_align)
