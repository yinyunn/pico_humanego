# -*- coding: utf-8 -*-
# ==============================================================================
# FileName: FlowMatchingModel.py
# 
# DESCRIPTION:
# This module implements "FlowMatchingModel", the Ultimate Flow Matching policy with 
# advanced Co-Training paradigms designed for few-shot robotic manipulation.
#
# ABLATIONS & FEATURES:
# 1. ICT: Handles variable number of objects natively.
# 2. Point Cloud Injection: Fuses explicit 3D geometry (x_pcd) into ICTs via PointNet.
# 3. Object Dynamics Co-Training: Joint manifold flow matching (19D/29D).
# 4. Visual Foresight: Dynamic Deconv head predicting future 2D spatial heatmaps.
# 5. Temporal Contrastive: Predicts future ICTs in latent space.
# 6. Learnable Region-Aware Attention: Gaussian bias with learnable spotlight (sigma).
# ================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# ------------------------------------------------------------
# 1. Positional Embeddings
# ------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * 1000.0 * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H_patch, W_patch)w
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

# ------------------------------------------------------------
# 2. Explicit 3D Geometry Encoder (PointNet for x_pcd)
# ------------------------------------------------------------
class MiniPointNet(nn.Module):
    """ Encodes (64, 3) point cloud into a single feature vector per token. """
    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, T_ict, 64, 3)
        B, T, N, _ = pts.shape
        x = pts.view(B * T, N, 3).transpose(1, 2)  # (B*T, 3, 64)
        
        x = self.mlp1(x)               # (B*T, 128, 64)
        x = torch.max(x, dim=2)[0]     # Max Pooling -> (B*T, 128)
        
        x = self.mlp2(x)               # (B*T, out_dim)
        return x.view(B, T, -1)        # (B, T, out_dim)

# ------------------------------------------------------------
# 3. Region-Aware Transformer Decoder Block
# ------------------------------------------------------------
class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1, use_pre_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.use_pre_norm = use_pre_norm

        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        q: torch.Tensor,
        ctx: torch.Tensor,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
        spatial_attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.use_pre_norm:
            # Pre-Norm (new default): norm before attention
            q2, _ = self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q))
            q = q + q2

            q2, _ = self.cross_attn(
                self.norm2(q), ctx, ctx,
                key_padding_mask=ctx_key_padding_mask,
                attn_mask=spatial_attn_bias
            )
            q = q + q2

            q2 = self.mlp(self.norm3(q))
            q = q + q2
        else:
            # Post-Norm (legacy): norm after residual add
            q2, _ = self.self_attn(q, q, q)
            q = self.norm1(q + q2)

            q2, _ = self.cross_attn(
                q, ctx, ctx,
                key_padding_mask=ctx_key_padding_mask,
                attn_mask=spatial_attn_bias
            )
            q = self.norm2(q + q2)

            q2 = self.mlp(q)
            q = self.norm3(q + q2)

        return q


# ------------------------------------------------------------
# 5. Main Flow Matching Policy Model
# ------------------------------------------------------------
class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        *,
        single_hand: bool = True,
        pred_horizon: int = 50,
        max_ict: int = 8, 

        img_size: Tuple[int, int] = (240, 320),
        patch_size: int = 16,
        vision_embed_dim: int = 384,

        num_decoder_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,

        horizon_weighting: str = "uniform",
        horizon_beta: float = 0.0,
        
        # --- Ablations & Co-Training ---
        use_pcd_features: bool = True,
        use_aux_obj_dynamics: bool = True,
        use_aux_visual_foresight: bool = True,
        use_aux_temporal_contrastive: bool = True,
        use_region_attn: bool = True,

        # --- Legacy Compatibility Switches ---
        use_pre_norm: bool = True,           # False = Post-Norm (legacy)
        use_ctx_norm: bool = True,           # False = no LayerNorm on context (legacy)
        use_done_in_flow: bool = False,      # True = Done flag in flow matching dim (legacy)
    ):
        super().__init__()
        self.single_hand = single_hand
        self.pred_horizon = pred_horizon
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.use_pcd_features = use_pcd_features
        self.use_aux_obj_dynamics = use_aux_obj_dynamics
        self.use_aux_visual_foresight = use_aux_visual_foresight
        self.use_aux_temporal_contrastive = use_aux_temporal_contrastive
        self.use_region_attn = use_region_attn

        self.use_pre_norm = use_pre_norm
        self.use_ctx_norm = use_ctx_norm
        self.use_done_in_flow = use_done_in_flow

        # --- Action Dimensions (Unified Joint Manifold Router) ---
        self.num_hands = 1 if single_hand else 2
        self.base_action_dim = 10 * self.num_hands
        self.obj_action_dim = 9 if self.use_aux_obj_dynamics else 0
        self.done_in_flow_dim = 1 if self.use_done_in_flow else 0
        self.action_dim = self.base_action_dim + self.obj_action_dim + self.done_in_flow_dim

        # -------------------------------
        # Core Embeddings
        # -------------------------------
        self.rgb_embed = PatchEmbed(3, vision_embed_dim, patch_size)

        # ICT Dimension:[TypeID(1) + Pose_in_Ref(9) + HandL_in_This(9) + (HandR_in_This(9)) + Flag(1)]
        self.ict_dim = 20 if single_hand else 29
        # NOTE: attr names `state_proj` / `state_pos_emb` (and `head_future_state` below) are
        # intentionally kept (not renamed to ict_*) for backward-compat with existing
        # checkpoints (these become state_dict keys). They project / position-embed the ICTs.
        self.state_proj = nn.Linear(self.ict_dim, vision_embed_dim)
        self.state_pos_emb = nn.Parameter(torch.randn(1, max_ict, vision_embed_dim))
        
        # Explicit 3D PCD Features
        if self.use_pcd_features:
            self.pcd_encoder = MiniPointNet(out_dim=vision_embed_dim)
            self.pcd_alpha = nn.Parameter(torch.tensor(0.5))

        self.ctx_norm = nn.LayerNorm(vision_embed_dim) if self.use_ctx_norm else nn.Identity()

        # -------------------------------
        # Flow Matching Components
        # -------------------------------
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(vision_embed_dim),
            nn.Linear(vision_embed_dim, vision_embed_dim * 2),
            nn.Mish(),
            nn.Linear(vision_embed_dim * 2, vision_embed_dim),
        )
        self.action_proj = nn.Linear(self.action_dim, vision_embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(pred_horizon, vision_embed_dim))

        # -------------------------------
        # Transformer Decoder
        # -------------------------------
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(vision_embed_dim, num_heads, mlp_ratio, dropout, use_pre_norm=self.use_pre_norm)
            for _ in range(num_decoder_layers)
        ])

        # -------------------------------
        # Prediction Heads
        # -------------------------------
        # 1. Main Vector Field Head 
        self.head_v = nn.Linear(vision_embed_dim, self.action_dim)
        
        # 2. 2D Trace Regression Head (Replaces Deconv Heatmap)
        if self.use_aux_visual_foresight:
            self.num_trace_targets = self.num_hands + (1 if self.use_aux_obj_dynamics else 0)
            self.head_2d_trace = nn.Sequential(
                nn.Linear(vision_embed_dim, 256),
                nn.Mish(),
                nn.Linear(256, self.pred_horizon * self.num_trace_targets * 2)
            )
            
        # 3. Temporal Contrastive Head
        if self.use_aux_temporal_contrastive:
            self.head_future_state = nn.Sequential(
                nn.Linear(vision_embed_dim, vision_embed_dim),
                nn.Mish(),
                nn.Linear(vision_embed_dim, self.ict_dim)
            )

        # 4. Done (Task Termination) Head — only when Done is NOT in flow matching
        if not self.use_done_in_flow:
            self.head_done = nn.Sequential(
                nn.Linear(vision_embed_dim, 128),
                nn.Mish(),
                nn.Linear(128, 1)
            )

        # Learnable Gaussian Spotlight for Region Attention
        if self.use_region_attn:
            self.spatial_sigma = nn.Parameter(torch.tensor(0.15))

        w = self._build_horizon_weights(pred_horizon, horizon_weighting, horizon_beta)
        self.register_buffer("horizon_weights", w, persistent=False)

    @staticmethod
    def _build_horizon_weights(K: int, mode: str, beta: float) -> torch.Tensor:
        if mode == "linear":
            w = torch.linspace(K, 1, K)
        elif mode == "exp":
            w = torch.exp(-beta * torch.arange(K, dtype=torch.float32))
        else:
            w = torch.ones(K)
        return w / w.sum()

    def _generate_spatial_bias(self, anchor_uv: torch.Tensor, T_ict: int, N_vis: int) -> torch.Tensor:
        """ Generates a dynamic Gaussian bias mask centered at the Anchor Object. """
        B = anchor_uv.size(0)
        grid_h, grid_w = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        
        center_x = anchor_uv[:, 0] * grid_w
        center_y = anchor_uv[:, 1] * grid_h
        
        y = torch.arange(grid_h, device=anchor_uv.device).view(-1, 1).expand(grid_h, grid_w)
        x = torch.arange(grid_w, device=anchor_uv.device).view(1, -1).expand(grid_h, grid_w)
        
        dist_sq = (x.unsqueeze(0) - center_x.view(-1, 1, 1))**2 + (y.unsqueeze(0) - center_y.view(-1, 1, 1))**2
        
        # Prevent division by zero and extreme sharpening
        sigma = torch.clamp(self.spatial_sigma, min=0.05, max=1.0)
        sigma_sq = (grid_w * sigma) ** 2 
        
        vis_bias = - (dist_sq / (2 * sigma_sq)) 
        vis_bias = vis_bias.view(B, N_vis)      
        
        state_bias = torch.zeros((B, T_ict), device=anchor_uv.device)
        full_bias = torch.cat([state_bias, vis_bias], dim=1) 
        
        full_bias = full_bias.unsqueeze(1).unsqueeze(1) 
        full_bias = full_bias.expand(B, self.num_heads, self.pred_horizon, T_ict + N_vis)
        return full_bias.reshape(B * self.num_heads, self.pred_horizon, T_ict + N_vis)

    def forward(
        self,
        *,
        x_rgb: torch.Tensor,       
        x_ict: torch.Tensor,     
        ict_mask: torch.Tensor,
        x_t: torch.Tensor,         
        t: torch.Tensor,
        x_pcd: Optional[torch.Tensor] = None,
        anchor_uv: Optional[torch.Tensor] = None, 
    ) -> Dict[str, torch.Tensor]:

        B = x_rgb.shape[0]
        out_dict = {}

        # 1. Vision Tokens
        vis_tokens = self.rgb_embed(x_rgb)  
        N_vis = vis_tokens.size(1)

        # 2. ICTs (With Optional 3D PCD Fusion)
        T_ict = x_ict.size(1)
        ict_tokens = self.state_proj(x_ict)

        if self.use_pcd_features and x_pcd is not None:
            # x_pcd shape is (B, T_ict, 64, 3), fallback check for dummy 1D tensor
            if x_pcd.dim() == 4:
                pcd_feats = self.pcd_encoder(x_pcd)
                ict_tokens = ict_tokens + self.pcd_alpha * pcd_feats

        ict_tokens = ict_tokens + self.state_pos_emb[:, :T_ict, :]

        # 3. Context & Masking
        ctx = torch.cat([ict_tokens, vis_tokens], dim=1) 
        ctx = self.ctx_norm(ctx) 
        
        vis_pad = torch.zeros((B, N_vis), dtype=torch.bool, device=x_rgb.device)
        ctx_pad = torch.cat([~ict_mask, vis_pad], dim=1) 

        # 4. Action Query
        t_emb = self.time_mlp(t)
        act_emb = self.action_proj(x_t)
        q = act_emb + self.pos_embed.unsqueeze(0) + t_emb.unsqueeze(1)

        # 5. Region Attention Bias
        spatial_attn_bias = None
        if self.use_region_attn and anchor_uv is not None:
            spatial_attn_bias = self._generate_spatial_bias(anchor_uv, T_ict, N_vis)

        # 6. Transformer
        for blk in self.decoder:
            q = blk(q, ctx, ctx_key_padding_mask=ctx_pad, spatial_attn_bias=spatial_attn_bias)

        #[HEAD 1]: Velocity Field (10D / 19D / 20D / 29D)
        out_dict["v_pred"] = self.head_v(q)

        # --- Pool Context for Co-Training Heads ---
        needs_global_ctx = (
            (not self.use_done_in_flow) or
            self.use_aux_visual_foresight or
            self.use_aux_temporal_contrastive
        )

        if needs_global_ctx:
            ctx_mask_float = (~ctx_pad).float().unsqueeze(-1)
            global_ctx = (ctx * ctx_mask_float).sum(dim=1) / ctx_mask_float.sum(dim=1).clamp(min=1e-6)

        # [HEAD DONE]: Task Termination — only when Done is NOT in flow matching
        if not self.use_done_in_flow:
            out_dict["done_logit"] = self.head_done(global_ctx)  # (B, 1)

        # [HEAD 2]: 2D Trace Regression
        if self.use_aux_visual_foresight:
            trace_flat = self.head_2d_trace(global_ctx) # (B, K * targets * 2)
            # Reshape back to (B, K, targets, 2)
            out_dict["trace_pred"] = trace_flat.view(B, self.pred_horizon, self.num_trace_targets, 2)

        #[HEAD 3]: Temporal Contrastive
        if self.use_aux_temporal_contrastive:
            out_dict["ict_fut_pred"] = self.head_future_state(ict_tokens)

        return out_dict

    def compute_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        loss_lambdas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:

        # --- 1. Flow Matching Loss ---
        v_pred = preds["v_pred"].float()
        v_target = targets["v_target"].float()

        w_time = self.horizon_weights.to(v_pred.device).view(1, -1, 1).float()
        w_p, w_r, w_g = (weights.get(k, 1.0) for k in ["w_pos", "w_rot", "w_g"]) if weights else (1.0, 1.0, 1.0)
        w_flow = weights.get("w_flow", 1.0) if weights else 1.0

        D = v_pred.shape[-1]
        w_dim = torch.ones((1, 1, D), device=v_pred.device, dtype=torch.float32)

        # Apply strict routing for Manifold Dimensions.
        # Action layout is MODALITY-MAJOR (matches dataloader/inference/evaluator):
        #   single: [pos(0:3), o6d(3:9), g(9:10)]
        #   dual  : [L_pos,R_pos(0:6) | L_o6d,R_o6d(6:18) | L_g,R_g(18:20)]
        base_D = self.base_action_dim  # 10 or 20
        nh = self.num_hands
        w_dim[..., 0 : nh*3]     = w_p   # all hands' positions
        w_dim[..., nh*3 : nh*9]  = w_r   # all hands' rotations
        w_dim[..., nh*9 : nh*10] = w_g   # all hands' grasps

        if self.use_aux_obj_dynamics:
            w_dim[..., base_D : base_D+3] = w_p * 0.5
            w_dim[..., base_D+3 : base_D+9] = w_r * 0.5

        # Done-in-flow: apply w_done weight to the last dimension
        if self.use_done_in_flow:
            w_done = weights.get("w_done", 1.0) if weights else 1.0
            w_dim[..., -1] = w_done

        diff = (v_pred - v_target) ** 2
        loss_flow = (diff * w_time * w_dim).sum(dim=1).mean()

        loss_dict = {"loss_flow": loss_flow.detach(), "loss_unweighted": diff.mean().detach()}
        total_loss = w_flow * loss_flow

        # --- Extract individual raw MSE components for console logging ---
        # MODALITY-MAJOR: pos / rot / grasp are each contiguous across all hands.
        pos_diffs = [diff[..., 0 : nh*3]]
        rot_diffs = [diff[..., nh*3 : nh*9]]
        g_diffs   = [diff[..., nh*9 : nh*10]]

        if self.use_aux_obj_dynamics:
            pos_diffs.append(diff[..., base_D : base_D+3])
            rot_diffs.append(diff[..., base_D+3 : base_D+9])

        loss_dict["loss_pos"] = torch.cat(pos_diffs, dim=-1).mean().detach()
        loss_dict["loss_rot"] = torch.cat(rot_diffs, dim=-1).mean().detach()
        loss_dict["loss_g"] = torch.cat(g_diffs, dim=-1).mean().detach()

        # --- Done Loss ---
        if self.use_done_in_flow:
            # Done is already inside flow matching loss via w_dim, just log it
            loss_dict["loss_done"] = diff[..., -1:].mean().detach()
        elif "done_logit" in preds and "y_done" in targets:
            # Independent BCE head
            w_done = weights.get("w_done", 1.0) if weights else 1.0
            loss_done = F.binary_cross_entropy_with_logits(
                preds["done_logit"].float(), targets["y_done"].float()
            )
            total_loss = total_loss + w_done * loss_done
            loss_dict["loss_done"] = loss_done.detach()

        # --- 2. 2D Trace MSE Loss ---
        if self.use_aux_visual_foresight and "trace_pred" in preds and "y_2d_trace" in targets:
            lam_hf = loss_lambdas.get("lambda_foresight", 1.0) if loss_lambdas else 1.0
            # Coordinates are directly in [0, 1] range, no sigmoid needed
            loss_trace = F.mse_loss(preds["trace_pred"].float(), targets["y_2d_trace"].float())
            total_loss = total_loss + lam_hf * loss_trace
            loss_dict["loss_foresight"] = loss_trace.detach()

        # --- 3. Temporal Contrastive Loss ---
        if self.use_aux_temporal_contrastive and "ict_fut_pred" in preds and "x_ict_future" in targets:
            lam_tc = loss_lambdas.get("lambda_contrastive", 1.0) if loss_lambdas else 1.0
            
            # 🌟 Core Slicing: The Dataloader strictly guarantees that the first 'num_hands' tokens are always hand tokens!
            num_h = self.num_hands
            s_pred_hand = preds["ict_fut_pred"][:, :num_h].float()
            s_target_hand = targets["x_ict_future"][:, :num_h].float()
            s_mask_hand = targets["ict_mask_future"][:, :num_h].float().unsqueeze(-1) 
            
            # Penalize only the deviation of future hand features, effectively removing 
            # interference/noise from environmental objects in the latent space.
            loss_tc = (F.mse_loss(s_pred_hand, s_target_hand, reduction="none") * s_mask_hand).sum() / s_mask_hand.sum().clamp(min=1.0)
            total_loss = total_loss + lam_tc * loss_tc
            loss_dict["loss_contrastive"] = loss_tc.detach()

        loss_dict["loss"] = total_loss
        return loss_dict

if __name__ == "__main__":
    # --- Sanity Check Script ---
    print("\n[FlowMatchingModel] Running Dimension Check...")
    
    # Dual Hand + Obj Dynamics = 29D Action Space
    model = FlowMatchingModel(
        single_hand=False, 
        use_aux_obj_dynamics=True, 
        use_aux_visual_foresight=True,
        use_aux_temporal_contrastive=True,
        use_pcd_features=True
    )
    
    B, K, H, W = 2, 50, 240, 320
    x_rgb = torch.randn(B, 3, H, W)
    x_ict = torch.randn(B, 8, 29) # 29D Token (Dual Hand)
    x_pcd = torch.randn(B, 8, 64, 3) # 8 entities, 64 pts each
    ict_mask = torch.ones(B, 8, dtype=torch.bool)
    
    x_t = torch.randn(B, K, 29) # 29D Joint Manifold (no Done dim — Done uses separate head)
    t = torch.rand(B, 1)
    
    out = model(x_rgb=x_rgb, x_ict=x_ict, ict_mask=ict_mask, x_t=x_t, t=t, x_pcd=x_pcd)
    
    print(f"-> V_pred shape: {out['v_pred'].shape} (Expected B, K, 29)")
    print(f"-> Trace shape: {out['trace_pred'].shape} (Expected B, K, 3, 2)")
    print(f"-> State Future shape: {out['ict_fut_pred'].shape} (Expected B, 8, 29)")


# python -m training.FlowMatchingModel