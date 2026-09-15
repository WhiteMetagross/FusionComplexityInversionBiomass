"""
Model architectures for CSIRO Pasture Biomass experiments.

This module defines all neural network architectures used in the
Fusion Complexity Inversion study, including five cross-view fusion
blocks, and two dual-view model wrappers.

Fusion Blocks:
- GatedDepthwiseConvBlock: Gated depthwise-conv fusion (baselines B4, B5).
- MambaFusionBlock: Hardware-accelerated Mamba SSM fusion (B6, ablations A1-A6).
- BidirMamba: Bidirectional SSM with gating and local convolution.
- BabyMambaFusionBlock: Weight-tied bidirectional Mamba block.
- CVGABlock: Cross-View Gated Attention fusion.

Model Wrappers:
- BiomassModelTimm: Dual-view model with any timm backbone.
- VMambaBackbone and BiomassModelVMamba: VMamba VSSM-based model.

Author: Mridankan Mandal
Paper: Fusion Complexity Inversion: Why Simpler Cross-View Modules
       Outperform SSMs and Cross-View Attention Transformers for
       Pasture Biomass Regression (arXiv:2603.07819).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────────────────────────────
# Fusion Blocks.
# ──────────────────────────────────────────────────────────────────────

class GatedDepthwiseConvBlock(nn.Module):
    """Gated depthwise-conv fusion block for local spatial feature mixing."""
    def __init__(self, dim, kernel_size=5, dropout=0.1, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class MambaFusionBlock(nn.Module):
    """Mamba SSM fusion block using mamba_ssm for hardware-accelerated
    selective state space scanning through triton and CUDA kernels."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, **kwargs):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    @torch.compiler.disable  # mamba_ssm uses custom CUDA kernels incompatible with dynamo.
    def forward(self, x):
        shortcut = x
        # Mamba SSM CUDA kernels are NOT AMP-safe — force fp32
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x = self.norm(x)
            x = self.mamba(x)
            x = self.drop(x)
        return shortcut + x


class BidirMamba(nn.Module):
    """Bidirectional Mamba Fusion Block.

    Weight-tied bidirectional SSM scanning with
    gating and local convolution.

    Key mechanisms:
    - Sigmoid gating to filter features (proven in GatedDepthwiseConv).
    - Depthwise convolution for local spatial context.
    - Weight-tied bidirectional Mamba SSM: forward and flip(SSM(flip(x))), same weights.
    - Output projection and skip connection.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, filter_size=5, dropout=0.1, **kwargs):
        super().__init__()
        from mamba_ssm import Mamba

        self.pre_norm = nn.LayerNorm(d_model)

        # Sigmoid gating (proven effective in GatedDepthwiseConv).
        self.gate = nn.Linear(d_model, d_model)

        # Local spatial aggregation.
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=filter_size,
            padding=filter_size // 2, groups=d_model
        )

        # Weight-tied bidirectional Mamba SSM.
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    @torch.compiler.disable
    def forward(self, x):
        shortcut = x
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()

            # Layer normalization.
            x = self.pre_norm(x)

            # Sigmoid gating.
            g = torch.sigmoid(self.gate(x))
            x = x * g

            # Local spatial aggregation.
            x = x.transpose(1, 2)
            x = self.dwconv(x)
            x = x.transpose(1, 2)

            # Weight-tied bidirectional Mamba SSM.
            h_fwd = self.mamba(x)
            h_bwd = torch.flip(self.mamba(torch.flip(x, dims=[1])), dims=[1])
            x = h_fwd + h_bwd

            # Project and apply skip connection.
            x = self.proj(x)
            x = self.drop(x)

        return shortcut + x


class BabyMambaFusionBlock(nn.Module):
    """Weight-Tied Bidirectional Mamba Block (BabyMamba)

    Forward and backward passes share the exact same parameters.
    Matches the formulation from NanoMamba-Crossover-BiDir.
    """
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, dropout=0.1, **kwargs):
        super().__init__()
        from mamba_ssm import Mamba
        self.pre_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @torch.compiler.disable
    def forward(self, x):
        shortcut = x
        # Mamba SSM CUDA kernels are NOT AMP-safe — force fp32.
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x = self.pre_norm(x)

            # Forward pass: t -> T.
            h_fwd = self.mamba(x)

            # Backward pass: T -> t (weight-tied).
            x_flip = torch.flip(x, dims=[1])
            h_bwd = self.mamba(x_flip)
            h_bwd = torch.flip(h_bwd, dims=[1])

            h_out = h_fwd + h_bwd
            h_out = self.drop(h_out)

            out = shortcut + h_out
            out = self.post_norm(out)
        return out


class CVGABlock(nn.Module):
    """Cross-View Gated Attention Block.

    Splits the concatenated left+right token sequence, performs bidirectional
    cross-attention between views (left queries attend to right keys/values
    and vice versa), applies sigmoid gating, and recombines with a skip
    connection.

    Uses F.scaled_dot_product_attention for automatic flash and memory-efficient
    dispatch. Fully AMP-safe and torch.compile compatible.
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)

        # Weight-tied Q/K/V projections (same weights for both views).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Sigmoid gating (same structure proven in GatedDepthwiseConv).
        self.gate = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_drop_p = dropout

    def forward(self, x):
        B, S, D = x.shape
        N = S // 2
        shortcut = x

        x = self.norm(x)

        # Input gating — filter before computing Q/K/V.
        g = torch.sigmoid(self.gate(x))
        x = x * g

        # Split into views.
        left, right = x[:, :N], x[:, N:]

        # Compute Q, K, V (weight-tied across views).
        def qkv(t):
            q = self.q_proj(t).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(t).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(t).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            return q, k, v

        q_l, k_l, v_l = qkv(left)
        q_r, k_r, v_r = qkv(right)

        # Bidirectional cross-attention.
        drop_p = self.attn_drop_p if self.training else 0.0
        left_out = F.scaled_dot_product_attention(q_l, k_r, v_r, dropout_p=drop_p)
        right_out = F.scaled_dot_product_attention(q_r, k_l, v_l, dropout_p=drop_p)

        # Reshape back: (B, heads, N, head_dim) -> (B, N, D).
        left_out = left_out.transpose(1, 2).contiguous().view(B, N, D)
        right_out = right_out.transpose(1, 2).contiguous().view(B, N, D)

        x = torch.cat([left_out, right_out], dim=1)
        x = self.out_proj(x)
        x = self.drop(x)

        return shortcut + x


# ──────────────────────────────────────────────────────────────────────
# Shared regression head builder.
# ──────────────────────────────────────────────────────────────────────

def _make_head(nf, dropout):
    return nn.Sequential(
        nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(nf // 2, 1), nn.Softplus()
    )


# ──────────────────────────────────────────────────────────────────────
# BiomassModelTimm — any timm backbone and fusion.
# ──────────────────────────────────────────────────────────────────────

class BiomassModelTimm(nn.Module):
    """Dual-view biomass model with a timm backbone and optional Mamba SSM fusion."""
    def __init__(self, model_name, dropout=0.2, use_mamba_ssm=False, use_babymamba=False, use_biobabymamba=False, use_cvga=False, pretrained=True,
                 use_metadata=False, meta_input_dim=23, img_size=None, freeze_backbone=False):
        super().__init__()
        backbone_kwargs = dict(pretrained=pretrained, num_classes=0, global_pool='')
        if img_size is not None and model_name.startswith('vit_'):
            backbone_kwargs['img_size'] = img_size

        if pretrained and model_name == 'vit_large_patch14_dinov2.lvd142m':
            pretrained_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained')
            local_pth = os.path.join(pretrained_dir, 'dinov2_vitl14_pretrain.pth')
            hub_pth = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth')
            pth_file = local_pth if os.path.exists(local_pth) else (hub_pth if os.path.exists(hub_pth) else None)
            if pth_file:
                backbone_kwargs['pretrained'] = False
                self.backbone = timm.create_model(model_name, **backbone_kwargs)
                sd = torch.load(pth_file, map_location='cpu')
                if 'mask_token' in sd:
                    del sd['mask_token']
                self.backbone.load_state_dict(sd, strict=True)
                print(f"Loaded {model_name} from local file: {pth_file}")
            else:
                self.backbone = timm.create_model(model_name, **backbone_kwargs)
        else:
            self.backbone = timm.create_model(model_name, **backbone_kwargs)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        elif hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        nf = self.backbone.num_features
        print(f"Backbone: {model_name}, features={nf}, frozen={freeze_backbone}")

        if use_cvga:
            FusionBlock = CVGABlock
            fusion_label = "CVGA"
        elif use_biobabymamba:
            FusionBlock = BidirMamba
            fusion_label = "BidirMamba"
        elif use_babymamba:
            FusionBlock = BabyMambaFusionBlock
            fusion_label = "BabyMamba"
        elif use_mamba_ssm:
            FusionBlock = MambaFusionBlock
            fusion_label = "MambaSSM"
        else:
            FusionBlock = GatedDepthwiseConvBlock
            fusion_label = "GatedDepthwiseConv"

        print(f"Fusion: 2× {fusion_label}Block(dim={nf})")

        self.fusion = nn.Sequential(
            FusionBlock(nf, dropout=dropout),
            FusionBlock(nf, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Metadata MLP fusion.
        self.use_metadata = use_metadata
        if use_metadata:
            meta_hidden = 64
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_input_dim, meta_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(meta_hidden, meta_hidden),
            )
            self.meta_proj = nn.Sequential(
                nn.Linear(nf + meta_hidden, nf),
                nn.GELU(),
            )
            print(f"Metadata MLP: {meta_input_dim} → {meta_hidden}, fused with {nf}")

        self.head_green = _make_head(nf, dropout)
        self.head_dead = _make_head(nf, dropout)
        self.head_clover = _make_head(nf, dropout)

    @staticmethod
    def _as_tokens(x):
        if x.ndim == 4:
            return x.flatten(2).transpose(1, 2)
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Unsupported backbone feature shape: {tuple(x.shape)}")
        return x

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, left, right, metadata=None):
        if self.freeze_backbone:
            with torch.no_grad():
                x_l = self.backbone(left)
                x_r = self.backbone(right)
        else:
            x_l = self.backbone(left)
            x_r = self.backbone(right)
        x_l = self._as_tokens(x_l)
        x_r = self._as_tokens(x_r)
        x = torch.cat([x_l, x_r], dim=1)
        x = self.fusion(x)
        x = self.pool(x.transpose(1, 2)).flatten(1)

        if self.use_metadata and metadata is not None:
            meta_feat = self.meta_mlp(metadata)
            x = self.meta_proj(torch.cat([x, meta_feat], dim=1))

        green = self.head_green(x)
        dead = self.head_dead(x)
        clover = self.head_clover(x)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, dead, clover, gdm, total], dim=1)


# ──────────────────────────────────────────────────────────────────────
# VMamba backbone (uses vmamba.VSSM).
# ──────────────────────────────────────────────────────────────────────

_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained'
)

# VMamba configs — tiny and small are v0 (vanilla) checkpoints, and base is v2.
# Each carries the full VSSM constructor kwargs matching the pretrained weights.
_VSSM_V0_KWARGS = dict(
    ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
    ssm_conv=3, ssm_conv_bias=True, ssm_init="v0", forward_type="v0",
    mlp_ratio=0.0, mlp_act_layer="gelu", patch_norm=True, norm_layer="ln",
    downsample_version="v1", patchembed_version="v1",
)
_VSSM_V2_KWARGS = dict(
    ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
    ssm_conv=3, ssm_conv_bias=False, ssm_init="v0", forward_type="v05_noz",
    mlp_ratio=4.0, mlp_act_layer="gelu", patch_norm=True, norm_layer="ln2d",
    downsample_version="v3", patchembed_version="v2",
)

VMAMBA_CONFIGS = {
    'vmamba_tiny': {
        'depths': [2, 2, 9, 2],   # v0 vanilla tiny
        'dims': 96,
        'num_features': 768,       # dims * 2^3
        'pretrained_weight': 'vssmtiny_dp01_ckpt_epoch_292.pth',
        'vssm_kwargs': _VSSM_V0_KWARGS,
    },
    'vmamba_small': {
        'depths': [2, 2, 27, 2],   # v0 vanilla small
        'dims': 96,
        'num_features': 768,
        'pretrained_weight': 'vssmsmall_dp03_ckpt_epoch_238.pth',
        'vssm_kwargs': _VSSM_V0_KWARGS,
    },
    'vmamba_base': {
        'depths': [2, 2, 15, 2],   # v2 s2l15
        'dims': 128,
        'num_features': 1024,
        'pretrained_weight': 'vssm_base_0229_ckpt_epoch_237.pth',
        'vssm_kwargs': _VSSM_V2_KWARGS,
    },
}


class VMambaBackbone(nn.Module):
    """VMamba backbone wrapping vmamba.VSSM with ImageNet-1K weights.

    Supports 'vmamba_tiny', 'vmamba_small', 'vmamba_base' via VMAMBA_CONFIGS.
    """
    def __init__(self, pretrained_path='', variant='vmamba_base'):
        super().__init__()
        cfg = VMAMBA_CONFIGS.get(variant)
        if cfg is None:
            raise ValueError(f"Unknown VMamba variant '{variant}'. "
                             f"Choose from: {list(VMAMBA_CONFIGS.keys())}")

        # Auto-resolve cached weight if no explicit path is given.
        if not pretrained_path:
            candidate = os.path.join(_PRETRAINED_DIR, cfg['pretrained_weight'])
            if os.path.exists(candidate):
                pretrained_path = candidate
        try:
            from vmamba import VSSM
            self.model = VSSM(
                depths=cfg['depths'],
                dims=cfg['dims'],
                **cfg['vssm_kwargs'],
            )
            self.num_features = cfg['num_features']
            if pretrained_path and os.path.exists(pretrained_path):
                ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                if 'model' in ckpt:
                    ckpt = ckpt['model']
                # Remove classifier head weights (not needed for feature extraction).
                ckpt = {k: v for k, v in ckpt.items() if not k.startswith('classifier.')}
                self.model.load_state_dict(ckpt, strict=False)
                print(f"Loaded VMamba-{variant} weights from {pretrained_path}")
            self._use_vssm = True
            self._is_v0 = (cfg['vssm_kwargs'].get('forward_type', '') == 'v0')
        except ImportError:
            print("[WARN] vmamba package not available — using timm ViT fallback")
            self.model = timm.create_model(
                'vit_base_patch16_224.augreg_in21k_ft_in1k',
                pretrained=True, num_classes=0, global_pool='',
                img_size=512,  # match experiment resolution
            )
            self.num_features = self.model.num_features
            self._use_vssm = False
            self._is_v0 = False
        print(f"VMamba backbone ({variant}), features={self.num_features}")

    @torch.compiler.disable
    def forward(self, x):
        if self._use_vssm:
            # Extract features before the classifier head.
            x = self.model.patch_embed(x)
            if self.model.pos_embed is not None:
                x = x + self.model.pos_embed
            for layer in self.model.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            # Apply final normalization (differs between v0 and v2).
            if hasattr(self.model.classifier, 'norm'):
                x = self.model.classifier.norm(x)
            elif hasattr(self.model, 'norm'):
                x = self.model.norm(x)
            # Reshape to (B, seq_len, C) for sequence fusion.
            if x.dim() == 4:
                B = x.shape[0]
                if self._is_v0:
                    # v0: (B, H, W, C) — channel-last.
                    _, H, W, C = x.shape
                    x = x.reshape(B, H * W, C)
                else:
                    # v2: (B, C, H, W) — channel-first.
                    _, C, H, W = x.shape
                    x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            return x
        return self.model(x)


class BiomassModelVMamba(nn.Module):
    """Dual-view biomass model with VMamba backbone + Mamba SSM fusion."""
    def __init__(self, pretrained_path='', dropout=0.2, use_mamba_ssm=True,
                 use_metadata=False, meta_input_dim=23, variant='vmamba_base'):
        super().__init__()
        self.backbone = VMambaBackbone(pretrained_path, variant=variant)
        nf = self.backbone.num_features

        FusionBlock = MambaFusionBlock if use_mamba_ssm else GatedDepthwiseConvBlock
        fusion_label = "MambaSSM" if use_mamba_ssm else "GatedDepthwiseConv"
        print(f"Fusion: 2× {fusion_label}Block(dim={nf})")

        self.fusion = nn.Sequential(
            FusionBlock(nf, dropout=dropout),
            FusionBlock(nf, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Metadata MLP fusion.
        self.use_metadata = use_metadata
        if use_metadata:
            meta_hidden = 64
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_input_dim, meta_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(meta_hidden, meta_hidden),
            )
            self.meta_proj = nn.Sequential(
                nn.Linear(nf + meta_hidden, nf),
                nn.GELU(),
            )
            print(f"Metadata MLP: {meta_input_dim} → {meta_hidden}, fused with {nf}")

        self.head_green = _make_head(nf, dropout)
        self.head_dead = _make_head(nf, dropout)
        self.head_clover = _make_head(nf, dropout)

    def forward(self, left, right, metadata=None):
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        x = torch.cat([x_l, x_r], dim=1)
        x = self.fusion(x)
        x = self.pool(x.transpose(1, 2)).flatten(1)

        if self.use_metadata and metadata is not None:
            meta_feat = self.meta_mlp(metadata)
            x = self.meta_proj(torch.cat([x, meta_feat], dim=1))

        green = self.head_green(x)
        dead = self.head_dead(x)
        clover = self.head_clover(x)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, dead, clover, gdm, total], dim=1)
