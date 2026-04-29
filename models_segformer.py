"""
    models_segformer.py - SegFormer semantic segmentation model
    Compatible with the DroneDeploy benchmark FastAI pipeline.

    Architecture:
        - Mix Transformer (MiT) encoder: hierarchical patch embedding + efficient self-attention
        - Lightweight MLP decoder: all-MLP head that fuses multi-scale features
        - 4 encoder stages producing features at 1/4, 1/8, 1/16, 1/32 of input resolution
        - No positional encoding - uses mix-FFN with depth-wise conv for positional information

    References:
        SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
        Xie et al., NeurIPS 2021 - https://arxiv.org/abs/2105.15203
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth regularisation (drop entire residual branch per sample)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand = torch.floor(rand + keep)
        return x * rand / keep


class DWConv(nn.Module):
    """Depth-wise convolution used inside Mix-FFN for local positional encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dw(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """Mix-FFN: replaces positional encodings with a depth-wise conv inside the FFN."""
    def __init__(self, dim, ffn_dim, drop=0.0):
        super().__init__()
        self.fc1    = nn.Linear(dim, ffn_dim)
        self.dw     = DWConv(ffn_dim)
        self.act    = nn.GELU()
        self.fc2    = nn.Linear(ffn_dim, dim)
        self.drop   = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dw(x, H, W) + x   # local context
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """
    Sequence Reduction Attention (SR-Attention).
    Reduces key/value sequence length by factor `sr_ratio` to keep complexity linear
    in feature map area, which is critical for high-res early stages.
    """
    def __init__(self, dim, num_heads, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.sr_ratio   = sr_ratio

        self.q          = nn.Linear(dim, dim)
        self.kv         = nn.Linear(dim, dim * 2)
        self.proj       = nn.Linear(dim, dim)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj_drop  = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).view(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """One transformer block: ESA + LayerNorm + MixFFN + DropPath."""
    def __init__(self, dim, num_heads, ffn_ratio=4, sr_ratio=1,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = EfficientSelfAttention(dim, num_heads, sr_ratio, attn_drop, drop)
        self.norm2  = nn.LayerNorm(dim)
        self.ffn    = MixFFN(dim, int(dim * ffn_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding via strided convolution.
    Overlap (kernel > stride) preserves local continuity across patch boundaries.
    """
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4):
        super().__init__()
        pad = patch_size // 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=pad)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                        # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H'*W', C)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Mix Transformer Encoder (MiT-B2 by default)
# ---------------------------------------------------------------------------

# Predefined configs matching the original paper variants
# embed_dims, num_heads, depths, sr_ratios are per-stage lists
MiT_CONFIGS = {
    # (embed_dims, num_heads, depths, sr_ratios)
    'B0': ([32,  64,  160, 256], [1, 2, 5,  8],  [2, 2, 2,  2],  [8, 4, 2, 1]),
    'B1': ([64,  128, 320, 512], [1, 2, 5,  8],  [2, 2, 2,  2],  [8, 4, 2, 1]),
    'B2': ([64,  128, 320, 512], [1, 2, 5,  8],  [3, 4, 6,  3],  [8, 4, 2, 1]),
    'B3': ([64,  128, 320, 512], [1, 2, 5,  8],  [3, 4, 18, 3],  [8, 4, 2, 1]),
    'B4': ([64,  128, 320, 512], [1, 2, 5,  8],  [3, 8, 27, 3],  [8, 4, 2, 1]),
    'B5': ([64,  128, 320, 512], [1, 2, 5,  8],  [3, 6, 40, 3],  [8, 4, 2, 1]),
}

class MixTransformerEncoder(nn.Module):
    """
    Hierarchical Mix Transformer encoder.
    Produces 4 feature maps at strides 4, 8, 16, 32.
    """
    def __init__(self, variant='B2', in_ch=3, drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()

        embed_dims, num_heads, depths, sr_ratios = MiT_CONFIGS[variant]
        self.embed_dims = embed_dims

        # stochastic depth decay across all blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0

        # Build 4 stages
        self.patch_embeds = nn.ModuleList()
        self.stages       = nn.ModuleList()
        self.norms        = nn.ModuleList()

        patch_sizes = [7, 3, 3, 3]
        strides     = [4, 2, 2, 2]
        in_chs      = [in_ch] + embed_dims[:-1]

        for i in range(4):
            self.patch_embeds.append(
                OverlapPatchEmbed(in_chs[i], embed_dims[i], patch_sizes[i], strides[i])
            )
            self.stages.append(nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    ffn_ratio=4,
                    sr_ratio=sr_ratios[i],
                    drop=drop_rate,
                    attn_drop=0.0,
                    drop_path=dpr[cur + j],
                )
                for j in range(depths[i])
            ]))
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for block in self.stages[i]:
                x = block(x, H, W)
            x = self.norms[i](x)
            B, _, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W)
            features.append(x)
        return features   # [1/4, 1/8, 1/16, 1/32] resolution feature maps


# ---------------------------------------------------------------------------
# All-MLP Decoder Head
# ---------------------------------------------------------------------------

class SegFormerHead(nn.Module):
    """
    Lightweight all-MLP decoder.
    1. Projects each stage feature to a unified embed_dim with a linear layer
    2. Upsamples all to 1/4 resolution
    3. Concatenates and fuses with a final linear + conv
    4. Upsamples to output resolution
    """
    def __init__(self, in_channels, embed_dim, num_classes):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(c, embed_dim) for c in in_channels
        ])
        self.fuse    = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        # features: list of (B, C_i, H_i, W_i) at strides 4,8,16,32
        target_H, target_W = features[0].shape[2:]

        projected = []
        for feat, lin in zip(features, self.linears):
            B, C, H, W = feat.shape
            # Linear operates on channel dim: reshape → linear → reshape
            f = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            f = lin(f)                            # (B, H*W, embed_dim)
            f = f.transpose(1, 2).view(B, -1, H, W)
            # Upsample to 1/4 resolution
            f = F.interpolate(f, size=(target_H, target_W),
                              mode='bilinear', align_corners=False)
            projected.append(f)

        x = torch.cat(projected, dim=1)  # (B, embed_dim*4, H/4, W/4)
        x = self.fuse(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Full SegFormer Model
# ---------------------------------------------------------------------------

class SegFormer(nn.Module):
    """
    SegFormer for semantic segmentation.

    Args:
        num_classes:    Number of output classes (6 for DroneDeploy benchmark)
        variant:        MiT encoder variant - 'B0' (tiny) to 'B5' (large)
        decoder_dim:    Unified channel dim in the MLP decoder head (256 default)
        in_ch:          Input image channels (3 for RGB)
        drop_rate:      Dropout in attention/FFN
        drop_path_rate: Stochastic depth rate
        output_size:    If set, upsample logits to this (H, W) in forward pass.
                        Leave None to output at 1/4 input resolution.

    Output:
        (B, num_classes, H, W) logits — at 1/4 input res, or `output_size` if set.
        For FastAI compatibility call with output_size=(input_H, input_W).
    """
    def __init__(self, num_classes=6, variant='B2', decoder_dim=256,
                 in_ch=3, drop_rate=0.0, drop_path_rate=0.1, output_size=None):
        super().__init__()
        self.output_size = output_size
        self.encoder = MixTransformerEncoder(variant, in_ch, drop_rate, drop_path_rate)
        self.decoder = SegFormerHead(
            in_channels=self.encoder.embed_dims,
            embed_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features   = self.encoder(x)
        logits     = self.decoder(features)

        target = self.output_size or input_size
        if logits.shape[2:] != torch.Size(target):
            logits = F.interpolate(logits, size=target,
                                   mode='bilinear', align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def segformer_b0(num_classes=6, **kwargs):
    """Tiny - 3.7M params. Good for quick experiments."""
    return SegFormer(num_classes=num_classes, variant='B0', decoder_dim=256, **kwargs)

def segformer_b1(num_classes=6, **kwargs):
    """Small - 13.7M params."""
    return SegFormer(num_classes=num_classes, variant='B1', decoder_dim=256, **kwargs)

def segformer_b2(num_classes=6, **kwargs):
    """Medium - 24.7M params. Recommended default."""
    return SegFormer(num_classes=num_classes, variant='B2', decoder_dim=768, **kwargs)

def segformer_b3(num_classes=6, **kwargs):
    """Large - 44.6M params."""
    return SegFormer(num_classes=num_classes, variant='B3', decoder_dim=768, **kwargs)
