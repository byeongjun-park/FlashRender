# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Set
import torch
from torch import nn


class Discriminator(torch.nn.Module):
    """
    Base class for Discriminators.
    Subclasses should override the forward method.
    """

    def __init__(self, feature_indices: Optional[Set[int]] = None):
        super().__init__()
        self.feature_indices = feature_indices

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the discriminator.

        Args:
            feats (List[torch.Tensor]): The features to use for the discriminator.

        Returns:
            torch.Tensor: The output of the discriminator.

        Raises:
            NotImplementedError: If the forward method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement forward()")


def _get_optimal_groups(num_channels: int) -> int:
    """
    Calculate optimal number of groups for GroupNorm.

    Args:
        num_channels: Number of input channels

    Returns:
        Optimal number of groups that divides num_channels evenly
    """
    if num_channels <= 32:
        # For small channel counts, use num_channels//4 but ensure at least 1 group
        groups = max(1, num_channels // 4)
    else:
        # For larger channel counts, try to use 32 groups, but find the largest divisor <= 32
        groups = 32
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
    assert num_channels % groups == 0, f"{num_channels} not divisible by {groups}"

    return groups


# =============================================================================
# EDM Discriminator
# =============================================================================


class Discriminator_EDM(Discriminator):
    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        all_res: List[int] = [32, 16, 8],
        in_channels: int = 256,
    ):
        super(Discriminator_EDM, self).__init__(feature_indices=feature_indices)
        if self.feature_indices is None:
            self.feature_indices = {len(all_res) - 1}  # use the middle bottleneck feature
        self.feature_indices = {
            i for i in self.feature_indices if i < len(all_res)
        }  # make sure feature indices are valid
        self.in_res = [all_res[i] for i in sorted(self.feature_indices)]
        self.in_channels = in_channels

        self.discriminator_heads = nn.ModuleList()
        for res in self.in_res:
            layers = []
            while res > 8:
                # reduce the resolution by half, until 8x8
                layers.extend(
                    [
                        nn.Conv2d(
                            kernel_size=4,
                            in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(num_groups=_get_optimal_groups(self.in_channels), num_channels=self.in_channels),
                        nn.SiLU(),
                    ]
                )
                res //= 2

            layers.extend(
                [
                    nn.Conv2d(
                        kernel_size=4, in_channels=self.in_channels, out_channels=self.in_channels, stride=2, padding=1
                    ),
                    # 8x8 -> 4x4
                    nn.GroupNorm(num_groups=_get_optimal_groups(self.in_channels), num_channels=self.in_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        kernel_size=4, in_channels=self.in_channels, out_channels=self.in_channels, stride=4, padding=0
                    ),
                    # 4x4 -> 1x1
                    nn.GroupNorm(num_groups=_get_optimal_groups(self.in_channels), num_channels=self.in_channels),
                    nn.SiLU(),
                    nn.Conv2d(kernel_size=1, in_channels=self.in_channels, out_channels=1, stride=1, padding=0),
                    # 1x1 -> 1x1
                ]
            )

            # append the layers for current resolution to the discriminator head
            self.discriminator_heads.append(nn.Sequential(*layers))

    def forward(self, feats: List[torch.Tensor]):
        assert isinstance(feats, list)
        all_logits = []

        if len(feats) != len(self.in_res):
            raise ValueError(
                f"Number of feature maps {len(feats)} does not match the number of resolutions {len(self.in_res)}"
            )

        for i, res in enumerate(self.in_res):
            assert res == feats[i].shape[-1]
            # perform average pooling over spatial dimension if necessary
            logits = self.discriminator_heads[i](feats[i]).reshape(-1, 1)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=1)

        return all_logits


# =============================================================================
# SD15 and SDXL, Flux Discriminator
# =============================================================================


class Discriminator_SD15(Discriminator_EDM):
    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        all_res: List[int] = [32, 16, 8, 8, 8],
        in_channels: int = 1280,
    ):
        super().__init__(feature_indices=feature_indices, all_res=all_res, in_channels=in_channels)


class Discriminator_SDXL(Discriminator_EDM):
    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        all_res: List[int] = [32, 16, 16, 16],
        in_channels: int = 1280,
    ):
        super().__init__(feature_indices=feature_indices, all_res=all_res, in_channels=in_channels)


# =============================================================================
# Image Diffusion Transformer (DiT) Discriminator
# =============================================================================


class Discriminator_ImageDiT(Discriminator):
    """
    Simple discriminator for image features from image DiT models (e.g., Flux).

    Uses a lightweight 2-layer Conv2D architecture (~0.5M params per head).

    Input: List of feature tensors with shape [B, inner_dim, H, W]
    Output: Concatenated logits [B, num_heads] for discrimination
    """

    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        num_blocks: int = 57,  # Flux: 19 joint + 38 single = 57
        inner_dim: int = 3072,  # Flux hidden dimension
    ):
        """
        Initialize the image DiT discriminator.

        Args:
            feature_indices: Which block indices to apply discrimination to.
                           Defaults to middle block if None.
            num_blocks: Total number of blocks in the model.
                       Flux: 57 (19 joint + 38 single)
            inner_dim: Input channel dimension of the features.
                      Flux: 3072
        """
        super().__init__(feature_indices=feature_indices)

        # Validate and set feature indices
        if self.feature_indices is None:
            self.feature_indices = {int(num_blocks // 2)}
        self.feature_indices = {i for i in self.feature_indices if i < num_blocks}
        self.num_features = len(self.feature_indices)
        self.inner_dim = inner_dim

        # Build discriminator heads - simple 2-layer conv2d
        hidden_channels = inner_dim // 2
        self.cls_pred_heads = nn.ModuleList()
        for _ in range(self.num_features):
            head = nn.Sequential(
                nn.Conv2d(
                    in_channels=inner_dim,
                    out_channels=hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(num_groups=_get_optimal_groups(hidden_channels), num_channels=hidden_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.cls_pred_heads.append(head)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            feats: List of feature tensors, one for each discriminator head.
                  Each tensor shape: [B, inner_dim, H, W]

        Returns:
            Concatenated logits from all heads: [B, num_features]
        """
        if not isinstance(feats, list) or len(feats) != self.num_features:
            raise ValueError(
                f"Expected list of {self.num_features} feature tensors, "
                f"got {type(feats)} with length {len(feats) if isinstance(feats, list) else 'N/A'}"
            )

        all_logits = []
        for head, feat in zip(self.cls_pred_heads, feats):
            logits = head(feat)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)


# =============================================================================
# Unified Video Diffusion Discriminator (DiT)
# =============================================================================


def _build_dit_simple_conv3d_discriminator_head(
    inner_dim: int,
    kernel_size=(2, 4, 4),
    stride=(2, 2, 2),
    padding=(0, 1, 1),
) -> nn.Sequential:
    """
    Builds a simple 2-layer Conv3D discriminator head

    This is a lightweight discriminator with just two conv3d layers, suitable for
    scenarios where computational efficiency is prioritized over discrimination power.

    Args:
        inner_dim: Input channel dimension.
        kernel_size: Kernel size for the first conv3d layer.
        stride: Stride for the first conv3d layer.
        padding: Padding for the first conv3d layer.

    Returns:
        Sequential module representing the discriminator head.
    """
    hidden_channels = inner_dim // 2

    return nn.Sequential(
        nn.Conv3d(
            kernel_size=kernel_size,
            in_channels=inner_dim,
            out_channels=hidden_channels,
            stride=stride,
            padding=padding,
        ),  # Default: reduces spatial/temporal dimensions
        nn.GroupNorm(num_groups=_get_optimal_groups(hidden_channels), num_channels=hidden_channels),
        nn.LeakyReLU(0.2),
        nn.Conv3d(kernel_size=1, in_channels=hidden_channels, out_channels=1, stride=1, padding=0),
        # Final 1x1 conv to get logits
        nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global average pooling
        nn.Flatten(),
    )


def _build_dit_conv3d_discriminator_head(
    inner_dim: int,
    channel_mults: List[int],
    mlp_hidden_dim: int,
) -> nn.Sequential:
    """
    Builds a 3D convolutional discriminator head with joint spatiotemporal processing.

    This architecture uses 3D convolutions to process temporal and spatial dimensions
    jointly, providing good performance but with higher parameter count.

    Input shape: [B, inner_dim, T, H, W] where T=21, H=30, W=52
    Output shape: [B, 1]

    Args:
        inner_dim: Input channel dimension (typically 384).
        channel_mults: Channel multipliers for each conv layer [c2, c3, c4].
        mlp_hidden_dim: Hidden dimension of the final MLP classifier.

    Returns:
        Sequential module representing the complete discriminator head.
    """
    assert len(channel_mults) == 3
    c1, c2, c3, c4 = inner_dim, *channel_mults

    conv_stack = nn.Sequential(
        nn.Conv3d(c1, c2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # Output: (B, c2, 21, 15, 26)
        nn.GroupNorm(num_groups=_get_optimal_groups(c2), num_channels=c2),
        nn.LeakyReLU(0.2),
        nn.Conv3d(c2, c3, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # Output: (B, c3, 11, 7, 13)
        nn.GroupNorm(num_groups=_get_optimal_groups(c3), num_channels=c3),
        nn.LeakyReLU(0.2),
        nn.Conv3d(c3, c4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),  # Output: (B, c3, 6, 4, 7)
        nn.GroupNorm(num_groups=_get_optimal_groups(c4), num_channels=c4),
        nn.LeakyReLU(0.2),
        nn.Conv3d(c4, c4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),  # Output: (B, c4, 3, 2, 4)
        nn.GroupNorm(num_groups=_get_optimal_groups(c4), num_channels=c4),
        nn.LeakyReLU(0.2),
    )

    mlp_head = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(c4, mlp_hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(mlp_hidden_dim, 1),
    )

    return nn.Sequential(conv_stack, mlp_head)


def _build_dit_conv1d_2d_discriminator_head(
    inner_dim: int,
    channel_progression: List[int],
    mlp_hidden_dim: int,
) -> nn.Sequential:
    """
    Builds a discriminator head using separate Conv1d (temporal) and Conv2d (spatial) operations.

    Input shape: [B, inner_dim, T, H, W] where T=21, H=30, W=52

    The architecture alternates between temporal (Conv1d) and spatial (Conv2d) processing:
    1. Conv1d along temporal dimension
    2. Conv2d along spatial dimensions
    3. Repeat to progressively downsample

    Args:
        inner_dim: The input channel dimension (384).
        channel_progression: List of channel dimensions for each conv layer.
        mlp_hidden_dim: The hidden dimension of the MLP head.

    Returns:
        An nn.Sequential module representing the discriminator head.
    """

    class Conv1d2dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, temp_kernel=3, spatial_kernel=3, temp_stride=1, spatial_stride=2):
            super().__init__()
            self.temp_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=temp_kernel, stride=temp_stride, padding=temp_kernel // 2
            )
            self.spatial_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=spatial_kernel,
                stride=spatial_stride,
                padding=spatial_kernel // 2,
            )
            self.temp_norm = nn.GroupNorm(num_groups=_get_optimal_groups(out_channels), num_channels=out_channels)
            self.spatial_norm = nn.GroupNorm(num_groups=_get_optimal_groups(out_channels), num_channels=out_channels)
            self.activation = nn.LeakyReLU(0.2)

        def forward(self, x):
            # Input: [B, C, T, H, W]
            B, C, T, H, W = x.shape

            # Temporal processing: Conv1d along T dimension
            # Reshape: [B, C, T, H, W] -> [B*H*W, C, T]
            x_temp = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, C, T)
            x_temp = self.temp_conv(x_temp)  # [B*H*W, C_out, T']
            _, C_out, T_new = x_temp.shape

            # Reshape back: [B*H*W, C_out, T'] -> [B, C_out, T', H, W]
            x = x_temp.view(B, H, W, C_out, T_new).permute(0, 3, 4, 1, 2).contiguous()
            x = self.temp_norm(x)
            x = self.activation(x)

            # Spatial processing: Conv2d along H, W dimensions
            # Reshape: [B, C_out, T', H, W] -> [B*T', C_out, H, W]
            B, C_out, T_new, H, W = x.shape
            x_spatial = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T_new, C_out, H, W)
            x_spatial = self.spatial_conv(x_spatial)  # [B*T', C_out, H', W']
            _, C_out, H_new, W_new = x_spatial.shape

            # Reshape back: [B*T', C_out, H', W'] -> [B, C_out, T', H', W']
            x = x_spatial.view(B, T_new, C_out, H_new, W_new).permute(0, 2, 1, 3, 4).contiguous()
            x = self.spatial_norm(x)
            x = self.activation(x)

            return x

    # Build the conv stack with alternating temporal/spatial processing
    assert len(channel_progression) == 3
    c1, c2, c3, c4 = inner_dim, *channel_progression

    conv_stack = nn.Sequential(
        # Block 1: 384 -> c2, downsample spatial by 2x
        Conv1d2dBlock(c1, c2, temp_stride=1, spatial_stride=2),  # [B, c2, 21, 15, 26]
        # Block 2: c2 -> c3, downsample temporal by 2x and spatial by 2x
        Conv1d2dBlock(c2, c3, temp_stride=2, spatial_stride=2),  # [B, c3, 11, 8, 13]
        # Block 3: c3 -> c4, downsample both dimensions
        Conv1d2dBlock(c3, c4, temp_stride=2, spatial_stride=2),  # [B, c4, 6, 4, 7]
    )

    mlp_head = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(c4, mlp_hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(mlp_hidden_dim, 1),
    )

    return nn.Sequential(conv_stack, mlp_head)


def _build_dit_attention_discriminator_head(
    inner_dim: int,
    num_heads: int = 8,
    num_layers: int = 2,
    mlp_hidden_dim: int = 256,
) -> nn.Module:
    """
    Builds an attention-based discriminator head using self-attention.

    Input shape: [B, inner_dim, T, H, W] where T=21, H=30, W=52

    Uses self-attention to capture long-range dependencies in spatiotemporal features.
    Good for: Global context understanding, long-range temporal relationships.

    Args:
        inner_dim: The input channel dimension (384).
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_hidden_dim: The hidden dimension of the final MLP head.

    Returns:
        An nn.Sequential module representing the discriminator head.
    """

    class SpatioTemporalAttention(nn.Module):
        def __init__(self, dim, num_heads, num_layers):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads

            # Project input channels to attention dimension
            self.input_proj = nn.Linear(inner_dim, dim)

            # Multi-head self-attention layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=0.1,
                activation=nn.LeakyReLU(0.2),
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_proj = nn.Linear(dim, 1)

        def forward(self, x):
            # Input: [B, inner_dim, T, H, W]
            B, C, T, H, W = x.shape

            # Reshape to sequence: [B, T*H*W, C]
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H * W, C)

            # Project to attention dimension
            x = self.input_proj(x)  # [B, T*H*W, dim]

            # Apply transformer layers
            x = self.transformer(x)  # [B, T*H*W, dim]

            # Global average pooling across sequence dimension
            x = x.mean(dim=1)  # [B, dim]

            # Final classification
            x = self.output_proj(x)  # [B, 1]

            return x

    return SpatioTemporalAttention(dim=mlp_hidden_dim, num_heads=num_heads, num_layers=num_layers)


def _build_dit_multiscale_discriminator_head(
    inner_dim: int,
    scales: List[int] = [1, 2, 4],
    channel_reduction: int = 4,
    mlp_hidden_dim: int = 256,
) -> nn.Module:
    """
    Builds a multi-scale discriminator head that processes features at multiple scales.

    Input shape: [B, inner_dim, T, H, W] where T=21, H=30, W=52

    Processes features at multiple temporal/spatial scales simultaneously and combines them.
    Good for: Multi-resolution discrimination, capturing both fine and coarse patterns.

    Args:
        inner_dim: The input channel dimension (384).
        scales: List of pooling scales to use.
        channel_reduction: Factor to reduce channels in each scale branch.
        mlp_hidden_dim: The hidden dimension of the final MLP head.

    Returns:
        An nn.Sequential module representing the discriminator head.
    """

    class MultiScaleProcessor(nn.Module):
        def __init__(self, inner_dim, scales, channel_reduction, mlp_hidden_dim):
            super().__init__()
            self.scales = scales
            reduced_dim = inner_dim // channel_reduction

            # Create processing branches for each scale
            self.scale_branches = nn.ModuleList()
            for scale in scales:
                branch = nn.Sequential(
                    # Channel reduction
                    nn.Conv3d(inner_dim, reduced_dim, kernel_size=1),
                    nn.GroupNorm(num_groups=_get_optimal_groups(reduced_dim), num_channels=reduced_dim),
                    nn.LeakyReLU(0.2),
                    # Scale-specific pooling
                    nn.AvgPool3d(kernel_size=scale, stride=scale) if scale > 1 else nn.Identity(),
                    # Feature processing
                    nn.Conv3d(reduced_dim, reduced_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=_get_optimal_groups(reduced_dim), num_channels=reduced_dim),
                    nn.LeakyReLU(0.2),
                    # Global pooling
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten(),
                )
                self.scale_branches.append(branch)

            # Fusion and classification
            total_features = len(scales) * reduced_dim
            self.classifier = nn.Sequential(
                nn.Linear(total_features, mlp_hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, 1),
            )

        def forward(self, x):
            # Process each scale branch
            scale_features = []
            for branch in self.scale_branches:
                feat = branch(x)
                scale_features.append(feat)

            # Concatenate features from all scales
            combined = torch.cat(scale_features, dim=1)

            # Final classification
            return self.classifier(combined)

    return MultiScaleProcessor(inner_dim, scales, channel_reduction, mlp_hidden_dim)


def _build_dit_factorized_discriminator_head(
    inner_dim: int,
    temporal_dim: int = 128,
    spatial_dim: int = 128,
    mlp_hidden_dim: int = 256,
) -> nn.Module:
    """
    Builds a factorized discriminator with separate temporal and spatial processing branches.

    Input shape: [B, inner_dim, T, H, W] where T=21, H=30, W=52

    Processes temporal and spatial dimensions in separate branches, then combines them.
    Good for: Explicit temporal vs spatial feature separation, interpretable processing.

    Args:
        inner_dim: The input channel dimension (384).
        temporal_dim: Output dimension of temporal branch.
        spatial_dim: Output dimension of spatial branch.
        mlp_hidden_dim: The hidden dimension of the fusion MLP.

    Returns:
        An nn.Sequential module representing the discriminator head.
    """

    class FactorizedProcessor(nn.Module):
        def __init__(self, inner_dim, temporal_dim, spatial_dim, mlp_hidden_dim):
            super().__init__()

            # Temporal processing branch
            self.temporal_branch = nn.Sequential(
                # Global spatial pooling: [B, C, T, H, W] -> [B, C, T]
                nn.AdaptiveAvgPool3d((None, 1, 1)),  # Keep T, pool H,W
                nn.Flatten(start_dim=2),  # [B, C, T]
                # Temporal convolutions with Group normalization
                nn.Conv1d(inner_dim, temporal_dim * 2, kernel_size=5, padding=2),
                nn.GroupNorm(num_groups=_get_optimal_groups(temporal_dim * 2), num_channels=temporal_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(temporal_dim * 2, temporal_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=_get_optimal_groups(temporal_dim), num_channels=temporal_dim),
                nn.LeakyReLU(0.2),
                # Global temporal pooling
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),  # [B, temporal_dim]
            )

            # Spatial processing branch
            self.spatial_branch = nn.Sequential(
                # Global temporal pooling: [B, C, T, H, W] -> [B, C, 1, H, W]
                nn.AdaptiveAvgPool3d((1, None, None)),  # Pool T, keep H,W
                # Spatial convolutions with Group normalization
                nn.Conv2d(inner_dim, spatial_dim * 2, kernel_size=5, padding=2),
                nn.GroupNorm(num_groups=_get_optimal_groups(spatial_dim * 2), num_channels=spatial_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(spatial_dim * 2, spatial_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=_get_optimal_groups(spatial_dim), num_channels=spatial_dim),
                nn.LeakyReLU(0.2),
                # Global spatial pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),  # [B, spatial_dim]
            )

            # Fusion and classification
            self.fusion = nn.Sequential(
                nn.Linear(temporal_dim + spatial_dim, mlp_hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(mlp_hidden_dim // 2, 1),
            )

        def forward(self, x):
            # Process temporal and spatial branches separately
            temporal_feat = self.temporal_branch(x)  # [B, temporal_dim]

            # For spatial branch, we need to handle the dimension properly
            # After AdaptiveAvgPool3d: [B, C, 1, H, W] -> squeeze to [B, C, H, W]
            x_spatial = self.spatial_branch[0](x)  # AdaptiveAvgPool3d: [B, C, 1, H, W]
            x_spatial = x_spatial.squeeze(2)  # Remove temporal dim: [B, C, H, W]

            # Apply the rest of the spatial branch
            for layer in self.spatial_branch[1:]:
                x_spatial = layer(x_spatial)

            spatial_feat = x_spatial  # [B, spatial_dim]

            # Concatenate and fuse
            combined = torch.cat([temporal_feat, spatial_feat], dim=1)

            # Final classification
            return self.fusion(combined)

    return FactorizedProcessor(inner_dim, temporal_dim, spatial_dim, mlp_hidden_dim)


class Discriminator_VideoDiT(Discriminator):
    """
    Advanced discriminator for video features from video diffusion models (DiT, Wan, etc.).

    This unified discriminator supports multiple architectures with different computational and
    performance characteristics. Each architecture is designed for specific use cases
    and parameter budgets, supporting both DiT-style and Wan-style video generation models.

    Architecture Categories:
    1. Simple Conv3D: Lightweight DiT-style discriminator (2-layer)
    2. 3D Convolution: Joint spatiotemporal processing (multi-layer)
    3. Factorized Conv: Separate 1D temporal + 2D spatial processing
    4. Attention: Self-attention for global context modeling
    5. Multi-scale: Parallel processing at multiple resolutions
    6. Factorized: Separate temporal and spatial branch processing

    Available Architectures:
    - dit_simple_conv3d: Simple DiT-style 2-layer conv3d (~1M params)
    - conv3d_down_mlp: Standard 3D conv (~65M params)
    - conv3d_down_mlp_efficient: Efficient 3D conv (~25M params)
    - conv1d_2d_down_mlp: Factorized conv (~17.6M params)
    - conv1d_2d_down_mlp_efficient: Efficient factorized conv (~6.9M params)
    - attention_down_mlp: Large attention model (~15M params)
    - attention_down_mlp_efficient: Compact attention (~5M params)
    - multiscale_down_mlp: Large multi-scale model (~15M params)
    - multiscale_down_mlp_efficient: Compact multi-scale (~4M params)
    - multiscale_down_mlp_medium: Medium multi-scale model (~30M params)
    - multiscale_down_mlp_large: Extra large multi-scale model (~50M params)
    - factorized_down_mlp: Large factorized model (~15M params)
    - factorized_down_mlp_efficient: Compact factorized (~3M params)
    - factorized_down_mlp_large: Extra large factorized model (~50M params)

    Input: List of feature tensors with shape [B, inner_dim, T, H, W]
           where T, H, W are temporal and spatial dimensions (varies by model)
    Output: Concatenated logits [B, num_heads] for discrimination
    """

    # Architecture configurations organized by type and capacity
    ARCHITECTURES = {
        # Simple DiT-style Architectures (lightweight, 2-layer)
        "dit_simple_conv3d": {
            "type": "dit_simple_conv3d",
            "kernel_size": (2, 4, 4),
            "stride": (2, 2, 2),
            "padding": (0, 1, 1),
            "params_estimate": "~1M",
            "description": "Simple DiT-style 2-layer conv3d for efficiency",
        },
        # 3D Convolution Architectures
        "conv3d_down_mlp": {
            "type": "conv3d",
            "channel_mults": [512, 512, 1024],
            "mlp_hidden_dim": 512,
            "params_estimate": "~65M",
            "description": "Standard 3D conv with high capacity",
        },
        "conv3d_down_mlp_efficient": {
            "type": "conv3d",
            "channel_mults": [256, 512, 512],
            "mlp_hidden_dim": 256,
            "params_estimate": "~25M",
            "description": "Efficient 3D conv with reduced channels",
        },
        # Conv1d + Conv2d Architectures
        "conv1d_2d_down_mlp": {
            "type": "conv1d_2d",
            "channel_progression": [512, 512, 1024],
            "mlp_hidden_dim": 512,
            "params_estimate": "~17M",
            "description": "Factorized conv with high capacity",
        },
        "conv1d_2d_down_mlp_efficient": {
            "type": "conv1d_2d",
            "channel_progression": [256, 512, 512],
            "mlp_hidden_dim": 256,
            "params_estimate": "~7M",
            "description": "Efficient factorized conv",
        },
        # Attention Architectures
        "attention_down_mlp": {
            "type": "attention",
            "num_heads": 16,
            "num_layers": 3,
            "mlp_hidden_dim": 512,
            "params_estimate": "~15M",
            "description": "Large attention model with deep layers",
        },
        "attention_down_mlp_efficient": {
            "type": "attention",
            "num_heads": 8,
            "num_layers": 2,
            "mlp_hidden_dim": 256,
            "params_estimate": "~5M",
            "description": "Self-attention for global context",
        },
        # Multiscale Architectures
        "multiscale_down_mlp": {
            "type": "multiscale",
            "scales": [1, 2, 4, 8],
            "channel_reduction": 2,
            "mlp_hidden_dim": 512,
            "params_estimate": "~15M",
            "description": "Large multi-scale with more scales",
        },
        "multiscale_down_mlp_efficient": {
            "type": "multiscale",
            "scales": [1, 2, 4],
            "channel_reduction": 4,
            "mlp_hidden_dim": 256,
            "params_estimate": "~4M",
            "description": "Multi-resolution feature processing",
        },
        "multiscale_down_mlp_medium": {
            "type": "multiscale",
            "scales": [1, 2, 4, 8],
            "channel_reduction": 2,
            "mlp_hidden_dim": 768,
            "params_estimate": "~25M",
            "description": "Medium multi-scale model balancing capacity and efficiency",
        },
        "multiscale_down_mlp_large": {
            "type": "multiscale",
            "scales": [1, 2, 4, 8, 16],
            "channel_reduction": 1,
            "mlp_hidden_dim": 1024,
            "params_estimate": "~50M",
            "description": "Large multi-scale with extensive scales and minimal channel reduction",
        },
        # Factorized Architectures
        "factorized_down_mlp": {
            "type": "factorized",
            "temporal_dim": 256,
            "spatial_dim": 256,
            "mlp_hidden_dim": 512,
            "params_estimate": "~15M",
            "description": "Large factorized with wider branches",
        },
        "factorized_down_mlp_efficient": {
            "type": "factorized",
            "temporal_dim": 128,
            "spatial_dim": 128,
            "mlp_hidden_dim": 256,
            "params_estimate": "~3M",
            "description": "Separate temporal/spatial branches",
        },
        "factorized_down_mlp_large": {
            "type": "factorized",
            "temporal_dim": 512,
            "spatial_dim": 512,
            "mlp_hidden_dim": 1024,
            "params_estimate": "~50M",
            "description": "Large factorized with very wide branches for high capacity",
        },
    }

    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        num_blocks: int = 30,
        disc_type: str = "conv3d_down_mlp_efficient",
        inner_dim: int = 384,
    ):
        """
        Initialize the unified video diffusion discriminator.

        Args:
            feature_indices: Which block indices to apply discrimination to.
                           Defaults to middle block if None.
            num_blocks: Total number of blocks in the model.
                       Common values: 30 (CogVideoX-2B, Wan2.1 2B)
            disc_type: Architecture type. See ARCHITECTURES for options.
                      Defaults to lightweight DiT-style for efficiency.
            inner_dim: Input channel dimension of the features.
                      Common values: 480 (CogVideoX-2B), 384 (Wan2.1 2B)
        """
        super().__init__(feature_indices=feature_indices)

        # Validate and set feature indices
        if self.feature_indices is None:
            self.feature_indices = {int(num_blocks // 2)}
        self.feature_indices = {i for i in self.feature_indices if i < num_blocks}
        self.num_features = len(self.feature_indices)
        self.disc_type = disc_type
        self.inner_dim = inner_dim

        # Validate architecture type
        if disc_type not in self.ARCHITECTURES:
            available = ", ".join(self.ARCHITECTURES.keys())
            raise ValueError(f"Unknown disc_type '{disc_type}'. Available: {available}")

        config = self.ARCHITECTURES[disc_type]

        # Build discriminator heads
        self.cls_pred_heads = nn.ModuleList()
        for _ in range(self.num_features):
            head = self._build_discriminator_head(config, inner_dim)
            self.cls_pred_heads.append(head)

    def _build_discriminator_head(self, config: dict, inner_dim: int) -> nn.Module:
        """Build a single discriminator head based on architecture config."""
        arch_type = config["type"]

        if arch_type == "dit_simple_conv3d":
            return _build_dit_simple_conv3d_discriminator_head(
                inner_dim=inner_dim,
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
            )
        elif arch_type == "conv3d":
            return _build_dit_conv3d_discriminator_head(
                inner_dim=inner_dim,
                channel_mults=config["channel_mults"],
                mlp_hidden_dim=config["mlp_hidden_dim"],
            )
        elif arch_type == "conv1d_2d":
            return _build_dit_conv1d_2d_discriminator_head(
                inner_dim=inner_dim,
                channel_progression=config["channel_progression"],
                mlp_hidden_dim=config["mlp_hidden_dim"],
            )
        elif arch_type == "attention":
            return _build_dit_attention_discriminator_head(
                inner_dim=inner_dim,
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                mlp_hidden_dim=config["mlp_hidden_dim"],
            )
        elif arch_type == "multiscale":
            return _build_dit_multiscale_discriminator_head(
                inner_dim=inner_dim,
                scales=config["scales"],
                channel_reduction=config["channel_reduction"],
                mlp_hidden_dim=config["mlp_hidden_dim"],
            )
        elif arch_type == "factorized":
            return _build_dit_factorized_discriminator_head(
                inner_dim=inner_dim,
                temporal_dim=config["temporal_dim"],
                spatial_dim=config["spatial_dim"],
                mlp_hidden_dim=config["mlp_hidden_dim"],
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            feats: List of feature tensors, one for each discriminator head.
                  Each tensor shape: [B, inner_dim, T, H, W]

        Returns:
            Concatenated logits from all heads: [B, num_features]
        """
        if not isinstance(feats, list) or len(feats) != self.num_features:
            raise ValueError(
                f"Expected list of {self.num_features} feature tensors, "
                f"got {type(feats)} with length {len(feats) if isinstance(feats, list) else 'N/A'}"
            )

        all_logits = []
        for head, feat in zip(self.cls_pred_heads, feats):
            logits = head(feat)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)

    @classmethod
    def list_architectures(cls) -> List[str]:
        """Return a list of all available architecture names."""
        return list(cls.ARCHITECTURES.keys())

    @classmethod
    def get_architecture_info(cls, disc_type: str) -> dict:
        """Get detailed information about a specific architecture."""
        if disc_type not in cls.ARCHITECTURES:
            available = ", ".join(cls.ARCHITECTURES.keys())
            raise ValueError(f"Unknown disc_type '{disc_type}'. Available: {available}")

        config = cls.ARCHITECTURES[disc_type].copy()
        return {
            "name": disc_type,
            "type": config.pop("type"),
            "params_estimate": config.pop("params_estimate"),
            "description": config.pop("description", "No description available"),
            "config": config,
        }

    def get_model_info(self) -> dict:
        """Get information about the current discriminator instance."""
        arch_info = self.get_architecture_info(self.disc_type)
        return {
            **arch_info,
            "num_heads": self.num_features,
            "feature_indices": sorted(self.feature_indices),
            "inner_dim": self.inner_dim,
        }


# =============================================================================
# End of Discriminators
# =============================================================================
