# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Edge-optimized block modules for real-time small object detection.

This module provides lightweight alternatives to standard YOLO blocks:
- RepGhostBlock: Reparameterization + Ghost convolution for edge devices
- C2f_Edge: C2f optimized with depthwise separable convolutions
- BiFPN_Add: Weighted bidirectional feature fusion
- ASFF_Lite: Adaptive spatial feature fusion (lightweight)
- DyHead_Edge: Dynamic detection head for small objects
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, autopad

__all__ = (
    "ASFF_Lite",
    "BiFPN_Add",
    "C2f_Edge",
    "C3_Edge",
    "DyHead_Edge",
    "ECAAttention",
    "EdgeSPPF",
    "RepConv_Edge",
    "RepGhostBlock",
)


class ECAAttention(nn.Module):
    """Efficient Channel Attention - lightweight attention for edge devices.

    Proposed in ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://arxiv.org/abs/1910.03155
    """

    def __init__(self, c: int, gamma: int = 2, b: int = 1):
        """Initialize ECA module.

        Args:
            c: Number of channels
            gamma: Parameter for kernel size calculation
            b: Bias for kernel size calculation
        """
        super().__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(c, dtype=torch.float32)) / gamma) + b / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ECA attention."""
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class RepConv_Edge(nn.Module):
    """RepConv optimized for edge inference with structural reparameterization.

    Training: Multi-branch (3x3, 1x1, identity) Inference: Single 3x3 conv (fused)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
        deploy: bool = False,
    ):
        """Initialize RepConv_Edge.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            d: Dilation
            act: Activation function
            deploy: Whether in deployment mode (fused)
        """
        super().__init__()
        self.deploy = deploy
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        else:
            # Training branches
            self.rbr_3x3 = Conv(c1, c2, k, s, p, g, d, act=False)
            self.rbr_1x1 = Conv(c1, c2, 1, s, 0, g, d, act=False) if c1 == c2 and s == 1 else None
            self.rbr_identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        out = self.rbr_3x3(x)
        if self.rbr_1x1 is not None:
            out = out + self.rbr_1x1(x)
        out = out + id_out
        return self.act(out)

    def switch_to_deploy(self):
        """Fuse training branches into single conv for inference."""
        if self.deploy:
            return

        # Get fused weight and bias
        w_3x3 = self.rbr_3x3.conv.weight
        b_3x3 = self.rbr_3x3.bn.bias if hasattr(self.rbr_3x3, "bn") else torch.zeros(w_3x3.size(0))

        if self.rbr_1x1 is not None:
            w_1x1 = self.rbr_1x1.conv.weight
            # Pad 1x1 to 3x3
            w_1x1_padded = F.pad(w_1x1, [1, 1, 1, 1])
            w_fused = w_3x3 + w_1x1_padded
            b_1x1 = self.rbr_1x1.bn.bias if hasattr(self.rbr_1x1, "bn") else torch.zeros(w_1x1.size(0))
            b_fused = b_3x3 + b_1x1
        else:
            w_fused = w_3x3
            b_fused = b_3x3

        if self.rbr_identity is not None:
            # Identity as 1x1 conv centered
            identity_weight = torch.zeros_like(w_3x3)
            for i in range(w_3x3.size(0)):
                identity_weight[i, i, 1, 1] = 1.0
            w_fused = w_fused + identity_weight
            b_fused = b_fused + self.rbr_identity.bias

        # Create reparam conv
        self.rbr_reparam = nn.Conv2d(
            w_fused.size(1),
            w_fused.size(0),
            w_fused.size(2),
            self.rbr_3x3.conv.stride,
            self.rbr_3x3.conv.padding,
            groups=self.rbr_3x3.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = w_fused
        self.rbr_reparam.bias.data = b_fused

        # Delete training branches
        del self.rbr_3x3
        if hasattr(self, "rbr_1x1"):
            del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity

        self.deploy = True


class RepGhostBlock(nn.Module):
    """RepGhost Block - combines RepVGG reparameterization with Ghost for ultra-lightweight inference.

    Training: Ghost bottleneck structure Inference: Fused single path
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, shortcut: bool = True):
        """Initialize RepGhostBlock.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            shortcut: Whether to use shortcut
        """
        super().__init__()
        c_ = c2 // 2
        self.stride = s
        self.shortcut = shortcut and c1 == c2 and s == 1

        # Primary convolutions (training only, will be fused)
        self.conv1 = Conv(c1, c_, 1, 1, act=False)  # Reduce
        self.conv2 = DWConv(c_, c_, k, s, act=False)  # Depthwise
        self.conv3 = Conv(c_, c2, 1, 1, act=False)  # Expand

        # Ghost branch
        self.ghost = GhostConv(c1, c2, 1, s) if s == 1 else None

        # Activation
        self.act = nn.SiLU()

        # Batch norms for fusion
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c_)
        self.bn3 = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.stride == 2:
            # For stride=2, can't easily fuse, use standard path
            return self.act(self.conv3(self.conv2(self.conv1(x))))

        # Main path with BN
        y = self.bn1(self.conv1.conv(x))
        y = self.bn2(self.conv2.conv(y))
        y = self.bn3(self.conv3.conv(y))

        # Ghost shortcut
        if self.ghost is not None and self.shortcut:
            y = y + self.ghost(x)

        return self.act(y)

    def fuse(self):
        """Fuse conv-bn for faster inference."""
        self.conv1 = fuse_conv_and_bn(self.conv1.conv, self.bn1)
        self.conv2 = fuse_conv_and_bn(self.conv2.conv, self.bn2)
        self.conv3 = fuse_conv_and_bn(self.conv3.conv, self.bn3)


class C2f_Edge(nn.Module):
    """C2f optimized for edge devices using depthwise separable convolutions.

    Combines the efficiency of C2f with DWConv for reduced FLOPs.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        """Initialize C2f_Edge.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of EdgeBottleneck blocks
            shortcut: Whether to use shortcut connections
            e: Expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(EdgeBottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class EdgeBottleneck(nn.Module):
    """Lightweight bottleneck using depthwise separable convolutions."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        """Initialize EdgeBottleneck.

        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Whether to use shortcut
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)  # Pointwise
        self.cv2 = DWConv(c2, c2, 3, 1)  # Depthwise
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_Edge(nn.Module):
    """C3 variant optimized for edge with DWConv and ECA attention."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5, use_eca: bool = False):
        """Initialize C3_Edge.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of blocks
            shortcut: Whether to use shortcut
            e: Expansion ratio
            use_eca: Whether to use ECA attention
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(EdgeBottleneck(c_, c_, shortcut) for _ in range(n)))
        self.eca = ECAAttention(c2) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x1 = self.m(x1)
        out = self.cv3(torch.cat([x1, x2], 1))
        return self.eca(out)


class BiFPN_Add(nn.Module):
    """BiFPN weighted addition for feature fusion.

    Learnable weights for efficient multi-scale feature fusion.
    """

    def __init__(self, c: int, num_inputs: int = 2, epsilon: float = 1e-4):
        """Initialize BiFPN_Add.

        Args:
            c: Number of channels (for compatibility, not used in addition)
            num_inputs: Number of input tensors to fuse
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
        self.num_inputs = num_inputs
        # Learnable weights (initialized to 1)
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Weighted addition of input tensors.

        Args:
            x: List of tensors to fuse

        Returns:
            Fused tensor
        """
        # Normalize weights with softmax
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.epsilon)

        # Weighted sum
        out = weights[0] * x[0]
        for i in range(1, len(x)):
            out = out + weights[i] * x[i]
        return out


class ASFF_Lite(nn.Module):
    """Adaptive Spatial Feature Fusion - Lite version for edge devices.

    Dynamically learns to fuse multi-scale features with spatial attention.
    """

    def __init__(self, c: int, level: int = 0, num_levels: int = 3):
        """Initialize ASFF_Lite.

        Args:
            c: Number of channels
            level: Current level (0=high res, 1=med, 2=low)
            num_levels: Total number of feature levels
        """
        super().__init__()
        self.level = level
        self.num_levels = num_levels

        # Spatial attention for each level
        self.weight_convs = nn.ModuleList()
        for i in range(num_levels):
            # Use DWConv for efficiency
            self.weight_convs.append(
                nn.Sequential(
                    DWConv(c, c, 3, 1),
                    nn.Conv2d(c, 1, 1),  # Single channel weight map
                    nn.Sigmoid(),
                )
            )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features with adaptive spatial weights.

        Args:
            x: List of feature maps from different scales

        Returns:
            Fused feature map at target scale
        """
        target_h, target_w = x[self.level].shape[2:]

        fused = []
        for i, feat in enumerate(x):
            # Resize to target size
            if i != self.level:
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)

            # Apply spatial attention
            weight = self.weight_convs[i](feat)
            feat = feat * weight
            fused.append(feat)

        # Sum fusion
        return sum(fused)


class EdgeSPPF(nn.Module):
    """SPPF optimized for edge - faster pooling with reduced channels."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize EdgeSPPF.

        Args:
            c1: Input channels
            c2: Output channels
            k: Pool kernel size
        """
        super().__init__()
        c_ = c1 // 2  # Reduce channels first
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class DyHead_Edge(nn.Module):
    """Dynamic Detection Head - Edge optimized for small object detection.

    Combines scale-aware, spatial-aware, and task-aware attention efficiently.
    """

    def __init__(self, c: int, num_anchors: int = 1, num_classes: int = 80):
        """Initialize DyHead_Edge.

        Args:
            c: Input channels
            num_anchors: Number of anchors per location
            num_classes: Number of classes
        """
        super().__init__()
        self.c = c
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Scale-aware attention (simplified)
        self.scale_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, 1, 1), nn.Sigmoid())

        # Spatial-aware (offset learning for small objects)
        self.spatial_conv = DWConv(c, c, 3, 1)
        self.spatial_attn = nn.Conv2d(c, 1, 1)

        # Task-aware (box vs cls)
        self.task_conv = Conv(c, c, 1, 1)

        # Output layers
        self.box_conv = nn.Conv2d(c, num_anchors * 4, 1)
        self.cls_conv = nn.Conv2d(c, num_anchors * num_classes, 1)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through dynamic head.

        Args:
            x: List of multi-scale features

        Returns:
            box_predictions, cls_predictions
        """
        box_outputs = []
        cls_outputs = []

        for feat in x:
            # Scale-aware modulation
            scale = self.scale_attn(feat)
            feat = feat * scale

            # Spatial-aware (emphasize small object regions)
            spatial_feat = self.spatial_conv(feat)
            spatial_weight = torch.sigmoid(self.spatial_attn(spatial_feat))
            feat = feat + spatial_feat * spatial_weight

            # Task-aware feature
            task_feat = self.task_conv(feat)

            # Predictions
            box_outputs.append(self.box_conv(task_feat))
            cls_outputs.append(self.cls_conv(task_feat))

        return torch.cat(box_outputs, 1), torch.cat(cls_outputs, 1)
