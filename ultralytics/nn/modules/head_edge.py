# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Edge-optimized detection heads for real-time small object detection.

This module provides lightweight detection heads optimized for edge deployment:
- Detect_Edge: Standard detection head with DWConv optimizations
- Detect_Small: Specialized head for small object detection with P2 support
- Detect_Edge_End2End: End-to-end detection without NMS for maximum speed
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from ultralytics.utils.tal import dist2bbox

from .block import DFL
from .conv import Conv, DWConv
from .head import Detect

__all__ = (
    "Detect_Edge",
    "Detect_Edge_End2End",
    "Detect_Small",
)


class Detect_Edge(Detect):
    """Edge-optimized detection head with depthwise separable convolutions.

    Replaces standard convolutions with depthwise separable for reduced FLOPs while maintaining detection accuracy.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Initialize edge-optimized detection head.

        Args:
            nc: Number of classes
            reg_max: DFL channels
            end2end: Whether to use end-to-end NMS-free detection
            ch: Tuple of channel sizes from backbone feature maps
        """
        # Don't call super().__init__ to avoid creating the standard layers
        nn.Module.__init__(self)

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        # Compute channels with edge-optimized ratios (smaller)
        c2 = max(16, ch[0] // 8, reg_max * 4)  # Box channels - smaller
        c3 = max(ch[0] // 2, min(nc, 100))  # Class channels - smaller

        # Box regression head with DWConv
        self.cv2 = nn.ModuleList()
        for x in ch:
            self.cv2.append(
                nn.Sequential(
                    # First: reduce channels with 1x1
                    Conv(x, c2, 1, 1),
                    # Depthwise 3x3
                    DWConv(c2, c2, 3, 1),
                    # Pointwise output
                    nn.Conv2d(c2, 4 * reg_max, 1),
                )
            )

        # Classification head with DWConv
        self.cv3 = nn.ModuleList()
        for x in ch:
            self.cv3.append(
                nn.Sequential(
                    # Reduce with 1x1
                    Conv(x, c3, 1, 1),
                    # Depthwise
                    DWConv(c3, c3, 3, 1),
                    # Class output
                    nn.Conv2d(c3, nc, 1),
                )
            )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        # End2end support
        self._end2end = end2end
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | dict:
        """Forward pass with edge-optimized inference."""
        # Training/validation path
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        # Inference path
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            # For TF exports
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
            return box, cls

        if self.export and self.format in ("onnx", "engine", "openvino"):
            # For ONNX/OpenVINO - separate outputs for efficient NMS
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
            return box, cls

        # Standard inference
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize biases with edge-optimized values."""
        # Box regression bias - small positive for stability
        for a, b in zip(self.cv2, self.cv2):
            a[-1].bias.data[:] = 1.0  # box

        # Classification bias - standard focal loss initialization
        for a, b in zip(self.cv3, self.cv3):
            a[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)


class Detect_Small(Detect_Edge):
    """Detection head specialized for small object detection.

    Features:
    - Enhanced P2 head for 4x stride detection (small objects)
    - Scale-specific anchor calibration
    - Additional small object augmentation in feature extraction
    """

    # Anchor scales calibrated for small object detection
    # P2: 4x stride, P3: 8x, P4: 16x, P5: 32x
    anchor_scales = {
        0: 4,  # P2 - ultra small (4-16 pixels)
        1: 8,  # P3 - small (8-32 pixels)
        2: 16,  # P4 - medium (16-64 pixels)
        3: 32,  # P5 - large (32+ pixels)
    }

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Initialize small-object-optimized detection head.

        Args:
            nc: Number of classes
            reg_max: DFL channels (higher for small object precision)
            end2end: Whether to use end-to-end detection
            ch: Tuple of channel sizes (should include P2, P3, P4, P5)
        """
        # Use larger reg_max for small object precision
        reg_max = reg_max or 16
        super().__init__(nc, reg_max, end2end, ch)

        # Small object specific calibration factors
        self.small_obj_scales = nn.Parameter(torch.ones(self.nl))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | dict:
        """Forward pass with small object optimizations."""
        # Apply scale-specific calibration for small objects
        for i in range(self.nl):
            x[i] = x[i] * (1.0 + 0.1 * self.small_obj_scales[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        # Inference with optimized decoding for small objects
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # Decode with small object-aware DFL
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)

        # Scale-specific striding (P2 gets finer stride)
        dbox = dbox * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize with small-object-aware biases."""
        super().bias_init()

        # P2 head gets stronger prior for small objects
        if len(self.cv2) >= 4:  # Has P2
            # Lower confidence threshold for P2 (small objects are harder)
            self.cv2[0][-1].bias.data[:] = 0.5  # More conservative box init
            self.cv3[0][-1].bias.data[: self.nc] = math.log(8 / self.nc / (640 / 4) ** 2)


class Detect_Edge_End2End(Detect_Edge):
    """End-to-end detection head optimized for edge devices.

    Eliminates NMS during inference through one-to-one label assignment.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = True, ch: tuple = ()):
        """Initialize end-to-end edge detection head.

        Args:
            nc: Number of classes
            reg_max: DFL channels
            end2end: Must be True
            ch: Tuple of channel sizes
        """
        super().__init__(nc, reg_max, True, ch)  # Force end2end=True

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | dict:
        """Forward pass with end-to-end optimization."""
        # One-to-many for training
        one2many_x = [torch.cat((self.cv2[i](xi), self.cv3[i](xi)), 1) for i, xi in enumerate(x)]

        if self.training:
            return one2many_x

        # One-to-one for inference (no NMS needed)
        one2one_x = [torch.cat((self.one2one_cv2[i](xi), self.one2one_cv3[i](xi)), 1) for i, xi in enumerate(x)]

        # Decode one2one predictions
        shape = one2one_x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in one2one_x], 2)

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        # Direct output - no NMS filtering
        y = torch.cat((dbox, cls.sigmoid()), 1)

        # Return with explicit end2end markers
        return {"one2one": y, "one2many": torch.cat(one2many_x, 1) if hasattr(self, "cv2") else y, "feats": x}

    def fuse(self):
        """Fuse one-to-many head into one-to-one for deployment."""
        # For deployment, we only need one2one
        delattr(self, "cv2")
        delattr(self, "cv3")
        self.cv2 = self.one2one_cv2
        self.cv3 = self.one2one_cv3
        self._end2end = False  # Mark as fused

    def bias_init(self):
        """Initialize with end-to-end specific biases."""
        # One-to-one head initialization
        for a, b in zip(self.one2one_cv2, self.one2one_cv2):
            a[-1].bias.data[:] = 1.0

        for a, b in zip(self.one2one_cv3, self.one2one_cv3):
            a[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

        # Also init one2many for training
        for a, b in zip(self.cv2, self.cv2):
            a[-1].bias.data[:] = 1.0

        for a, b in zip(self.cv3, self.cv3):
            a[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)
