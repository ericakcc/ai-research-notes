"""YOLOv1 model architecture.

Paper: You Only Look Once: Unified, Real-Time Object Detection (2016)
Architecture: 24 conv layers + 2 FC layers, inspired by GoogLeNet
"""

import torch
import torch.nn as nn


# Architecture config: list of layer specifications
# Each tuple: (kernel_size, num_filters, stride, padding)
# "M" = MaxPool2d(2, 2)
#
# TODO(human): Complete the architecture config based on the paper
# The network should:
#   - Start from 448x448x3 input
#   - End at 7x7 spatial resolution before FC layers
#   - Use 1x1 convs to reduce channels, 3x3 convs to extract features
#
# Hint: The paper describes the architecture in Figure 3 and Section 2.1
# Key pattern: 1x1 reduction → 3x3 conv, repeated multiple times
ARCHITECTURE_CONFIG: list[tuple[int, int, int, int] | str] = [
    # Block 1: 448 → 224 → 112
    (7, 64, 2, 3),    # 7×7 conv, 64 filters, stride 2 → 224×224
    "M",              # MaxPool → 112×112

    # Block 2: 112 → 56
    (3, 192, 1, 1),   # 3×3 conv, 192 filters → 112×112
    "M",              # MaxPool → 56×56

    # Block 3: 56 → 28 [1×1×128, 3×3×256, 1×1×256, 3×3×512]
    (1, 128, 1, 0),   # 1×1 conv, 128 filters
    (3, 256, 1, 1),   # 3×3 conv, 256 filters
    (1, 256, 1, 0),   # 1×1 conv, 256 filters
    (3, 512, 1, 1),   # 3×3 conv, 512 filters
    "M",              # MaxPool → 28×28

    # Block 4: 28 → 14 [1×1×256, 3×3×512 ×4, 1×1×512, 3×3×1024]
    (1, 256, 1, 0),   # 1×1 conv, 256 filters
    (3, 512, 1, 1),   # 3×3 conv, 512 filters ─┐
    (1, 256, 1, 0),   # 1×1 conv, 256 filters  │
    (3, 512, 1, 1),   # 3×3 conv, 512 filters  │ ×4 (1×1 + 3×3 pairs)
    (1, 256, 1, 0),   # 1×1 conv, 256 filters  │
    (3, 512, 1, 1),   # 3×3 conv, 512 filters  │
    (1, 256, 1, 0),   # 1×1 conv, 256 filters  │
    (3, 512, 1, 1),   # 3×3 conv, 512 filters ─┘
    (1, 512, 1, 0),   # 1×1 conv, 512 filters
    (3, 1024, 1, 1),  # 3×3 conv, 1024 filters
    "M",              # MaxPool → 14×14

    # Block 5: 14 → 7 [1×1×512, 3×3×1024 ×2, 3×3×1024, 3×3×1024-s2]
    (1, 512, 1, 0),   # 1×1 conv, 512 filters
    (3, 1024, 1, 1),  # 3×3 conv, 1024 filters ─┐ ×2
    (1, 512, 1, 0),   # 1×1 conv, 512 filters   │
    (3, 1024, 1, 1),  # 3×3 conv, 1024 filters ─┘
    (3, 1024, 1, 1),  # 3×3 conv, 1024 filters
    (3, 1024, 2, 1),  # 3×3 conv, 1024 filters, stride 2 → 7×7
]


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    """YOLOv1 object detection network."""

    def __init__(self, in_channels: int = 3, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20) -> None:
        """Initialize YOLOv1.

        Args:
            in_channels: Number of input channels (3 for RGB)
            split_size: Grid size S (7 for YOLOv1)
            num_boxes: Number of bounding boxes per cell B (2 for YOLOv1)
            num_classes: Number of object classes C (20 for VOC)
        """
        super().__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.backbone = self._build_backbone(in_channels)
        self.fc = self._build_fc(split_size, num_boxes, num_classes)

    def _build_backbone(self, in_channels: int) -> nn.Sequential:
        """Build convolutional backbone from config."""
        layers: list[nn.Module] = []
        channels = in_channels

        for spec in ARCHITECTURE_CONFIG:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                kernel_size, out_channels, stride, padding = spec
                layers.append(ConvBlock(channels, out_channels, kernel_size, stride, padding))
                channels = out_channels

        return nn.Sequential(*layers)

    def _build_fc(self, split_size: int, num_boxes: int, num_classes: int) -> nn.Sequential:
        """Build fully connected head."""
        # Output: S x S x (B * 5 + C) = 7 x 7 x 30 = 1470
        output_size = split_size * split_size * (num_boxes * 5 + num_classes)

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (batch, 3, 448, 448)

        Returns:
            Predictions (batch, S, S, B*5+C) = (batch, 7, 7, 30)
        """
        x = self.backbone(x)
        x = self.fc(x)
        # Reshape to (batch, S, S, B*5+C)
        return x.view(-1, self.split_size, self.split_size, self.num_boxes * 5 + self.num_classes)
