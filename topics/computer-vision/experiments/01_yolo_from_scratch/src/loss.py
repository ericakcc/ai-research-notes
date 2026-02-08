"""YOLOv1 loss function.

The loss consists of 5 parts:
1. Box center (x, y) loss - only responsible predictor
2. Box size (√w, √h) loss - only responsible predictor
3. Confidence loss (object exists) - only responsible predictor
4. Confidence loss (no object) - all non-responsible predictors
5. Class probability loss - cell level (not predictor level)
"""

import torch
import torch.nn as nn

from utils import iou


class YOLOv1Loss(nn.Module):
    """YOLOv1 multi-part loss function."""

    def __init__(
        self,
        split_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        """Initialize YOLOv1 loss.

        Args:
            split_size: Grid size S (7)
            num_boxes: Boxes per cell B (2)
            num_classes: Number of classes C (20)
            lambda_coord: Weight for coordinate loss (5.0)
            lambda_noobj: Weight for no-object confidence loss (0.5)
        """
        super().__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate YOLOv1 loss.

        Args:
            predictions: Model output (batch, S, S, B*5 + C) = (batch, 7, 7, 30)
            target: Ground truth (batch, S, S, 5 + C) = (batch, 7, 7, 25)
                    Format per cell: [x, y, w, h, confidence, class_0, ..., class_19]
                    Note: target only has 1 box per cell (the GT box)

        Returns:
            Total loss (scalar)
        """
        # === Step 1: Extract predictions ===
        # predictions: [box1: x,y,w,h,conf (5)] + [box2: x,y,w,h,conf (5)] + [classes (20)]
        box1_pred = predictions[..., 0:4]
        box2_pred = predictions[..., 5:9]
        conf1_pred = predictions[..., 4:5]
        conf2_pred = predictions[..., 9:10]
        classes_pred = predictions[..., 10:]

        # === Step 2: Extract target ===
        box_target = target[..., 0:4]
        exists_obj = target[..., 4:5]  # 𝟙_i^obj: 1 if object, 0 if not
        class_target = target[..., 5:]

        # === Step 3: Determine responsible predictor ===
        # Calculate IoU for each predicted box with target box
        iou1 = iou(box1_pred, box_target)
        iou2 = iou(box2_pred, box_target)

        # Responsible = higher IoU with GT
        responsible_box1 = (iou1 > iou2).unsqueeze(-1)  # (batch, 7, 7, 1)
        responsible_box2 = ~responsible_box1

        # Select responsible predictor's outputs
        box_pred = responsible_box1 * box1_pred + responsible_box2 * box2_pred
        conf_pred = responsible_box1 * conf1_pred + responsible_box2 * conf2_pred

        # === Step 4: Calculate 5 loss components ===

        # Loss 1: xy_loss - center coordinate loss (responsible + has object)
        xy_loss = (exists_obj * (box_pred[..., 0:2] - box_target[..., 0:2]) ** 2).sum()

        # Loss 2: wh_loss - size loss with sqrt (responsible + has object)
        # Use abs + epsilon to handle potential negative predictions
        pred_wh = torch.sign(box_pred[..., 2:4]) * torch.sqrt(
            torch.abs(box_pred[..., 2:4]) + 1e-6
        )
        target_wh = torch.sqrt(box_target[..., 2:4] + 1e-6)
        wh_loss = (exists_obj * (pred_wh - target_wh) ** 2).sum()

        # Loss 3: obj_loss - confidence loss when object exists (responsible)
        obj_loss = (exists_obj * (conf_pred - exists_obj) ** 2).sum()

        # Loss 4: noobj_loss - confidence loss when no object (both predictors)
        no_obj = 1 - exists_obj
        noobj_loss = (
            no_obj * (conf1_pred ** 2) + no_obj * (conf2_pred ** 2)
        ).sum()

        # Loss 5: class_loss - classification loss (cell level, has object)
        class_loss = (exists_obj * (classes_pred - class_target) ** 2).sum()

        # === Step 5: Weighted sum ===
        total_loss = (
            self.lambda_coord * xy_loss
            + self.lambda_coord * wh_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )

        return total_loss
