"""YOLOv1 utility functions.

Core utilities for IoU calculation, NMS, and coordinate conversion.
"""

import torch


def iou(
    box_preds: torch.Tensor,
    box_labels: torch.Tensor,
    box_format: str = "midpoint",
) -> torch.Tensor:
    """Calculate Intersection over Union between two sets of boxes.

    Args:
        box_preds: Predicted boxes (..., 4)
        box_labels: Ground truth boxes (..., 4)
        box_format: "midpoint" for (x, y, w, h) or "corners" for (x1, y1, x2, y2)

    Returns:
        IoU values with same batch dimensions
    """
    # Step 1: Convert to corners format (x1, y1, x2, y2)
    if box_format == "midpoint":
        box1_x1 = box_preds[..., 0] - box_preds[..., 2] / 2
        box1_y1 = box_preds[..., 1] - box_preds[..., 3] / 2
        box1_x2 = box_preds[..., 0] + box_preds[..., 2] / 2
        box1_y2 = box_preds[..., 1] + box_preds[..., 3] / 2

        box2_x1 = box_labels[..., 0] - box_labels[..., 2] / 2
        box2_y1 = box_labels[..., 1] - box_labels[..., 3] / 2
        box2_x2 = box_labels[..., 0] + box_labels[..., 2] / 2
        box2_y2 = box_labels[..., 1] + box_labels[..., 3] / 2
    elif box_format == "corners":
        box1_x1 = box_preds[..., 0]
        box1_y1 = box_preds[..., 1]
        box1_x2 = box_preds[..., 2]
        box1_y2 = box_preds[..., 3]

        box2_x1 = box_labels[..., 0]
        box2_y1 = box_labels[..., 1]
        box2_x2 = box_labels[..., 2]
        box2_y2 = box_labels[..., 3]
    else:
        raise ValueError(f"Invalid box format: {box_format}")

    # Step 2: Compute intersection area
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Step 3: Compute union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)
