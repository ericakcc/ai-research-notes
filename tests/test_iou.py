"""Tests for IoU utility function."""

import sys
from pathlib import Path

import torch

# Add the src directory to path (directory names contain hyphens, can't use normal imports)
_src_dir = Path(__file__).resolve().parent.parent / "topics" / "computer-vision" / "experiments" / "01_yolo_from_scratch" / "src"
sys.path.insert(0, str(_src_dir))

from utils import iou


class TestIoU:
    """Test IoU calculation with various scenarios."""

    def test_identical_boxes_midpoint(self) -> None:
        """Identical boxes should have IoU = 1.0."""
        box1 = torch.tensor([0.5, 0.5, 1.0, 1.0])
        box2 = torch.tensor([0.5, 0.5, 1.0, 1.0])
        result = iou(box1, box2, box_format="midpoint")
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_no_overlap_midpoint(self) -> None:
        """Non-overlapping boxes should have IoU = 0.0."""
        box1 = torch.tensor([0.25, 0.25, 0.5, 0.5])
        box2 = torch.tensor([0.75, 0.75, 0.5, 0.5])
        result = iou(box1, box2, box_format="midpoint")
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_partial_overlap_corners(self) -> None:
        """Partially overlapping boxes: intersection=1, union=7, IoU=1/7."""
        box1 = torch.tensor([0.0, 0.0, 2.0, 2.0])
        box2 = torch.tensor([1.0, 1.0, 3.0, 3.0])
        result = iou(box1, box2, box_format="corners")
        assert torch.isclose(result, torch.tensor(1.0 / 7.0), atol=1e-5)

    def test_one_box_inside_another_corners(self) -> None:
        """Small box fully inside large box: intersection=1, union=16."""
        box1 = torch.tensor([0.0, 0.0, 4.0, 4.0])
        box2 = torch.tensor([1.0, 1.0, 2.0, 2.0])
        result = iou(box1, box2, box_format="corners")
        assert torch.isclose(result, torch.tensor(1.0 / 16.0), atol=1e-5)

    def test_batched_boxes(self) -> None:
        """IoU should work on batched inputs."""
        boxes1 = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.25, 0.25, 0.5, 0.5]])
        boxes2 = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.75, 0.75, 0.5, 0.5]])
        result = iou(boxes1, boxes2, box_format="midpoint")
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(result[1], torch.tensor(0.0), atol=1e-5)

    def test_invalid_format_raises(self) -> None:
        """Invalid box_format should raise ValueError."""
        box1 = torch.tensor([0.0, 0.0, 1.0, 1.0])
        box2 = torch.tensor([0.0, 0.0, 1.0, 1.0])
        try:
            iou(box1, box2, box_format="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
