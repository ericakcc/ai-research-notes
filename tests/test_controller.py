"""Tests for Controller: parameter count, roundtrip, action range."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
        / "topics/world-models/experiments/01_car_racing"
    ),
)

from src.controller import Controller


class TestController:
    def test_num_params(self) -> None:
        """Controller should have (32+256)*3 + 3 = 867 parameters."""
        ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
        assert ctrl.num_params == (32 + 256) * 3 + 3  # weight + bias

    def test_action_shape(self) -> None:
        ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
        z = torch.randn(4, 32)
        h = torch.randn(4, 256)
        action = ctrl(z, h)
        assert action.shape == (4, 3)

    def test_steering_range(self) -> None:
        """Steering (index 0) should be in [-1, 1]."""
        ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
        z = torch.randn(100, 32) * 5  # large inputs to stress test
        h = torch.randn(100, 256) * 5
        action = ctrl(z, h)
        assert action[:, 0].min() >= -1.0
        assert action[:, 0].max() <= 1.0

    def test_gas_brake_range(self) -> None:
        """Gas (index 1) and brake (index 2) should be in [0, 1]."""
        ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
        z = torch.randn(100, 32) * 5
        h = torch.randn(100, 256) * 5
        action = ctrl(z, h)
        assert action[:, 1].min() >= 0.0
        assert action[:, 1].max() <= 1.0
        assert action[:, 2].min() >= 0.0
        assert action[:, 2].max() <= 1.0

    def test_params_roundtrip(self) -> None:
        """get_params -> set_params should preserve parameters exactly."""
        ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)

        # Modify params
        new_params = np.random.randn(ctrl.num_params)
        ctrl.set_params(new_params)
        retrieved = ctrl.get_params()

        np.testing.assert_array_almost_equal(new_params, retrieved)

    def test_set_params_changes_output(self) -> None:
        """Different parameters should produce different actions."""
        ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
        z = torch.randn(1, 32)
        h = torch.randn(1, 256)

        action1 = ctrl(z, h).detach()
        ctrl.set_params(np.random.randn(ctrl.num_params) * 2)
        action2 = ctrl(z, h).detach()

        assert not torch.allclose(action1, action2)
