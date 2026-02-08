"""Tests for MDN-RNN: shape verification, numerical stability, mixture weights."""

import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
        / "topics/world-models/experiments/01_car_racing"
    ),
)

from src.mdn_rnn import MDNRNN, mdn_loss


class TestMDNRNN:
    def test_output_shapes(self) -> None:
        model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)
        z = torch.randn(4, 10, 32)  # (B, T, latent_dim)
        a = torch.randn(4, 10, 3)  # (B, T, action_dim)
        pi, mu, sigma, hidden = model(z, a)
        assert pi.shape == (4, 10, 5, 32)
        assert mu.shape == (4, 10, 5, 32)
        assert sigma.shape == (4, 10, 5, 32)
        assert hidden[0].shape == (1, 4, 256)
        assert hidden[1].shape == (1, 4, 256)

    def test_pi_log_softmax(self) -> None:
        """Mixture weights (in log space) should sum to 1 after exp."""
        model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)
        z = torch.randn(2, 5, 32)
        a = torch.randn(2, 5, 3)
        pi, _, _, _ = model(z, a)
        # exp(log_softmax) should sum to ~1 along mixture dim
        pi_sum = pi.exp().sum(dim=2)
        assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5)

    def test_sigma_positive(self) -> None:
        """Standard deviations must be strictly positive."""
        model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)
        z = torch.randn(2, 5, 32)
        a = torch.randn(2, 5, 3)
        _, _, sigma, _ = model(z, a)
        assert (sigma > 0).all()

    def test_hidden_state_persistence(self) -> None:
        """Hidden state from one call should be usable in the next."""
        model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)
        z1 = torch.randn(2, 5, 32)
        a1 = torch.randn(2, 5, 3)
        _, _, _, hidden = model(z1, a1)

        z2 = torch.randn(2, 3, 32)
        a2 = torch.randn(2, 3, 3)
        pi, mu, sigma, _ = model(z2, a2, hidden)
        assert pi.shape == (2, 3, 5, 32)


class TestMDNLoss:
    def test_loss_positive(self) -> None:
        """Loss should be a positive scalar."""
        model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)
        z = torch.randn(4, 10, 32)
        a = torch.randn(4, 10, 3)
        target = torch.randn(4, 10, 32)
        pi, mu, sigma, _ = model(z, a)
        loss = mdn_loss(pi, mu, sigma, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_loss_decreases_for_matching_target(self) -> None:
        """Loss for targets near the predicted mean should be lower."""
        model = MDNRNN(latent_dim=4, action_dim=3, hidden_dim=32, num_gaussians=2)
        z = torch.randn(2, 5, 4)
        a = torch.randn(2, 5, 3)
        pi, mu, sigma, _ = model(z, a)

        # Target far from means
        far_target = torch.randn(2, 5, 4) * 10
        loss_far = mdn_loss(pi, mu, sigma, far_target)

        # Target close to first mixture component mean
        close_target = mu[:, :, 0, :].detach() + torch.randn(2, 5, 4) * 0.01
        loss_close = mdn_loss(pi, mu, sigma, close_target)

        assert loss_close.item() < loss_far.item()

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the MDN loss."""
        model = MDNRNN(latent_dim=4, action_dim=3, hidden_dim=32, num_gaussians=2)
        z = torch.randn(2, 5, 4)
        a = torch.randn(2, 5, 3)
        target = torch.randn(2, 5, 4)
        pi, mu, sigma, _ = model(z, a)
        loss = mdn_loss(pi, mu, sigma, target)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
