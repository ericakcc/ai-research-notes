"""Tests for VAE model: shape verification, loss correctness, gradient flow."""

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

from src.train_vae import vae_loss
from src.vae import ConvVAE, Decoder, Encoder


class TestEncoder:
    def test_output_shape(self) -> None:
        encoder = Encoder(img_channels=3, latent_dim=32)
        x = torch.randn(4, 3, 64, 64)
        mu, log_var = encoder(x)
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)

    def test_different_latent_dim(self) -> None:
        encoder = Encoder(img_channels=3, latent_dim=64)
        x = torch.randn(2, 3, 64, 64)
        mu, log_var = encoder(x)
        assert mu.shape == (2, 64)


class TestDecoder:
    def test_output_shape(self) -> None:
        decoder = Decoder(img_channels=3, latent_dim=32)
        z = torch.randn(4, 32)
        x_recon = decoder(z)
        assert x_recon.shape == (4, 3, 64, 64)

    def test_output_range(self) -> None:
        decoder = Decoder(img_channels=3, latent_dim=32)
        z = torch.randn(4, 32)
        x_recon = decoder(z)
        assert x_recon.min() >= 0.0
        assert x_recon.max() <= 1.0


class TestConvVAE:
    def test_forward_shapes(self) -> None:
        model = ConvVAE(img_channels=3, latent_dim=32)
        x = torch.randn(4, 3, 64, 64)
        x_recon, z, mu, log_var = model(x)
        assert x_recon.shape == (4, 3, 64, 64)
        assert z.shape == (4, 32)
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)

    def test_encode_returns_mean(self) -> None:
        model = ConvVAE(img_channels=3, latent_dim=32)
        x = torch.randn(2, 3, 64, 64)
        z = model.encode(x)
        assert z.shape == (2, 32)

    def test_gradient_flow(self) -> None:
        """Verify gradients flow from loss back to encoder parameters."""
        model = ConvVAE(img_channels=3, latent_dim=32)
        x = torch.randn(2, 3, 64, 64)
        x_recon, _z, mu, log_var = model(x)
        loss, _, _ = vae_loss(x_recon, x, mu, log_var)
        loss.backward()

        # Check that encoder conv layers have gradients
        for param in model.encoder.convs.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


class TestVAELoss:
    def test_loss_positive(self) -> None:
        x = torch.rand(4, 3, 64, 64)
        x_recon = torch.rand(4, 3, 64, 64)
        mu = torch.randn(4, 32)
        log_var = torch.randn(4, 32)
        total, recon, kl = vae_loss(x_recon, x, mu, log_var)
        assert total.item() > 0
        assert recon.item() > 0

    def test_kl_zero_for_standard_normal(self) -> None:
        """KL should be ~0 when mu=0, log_var=0 (i.e., q = N(0,1))."""
        x = torch.rand(4, 3, 64, 64)
        mu = torch.zeros(4, 32)
        log_var = torch.zeros(4, 32)
        _, _, kl = vae_loss(x, x, mu, log_var)
        assert abs(kl.item()) < 1e-5
