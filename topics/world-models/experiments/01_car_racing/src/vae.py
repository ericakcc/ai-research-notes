"""Convolutional VAE for compressing 64x64x3 frames to 32-dim latent vectors.

Architecture follows Ha & Schmidhuber (2018):
  Encoder: 4 Conv layers (stride=2) -> flatten -> FC -> (mu, log_var)
  Decoder: FC -> unflatten -> 4 ConvTranspose layers (stride=2)
"""

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """4-layer CNN encoder: 64x64x3 -> latent_dim."""

    def __init__(self, img_channels: int = 3, latent_dim: int = 32) -> None:
        super().__init__()
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.convs = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        # 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input image to (mu, log_var).

        Args:
            x: (B, C, H, W) float tensor in [0, 1].

        Returns:
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
        """
        h = self.convs(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """4-layer transposed CNN decoder: latent_dim -> 64x64x3."""

    def __init__(self, img_channels: int = 3, latent_dim: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vector to reconstructed image.

        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, C, H, W) in [0, 1]
        """
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconvs(h)


class ConvVAE(nn.Module):
    """Convolutional VAE combining Encoder and Decoder."""

    def __init__(self, img_channels: int = 3, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z from q(z|x) using the reparameterization trick.

        Args:
            mu: (B, latent_dim) — mean of the approximate posterior.
            log_var: (B, latent_dim) — log variance of the approximate posterior.

        Returns:
            z: (B, latent_dim) — sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass: encode -> reparameterize -> decode.

        Args:
            x: (B, C, H, W) input images in [0, 1].

        Returns:
            x_recon: (B, C, H, W) reconstructed images.
            z: (B, latent_dim) sampled latent vectors.
            mu: (B, latent_dim) encoder means.
            log_var: (B, latent_dim) encoder log variances.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, z, mu, log_var

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent vector (using mean, no sampling)."""
        mu, _ = self.encoder(x)
        return mu
