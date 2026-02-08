"""VAE training loop with reconstruction + KL divergence loss."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import VAEConfig
from .vae import ConvVAE
from .vae_dataset import VAEDataset


def vae_loss(
    x_recon: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, kl_weight: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute VAE loss = reconstruction + KL divergence.

    Args:
        x_recon: Reconstructed images (B, C, H, W).
        x: Original images (B, C, H, W).
        mu: Encoder means (B, latent_dim).
        log_var: Encoder log variances (B, latent_dim).
        kl_weight: Weight for KL term.

    Returns:
        total_loss, recon_loss, kl_loss (all scalar tensors).
    """
    # MSE reconstruction loss (per pixel, averaged over batch)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def train_vae(config: VAEConfig, data_dir: Path, device: str = "cpu") -> ConvVAE:
    """Train the VAE model.

    Args:
        config: VAE configuration.
        data_dir: Path to directory containing .npz rollout files.
        device: Device to train on.

    Returns:
        Trained ConvVAE model.
    """
    dataset = VAEDataset(data_dir)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    model = ConvVAE(img_channels=config.img_channels, latent_dim=config.latent_dim).to(
        device
    )
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.num_epochs):
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            x_recon, _z, mu, log_var = model(batch)
            loss, recon, kl = vae_loss(x_recon, batch, mu, log_var, config.kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            recon_loss_sum += recon.item()
            kl_loss_sum += kl.item()
            num_batches += 1
            pbar.set_postfix(
                loss=f"{total_loss_sum / num_batches:.4f}",
                recon=f"{recon_loss_sum / num_batches:.4f}",
                kl=f"{kl_loss_sum / num_batches:.4f}",
            )

        avg_loss = total_loss_sum / num_batches
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    # Save checkpoint
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / "vae_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved VAE checkpoint to {checkpoint_path}")

    return model
