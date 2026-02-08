"""MDN-RNN training loop."""

from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RNNConfig
from .mdn_rnn import MDNRNN, mdn_loss
from .rnn_dataset import RNNDataset


def train_rnn(config: RNNConfig, data_dir: Path, device: str = "cpu") -> MDNRNN:
    """Train the MDN-RNN model.

    Args:
        config: RNN configuration.
        data_dir: Path to directory containing encoded .npz files (z, actions).
        device: Device to train on.

    Returns:
        Trained MDNRNN model.
    """
    dataset = RNNDataset(data_dir, sequence_length=config.sequence_length)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    model = MDNRNN(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_gaussians=config.num_gaussians,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.num_epochs):
        total_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for z_input, actions, z_target in pbar:
            z_input = z_input.to(device)
            actions = actions.to(device)
            z_target = z_target.to(device)

            pi, mu, sigma, _ = model(z_input, actions)
            loss = mdn_loss(pi, mu, sigma, z_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss_sum += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss_sum / num_batches:.4f}")

        avg_loss = total_loss_sum / num_batches
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    # Save checkpoint
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / "rnn_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved RNN checkpoint to {checkpoint_path}")

    return model
