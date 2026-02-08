"""Encode raw rollout observations into latent vectors using trained VAE."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.vae import ConvVAE


def encode_rollouts(
    vae_checkpoint: Path,
    input_dir: Path,
    output_dir: Path,
    config: Config,
) -> None:
    """Encode all rollout .npz files using the trained VAE.

    For each episode, produces a new .npz with:
        z: (T, latent_dim) encoded latent vectors
        actions: (T, action_dim) original actions
        rewards: (T,) original rewards
    """
    device = config.device
    model = ConvVAE(
        img_channels=config.vae.img_channels, latent_dim=config.vae.latent_dim
    )
    model.load_state_dict(
        torch.load(vae_checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(input_dir.glob("*.npz"))

    for f in tqdm(npz_files, desc="Encoding episodes"):
        episode = np.load(f)
        obs = episode["observations"]  # (T, H, W, C) uint8
        actions = episode["actions"]
        rewards = episode["rewards"]

        # Convert to tensor: (T, C, H, W) float [0, 1]
        obs_tensor = (
            torch.from_numpy(obs).permute(0, 3, 1, 2).float().div(255.0).to(device)
        )

        # Encode in batches to avoid OOM
        z_list = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(obs_tensor), batch_size):
                batch = obs_tensor[i : i + batch_size]
                z = model.encode(batch)
                z_list.append(z.cpu().numpy())

        z_all = np.concatenate(z_list, axis=0)

        np.savez_compressed(
            output_dir / f.name,
            z=z_all,
            actions=actions,
            rewards=rewards,
        )

    print(f"Encoded {len(npz_files)} episodes to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode rollouts with trained VAE")
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="checkpoints/vae/vae_final.pt",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/rollouts",
        help="Input rollout directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/encoded",
        help="Output encoded directory",
    )
    args = parser.parse_args()

    config = Config()
    encode_rollouts(
        vae_checkpoint=Path(args.vae_checkpoint),
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        config=config,
    )


if __name__ == "__main__":
    main()
