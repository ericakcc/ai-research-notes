"""Dataset for loading flattened frames from .npz rollout files for VAE training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """Loads all frames from .npz files, flattened across episodes.

    VAE training doesn't need temporal ordering — each frame is independent.
    Images are stored as uint8 and converted to float32 [0, 1] on access.
    """

    def __init__(self, data_dir: Path) -> None:
        self.frames: list[np.ndarray] = []
        npz_files = sorted(data_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        for f in npz_files:
            episode = np.load(f)
            self.frames.append(episode["observations"])  # (T, H, W, C)

        # Concatenate all frames: (total_frames, H, W, C)
        self.frames = np.concatenate(self.frames, axis=0)
        print(f"Loaded {len(self.frames)} frames from {len(npz_files)} episodes")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single frame as (C, H, W) float tensor in [0, 1]."""
        frame = self.frames[idx]  # (H, W, C) uint8
        # HWC -> CHW, uint8 -> float32 [0, 1]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor
