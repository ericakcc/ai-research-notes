"""Sequence dataset for MDN-RNN training from pre-encoded latent vectors."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    """Loads encoded z sequences and actions for MDN-RNN training.

    Each item is a fixed-length subsequence (z_seq, action_seq, target_seq)
    where target_seq is z shifted by one timestep.
    """

    def __init__(self, data_dir: Path, sequence_length: int = 64) -> None:
        self.sequence_length = sequence_length
        self.segments: list[tuple[np.ndarray, np.ndarray]] = []

        npz_files = sorted(data_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        for f in npz_files:
            data = np.load(f)
            z_seq = data["z"]  # (T, latent_dim)
            actions = data["actions"]  # (T, action_dim)

            # Cut into non-overlapping segments of sequence_length + 1
            # (+1 because we need next-step target)
            total_len = len(z_seq)
            needed = sequence_length + 1
            for start in range(0, total_len - needed + 1, sequence_length):
                z_segment = z_seq[start : start + needed]
                a_segment = actions[start : start + needed - 1]
                self.segments.append((z_segment, a_segment))

        print(f"Created {len(self.segments)} sequences from {len(npz_files)} episodes")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_input, actions, z_target) tensors.

        z_input: (seq_len, latent_dim) — z_0 to z_{T-1}
        actions: (seq_len, action_dim) — a_0 to a_{T-1}
        z_target: (seq_len, latent_dim) — z_1 to z_T
        """
        z_seg, a_seg = self.segments[idx]
        z_input = torch.from_numpy(z_seg[:-1]).float()
        z_target = torch.from_numpy(z_seg[1:]).float()
        actions = torch.from_numpy(a_seg).float()
        return z_input, actions, z_target
