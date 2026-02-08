"""Linear Controller: maps [z, h] -> action using CMA-ES optimized weights.

The controller is intentionally simple — a single linear layer — so that
CMA-ES can optimize its ~867 parameters efficiently.
"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor


class Controller(nn.Module):
    """Linear controller: action = activation(W @ [z, h] + b).

    Args:
        latent_dim: Dimension of VAE latent vector z.
        hidden_dim: Dimension of RNN hidden state h.
        action_dim: Dimension of action space.
    """

    def __init__(
        self, latent_dim: int = 32, hidden_dim: int = 256, action_dim: int = 3
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """Compute action from latent state and RNN hidden state.

        TODO(human): Implement the action activation mapping.

        Given the raw linear output, apply appropriate activation functions
        to map to CarRacing's action space:
          - steering: [-1, 1]
          - gas: [0, 1]
          - brake: [0, 1]

        Args:
            z: (B, latent_dim) VAE latent vector.
            h: (B, hidden_dim) RNN hidden state.

        Returns:
            action: (B, 3) with [steering, gas, brake] in correct ranges.

        Hints:
            1. Concatenate z and h, pass through self.fc to get raw output (B, 3)
            2. Apply tanh to steering (index 0) for [-1, 1]
            3. Apply sigmoid to gas and brake (indices 1, 2) for [0, 1]
            4. Combine into a single (B, 3) tensor
        """
        raise NotImplementedError("Implement the action activation mapping")

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_params(self) -> NDArray[np.float64]:
        """Flatten all parameters into a 1D numpy array for CMA-ES."""
        return np.concatenate(
            [p.data.cpu().numpy().flatten() for p in self.parameters()]
        )

    def set_params(self, params: NDArray[np.float64]) -> None:
        """Load flattened parameter array back into the model."""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = torch.from_numpy(
                params[idx : idx + numel].reshape(p.shape)
            ).float()
            idx += numel
