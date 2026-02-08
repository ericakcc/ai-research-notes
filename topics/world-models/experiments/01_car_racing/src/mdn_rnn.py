"""MDN-RNN: LSTM + Mixture Density Network for next-latent prediction.

Given (z_t, a_t), predicts a mixture of Gaussians over z_{t+1}.
Architecture follows Ha & Schmidhuber (2018).
"""

import torch
import torch.nn as nn
from torch import Tensor


class MDNRNN(nn.Module):
    """LSTM-based MDN-RNN for world model dynamics prediction.

    Input: concatenated (z_t, a_t) at each timestep.
    Output: mixture parameters (pi, mu, sigma) for z_{t+1}.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_gaussians: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # MDN output heads: for each latent dim, predict num_gaussians * 3 params
        # (pi, mu, sigma) for each Gaussian component
        n_out = num_gaussians * latent_dim  # per-param count
        self.fc_pi = nn.Linear(hidden_dim, n_out)  # mixture weights
        self.fc_mu = nn.Linear(hidden_dim, n_out)  # means
        self.fc_sigma = nn.Linear(hidden_dim, n_out)  # std devs

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through LSTM + MDN heads.

        Args:
            z: (B, T, latent_dim) encoded observations.
            action: (B, T, action_dim) actions taken.
            hidden: Optional initial LSTM hidden state.

        Returns:
            pi: (B, T, num_gaussians, latent_dim) mixture log-weights.
            mu: (B, T, num_gaussians, latent_dim) mixture means.
            sigma: (B, T, num_gaussians, latent_dim) mixture std devs.
            hidden: Updated LSTM hidden state.
        """
        B, T, _ = z.shape
        x = torch.cat([z, action], dim=-1)  # (B, T, latent_dim + action_dim)
        lstm_out, hidden = self.lstm(x, hidden)  # (B, T, hidden_dim)

        # Reshape MDN outputs: (B, T, num_gaussians, latent_dim)
        pi = self.fc_pi(lstm_out).view(B, T, self.num_gaussians, self.latent_dim)
        mu = self.fc_mu(lstm_out).view(B, T, self.num_gaussians, self.latent_dim)
        sigma = self.fc_sigma(lstm_out).view(B, T, self.num_gaussians, self.latent_dim)

        # Log-softmax for numerical stability on mixture weights
        pi = torch.log_softmax(pi, dim=2)
        # Ensure positive sigma with lower bound
        sigma = torch.exp(sigma).clamp(min=1e-6)

        return pi, mu, sigma, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Initialize LSTM hidden state with zeros."""
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )


def mdn_loss(pi: Tensor, mu: Tensor, sigma: Tensor, target: Tensor) -> Tensor:
    """Compute MDN negative log-likelihood loss.

    TODO(human): Implement the MDN loss function.

    Given mixture parameters and target z_{t+1}, compute the negative
    log-likelihood of the target under the predicted mixture of Gaussians.

    Args:
        pi: (B, T, num_gaussians, latent_dim) log mixture weights (already log-softmax).
        mu: (B, T, num_gaussians, latent_dim) Gaussian means.
        sigma: (B, T, num_gaussians, latent_dim) Gaussian std devs (positive).
        target: (B, T, latent_dim) actual next-step z values.

    Returns:
        Scalar loss (mean negative log-likelihood over batch and time).

    Hints:
        1. Expand target to match mixture dim: target.unsqueeze(2)
        2. Compute log probability of each Gaussian component (log of normal PDF)
        3. Add log mixture weights (pi) to each component's log probability
        4. Use torch.logsumexp across the mixture dimension to combine
        5. Negate and average for the final loss
    """
    raise NotImplementedError("Implement the MDN negative log-likelihood loss")
