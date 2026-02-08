"""Centralized hyperparameter configuration for the World Models experiment."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data collection settings."""

    num_episodes: int = 200  # 論文 10,000; 本機小規模先用 200
    max_steps: int = 1000
    img_size: tuple[int, int] = (64, 64)
    data_dir: Path = field(default_factory=lambda: Path("datasets/rollouts"))


@dataclass
class VAEConfig:
    """VAE architecture and training settings."""

    latent_dim: int = 32
    img_channels: int = 3
    img_size: int = 64
    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 10  # 論文更多; 本機先跑通
    kl_weight: float = 1.0
    # Checkpoints
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/vae"))


@dataclass
class RNNConfig:
    """MDN-RNN architecture and training settings."""

    latent_dim: int = 32
    action_dim: int = 3
    hidden_dim: int = 256
    num_gaussians: int = 5
    sequence_length: int = 64
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    grad_clip: float = 1.0
    # Checkpoints
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/rnn"))


@dataclass
class ControllerConfig:
    """Controller and CMA-ES settings."""

    latent_dim: int = 32
    hidden_dim: int = 256  # RNN hidden size
    action_dim: int = 3
    # CMA-ES
    population_size: int = 64
    num_rollouts: int = 16  # rollouts per candidate
    max_generations: int = 50
    sigma_init: float = 0.1
    # Checkpoints
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/controller"))


@dataclass
class Config:
    """Top-level configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    env_name: str = "CarRacing-v3"
    device: str = "cpu"  # "cpu", "mps", or "cuda"
    seed: int = 42
