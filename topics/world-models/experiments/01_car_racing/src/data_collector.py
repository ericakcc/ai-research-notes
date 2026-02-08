"""Collect random rollouts from CarRacing for VAE/RNN training."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .config import DataConfig
from .env_wrapper import make_env


def random_action() -> NDArray[np.float32]:
    """Sample a random action for CarRacing.

    Returns:
        action: [steering, gas, brake] with appropriate ranges.
    """
    steering = np.random.uniform(-1.0, 1.0)
    gas = np.random.uniform(0.0, 1.0)
    brake = np.random.uniform(0.0, 1.0)
    return np.array([steering, gas, brake], dtype=np.float32)


def collect_episode(
    env_name: str,
    img_size: tuple[int, int],
    max_steps: int,
    seed: int | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
    """Collect a single episode with random actions.

    Returns:
        observations: (T, H, W, 3) uint8
        actions: (T, 3) float32
        rewards: (T,) float32
    """
    env = make_env(env_name=env_name, img_size=img_size, seed=seed)
    obs, _ = env.reset(seed=seed)

    observations = [obs]
    actions = []
    rewards = []

    for _ in range(max_steps):
        action = random_action()
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    env.close()
    return (
        np.array(observations[:-1], dtype=np.uint8),  # align with actions
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
    )


def collect_data(
    config: DataConfig,
    env_name: str = "CarRacing-v3",
    base_seed: int = 42,
) -> Path:
    """Collect multiple episodes and save as .npz files.

    Args:
        config: Data collection configuration.
        env_name: Gymnasium environment name.
        base_seed: Base random seed for reproducibility.

    Returns:
        Path to the output directory containing .npz files.
    """
    output_dir = config.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(config.num_episodes), desc="Collecting episodes"):
        seed = base_seed + i
        observations, actions, rewards = collect_episode(
            env_name=env_name,
            img_size=config.img_size,
            max_steps=config.max_steps,
            seed=seed,
        )
        np.savez_compressed(
            output_dir / f"episode_{i:04d}.npz",
            observations=observations,
            actions=actions,
            rewards=rewards,
        )

    print(f"Collected {config.num_episodes} episodes to {output_dir}")
    return output_dir
