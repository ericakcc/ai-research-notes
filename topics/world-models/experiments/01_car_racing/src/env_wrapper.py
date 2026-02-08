"""Gymnasium environment wrapper with frame resizing for CarRacing."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class CarRacingWrapper(gym.Wrapper):
    """Wraps CarRacing to resize frames to target size (default 64x64).

    Also handles the initial zoom-in phase by repeating a no-op action.
    """

    def __init__(self, env: gym.Env, img_size: tuple[int, int] = (64, 64)) -> None:
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_size[0], img_size[1], 3),
            dtype=np.uint8,
        )

    def _resize(self, obs: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Resize observation using simple area interpolation."""
        from PIL import Image

        img = Image.fromarray(obs)
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Skip initial zoom-in frames (roughly 50 no-op steps)
        no_op = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            obs, _, terminated, truncated, info = self.env.step(no_op)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
        return self._resize(obs), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._resize(obs), float(reward), terminated, truncated, info


def make_env(
    env_name: str = "CarRacing-v3",
    img_size: tuple[int, int] = (64, 64),
    seed: int | None = None,
) -> CarRacingWrapper:
    """Create a wrapped CarRacing environment."""
    env = gym.make(env_name, render_mode=None)
    wrapped = CarRacingWrapper(env, img_size=img_size)
    return wrapped
