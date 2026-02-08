"""Full VMC pipeline rollout: VAE encodes, RNN tracks state, Controller acts."""

import torch
import torch.nn as nn

from .env_wrapper import make_env


@torch.no_grad()
def rollout_episode(
    vae: nn.Module,
    rnn: nn.Module,
    controller: nn.Module,
    env_name: str = "CarRacing-v3",
    img_size: tuple[int, int] = (64, 64),
    max_steps: int = 1000,
    device: str = "cpu",
    seed: int | None = None,
) -> float:
    """Run a single episode with the full VMC pipeline.

    Flow at each step:
        1. VAE encodes observation -> z
        2. Controller maps [z, h] -> action
        3. Environment steps with action
        4. RNN updates hidden state with (z, action)

    Args:
        vae: Trained VAE model.
        rnn: Trained MDN-RNN model.
        controller: Trained Controller.
        env_name: Gymnasium environment name.
        img_size: Observation resize target.
        max_steps: Maximum steps per episode.
        device: Compute device.
        seed: Random seed for environment.

    Returns:
        Total episode reward.
    """
    env = make_env(env_name=env_name, img_size=img_size, seed=seed)
    obs, _ = env.reset(seed=seed)

    # Initialize RNN hidden state
    hidden = rnn.init_hidden(batch_size=1, device=torch.device(device))

    total_reward = 0.0

    for _ in range(max_steps):
        # 1. Encode observation
        obs_tensor = (
            torch.from_numpy(obs)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )  # (1, C, H, W)
        z = vae.encode(obs_tensor)  # (1, latent_dim)

        # 2. Controller: [z, h] -> action
        h = hidden[0].squeeze(0)  # (1, hidden_dim)
        action = controller(z, h)  # (1, action_dim)
        action_np = action.squeeze(0).cpu().numpy()

        # 3. Step environment
        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += reward

        if terminated or truncated:
            break

        # 4. Update RNN hidden state
        action_tensor = action.unsqueeze(1).to(device)  # (1, 1, action_dim)
        z_input = z.unsqueeze(1)  # (1, 1, latent_dim)
        _, _, _, hidden = rnn(z_input, action_tensor, hidden)

    env.close()
    return total_reward
