"""CMA-ES training loop for the linear Controller.

Each candidate's fitness is evaluated by running rollouts in the real
environment (or in the MDN-RNN 'dream' environment).
"""

import cma
import numpy as np
import torch
from numpy.typing import NDArray

from .config import Config, ControllerConfig
from .controller import Controller
from .rollout import rollout_episode


def evaluate_candidate(
    params: NDArray[np.float64],
    config: Config,
    vae: torch.nn.Module,
    rnn: torch.nn.Module,
    num_rollouts: int = 16,
) -> float:
    """Evaluate a controller candidate by averaging rewards over rollouts.

    Args:
        params: Flattened controller parameters.
        config: Full experiment configuration.
        vae: Trained VAE model.
        rnn: Trained MDN-RNN model.
        num_rollouts: Number of rollouts to average.

    Returns:
        Negative mean reward (CMA-ES minimizes).
    """
    controller = Controller(
        latent_dim=config.controller.latent_dim,
        hidden_dim=config.controller.hidden_dim,
        action_dim=config.controller.action_dim,
    )
    controller.set_params(params)
    controller.eval()

    rewards = []
    for i in range(num_rollouts):
        total_reward = rollout_episode(
            vae=vae,
            rnn=rnn,
            controller=controller,
            env_name=config.env_name,
            img_size=config.data.img_size,
            device=config.device,
            seed=i,
        )
        rewards.append(total_reward)

    return -float(np.mean(rewards))  # negate for CMA-ES minimization


def train_controller(
    controller_config: ControllerConfig,
    config: Config,
    vae: torch.nn.Module,
    rnn: torch.nn.Module,
) -> Controller:
    """Train controller using CMA-ES.

    Args:
        controller_config: Controller-specific configuration.
        config: Full experiment configuration.
        vae: Trained VAE model (eval mode).
        rnn: Trained MDN-RNN model (eval mode).

    Returns:
        Best Controller found by CMA-ES.
    """
    controller = Controller(
        latent_dim=controller_config.latent_dim,
        hidden_dim=controller_config.hidden_dim,
        action_dim=controller_config.action_dim,
    )
    initial_params = controller.get_params()
    print(f"Controller parameters: {controller.num_params}")

    es = cma.CMAEvolutionStrategy(
        initial_params,
        controller_config.sigma_init,
        {"popsize": controller_config.population_size, "seed": config.seed},
    )

    best_reward = -float("inf")
    best_params = initial_params.copy()

    for gen in range(controller_config.max_generations):
        candidates = es.ask()
        fitnesses = []

        for candidate in candidates:
            fitness = evaluate_candidate(
                candidate, config, vae, rnn, controller_config.num_rollouts
            )
            fitnesses.append(fitness)

        es.tell(candidates, fitnesses)
        es.disp()

        # Track best (remember fitness is negated reward)
        gen_best = -min(fitnesses)
        gen_mean = -np.mean(fitnesses)
        if gen_best > best_reward:
            best_reward = gen_best
            best_params = candidates[np.argmin(fitnesses)].copy()

        print(
            f"Gen {gen + 1}: best_reward={gen_best:.1f}, "
            f"mean_reward={gen_mean:.1f}, overall_best={best_reward:.1f}"
        )

    # Load best params
    best_controller = Controller(
        latent_dim=controller_config.latent_dim,
        hidden_dim=controller_config.hidden_dim,
        action_dim=controller_config.action_dim,
    )
    best_controller.set_params(best_params)

    # Save checkpoint
    controller_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = controller_config.checkpoint_dir / "controller_best.pt"
    torch.save(
        {"params": best_params, "reward": best_reward},
        checkpoint_path,
    )
    print(f"Saved controller (reward={best_reward:.1f}) to {checkpoint_path}")

    return best_controller
