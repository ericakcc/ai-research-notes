# World Models (VMC) — From-Scratch Implementation

## Paper Summary

Ha & Schmidhuber (2018) propose learning a compressed spatial-temporal model of the environment, then training a compact controller entirely within this learned "dream." The architecture separates perception (VAE), dynamics (MDN-RNN), and decision-making (Controller), allowing the agent to plan using an internal model rather than raw pixels.

## Architecture

```
                         World Models Pipeline
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   Observation (64x64x3)                                  │
  │         │                                                │
  │         ▼                                                │
  │   ┌───────────┐                                          │
  │   │    VAE     │  4-layer CNN encoder → 32-dim latent     │
  │   │  (Vision)  │  4-layer CNN decoder ← reconstruction   │
  │   └─────┬─────┘                                          │
  │         │ z_t (32-dim)                                   │
  │         ▼                                                │
  │   ┌───────────┐                                          │
  │   │  MDN-RNN  │  LSTM (256 hidden) + 5-component MDN    │
  │   │ (Memory)  │  Input: [z_t, a_t] → Predict: z_{t+1}   │
  │   └─────┬─────┘                                          │
  │         │ h_t (256-dim)                                  │
  │         ▼                                                │
  │   ┌───────────┐                                          │
  │   │Controller │  Linear: [z_t, h_t] → [steer, gas, brake]│
  │   │ (Action)  │  Optimized by CMA-ES (~867 params)       │
  │   └───────────┘                                          │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

## Implementation Notes

**VAE:** Standard convolutional VAE with 4 stride-2 layers. Output uses sigmoid activation to match [0, 1] pixel range. Reparameterization trick enables gradient flow through the stochastic sampling step.

**MDN-RNN:** The mixture density network outputs per-dimension parameters — each of the 32 latent dimensions gets its own 5-component Gaussian mixture. This is more expressive than a single shared mixture but increases output dimensionality. Log-softmax is used for numerical stability on mixture weights.

**Controller:** Deliberately minimal — a single linear layer mapping the 288-dim input [z, h] to 3 actions. This keeps the parameter count low (~867) so CMA-ES can optimize effectively without gradient computation.

**Simplifications from the paper:**
- 200 rollouts for data collection (paper uses 10,000)
- 10 training epochs per component (paper trains longer)
- Single-layer LSTM (paper also uses single-layer)

## Results

| Metric | Paper | Ours | Notes |
|--------|-------|------|-------|
| VAE reconstruction loss | — | -- | Qualitative: check `notebooks/02_train_vae.ipynb` |
| MDN-RNN NLL | — | -- | Latent prediction loss |
| CarRacing reward (no dream) | 906 ± 21 | -- | Controller in real environment |
| CarRacing reward (dream) | 868 ± 11 | -- | Controller trained in dream |

<!-- TODO: Fill in after training each component -->

## Key Takeaways

<!-- To be written after completing the full pipeline. Preliminary observations: -->

1. **Separation of concerns matters** — Training V, M, C independently means each component can be debugged and validated in isolation. The VAE can be checked visually, the MDN-RNN by prediction loss, and the controller by reward.

2. **CMA-ES over gradients for tiny models** — With only ~867 parameters, evolutionary strategies avoid the need for differentiable dynamics. This is a key insight: the controller is intentionally kept simple so that gradient-free optimization is tractable.

3. **Latent space as information bottleneck** — The 32-dim latent forces the VAE to learn a compressed representation. Whether this bottleneck captures the right features for driving is an empirical question the training will answer.

## How to Reproduce

```bash
# From the repository root
uv sync

# Step 1: Collect rollouts from CarRacing
uv run python -m topics.world_models.experiments.01_car_racing.scripts.collect_data

# Step 2: Train VAE
uv run python -m topics.world_models.experiments.01_car_racing.src.train_vae

# Step 3: Encode rollouts to latent space
uv run python -m topics.world_models.experiments.01_car_racing.scripts.encode_rollouts

# Step 4: Train MDN-RNN
uv run python -m topics.world_models.experiments.01_car_racing.src.train_rnn

# Step 5: Train Controller via CMA-ES
uv run python -m topics.world_models.experiments.01_car_racing.src.train_controller
```

**Hardware:** Tested on Apple Silicon (MPS). CUDA also supported via config.
**Data:** ~200 episodes of random-policy rollouts (~1 GB).
**Runtime:** VAE training ~20 min on MPS; full pipeline TBD.
