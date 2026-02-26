# AI Research Portfolio

Hands-on, from-scratch implementations of modern deep learning architectures,
from Vision Transformers to World Models and Joint-Embedding Predictive Architectures.

> **Focus:** Self-supervised learning, world models, joint-embedding predictive architectures (JEPA),
> multimodal LLMs, and agentic systems.

## Highlighted Experiments

| Experiment | Paper | Key Result | Code |
|------------|-------|------------|------|
| World Models (VMC) | Ha & Schmidhuber 2018 | CarRacing reward: -- | [code](topics/world-models/experiments/01_car_racing/) |
| ViT from Scratch | Dosovitskiy et al. 2020 | CIFAR-10 acc: --% | [code](topics/vision-transformers/experiments/01_vit_from_scratch/) |
| YOLOv1 | Redmon et al. 2016 | -- | [code](topics/computer-vision/experiments/01_yolo_from_scratch/) |

> Results are filled in as each experiment completes training.
> Each experiment follows a [standardized write-up format](docs/EXPERIMENT_TEMPLATE.md).

## Topics

| Topic | Description | Status |
|-------|-------------|--------|
| [World Models](topics/world-models/) | VMC, DreamerV3, DIAMOND | Active |
| [Vision Transformers](topics/vision-transformers/) | ViT, MAE | Active |
| [Computer Vision](topics/computer-vision/) | YOLO evolution, detection | Active |
| [Self-Supervised Learning](topics/self-supervised-learning/) | VICReg, SimCLR, Barlow Twins | Active |
| [JEPA](topics/jepa/) | I-JEPA, V-JEPA, MC-JEPA | Active |
| [MLLM](topics/mllm/) | CLIP, LLaVA — vision-language alignment | Planned |
| [Reinforcement Learning](topics/reinforcement-learning/) | PPO, SAC, JEPA+RL | Planned |
| [Agent + Tool Use](topics/agent-tool-use/) | ReAct agent, Anthropic tool use, JEPA vision input | Planned |
| [Embodied AI](topics/embodied-ai/) | Isaac Gym, JEPA Navigator | Planned |

## Technical Stack

- **PyTorch** — from-scratch implementations, no high-level wrappers
- **uv + ruff + pytest** — modern Python tooling (fast dependency management, linting, testing)
- Type hints and Google-style docstrings throughout

## Repository Structure

```
ai-research-notes/
├── topics/
│   ├── world-models/              # VAE + MDN-RNN + Controller
│   ├── vision-transformers/       # ViT, MAE from scratch
│   ├── computer-vision/           # YOLOv1 model + loss
│   ├── self-supervised-learning/  # VICReg, SimCLR, Barlow Twins
│   ├── jepa/                      # I-JEPA, V-JEPA, MC-JEPA
│   ├── mllm/                      # CLIP, LLaVA (planned)
│   ├── reinforcement-learning/    # PPO, SAC (planned)
│   ├── agent-tool-use/            # ReAct agent, Anthropic tool use (planned)
│   └── embodied-ai/              # Isaac Gym, Habitat (planned)
├── docs/                          # Templates and guides
├── pyproject.toml                 # Shared dependencies
└── Makefile                       # Top-level commands
```

## Getting Started

```bash
# Setup environment (installs uv if needed + syncs dependencies)
make setup

# Run tests
make test

# Run linter
make lint

# Train an experiment (example: VMC VAE)
uv run python -m topics.world_models.experiments.01_car_racing.src.train_vae
```
