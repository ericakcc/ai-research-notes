# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-topic ML/AI research and learning repository. Each topic is organized as a self-contained sub-project under `topics/`.

## Topics

| Topic | Path | Status | NYUAD 對應 |
|-------|------|--------|-----------|
| World Models | `topics/world-models/` | Active | 基礎 |
| Computer Vision | `topics/computer-vision/` | Active | 輔助 |
| Vision Transformers | `topics/vision-transformers/` | Active | 基礎架構 |
| Self-Supervised Learning | `topics/self-supervised-learning/` | Active | 第一階段 |
| JEPA | `topics/jepa/` | Active | 第二階段 |
| Reinforcement Learning | `topics/reinforcement-learning/` | Planned | 第三階段 |
| Embodied AI | `topics/embodied-ai/` | Planned | 第三-四階段 |

## Commands

```bash
# Quick setup
make setup                       # Install uv + sync dependencies
make lint                        # Run ruff check --fix + format
make test                        # Run pytest -v
make clean                       # Remove .venv, __pycache__, etc.

# Direct uv commands
uv add <package>                 # Add dependency
uv add --dev <package>           # Add dev dependency
uv run pytest -k "test_name"     # Run specific test
```

## Project Structure

```
ai-research-notes/
├── topics/
│   ├── world-models/              # World Models research (VMC, DreamerV3, DIAMOND)
│   ├── computer-vision/           # YOLO evolution & defect detection
│   ├── vision-transformers/       # ViT, MAE implementations
│   ├── self-supervised-learning/  # VICReg, SimCLR, Barlow Twins
│   ├── jepa/                      # I-JEPA, V-JEPA, MC-JEPA
│   ├── reinforcement-learning/    # PPO, SAC, JEPA+RL
│   └── embodied-ai/              # Isaac Gym, Habitat, JEPA Navigator
├── NYUAD_PROGRESS.md              # NYUAD Research Engineer sprint progress
├── pyproject.toml                 # Shared dependencies
└── Makefile                       # Top-level commands
```

## Topic-Specific Instructions

Each topic has its own `CLAUDE.md` with detailed instructions. When working on a specific topic, refer to:
- `topics/world-models/CLAUDE.md` for World Models
- `topics/computer-vision/CLAUDE.md` for Computer Vision
- `topics/vision-transformers/CLAUDE.md` for Vision Transformers
- `topics/self-supervised-learning/CLAUDE.md` for Self-Supervised Learning
- `topics/jepa/CLAUDE.md` for JEPA
- `topics/reinforcement-learning/CLAUDE.md` for Reinforcement Learning
- `topics/embodied-ai/CLAUDE.md` for Embodied AI
