# CLAUDE.md - World Models

This file provides guidance to Claude Code when working with the World Models topic.

## Topic Overview

A structured learning project for World Models research, progressing through 4 phases:
1. **Foundations** - VMC architecture (VAE + RNN) with CarRacing
2. **Standard** - DreamerV3 RSSM framework with Atari 100k
3. **High-Fidelity** - Diffusion models (DIAMOND) for neural simulation
4. **Advanced** - Driving world models (GAIA-1, OpenDV). For V-JEPA deep implementation, see `topics/jepa/`

## Commands

```bash
# From this directory (topics/world-models/)
make papers                      # Download all 6 papers from arXiv

# From project root
make lint                        # Run ruff check --fix + format
make test                        # Run pytest -v
```

## Structure

- `experiments/01_car_racing/` - Phase 1: VAE + RNN implementation
- `experiments/02_dreamerv3_atari/` - Phase 2: DreamerV3 on Atari
- `experiments/03_diamond_diffusion/` - Phase 3: Diffusion world model
- `experiments/04_opendv_driving/` - Phase 4: Driving world model
- `papers/` - Research papers organized by phase (PDFs gitignored)
- `notes/` - Reading notes organized by phase
- `docs/` - Reference documents (Chinese)
- `datasets/` - Training data (gitignored, large files)
- `scripts/` - Utility scripts (download_papers.sh)

## Key Reference Repositories

When implementing experiments, reference these codebases:
- Phase 1: `world-models-jax` (JAX/Equinox)
- Phase 2: `danijar/dreamerv3` (official JAX) or `NM512/dreamerv3-torch`
- Phase 3: `eloialonso/diamond` (Diffusion world model)
- Phase 4: `wayveai/mile`, `waymo-research/waymax` (V-JEPA moved to `topics/jepa/`)
