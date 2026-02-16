# CLAUDE.md - Embodied AI

This file provides guidance to Claude Code when working with the Embodied AI topic.

## Topic Overview

A structured learning project for Embodied AI, progressing through 3 phases:
1. **Isaac Gym Basics** — NVIDIA Isaac Gym setup and basic environments
2. **Habitat Navigation** — Habitat-Sim visual navigation tasks
3. **JEPA-Driven Navigator** — Final portfolio project integrating JEPA + RL for embodied navigation

## NYUAD Relevance

Embodied AI is the **end goal** of the CAIR research agenda: world models (JEPA) → policy learning (RL) → embodied agent. The JEPA Navigator project demonstrates full-stack capability.

## Prerequisites

- Vision Transformers (backbone)
- Self-Supervised Learning (representation learning)
- JEPA (world model features)
- Reinforcement Learning (policy learning)

## Learning Path

1. Install Isaac Gym, run official demos
2. Create custom environment in Isaac Gym
3. Set up Habitat-Sim, run navigation benchmarks
4. Build JEPA-driven navigator: JEPA encoder → RL policy → sim navigation
5. Document as portfolio project

## Paper Checklist

- [ ] Isaac Gym — GPU-Accelerated Robot Simulation (Makoviychuk et al., 2021)
- [ ] Habitat — A Platform for Embodied AI Research (Savva et al., 2019)
- [ ] VC-1 — Visual Cortex for Embodied AI (Majumdar et al., 2023)

## Reference Repositories

- `NVIDIA-Omniverse/IsaacGymEnvs` — Isaac Gym environments
- `facebookresearch/habitat-sim` — Habitat simulator
- `facebookresearch/habitat-lab` — Habitat tasks and baselines

## Structure

- `experiments/01_isaac_gym_basics/` — Isaac Gym setup and demos
- `experiments/02_habitat_navigation/` — Habitat-Sim navigation
- `experiments/03_jepa_navigator/` — Portfolio: JEPA + RL navigator
- `papers/` — Research papers (PDFs gitignored)
- `notes/` — Reading notes

## Deliverables

| Experiment | Target | Output |
|-----------|--------|--------|
| 01_isaac_gym_basics | Run demos + custom env | working Isaac Gym setup |
| 02_habitat_navigation | Point-goal navigation | navigation metrics |
| 03_jepa_navigator | JEPA + RL integration | portfolio-ready demo |
