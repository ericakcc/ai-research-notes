# CLAUDE.md - Reinforcement Learning

This file provides guidance to Claude Code when working with the Reinforcement Learning topic.

## Topic Overview

A structured learning project for Deep RL, progressing through 3 phases:
1. **PPO on CartPole** — Policy gradient from scratch
2. **SAC on Continuous** — Off-policy actor-critic for continuous action spaces
3. **World Model + RL** — Integrating JEPA representations as RL state inputs

## NYUAD Relevance

Deep RL is essential for the **embodied AI pipeline**: JEPA features → RL policy → robot control. Understanding PPO/SAC is prerequisite for Phase 3 integration with JEPA representations.

## Prerequisites

- JEPA topic (for experiment 03)
- Basic PyTorch proficiency

## Learning Path

1. Read PPO paper (Schulman et al., 2017)
2. Implement PPO from scratch on CartPole — clipped objective, GAE
3. Read SAC paper (Haarnoja et al., 2018)
4. Implement SAC on LunarLander-Continuous or HalfCheetah
5. Integrate JEPA features as state representation for RL agent

## Paper Checklist

- [ ] PPO — Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- [ ] SAC — Soft Actor-Critic (Haarnoja et al., 2018)
- [ ] Dreamer — Dream to Control (Hafner et al., 2020)

## Reference Repositories

- `openai/baselines` — OpenAI baselines
- `DLR-RM/stable-baselines3` — Stable Baselines3
- `vwxyzjn/cleanrl` — Clean single-file RL implementations

## Structure

- `experiments/01_ppo_cartpole/` — PPO from scratch
- `experiments/02_sac_continuous/` — SAC for continuous actions
- `experiments/03_world_model_rl/` — JEPA features + RL integration
- `papers/` — Research papers (PDFs gitignored)
- `notes/` — Reading notes

## Deliverables

| Experiment | Target | Output |
|-----------|--------|--------|
| 01_ppo_cartpole | CartPole-v1 | reward curve + policy visualization |
| 02_sac_continuous | LunarLander/HalfCheetah | training curves |
| 03_world_model_rl | JEPA + RL | agent using learned representations |
