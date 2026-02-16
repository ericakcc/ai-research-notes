# CLAUDE.md - Vision Transformers

This file provides guidance to Claude Code when working with the Vision Transformers topic.

## Topic Overview

A structured learning project for Vision Transformers, progressing through 2 phases:
1. **ViT from Scratch** — Implement the original Vision Transformer for CIFAR-10 classification
2. **Masked Autoencoder (MAE)** — Self-supervised pre-training via masked image modeling

## NYUAD Relevance

ViT is the **backbone architecture** for I-JEPA, V-JEPA, and MC-JEPA. Mastering patch embedding, multi-head self-attention, and positional encoding here directly transfers to JEPA implementations.

## Learning Path

1. Read "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
2. Implement ViT from scratch — PatchEmbedding, MHSA, TransformerBlock
3. Train on CIFAR-10, achieve > 85% accuracy
4. Read "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
5. Implement MAE — masked patch prediction + decoder reconstruction
6. Visualize reconstruction quality

## Paper Checklist

- [ ] ViT — An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
- [ ] DeiT — Training data-efficient image transformers (Touvron et al., 2021)
- [ ] MAE — Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)

## Reference Repositories

- `google-research/vision_transformer` — Official JAX implementation
- `lucidrains/vit-pytorch` — Clean PyTorch implementations of ViT variants
- `facebookresearch/mae` — Official MAE implementation

## Structure

- `experiments/01_vit_from_scratch/` — ViT implementation + CIFAR-10 training
- `experiments/02_masked_autoencoder/` — MAE implementation
- `papers/` — Research papers (PDFs gitignored)
- `notes/` — Reading notes

## Commands

```bash
# Run ViT tests
uv run pytest topics/vision-transformers/experiments/01_vit_from_scratch/tests/ -v

# Train ViT on CIFAR-10
uv run python -m topics.vision_transformers.experiments.01_vit_from_scratch.src.train
```

## Deliverables

| Experiment | Target | Metric |
|-----------|--------|--------|
| 01_vit_from_scratch | CIFAR-10 classifier | accuracy > 85% |
| 02_masked_autoencoder | Image reconstruction | qualitative visualization |
