# CLAUDE.md - Self-Supervised Learning

This file provides guidance to Claude Code when working with the Self-Supervised Learning topic.

## Topic Overview

A structured learning project covering non-contrastive and contrastive self-supervised learning methods:
1. **VICReg** — Variance-Invariance-Covariance Regularization on CIFAR-10
2. **SimCLR** — Contrastive learning with NT-Xent loss
3. **Barlow Twins** — Redundancy reduction via cross-correlation

## NYUAD Relevance

SSL methods are the **theoretical foundation** for JEPA. Understanding how representations are learned without labels — and the collapse problem — is essential before implementing I-JEPA and V-JEPA.

## Learning Path

1. Read LeCun's "A Tutorial on Energy-Based Learning" for EBM foundations
2. Read VICReg paper — understand variance, invariance, covariance terms
3. Implement VICReg on CIFAR-10 with feature decorrelation visualization
4. Read SimCLR paper — contrastive learning and data augmentation strategies
5. Implement SimCLR with t-SNE visualization of learned features
6. Read Barlow Twins — cross-correlation matrix analysis
7. Implement Barlow Twins with correlation heatmap

## Paper Checklist

- [ ] EBM Tutorial — A Tutorial on Energy-Based Learning (LeCun et al., 2006)
- [ ] VICReg — Variance-Invariance-Covariance Regularization (Bardes et al., 2022)
- [ ] SimCLR — A Simple Framework for Contrastive Learning (Chen et al., 2020)
- [ ] Barlow Twins — Self-Supervised Learning via Redundancy Reduction (Zbontar et al., 2021)

## Reference Repositories

- `facebookresearch/vicreg` — Official VICReg implementation
- `google-research/simclr` — Official SimCLR
- `facebookresearch/barlowtwins` — Official Barlow Twins

## Structure

- `experiments/01_vicreg_cifar10/` — VICReg with feature decorrelation analysis
- `experiments/02_simclr/` — SimCLR with t-SNE visualization
- `experiments/03_barlow_twins/` — Barlow Twins with correlation heatmap
- `papers/` — Research papers (PDFs gitignored)
- `notes/` — Reading notes

## Deliverables

| Experiment | Target | Output |
|-----------|--------|--------|
| 01_vicreg_cifar10 | Feature learning on CIFAR-10 | decorrelation analysis plot |
| 02_simclr | Contrastive features | t-SNE visualization |
| 03_barlow_twins | Cross-correlation | correlation matrix heatmap |
