# CLAUDE.md - JEPA

This file provides guidance to Claude Code when working with the JEPA topic.

## Topic Overview

A structured learning project covering the full JEPA ecosystem:
1. **I-JEPA** — Image-based Joint Embedding Predictive Architecture (masked prediction in latent space)
2. **V-JEPA** — Video JEPA (temporal prediction for video understanding)
3. **MC-JEPA** — Multimodal Conditional JEPA (cross-modal conditioning)

## NYUAD Relevance

JEPA is the **core research direction** at NYUAD CAIR (Yann LeCun's lab collaboration). Demonstrating deep understanding of I-JEPA → V-JEPA → MC-JEPA progression is critical for the Research Engineer position.

## Prerequisites

- Vision Transformers (ViT backbone, patch embedding, MHSA)
- Self-Supervised Learning (collapse prevention, EBM foundations)

## Learning Path

1. Read I-JEPA paper — understand context encoder, target encoder, predictor
2. Implement I-JEPA on ImageNet subset — masked latent prediction + linear probe
3. Read V-JEPA paper — temporal extension, video patch masking
4. Implement V-JEPA on Kinetics subset — video feature prediction
5. Read V-JEPA 2 paper — scaling and improvements
6. Read MC-JEPA paper — multimodal conditional injection
7. Implement MC-JEPA — cross-modal conditioning experiments
8. Read LeJEPA — language-enhanced JEPA

## Paper Checklist

- [ ] I-JEPA — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (Assran et al., 2023)
- [ ] V-JEPA — Video Joint Embedding Predictive Architecture (Bardes et al., 2024)
- [ ] V-JEPA 2 — Self-Supervised Video Models with Large-Scale Joint Embedding (Bardes et al., 2025)
- [ ] MC-JEPA — Multimodal Conditional JEPA (Bardes et al., 2025)
- [ ] LeJEPA — Language-Enhanced JEPA (Li et al., 2025)

## Reference Repositories

- `facebookresearch/ijepa` — Official I-JEPA implementation
- `facebookresearch/jepa` — Official V-JEPA implementation

## Structure

- `experiments/01_ijepa/` — I-JEPA: masked prediction in latent space
- `experiments/02_vjepa/` — V-JEPA: video temporal prediction
- `experiments/03_mcjepa/` — MC-JEPA: multimodal conditional injection
- `papers/` — Research papers (PDFs gitignored)
- `notes/` — Reading notes

## Key Concepts

- **Context Encoder**: Processes visible patches → latent representations
- **Target Encoder**: EMA of context encoder, processes target patches
- **Predictor**: Predicts target representations from context (operates in latent space)
- **No pixel reconstruction**: Predictions are in representation space, not pixel space

## Deliverables

| Experiment | Target | Output |
|-----------|--------|--------|
| 01_ijepa | ImageNet subset | masked prediction + linear probe accuracy |
| 02_vjepa | Kinetics subset | video feature prediction quality |
| 03_mcjepa | Multimodal | cross-modal conditioning results |
