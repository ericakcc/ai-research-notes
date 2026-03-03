# Ablation Study: Progressive Training Tricks for ViT-Tiny on CIFAR-10

Investigating which training tricks contribute most to ViT-Tiny performance,
building up from a bare baseline to a fully-equipped training pipeline.

---

## Background

Our custom ViT-Tiny (dim=256, depth=6, heads=8, ~2.7M params) uses patch_size=4
on CIFAR-10 (32x32). Without training tricks, ViT struggles on small datasets
due to its lack of inductive biases (no locality, no translation equivariance).

This ablation study progressively adds 5 training tricks — each backed by a
specific paper — to measure their individual and cumulative contribution.

### Training Tricks & References

| Trick | Paper | Key Idea |
|-------|-------|----------|
| Gradient Clipping | Pascanu et al., 2013 | Clip gradient norm to prevent explosion |
| LR Warmup | Goyal et al., 2017 | Linear warmup for Adam momentum stability |
| Label Smoothing | Szegedy et al., 2016 (Inception v2) | Soften one-hot targets to reduce overconfidence |
| RandAugment | Cubuk et al., 2020 (Google Brain) | Random augmentation with 2 hyperparams (N, M) |
| Mixup | Zhang et al., 2018 (FAIR) | Linear interpolation of input-label pairs |

---

## Experiment Setup

|  | A: Baseline | B: +Warmup | C: +Smooth | D: +RandAug | E: +Mixup |
|--|-------------|------------|------------|-------------|-----------|
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW |
| LR | 3e-4 | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| Weight decay | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| LR schedule | Cosine | Warmup 10ep + Cosine | same as B | same as B | same as B |
| Grad clip | - | 1.0 | 1.0 | 1.0 | 1.0 |
| Label smooth | - | - | 0.1 | 0.1 | 0.1 |
| RandAugment | - | - | - | N=2, M=9 | N=2, M=9 |
| Mixup alpha | - | - | - | - | 0.8 |
| Epochs | 100 | 100 | 100 | 100 | 100 |
| Batch size | 128 | 128 | 128 | 128 | 128 |

**Hardware:** <!-- TODO: fill in after running -->

---

## Results

|  | Best val acc | Best epoch | Final train acc | Train-val gap |
|--|-------------|------------|-----------------|---------------|
| A: Baseline | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| B: +Warmup+Clip | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| C: +LabelSmooth | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| D: +RandAugment | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| E: +Mixup (Full) | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

<!-- Insert W&B val/acc comparison chart here -->

<!-- Insert W&B train/loss comparison chart here -->

---

## Analysis

### Track A: Baseline

<!-- TODO: Describe training dynamics, overfitting behavior, loss curves -->

### Track B: +Warmup + Gradient Clipping

<!-- TODO: Did warmup stabilize early training? Compare LR curve shape -->

### Track C: +Label Smoothing

<!-- TODO: Effect on overconfidence? Compare val/loss divergence with Track B -->

### Track D: +RandAugment

<!-- TODO: Impact on train-val gap? Did stronger augmentation slow convergence? -->

### Track E: +Mixup (Full Pipeline)

<!-- TODO: Final accuracy improvement? Effect on decision boundary smoothness? -->

### Cumulative Effect

<!-- TODO: Which trick contributed the most? Was the 85% target reached? At which track? -->

---

## Key Takeaways

<!-- TODO: 3-5 bullet points summarizing findings -->

---

## Connection to Downstream Work

- **DeiT (Touvron et al., 2021):** Uses all 5 tricks (plus more) to train ViT
  without large-scale pretraining — our ablation validates their recipe
- **MAE (He et al., 2022):** Pre-training may reduce dependence on augmentation
  tricks — worth comparing MAE-pretrained ViT vs. trick-equipped scratch ViT
- **I-JEPA (Assran et al., 2023):** Uses minimal augmentation by design — the
  latent prediction objective provides its own regularization

---

## Reproducibility

```bash
# Run all tracks
cd topics/vision-transformers/experiments/01_vit_from_scratch/src
uv run python train_ablation.py --track all --epochs 100

# Run a single track
uv run python train_ablation.py --track baseline --epochs 100

# Run without W&B
uv run python train_ablation.py --track all --epochs 100 --no-wandb
```

W&B Project: `vit-cifar10`
Run naming: `ablation-{track_name}`
