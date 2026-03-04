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
| Epochs | 300 | 300 | 300 | 300 | 300 |
| Batch size | 128 | 128 | 128 | 128 | 128 |

**Hardware:** NVIDIA GPU (via remote server)

---

## Results

|  | Best val acc | Delta | Best epoch | Final train acc | Train-val gap |
|--|-------------|-------|------------|-----------------|---------------|
| A: Baseline | 83.3% | — | 264 | 99.96% | 16.7% |
| B: +Warmup+Clip | 85.0% | +1.7% | 261 | 99.97% | 15.0% |
| C: +LabelSmooth | 85.7% | +0.8% | 279 | 99.98% | 14.3% |
| D: +RandAugment | 88.5% | +2.7% | 259 | 97.67% | 9.2% |
| E: +Mixup (Full) | **91.8%** | **+3.3%** | 286 | 83.30% | -8.5% |

<!-- Insert W&B val/acc comparison chart here -->

<!-- Insert W&B train/loss comparison chart here -->

---

## Analysis

### Track A: Baseline

Classic overfitting pattern. Train acc reached 99.96% while val acc plateaued at 83.3%
— a 16.7% gap. Val loss diverged early (final val_loss=1.136 vs train_loss=0.001),
indicating the model memorized training data rather than learning generalizable features.
This is the expected behavior for a ViT without regularization on a small dataset.

### Track B: +Warmup + Gradient Clipping

Warmup stabilized early training and improved best val acc by +1.7% (83.3% → 85.0%).
The 10-epoch linear warmup gave AdamW's momentum estimates time to calibrate before
applying full learning rate. Gradient clipping (max_norm=1.0) prevented occasional
gradient spikes from destabilizing training. However, the train-val gap remained large
(15.0%) — optimizer tricks alone don't address the fundamental overfitting problem.

### Track C: +Label Smoothing

Label smoothing added a modest +0.8% (85.0% → 85.7%). The key effect is visible in
val_loss: it dropped from 1.125 (Track B) to 0.969 (Track C). By softening targets
from hard one-hot to ε=0.1, the model produces less overconfident predictions.
Train acc barely changed (99.97% → 99.98%), confirming that label smoothing primarily
acts as a regularizer on the output distribution, not on the learning capacity.

### Track D: +RandAugment

The first major breakthrough: +2.7% (85.7% → 88.5%). RandAugment(N=2, M=9) was the
first trick to significantly reduce the train-val gap (14.3% → 9.2%). Train acc
dropped to 97.67% — the augmented training data is now harder to memorize.
This validates the DeiT insight: for ViT without inductive biases, strong data
augmentation is more impactful than optimizer-level tricks.

### Track E: +Mixup (Full Pipeline)

The largest single improvement: +3.3% (88.5% → **91.8%**). Mixup fundamentally
changed the training dynamics — train acc (83.3%) is now **lower** than val acc (91.8%).
This is not a bug: Mixup creates artificially harder training samples by blending
images and labels, so the training task is inherently more difficult than clean
evaluation. The negative train-val gap (-8.5%) is a strong signal of effective
regularization without overfitting.

### Cumulative Effect

Total improvement from baseline to full pipeline: **+8.5%** (83.3% → 91.8%).

Contribution breakdown:
- **Data regularization (RandAugment + Mixup): +6.0%** — 71% of total improvement
- **Optimizer tricks (Warmup + Clip): +1.7%** — 20% of total improvement
- **Label regularization (Label Smoothing): +0.8%** — 9% of total improvement

The 85% target was reached at Track B (+Warmup+Clip). The final 91.8% far exceeded
the original goal, demonstrating that proper training tricks can compensate for
ViT's lack of inductive biases even on small datasets.

---

## Key Takeaways

- **Data regularization >> optimizer tricks** for ViT on small datasets.
  RandAugment + Mixup contributed 71% of total improvement (+6.0% out of +8.5%).
- **Mixup is the single most impactful trick** (+3.3%), producing a negative train-val
  gap that signals strong regularization without overfitting.
- **85% is achievable with just warmup + gradient clipping**, but reaching 90%+ requires
  data augmentation (RandAugment + Mixup).
- **Training tricks alone brought ViT-Tiny from 83.3% to 91.8%** without any model
  architecture changes — validating DeiT's approach of compensating for ViT's lack
  of inductive biases through training recipe rather than architecture modifications.
- **Label smoothing had the smallest individual impact** (+0.8%), but its effect on
  calibration (reducing val_loss) may matter more for downstream tasks than accuracy alone.

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
uv run python train_ablation.py --track all --epochs 300

# Run a single track
uv run python train_ablation.py --track baseline --epochs 300

# Run without W&B
uv run python train_ablation.py --track all --epochs 300 --no-wandb
```

W&B Project: `vit-cifar10`
Run naming: `ablation-{track_name}`
