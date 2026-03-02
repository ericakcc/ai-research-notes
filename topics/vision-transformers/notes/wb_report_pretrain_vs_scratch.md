# Pretrain vs Scratch: Validating ViT Section 3.2 on CIFAR-10

Reproducing the core claim of "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020):
a ViT pretrained on large-scale data significantly outperforms one trained from scratch on small datasets,
using fewer epochs and a lower learning rate.

---

## Background

Vision Transformer (ViT) introduces a fundamental claim in Section 3.2:

> "When trained on mid-sized datasets such as ImageNet without strong regularization,
> ViT yields modest accuracies of a few percent below ResNets of comparable size."
> — Dosovitskiy et al., 2020

Unlike CNNs, ViT has **no inductive biases** — no built-in assumptions about local pixel relationships
or translation invariance. CNNs get these for free by design; ViT must learn them from data.

The consequence: on small datasets (<14M images), ViT underperforms CNNs.
On large datasets (14M+), ViT's capacity to learn richer representations takes over.

This experiment directly tests that claim on CIFAR-10 (50k training images):
comparing a randomly initialized ViT against one pretrained on ImageNet-21k (14M images).

---

## Experiment Setup

|  | Track A — From Scratch | Track B — Fine-tune |
|--|--|--|
| Model | vit_small_patch16_224 | vit_small_patch16_224 |
| Initialization | Random | ImageNet-21k pretrained |
| Dataset | CIFAR-10 (45k train / 5k val) | same |
| Input resolution | 224×224 (resized from 32×32) | same |
| Epochs | 50 | 20 |
| Batch size | 64 | 64 |
| Base LR | 3e-4 | 1e-4 |
| Weight decay | 0.05 | 0.05 |
| LR schedule | CosineAnnealingLR | CosineAnnealingLR |
| LLRD | No | Yes (decay=0.75) |
| Hardware | RTX 5090 | same |

**Why different LR?**
Pretrained weights are already well-optimized. A smaller LR avoids destroying the learned representations.

**What is LLRD?**
Layer-wise Learning Rate Decay assigns lower LR to deeper (earlier) layers:
`lr_layer = base_lr × 0.75^(num_layers - layer_id)`
Bottom layers (edges, textures) are preserved; top layers (task-specific) are allowed to adapt.

---

## Results

|  | Track A (scratch) | Track B (finetune) |
|--|--|--|
| Best val acc | 72.3% | ~98.5% |
| Best epoch | ~50 | ~20 |
| Final train acc | ~99% | 100% |

<!-- 在這裡插入 W&B 的 val/acc 對比圖 panel -->

---

## Analysis

### Track A: Classic Overfitting Pattern

Train/acc reached ~99% while val/acc plateaued at ~72% — a 27-point gap.
This is the textbook overfitting signature: the model memorized training data
instead of learning generalizable features.

Notably, val/loss diverged (started rising around epoch 20) while val/acc kept improving.
This is not a contradiction: loss penalizes overconfident wrong predictions,
while accuracy only cares whether the top class is correct.
The model became increasingly overconfident on training patterns,
hurting loss but not yet flipping predictions.

<!-- 在這裡插入 val/loss vs val/acc 的圖 -->

### Track B: Pretrained Representations Transfer Immediately

Key observations:
- **Epoch 1 val/acc already ~98.4%** — surpassing Track A's 50-epoch best by over 26 percentage points
- **Effectively converged by epoch 1** — remaining 19 epochs contributed only ~0.1% marginal gain
- **val/loss monotonically decreased** — no divergence, no overconfidence signal

Why does this happen so cleanly?

ImageNet-21k pretraining gives the model 14M images worth of visual priors:
edge detectors, texture filters, shape outlines — the exact features CIFAR-10 classes depend on.
Fine-tuning only needs to re-map the final classification head from 21k classes to 10.
LLRD ensures the lower layers (which hold these general features) barely move,
while the top layers adapt freely.

Track A has to build all of this from 45,000 images. It never gets there.

### Overfitting Gap: Why No Inductive Bias Hurts

The train-val gap tells a clear story:

|  | Train acc | Val acc | Gap |
|--|--|--|--|
| Track A (scratch) | ~99% | 72.3% | **~27%** |
| Track B (finetune) | 100% | ~98.5% | **~1.5%** |

ViT has no inductive biases — no locality, no translation equivariance.
CNNs get these for free from their architecture, constraining the hypothesis space
so the model can generalize even with limited data.

A randomly initialized ViT faces an unconstrained search space:
every attention head can learn arbitrary patch-to-patch relationships.
With only 45,000 CIFAR-10 images, there is not enough signal to constrain 22M parameters,
so the model memorizes training data instead of learning generalizable features — hence the 27% gap.

Pretraining on ImageNet-21k (14M images) effectively gives ViT the priors that CNNs get by design:
low-level edge/texture detectors, mid-level shape compositions.
Combined with LLRD freezing these lower layers, the model's effective search space
shrinks to only the task-specific upper layers — producing a 1.5% gap instead of 27%.

---

## Conclusion

This experiment reproduces the core finding of ViT Section 3.2:

**On small datasets, pretrained ViT >> scratch ViT — by a wide margin, in fewer epochs.**

The result is not surprising given the theoretical framing:
- CNN inductive biases (locality, translation equivariance) act as free supervision signals
- ViT must learn equivalent priors from data
- 50,000 CIFAR-10 images is nowhere near enough for ViT to learn these priors from scratch
- ImageNet-21k pretraining provides exactly what's missing

### Connection to Downstream Work

This backbone understanding directly motivates:
- **MAE (He et al., 2022)**: self-supervised pretraining via masked image modeling,
  enabling ViT to learn representations *without labels* at scale
- **I-JEPA (Assran et al., 2023)**: predicts masked latent representations instead of pixels,
  learning more abstract features than MAE — building on the same ViT backbone

---

## Reproducibility

All code available at: [your GitHub repo link]

```bash
# Train from scratch (Track A)
uv run python train_scratch.py --epochs 50 --lr 3e-4

# Fine-tune pretrained (Track B)
uv run python train_finetune.py --epochs 20 --lr 1e-4
```
