# Vision Transformers

ViT is the backbone architecture for I-JEPA, V-JEPA, and MC-JEPA. This section covers the foundational attention mechanism and how it was adapted from NLP to computer vision.

## Learning Path

1. Understand the Transformer's core — scaled dot-product attention and multi-head attention
2. See how ViT applies Transformers to images via patch embedding
3. Validate the pretrain vs scratch claim through hands-on experiments

## Notes

| Paper / Report | Summary |
|---|---|
| [Attention Is All You Need](attention-paper-notes.md) | The original Transformer — self-attention, multi-head mechanism, positional encoding |
| [ViT Paper Notes](vit-paper-notes.md) | Vision Transformer — patch embedding, CLS token, position interpolation |
| [Pretrain vs Scratch](pretrain-vs-scratch.md) | Experiment reproducing ViT Section 3.2 on CIFAR-10 |
