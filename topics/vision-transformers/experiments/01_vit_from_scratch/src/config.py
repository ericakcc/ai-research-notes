"""ViT configuration using dataclass."""

from dataclasses import dataclass


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer.

    Defaults are ViT-Tiny sized, optimized for CIFAR-10 (32x32 images).

    Args:
        image_size: Input image resolution (square).
        patch_size: Size of each patch (square).
        num_classes: Number of classification classes.
        dim: Embedding dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_dim: Hidden dimension of feed-forward network.
        dropout: Dropout rate.
        channels: Number of input channels.
    """

    image_size: int = 32
    patch_size: int = 4
    num_classes: int = 10
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    channels: int = 3

    @property
    def num_patches(self) -> int:
        """Total number of patches in the image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.dim // self.heads
