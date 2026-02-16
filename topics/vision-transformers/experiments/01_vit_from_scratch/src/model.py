"""Vision Transformer (ViT) implementation from scratch.

Architecture: PatchEmbedding → [CLS] + pos_embed → TransformerBlocks → LN → MLP Head
Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
"""

import torch
import torch.nn as nn
from einops import rearrange

from config import ViTConfig


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension.

    Uses Conv2d with kernel_size=stride=patch_size to extract non-overlapping patches,
    then flattens and projects each patch to the embedding dimension.
    Adds learnable positional embeddings.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            config.channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.num_patches, config.dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches to embeddings.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Patch embeddings with positional encoding (B, num_patches, dim).
        """
        # Conv2d: (B, C, H, W) -> (B, dim, H/P, W/P)
        x = self.projection(x)
        # Flatten spatial dims: (B, dim, H/P, W/P) -> (B, num_patches, dim)
        x = rearrange(x, "b d h w -> b (h w) d")
        # Add positional embeddings
        x = x + self.pos_embedding
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism.

    Splits input into multiple heads, computes scaled dot-product attention
    independently per head, then concatenates and projects back.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.proj = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor (B, seq_len, dim).

        Returns:
            Attention output (B, seq_len, dim).

        TODO(human): Implement the forward pass for multi-head self-attention.

        Steps:
            1. Project x to Q, K, V using self.qkv (single linear layer, split into 3)
            2. Reshape Q, K, V to (B, heads, seq_len, head_dim) using einops rearrange
            3. Compute attention scores: (Q @ K^T) * self.scale
            4. Apply softmax to get attention weights
            5. Apply dropout to attention weights (self.dropout)
            6. Compute weighted sum: attention_weights @ V
            7. Reshape back to (B, seq_len, dim)
            8. Project through self.proj

        Available attributes:
            - self.heads: number of attention heads
            - self.head_dim: dimension per head
            - self.scale: 1/sqrt(head_dim) for scaling
            - self.qkv: nn.Linear that projects to 3*dim (for Q, K, V)
            - self.proj: nn.Linear output projection
            - self.dropout: nn.Dropout

        Hint: Use `rearrange` from einops for reshaping.
              rearrange(tensor, "b n (h d) -> b h n d", h=self.heads)
        """
        raise NotImplementedError("TODO(human): Implement MHSA forward pass")


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Two linear layers with GELU activation and dropout.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor (B, seq_len, dim).

        Returns:
            Output tensor (B, seq_len, dim).
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN → MHSA → residual → LN → FFN → residual.

    Pre-LayerNorm is more stable for training than Post-LayerNorm.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections.

        Args:
            x: Input tensor (B, seq_len, dim).

        Returns:
            Output tensor (B, seq_len, dim).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification.

    Pipeline: PatchEmbed → prepend [CLS] → TransformerBlocks → LN → MLP Head
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.depth)]
        )
        self.ln = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images using Vision Transformer.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Classification logits (B, num_classes).
        """
        b = x.shape[0]
        # Patch embedding: (B, C, H, W) -> (B, num_patches, dim)
        x = self.patch_embed(x)
        # Prepend [CLS] token: (B, num_patches+1, dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Transformer blocks
        x = self.blocks(x)
        # Take [CLS] token output, normalize, classify
        cls_out = self.ln(x[:, 0])
        return self.head(cls_out)
