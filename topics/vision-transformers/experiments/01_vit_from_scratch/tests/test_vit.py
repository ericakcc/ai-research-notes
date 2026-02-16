"""Tests for ViT modules — written before implementation (TDD)."""

import torch
import pytest

from config import ViTConfig
from model import (
    FeedForward,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
    VisionTransformer,
)


@pytest.fixture
def config() -> ViTConfig:
    """Default ViT-Tiny config for CIFAR-10."""
    return ViTConfig()


@pytest.fixture
def batch() -> torch.Tensor:
    """Batch of 2 CIFAR-10 images."""
    return torch.randn(2, 3, 32, 32)


class TestPatchEmbedding:
    def test_output_shape(self, config: ViTConfig, batch: torch.Tensor) -> None:
        """PatchEmbedding should produce (B, num_patches, dim)."""
        embed = PatchEmbedding(config)
        out = embed(batch)
        assert out.shape == (2, config.num_patches, config.dim)

    def test_different_batch_sizes(self, config: ViTConfig) -> None:
        """Should work with any batch size."""
        embed = PatchEmbedding(config)
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 3, 32, 32)
            assert embed(x).shape[0] == bs


class TestMultiHeadSelfAttention:
    def test_output_shape(self, config: ViTConfig) -> None:
        """MHSA output should match input shape: (B, seq_len, dim)."""
        mhsa = MultiHeadSelfAttention(config)
        seq_len = config.num_patches + 1  # +1 for [CLS] token
        x = torch.randn(2, seq_len, config.dim)
        out = mhsa(x)
        assert out.shape == (2, seq_len, config.dim)

    def test_attention_is_equivariant_to_batch(self, config: ViTConfig) -> None:
        """Same input in different batch positions should give same output."""
        mhsa = MultiHeadSelfAttention(config)
        mhsa.eval()
        seq_len = config.num_patches + 1
        x = torch.randn(1, seq_len, config.dim)
        x_batched = x.repeat(3, 1, 1)
        out = mhsa(x_batched)
        torch.testing.assert_close(out[0], out[1], atol=1e-5, rtol=1e-5)


class TestFeedForward:
    def test_output_shape(self, config: ViTConfig) -> None:
        """FFN output should match input shape."""
        ffn = FeedForward(config)
        x = torch.randn(2, 65, config.dim)
        assert ffn(x).shape == (2, 65, config.dim)


class TestTransformerBlock:
    def test_output_shape(self, config: ViTConfig) -> None:
        """TransformerBlock should preserve input shape."""
        block = TransformerBlock(config)
        x = torch.randn(2, 65, config.dim)
        assert block(x).shape == (2, 65, config.dim)

    def test_residual_connection(self, config: ViTConfig) -> None:
        """Output should differ from input (transformation happened)."""
        block = TransformerBlock(config)
        block.eval()
        x = torch.randn(2, 65, config.dim)
        out = block(x)
        assert not torch.allclose(out, x, atol=1e-3)


class TestVisionTransformer:
    def test_forward_pass(self, config: ViTConfig, batch: torch.Tensor) -> None:
        """Full forward pass: (B, C, H, W) -> (B, num_classes)."""
        model = VisionTransformer(config)
        out = model(batch)
        assert out.shape == (2, config.num_classes)

    def test_parameter_count(self, config: ViTConfig) -> None:
        """ViT-Tiny for CIFAR-10 should have reasonable parameter count."""
        model = VisionTransformer(config)
        num_params = sum(p.numel() for p in model.parameters())
        # ViT-Tiny should be roughly 2-5M params
        assert 1_000_000 < num_params < 10_000_000, (
            f"Unexpected param count: {num_params:,}"
        )

    def test_output_logits_not_all_same(
        self, config: ViTConfig, batch: torch.Tensor
    ) -> None:
        """Different inputs should produce different outputs."""
        model = VisionTransformer(config)
        model.eval()
        out = model(batch)
        assert not torch.allclose(out[0], out[1], atol=1e-3)
