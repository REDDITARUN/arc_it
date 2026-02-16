"""Unit tests for all ARC-IT model components.

Tests verify shapes, gradients, and end-to-end integration on CPU/MPS.
Uses stub encoder (no download needed) for fast local testing.

Run with: python -m pytest tests/test_models.py -v
"""

import pytest
import torch
import torch.nn as nn

from arc_it.models.encoder import FrozenEncoder
from arc_it.models.bridge import Bridge
from arc_it.models.sana_backbone import (
    TimestepEmbedder,
    LinearAttention,
    CrossAttention,
    MixFFN,
    SanaBlock,
    SanaBackbone,
)
from arc_it.models.decoder import SpatialDecoder
from arc_it.models.diffusion import DiffusionScheduler
from arc_it.models.arc_it_model import ARCITModel, OutputEmbedder


B = 2    # batch size for tests
N = 256  # num patches (16x16)
C = 1152 # Sana hidden size


# ─── Encoder Tests ──────────────────────────────────────────────────

class TestEncoder:
    def test_stub_encoder_shape(self):
        enc = FrozenEncoder(encoder_name="stub", embed_dim=1024, num_patches=256)
        x = torch.randn(B, 3, 224, 224)
        out = enc(x)
        assert out.shape == (B, 256, 1024)

    def test_stub_encoder_frozen(self):
        enc = FrozenEncoder(encoder_name="stub", embed_dim=1024)
        for param in enc.parameters():
            assert not param.requires_grad

    def test_stub_encoder_no_grad(self):
        enc = FrozenEncoder(encoder_name="stub", embed_dim=1024)
        x = torch.randn(B, 3, 224, 224)
        out = enc(x)
        assert not out.requires_grad

    def test_stub_encoder_eval_mode_persists(self):
        enc = FrozenEncoder(encoder_name="stub", embed_dim=1024)
        enc.train()
        assert not enc.encoder.training


# ─── Bridge Tests ───────────────────────────────────────────────────

class TestBridge:
    def test_shape(self):
        bridge = Bridge(encoder_dim=1024, hidden_dim=2048, sana_dim=C)
        x = torch.randn(B, N, 1024)
        out = bridge(x)
        assert out.shape == (B, N, C)

    def test_gradients_flow(self):
        bridge = Bridge(encoder_dim=1024, hidden_dim=2048, sana_dim=C)
        x = torch.randn(B, N, 1024)
        out = bridge(x)
        loss = out.sum()
        loss.backward()
        for param in bridge.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_pos_embed_shape(self):
        bridge = Bridge(encoder_dim=1024, sana_dim=C, num_patches=N, use_2d_pos_embed=True)
        assert bridge.pos_embed is not None
        assert bridge.pos_embed.shape == (1, N, C)

    def test_param_count(self):
        bridge = Bridge(encoder_dim=1024, hidden_dim=2048, sana_dim=C)
        n_params = sum(p.numel() for p in bridge.parameters())
        assert n_params > 0
        assert n_params < 20_000_000  # should be under 20M


# ─── Sana Component Tests ──────────────────────────────────────────

class TestTimestepEmbedder:
    def test_shape(self):
        emb = TimestepEmbedder(C)
        t = torch.tensor([0, 500, 999])
        out = emb(t)
        assert out.shape == (3, C)

    def test_different_timesteps_different_embeddings(self):
        emb = TimestepEmbedder(C)
        t = torch.tensor([0, 999])
        out = emb(t)
        assert not torch.allclose(out[0], out[1])


class TestLinearAttention:
    def test_shape(self):
        attn = LinearAttention(hidden_size=C, num_heads=36, head_dim=32)
        x = torch.randn(B, N, C)
        out = attn(x)
        assert out.shape == (B, N, C)

    def test_output_changes_with_input(self):
        attn = LinearAttention(hidden_size=C, num_heads=36, head_dim=32)
        x1 = torch.randn(B, N, C)
        x2 = torch.randn(B, N, C)
        assert not torch.allclose(attn(x1), attn(x2))


class TestCrossAttention:
    def test_shape(self):
        ca = CrossAttention(hidden_size=C, num_heads=16)
        x = torch.randn(B, N, C)
        cond = torch.randn(B, N, C)
        out = ca(x, cond)
        assert out.shape == (B, N, C)

    def test_different_conditioning(self):
        ca = CrossAttention(hidden_size=C, num_heads=16)
        x = torch.randn(B, N, C)
        cond1 = torch.randn(B, N, C)
        cond2 = torch.randn(B, N, C)
        out1 = ca(x, cond1)
        out2 = ca(x, cond2)
        assert not torch.allclose(out1, out2)


class TestMixFFN:
    def test_shape(self):
        ffn = MixFFN(hidden_size=C, mlp_ratio=2.5)
        x = torch.randn(B, N, C)
        out = ffn(x)
        assert out.shape == (B, N, C)


class TestSanaBlock:
    def test_shape(self):
        block = SanaBlock(hidden_size=C, self_attn_heads=36, self_attn_head_dim=32)
        x = torch.randn(B, N, C)
        cond = torch.randn(B, N, C)
        t_emb = torch.randn(B, C)
        out = block(x, cond, t_emb)
        assert out.shape == (B, N, C)

    def test_gradients_flow(self):
        block = SanaBlock(hidden_size=C, self_attn_heads=36, self_attn_head_dim=32)
        x = torch.randn(B, N, C, requires_grad=True)
        cond = torch.randn(B, N, C)
        t_emb = torch.randn(B, C)
        out = block(x, cond, t_emb)
        out.sum().backward()
        assert x.grad is not None


class TestSanaBackbone:
    def test_shape_small(self):
        """Test with small depth for speed."""
        backbone = SanaBackbone(hidden_size=C, depth=2, num_patches=N)
        x = torch.randn(B, N, C)
        cond = torch.randn(B, N, C)
        t = torch.tensor([0, 500])
        out = backbone(x, cond, t)
        assert out.shape == (B, N, C)

    def test_depth_configurable(self):
        for depth in [1, 4]:
            backbone = SanaBackbone(hidden_size=C, depth=depth, num_patches=N)
            assert len(backbone.blocks) == depth


# ─── Decoder Tests ──────────────────────────────────────────────────

class TestSpatialDecoder:
    def test_shape(self):
        dec = SpatialDecoder(hidden_size=C, num_patches=N, canvas_size=64, num_colors=12)
        x = torch.randn(B, N, C)
        out = dec(x)
        assert out.shape == (B, 12, 64, 64)

    def test_argmax_gives_valid_colors(self):
        dec = SpatialDecoder(hidden_size=C, num_patches=N, canvas_size=64, num_colors=12)
        x = torch.randn(B, N, C)
        out = dec(x)
        pred = out.argmax(dim=1)
        assert pred.min() >= 0
        assert pred.max() < 12

    def test_gradients_flow(self):
        dec = SpatialDecoder(hidden_size=C, num_patches=N, canvas_size=64, num_colors=12)
        x = torch.randn(B, N, C, requires_grad=True)
        out = dec(x)
        out.sum().backward()
        assert x.grad is not None


# ─── Diffusion Tests ────────────────────────────────────────────────

class TestDiffusionScheduler:
    def test_add_noise(self):
        sched = DiffusionScheduler(num_train_timesteps=1000)
        x_0 = torch.randn(B, N, C)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0, 999])
        noisy = sched.add_noise(x_0, noise, t)
        assert noisy.shape == x_0.shape

    def test_t0_is_mostly_signal(self):
        sched = DiffusionScheduler(num_train_timesteps=1000)
        x_0 = torch.ones(1, N, C)
        noise = torch.zeros(1, N, C)
        t = torch.tensor([0])
        noisy = sched.add_noise(x_0, noise, t)
        # At t=0, should be almost entirely x_0
        assert torch.allclose(noisy, x_0, atol=0.01)

    def test_t999_is_mostly_noise(self):
        sched = DiffusionScheduler(num_train_timesteps=1000)
        x_0 = torch.zeros(1, N, C)
        noise = torch.ones(1, N, C)
        t = torch.tensor([999])
        noisy = sched.add_noise(x_0, noise, t)
        # At t=999, should be mostly noise
        assert noisy.abs().mean() > 0.5

    def test_sample_timesteps(self):
        sched = DiffusionScheduler(num_train_timesteps=1000)
        t = sched.sample_timesteps(B, torch.device("cpu"))
        assert t.shape == (B,)
        assert t.min() >= 0
        assert t.max() < 1000


# ─── Output Embedder Tests ──────────────────────────────────────────

class TestOutputEmbedder:
    def test_shape(self):
        emb = OutputEmbedder(num_colors=12, canvas_size=64, hidden_size=C, patch_size=4)
        grid = torch.randint(0, 12, (B, 64, 64))
        out = emb(grid)
        expected_patches = (64 // 4) ** 2  # 256
        assert out.shape == (B, expected_patches, C)


# ─── Full Model Integration ─────────────────────────────────────────

class TestARCITModel:
    @pytest.fixture
    def model(self):
        """Create a small model for testing (2 Sana layers instead of 28)."""
        return ARCITModel(
            encoder_name="stub",
            encoder_dim=1024,
            encoder_pretrained=False,
            sana_hidden=256,      # Small for testing
            sana_depth=2,         # 2 layers instead of 28
            sana_self_attn_heads=8,
            sana_self_attn_head_dim=32,
            sana_cross_attn_heads=4,
            sana_mlp_ratio=2.0,
            canvas_size=64,
            num_colors=12,
            decoder_channels=(128, 64),
            num_train_timesteps=100,
            num_inference_steps=5,
            output_patch_size=4,
        )

    def test_training_forward(self, model):
        input_rgb = torch.randn(B, 3, 224, 224)
        target = torch.randint(0, 10, (B, 64, 64))
        result = model(input_rgb, target=target)
        assert "loss" in result
        assert "pixel_accuracy" in result
        assert "logits" in result
        assert result["logits"].shape == (B, 12, 64, 64)
        assert result["loss"].ndim == 0  # scalar

    def test_training_loss_is_finite(self, model):
        input_rgb = torch.randn(B, 3, 224, 224)
        target = torch.randint(0, 10, (B, 64, 64))
        result = model(input_rgb, target=target)
        assert torch.isfinite(result["loss"])

    def test_training_backward(self, model):
        input_rgb = torch.randn(B, 3, 224, 224)
        target = torch.randint(0, 10, (B, 64, 64))
        result = model(input_rgb, target=target)
        result["loss"].backward()
        # Check that trainable params got gradients
        trainable_params = model.get_trainable_params()
        assert len(trainable_params) > 0
        grads_ok = sum(1 for p in trainable_params if p.grad is not None)
        assert grads_ok > 0

    def test_inference_forward(self, model):
        model.eval()
        input_rgb = torch.randn(B, 3, 224, 224)
        result = model(input_rgb, target=None)
        assert "prediction" in result
        assert "logits" in result
        assert result["prediction"].shape == (B, 64, 64)
        assert result["prediction"].dtype == torch.long
        assert result["prediction"].min() >= 0
        assert result["prediction"].max() < 12

    def test_difficulty_weighting(self, model):
        input_rgb = torch.randn(B, 3, 224, 224)
        target = torch.randint(0, 10, (B, 64, 64))

        # Without difficulty
        r1 = model(input_rgb, target=target)

        # With high difficulty weight
        difficulty = torch.tensor([10.0, 10.0])
        r2 = model(input_rgb, target=target, difficulty=difficulty)

        # Weighted loss should be higher
        assert r2["loss"] > r1["loss"]

    def test_encoder_frozen_after_training_step(self, model):
        input_rgb = torch.randn(B, 3, 224, 224)
        target = torch.randint(0, 10, (B, 64, 64))
        result = model(input_rgb, target=target)
        result["loss"].backward()
        for param in model.encoder.parameters():
            assert not param.requires_grad

    def test_param_count(self, model):
        counts = model.param_count()
        assert counts["encoder"]["trainable"] == 0
        assert counts["bridge"]["trainable"] > 0
        assert counts["sana"]["trainable"] > 0
        assert counts["decoder"]["trainable"] > 0
        assert counts["_total"]["trainable"] > 0
        assert counts["_total"]["trainable"] < counts["_total"]["total"]

    def test_from_config(self):
        config = {
            "data": {"canvas_size": 64, "num_colors": 12},
            "model": {
                "encoder": {"name": "stub", "embed_dim": 1024, "pretrained": False, "num_patches": 256},
                "bridge": {"hidden_dim": 512, "output_dim": 256, "dropout": 0.1, "use_2d_pos_embed": True},
                "sana": {
                    "hidden_size": 256, "depth": 2, "num_heads": 4,
                    "linear_head_dim": 32, "mlp_ratio": 2.0,
                    "attn_type": "linear", "ffn_type": "glumbconv",
                },
                "decoder": {"hidden_channels": [128, 64], "upsample_method": "transposed_conv"},
                "diffusion": {"num_train_timesteps": 100, "num_inference_steps": 5,
                              "noise_schedule": "linear", "prediction_type": "epsilon"},
            },
        }
        model = ARCITModel.from_config(config)
        assert model is not None
        input_rgb = torch.randn(1, 3, 224, 224)
        target = torch.randint(0, 10, (1, 64, 64))
        result = model(input_rgb, target=target)
        assert torch.isfinite(result["loss"])
