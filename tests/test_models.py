"""Unit tests for all ARC-IT Rule-Conditioned Transformer components.

Tests verify shapes, gradients, and end-to-end integration on CPU.
No pretrained encoder downloads needed.

Run with: python -m pytest tests/test_models.py -v
"""

import pytest
import torch

from arc_it.models.grid_tokenizer import GridTokenizer
from arc_it.models.rule_encoder import (
    StandardAttention,
    CrossAttention,
    FFN,
    PairEncoderBlock,
    AggregatorBlock,
    RuleEncoder,
)
from arc_it.models.rule_applier import RuleApplierBlock, RuleApplier
from arc_it.models.decoder import SpatialDecoder
from arc_it.models.arc_it_model import ARCITModel


B = 2    # batch size
N = 256  # num patches (16x16)
C = 128  # hidden size (small for tests)
K = 3    # demo pairs
M = 16   # rule tokens


# ─── GridTokenizer Tests ─────────────────────────────────────────────

class TestGridTokenizer:
    def test_shape(self):
        tok = GridTokenizer(num_colors=12, canvas_size=64, hidden_size=C, patch_size=4)
        grid = torch.randint(0, 12, (B, 64, 64))
        out = tok(grid)
        assert out.shape == (B, 256, C)

    def test_gradients_flow(self):
        tok = GridTokenizer(num_colors=12, canvas_size=64, hidden_size=C, patch_size=4)
        grid = torch.randint(0, 12, (B, 64, 64))
        out = tok(grid)
        out.sum().backward()
        for p in tok.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_inputs_different_outputs(self):
        tok = GridTokenizer(num_colors=12, canvas_size=64, hidden_size=C, patch_size=4)
        g1 = torch.zeros(1, 64, 64, dtype=torch.long)
        g2 = torch.ones(1, 64, 64, dtype=torch.long)
        assert not torch.allclose(tok(g1), tok(g2))


# ─── Attention Primitive Tests ───────────────────────────────────────

class TestStandardAttention:
    def test_shape(self):
        attn = StandardAttention(hidden_size=C, num_heads=4)
        x = torch.randn(B, N, C)
        out = attn(x)
        assert out.shape == (B, N, C)

    def test_masking(self):
        attn = StandardAttention(hidden_size=C, num_heads=4)
        x = torch.randn(B, 10, C)
        mask = torch.tensor([[True]*5 + [False]*5, [True]*8 + [False]*2])
        out = attn(x, mask=mask)
        assert out.shape == (B, 10, C)


class TestCrossAttention:
    def test_shape(self):
        ca = CrossAttention(hidden_size=C, num_heads=4)
        x = torch.randn(B, N, C)
        ctx = torch.randn(B, M, C)
        out = ca(x, ctx)
        assert out.shape == (B, N, C)

    def test_different_context(self):
        ca = CrossAttention(hidden_size=C, num_heads=4)
        x = torch.randn(B, N, C)
        c1 = torch.randn(B, M, C)
        c2 = torch.randn(B, M, C)
        assert not torch.allclose(ca(x, c1), ca(x, c2))


class TestFFN:
    def test_shape(self):
        ffn = FFN(hidden_size=C, mlp_ratio=2.5)
        x = torch.randn(B, N, C)
        assert ffn(x).shape == (B, N, C)


# ─── Rule Encoder Block Tests ───────────────────────────────────────

class TestPairEncoderBlock:
    def test_shape(self):
        block = PairEncoderBlock(hidden_size=C, num_heads=4)
        out_tok = torch.randn(B, N, C)
        in_tok = torch.randn(B, N, C)
        result = block(out_tok, in_tok)
        assert result.shape == (B, N, C)


class TestAggregatorBlock:
    def test_shape(self):
        block = AggregatorBlock(hidden_size=C, num_heads=4)
        x = torch.randn(B, K * N, C)
        out = block(x)
        assert out.shape == (B, K * N, C)

    def test_with_mask(self):
        block = AggregatorBlock(hidden_size=C, num_heads=4)
        x = torch.randn(B, 20, C)
        mask = torch.tensor([[True]*15 + [False]*5, [True]*20])
        out = block(x, mask=mask)
        assert out.shape == (B, 20, C)


# ─── Full Rule Encoder Tests ────────────────────────────────────────

class TestRuleEncoder:
    def test_shape(self):
        enc = RuleEncoder(
            hidden_size=C, num_pair_layers=1, num_agg_layers=1,
            num_heads=4, num_rule_tokens=M, max_demos=K, num_patches=N,
        )
        demo_in = torch.randn(B, K, N, C)
        demo_out = torch.randn(B, K, N, C)
        num_demos = torch.tensor([K, K])
        rule_tokens = enc(demo_in, demo_out, num_demos)
        assert rule_tokens.shape == (B, M, C)

    def test_variable_demos(self):
        enc = RuleEncoder(
            hidden_size=C, num_pair_layers=1, num_agg_layers=1,
            num_heads=4, num_rule_tokens=M, max_demos=5, num_patches=N,
        )
        demo_in = torch.randn(B, 5, N, C)
        demo_out = torch.randn(B, 5, N, C)
        num_demos = torch.tensor([2, 4])
        rule_tokens = enc(demo_in, demo_out, num_demos)
        assert rule_tokens.shape == (B, M, C)

    def test_gradients_flow(self):
        enc = RuleEncoder(
            hidden_size=C, num_pair_layers=1, num_agg_layers=1,
            num_heads=4, num_rule_tokens=M, max_demos=K, num_patches=N,
        )
        demo_in = torch.randn(B, K, N, C, requires_grad=True)
        demo_out = torch.randn(B, K, N, C, requires_grad=True)
        num_demos = torch.tensor([K, K])
        rule_tokens = enc(demo_in, demo_out, num_demos)
        rule_tokens.sum().backward()
        assert demo_in.grad is not None
        assert demo_out.grad is not None


# ─── Rule Applier Tests ─────────────────────────────────────────────

class TestRuleApplierBlock:
    def test_shape(self):
        block = RuleApplierBlock(hidden_size=C, num_heads=4)
        x = torch.randn(B, N, C)
        rules = torch.randn(B, M, C)
        out = block(x, rules)
        assert out.shape == (B, N, C)


class TestRuleApplier:
    def test_shape(self):
        applier = RuleApplier(
            hidden_size=C, num_layers=2, num_heads=4, num_patches=N,
        )
        test_tok = torch.randn(B, N, C)
        rules = torch.randn(B, M, C)
        out = applier(test_tok, rules)
        assert out.shape == (B, N, C)

    def test_different_rules_different_output(self):
        applier = RuleApplier(hidden_size=C, num_layers=2, num_heads=4)
        test_tok = torch.randn(B, N, C)
        r1 = torch.randn(B, M, C)
        r2 = torch.randn(B, M, C)
        o1 = applier(test_tok, r1)
        o2 = applier(test_tok, r2)
        assert not torch.allclose(o1, o2)


# ─── Decoder Tests ──────────────────────────────────────────────────

class TestSpatialDecoder:
    def test_shape(self):
        dec = SpatialDecoder(hidden_size=C, num_patches=N, canvas_size=64, num_colors=12,
                             hidden_channels=(64, 32))
        x = torch.randn(B, N, C)
        out = dec(x)
        assert out.shape == (B, 12, 64, 64)


# ─── Full Model Integration ─────────────────────────────────────────

class TestARCITModel:
    @pytest.fixture
    def model(self):
        """Small model for testing."""
        return ARCITModel(
            num_colors=12,
            canvas_size=64,
            patch_size=4,
            hidden_size=C,
            rule_encoder_pair_layers=1,
            rule_encoder_agg_layers=1,
            rule_encoder_heads=4,
            num_rule_tokens=M,
            max_demos=5,
            rule_applier_layers=1,
            rule_applier_heads=4,
            decoder_channels=(64, 32),
            mlp_ratio=2.0,
        )

    def _make_batch(self, num_demos=3, max_demos=5):
        di = torch.randint(0, 12, (B, max_demos, 64, 64))
        do = torch.randint(0, 12, (B, max_demos, 64, 64))
        qi = torch.randint(0, 12, (B, 64, 64))
        nd = torch.tensor([num_demos] * B)
        target = torch.randint(0, 10, (B, 64, 64))
        return di, do, qi, nd, target

    def test_training_forward(self, model):
        di, do, qi, nd, target = self._make_batch()
        result = model(di, do, qi, nd, target=target)
        assert "loss" in result
        assert "pixel_accuracy" in result
        assert "logits" in result
        assert result["logits"].shape == (B, 12, 64, 64)
        assert result["loss"].ndim == 0

    def test_training_loss_is_finite(self, model):
        di, do, qi, nd, target = self._make_batch()
        result = model(di, do, qi, nd, target=target)
        assert torch.isfinite(result["loss"])

    def test_training_backward(self, model):
        di, do, qi, nd, target = self._make_batch()
        result = model(di, do, qi, nd, target=target)
        result["loss"].backward()
        trainable = model.get_trainable_params()
        assert len(trainable) > 0
        grads_ok = sum(1 for p in trainable if p.grad is not None)
        assert grads_ok > 0

    def test_inference_forward(self, model):
        model.eval()
        di, do, qi, nd, _ = self._make_batch()
        result = model(di, do, qi, nd)
        assert "prediction" in result
        assert "logits" in result
        assert result["prediction"].shape == (B, 64, 64)
        assert result["prediction"].dtype == torch.long
        assert result["prediction"].min() >= 0
        assert result["prediction"].max() < 12

    def test_difficulty_weighting(self, model):
        di, do, qi, nd, target = self._make_batch()
        r1 = model(di, do, qi, nd, target=target)
        difficulty = torch.tensor([10.0, 10.0])
        r2 = model(di, do, qi, nd, target=target, difficulty=difficulty)
        assert r2["loss"] > r1["loss"]

    def test_variable_num_demos(self, model):
        di, do, qi, _, target = self._make_batch(max_demos=5)
        nd = torch.tensor([1, 4])
        result = model(di, do, qi, nd, target=target)
        assert torch.isfinite(result["loss"])

    def test_param_count(self, model):
        counts = model.param_count()
        assert counts["tokenizer"]["trainable"] > 0
        assert counts["rule_encoder"]["trainable"] > 0
        assert counts["rule_applier"]["trainable"] > 0
        assert counts["decoder"]["trainable"] > 0
        # All params should be trainable (no frozen encoder)
        assert counts["_total"]["trainable"] == counts["_total"]["total"]

    def test_from_config(self):
        config = {
            "data": {"canvas_size": 64, "num_colors": 12, "max_demos": 5},
            "model": {
                "hidden_size": 128,
                "mlp_ratio": 2.0,
                "tokenizer": {"patch_size": 4},
                "rule_encoder": {
                    "pair_layers": 1, "agg_layers": 1,
                    "num_heads": 4, "num_rule_tokens": 16,
                },
                "rule_applier": {"num_layers": 1, "num_heads": 4},
                "decoder": {"hidden_channels": [64, 32]},
            },
        }
        model = ARCITModel.from_config(config)
        assert model is not None
        di = torch.randint(0, 12, (1, 5, 64, 64))
        do = torch.randint(0, 12, (1, 5, 64, 64))
        qi = torch.randint(0, 12, (1, 64, 64))
        nd = torch.tensor([2])
        target = torch.randint(0, 10, (1, 64, 64))
        result = model(di, do, qi, nd, target=target)
        assert torch.isfinite(result["loss"])
