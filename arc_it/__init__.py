"""
ARC-IT: Rule-Conditioned Transformer for ARC-AGI Benchmarks

A novel architecture that explicitly extracts transformation rules from
demonstration pairs via paired cross-attention, then applies those rules
to new inputs. This mirrors how humans solve ARC: look at what changed
across examples, then apply that pattern.

Key components:
    - GridTokenizer: Discrete grid → continuous patch tokens
    - RuleEncoder: Demo pairs → rule tokens (paired cross-attention)
    - RuleApplier: Test input × rule tokens → output tokens
    - SpatialDecoder: Output tokens → 64x64 grid logits
"""

__version__ = "0.2.0"
