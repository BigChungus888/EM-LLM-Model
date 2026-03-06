import unittest
from unittest import mock

import torch
import torch.nn as nn

from em_llm.attention.hybrid_state import HybridLayerState
from em_llm.attention.mag_fusion import MAGFusion
from em_llm.attention import em_llm_ttt_mag as hybrid_mod


class _FakeContextManager:
    def __init__(self, *args, **kwargs):
        pass

    def append(self, local_q, local_k, local_v, global_q, global_k, global_v):
        return local_q


class _FakeAttn:
    def __init__(self):
        self.layer_idx = 0


class _FakeLinearSGD(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_qkv_callable(self, fn):
        self._qkv = fn

    def forward(self, x, past_key_values=None):
        return x, (torch.zeros(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype), None)


class _Block:
    def __init__(self, size):
        self.size = size


class _Episodic:
    def __init__(self):
        self.global_blocks = [[_Block(4), _Block(8)]]


class HybridRuntimeTests(unittest.TestCase):
    def test_mag_fusion_shape(self):
        fusion = MAGFusion(hidden_size=16)
        ep = torch.randn(2, 5, 16)
        ttt = torch.randn(2, 5, 16)
        out = fusion(ep, ttt)
        self.assertEqual(out.shape, ep.shape)

    def test_mag_fusion_baseline_safe_init_preserves_episodic_when_ttt_zero(self):
        fusion = MAGFusion(hidden_size=16, baseline_safe_init=True)
        ep = torch.randn(2, 5, 16)
        ttt = torch.zeros_like(ep)
        out = fusion(ep, ttt)
        expected = fusion.out_proj(fusion.ep_norm(ep))
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_hybrid_forward_disabled_returns_hybrid_state(self):
        with mock.patch.object(hybrid_mod, "ContextManager", _FakeContextManager):
            forward = hybrid_mod.em_llm_ttt_mag_attn_forward(
                model=None,
                n_local=8,
                n_init=2,
                max_block_size=4,
                max_cached_block=8,
                exc_block_size=4,
                ttt_mag={"enabled": False},
            )
            fake = _FakeAttn()
            q_proj = nn.Linear(16, 16, bias=False)
            k_proj = nn.Linear(16, 16, bias=False)
            v_proj = nn.Linear(16, 16, bias=False)
            o_proj = nn.Linear(16, 16, bias=False)
            x = torch.randn(1, 3, 16)

            out, _, state = forward(
                fake,
                x,
                x,
                position_bias=None,
                use_cache=True,
                past_key_value=None,
                project_q=q_proj,
                project_k=k_proj,
                project_v=v_proj,
                attention_out=o_proj,
                dim_head=4,
                num_heads=4,
                num_heads_kv=4,
            )
            self.assertEqual(out.shape, x.shape)
            self.assertIsInstance(state, HybridLayerState)

    def test_attach_hybrid_modules_fail_fast_is_deterministic(self):
        fake = _FakeAttn()
        with mock.patch.object(hybrid_mod, "_import_linear_sgd", side_effect=ImportError("missing")):
            with self.assertRaises(RuntimeError):
                hybrid_mod.attach_hybrid_modules(fake, hidden_size=16, num_heads=4, head_dim=4, ttt_cfg={})

    def test_attach_hybrid_modules_registers_modules(self):
        fake = _FakeAttn()
        with mock.patch.object(hybrid_mod, "_import_linear_sgd", return_value=_FakeLinearSGD):
            hybrid_mod.attach_hybrid_modules(fake, hidden_size=16, num_heads=4, head_dim=4, ttt_cfg={})
        self.assertTrue(hasattr(fake, "_em_ttt_linear_sgd"))
        self.assertTrue(hasattr(fake, "_em_ttt_mag_fusion"))

    def test_hybrid_layer_state_exposes_episodic_blocks_for_benchmark(self):
        state = HybridLayerState(episodic_state=_Episodic())
        episodic = state.episodic_state if isinstance(state, HybridLayerState) else state
        block_sizes = [block.size for block in episodic.global_blocks[0]]
        self.assertEqual(block_sizes, [4, 8])


if __name__ == "__main__":
    unittest.main()
