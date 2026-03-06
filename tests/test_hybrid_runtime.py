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
        # return [B, H, T, D]
        return local_q


class _FakeAttn:
    def __init__(self):
        self.layer_idx = 0


class HybridRuntimeTests(unittest.TestCase):
    def test_mag_fusion_shape(self):
        fusion = MAGFusion(hidden_size=16)
        ep = torch.randn(2, 5, 16)
        ttt = torch.randn(2, 5, 16)
        out = fusion(ep, ttt)
        self.assertEqual(out.shape, ep.shape)

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

    def test_hybrid_forward_fail_fast_when_vendor_missing(self):
        with mock.patch.object(hybrid_mod, "ContextManager", _FakeContextManager):
            forward = hybrid_mod.em_llm_ttt_mag_attn_forward(
                model=None,
                n_local=8,
                n_init=2,
                max_block_size=4,
                max_cached_block=8,
                exc_block_size=4,
                ttt_mag={"enabled": True},
            )
            fake = _FakeAttn()
            q_proj = nn.Linear(16, 16, bias=False)
            k_proj = nn.Linear(16, 16, bias=False)
            v_proj = nn.Linear(16, 16, bias=False)
            o_proj = nn.Linear(16, 16, bias=False)
            x = torch.randn(1, 3, 16)

            with self.assertRaises(RuntimeError):
                forward(
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


if __name__ == "__main__":
    unittest.main()
