import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from em_llm.attention.hybrid_state import HybridLayerState, TitansHybridLayerState
from em_llm.attention.mag_fusion import MAGFusion
from em_llm.attention import em_llm_ttt_mag as hybrid_mod
from em_llm.utils.greedy_search import _get_episodic_state


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


class _RecordingLinearSGD(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.last_qkv_shapes = None

    def set_qkv_callable(self, fn):
        self._qkv = fn

    def forward(self, x, past_key_values=None):
        q, k, v = self._qkv(x)
        self.last_qkv_shapes = (q.shape, k.shape, v.shape)
        return x, (torch.zeros(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype), None)


class _FakeMemoryAsGate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs
        self.last_forward_kwargs = None

    def forward(self, x, **kwargs):
        self.last_forward_kwargs = kwargs
        return x, None, ("fla-state", None)


class _Block:
    def __init__(self, size):
        self.size = size


class _Episodic:
    def __init__(self):
        self.global_blocks = [[_Block(4), _Block(8)]]


class _OffloadableEpisodic:
    def __init__(self):
        self.allow_disk_offload = None
        self.vector_offload = True
        self.block_repr_k = [torch.zeros(1)]
        self.offloaded = False

    def _offload_vector(self):
        self.offloaded = True


class _RecordingContextManager:
    def __init__(self, *args, **kwargs):
        self.append_shapes = []
        self.update_calls = []
        self.uniform_blocks = False
        self.similarity_refinement = False
        self.global_blocks = [[_Block(4), _Block(8)]]
        self.batch_size = 1

    def append(self, local_q, local_k, local_v, global_q, global_k, global_v):
        self.append_shapes.append((local_q.shape, local_k.shape, local_v.shape))
        return local_q

    def update_memory(self, exc_length, surprisal, surprisal_values=None):
        self.update_calls.append((exc_length, surprisal, surprisal_values))


class _FakeTitansNativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.head_dim = 4
        self.forward_calls = []

    def get_qkv(self, hidden_states):
        return hidden_states, hidden_states, hidden_states

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False, **kwargs):
        self.forward_calls.append(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
            }
        )
        return hidden_states + 1, None, {"step": len(self.forward_calls)}


class _KwargIdentity(nn.Module):
    def forward(self, x, **kwargs):
        return x


class _FakeTitansBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = nn.Identity()
        self.attn = _FakeTitansNativeAttention()
        self.mlp_norm = nn.Identity()
        self.mlp = _KwargIdentity()


class _FakeTitansDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_FakeTitansBlock(config, layer_idx=0)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, _, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return hidden_states, past_key_values, all_hidden_states, None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class _FakeTitansLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16,
            num_heads=4,
            head_dim=4,
            vocab_size=32,
            fuse_norm=False,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            memory_model_config=SimpleNamespace(norm_eps=1e-5),
        )
        self.model = _FakeTitansDecoder(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits, outputs.past_key_values


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
        self.assertIs(fake._em_ttt_recurrent_module, fake._em_ttt_linear_sgd)
        self.assertTrue(fake._em_ttt_use_host_qkv)

    def test_attach_hybrid_modules_registers_fla_memory_as_gate_when_enabled(self):
        fake = _FakeAttn()
        with mock.patch.object(hybrid_mod, "_import_linear_sgd", return_value=_FakeLinearSGD):
            with mock.patch.object(hybrid_mod, "_import_memory_as_gate", return_value=_FakeMemoryAsGate):
                hybrid_mod.attach_hybrid_modules(
                    fake,
                    hidden_size=16,
                    num_heads=4,
                    head_dim=4,
                    ttt_cfg={"use_fla_attention": True, "window_size": 32},
                )
        self.assertIsInstance(fake._em_ttt_recurrent_module, _FakeMemoryAsGate)
        self.assertFalse(fake._em_ttt_use_host_qkv)
        self.assertTrue(hasattr(fake, "_em_ttt_linear_sgd"))

    def test_hybrid_forward_expands_gqa_kv_heads_for_recurrent_branch(self):
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
            fake._em_ttt_linear_sgd = _RecordingLinearSGD()
            fake._em_ttt_recurrent_module = fake._em_ttt_linear_sgd
            fake._em_ttt_use_host_qkv = True
            fake._em_ttt_mag_fusion = MAGFusion(hidden_size=16)
            q_proj = nn.Linear(16, 16, bias=False)
            k_proj = nn.Linear(16, 8, bias=False)
            v_proj = nn.Linear(16, 8, bias=False)
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
                num_heads_kv=2,
            )

            self.assertEqual(out.shape, x.shape)
            self.assertIsNotNone(state.recurrent_state)
            self.assertEqual(fake._em_ttt_linear_sgd.last_qkv_shapes, ((1, 3, 16), (1, 3, 16), (1, 3, 16)))

    def test_hybrid_layer_state_exposes_episodic_blocks_for_benchmark(self):
        state = HybridLayerState(episodic_state=_Episodic())
        episodic = state.episodic_state if isinstance(state, HybridLayerState) else state
        block_sizes = [block.size for block in episodic.global_blocks[0]]
        self.assertEqual(block_sizes, [4, 8])

    def test_get_episodic_state_unwraps_hybrid_layer_state(self):
        episodic = _OffloadableEpisodic()
        wrapped = HybridLayerState(episodic_state=episodic)
        self.assertIs(_get_episodic_state(wrapped), episodic)
        self.assertIs(_get_episodic_state(episodic), episodic)

    def test_hybrid_forward_supports_fla_attention_recurrent_module(self):
        with mock.patch.object(hybrid_mod, "ContextManager", _FakeContextManager):
            forward = hybrid_mod.em_llm_ttt_mag_attn_forward(
                model=None,
                n_local=8,
                n_init=2,
                max_block_size=4,
                max_cached_block=8,
                exc_block_size=4,
                ttt_mag={"enabled": True, "use_fla_attention": True},
            )
            fake = _FakeAttn()
            fake._em_ttt_recurrent_module = _FakeMemoryAsGate()
            fake._em_ttt_linear_sgd = _FakeLinearSGD()
            fake._em_ttt_use_host_qkv = False
            fake._em_ttt_mag_fusion = MAGFusion(hidden_size=16)
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
            self.assertEqual(state.recurrent_state, ("fla-state", None))
            self.assertEqual(fake._em_ttt_recurrent_module.last_forward_kwargs["use_cache"], True)


class TitansNativeRuntimeTests(unittest.TestCase):
    def _create_model(self):
        model = _FakeTitansLM()
        hybrid_mod.patch_titans_mag_model(
            model,
            n_local=8,
            n_init=2,
            max_block_size=4,
            max_cached_block=8,
            exc_block_size=4,
            n_mem=8,
            repr_topk=1,
            similarity_refinement_kwargs={},
            contiguity_buffer_kwargs={},
        )
        return model

    def test_titans_native_block_preserves_native_cache_and_episodic_state(self):
        with mock.patch.object(hybrid_mod, "ContextManager", _RecordingContextManager):
            model = self._create_model()
            block = model.model.layers[0]

            x = torch.randn(1, 3, 16)
            out, _, past = block(x, use_cache=True)
            first_state = past[0]
            first_episodic = first_state.episodic_state

            self.assertEqual(out.shape, x.shape)
            self.assertIsInstance(first_state, TitansHybridLayerState)
            self.assertEqual(block.attn.forward_calls[0]["past_key_values"], None)
            self.assertEqual(first_episodic.append_shapes[0], ((1, 4, 3, 4), (1, 4, 3, 4), (1, 4, 3, 4)))

            decode_x = torch.randn(1, 1, 16)
            decode_out, _, past = block(decode_x, past_key_values=past, use_cache=True)

            self.assertEqual(decode_out.shape, decode_x.shape)
            self.assertEqual(block.attn.forward_calls[1]["past_key_values"], {"step": 1})
            self.assertIs(past[0].episodic_state, first_episodic)
            self.assertEqual(past[0].titans_state, {"step": 2})
            self.assertEqual(len(first_episodic.append_shapes), 2)

    def test_titans_native_causal_lm_forward_updates_episodic_memory(self):
        with mock.patch.object(hybrid_mod, "ContextManager", _RecordingContextManager):
            with mock.patch.object(hybrid_mod.torch.cuda, "is_available", return_value=False):
                model = self._create_model()
                input_ids = torch.tensor([[1, 2, 3]])

                out = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    em_labels=input_ids,
                )

                self.assertEqual(out.logits.shape, (1, 3, model.config.vocab_size))
                self.assertIsInstance(out.past_key_values[0], TitansHybridLayerState)

                episodic = out.past_key_values[0].episodic_state
                self.assertEqual(len(episodic.update_calls), 1)
                exc_length, surprisal, surprisal_values = episodic.update_calls[0]
                self.assertEqual(exc_length, input_ids.size(1))
                self.assertEqual(surprisal.shape, input_ids.shape)
                self.assertIsNone(surprisal_values)

    def test_get_episodic_state_unwraps_titans_hybrid_layer_state(self):
        episodic = _Episodic()
        wrapped = TitansHybridLayerState(episodic_state=episodic, titans_state={"step": 1})

        self.assertIs(_get_episodic_state(wrapped), episodic)


if __name__ == "__main__":
    unittest.main()
