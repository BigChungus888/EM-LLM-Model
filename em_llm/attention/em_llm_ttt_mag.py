import torch

from typing import Optional, List, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from .context_manager import ContextManager
from .hybrid_state import HybridLayerState
from .mag_fusion import MAGFusion
from .similarity_refinement import events_with_similarity_adjustment


def _project_qkv(query, key_value, project_q, project_k, project_v, dim_head, num_heads, num_heads_kv):
    batch_size = query.size(0)
    len_q = query.size(1)
    len_k = key_value.size(1)

    if project_k is not None:
        h_q = project_q(query)
        h_k = project_k(key_value)
        h_v = project_v(key_value)
    else:
        qkv = project_q(query)
        query_pos = num_heads * dim_head
        h_q = qkv[..., :query_pos]
        h_k = qkv[..., query_pos: query_pos + num_heads_kv * dim_head]
        h_v = qkv[..., query_pos + num_heads_kv * dim_head:]

    h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()
    h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
    h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
    return h_q, h_k, h_v


def _import_linear_sgd():
    from .vendor_ttt_mag import LinearSGD

    return LinearSGD


def attach_hybrid_modules(attn_module, hidden_size, num_heads, head_dim, ttt_cfg):
    ttt_cfg = ttt_cfg or {}

    if not hasattr(attn_module, "_em_ttt_linear_sgd"):
        try:
            LinearSGD = _import_linear_sgd()
        except Exception as exc:
            raise RuntimeError(
                "em-llm-ttt-mag requires vendored TTT dependencies; "
                "failed to import em_llm.attention.vendor_ttt_mag.LinearSGD."
            ) from exc

        attn_module._em_ttt_linear_sgd = LinearSGD(
            hidden_size=hidden_size,
            num_heads=ttt_cfg.get("num_heads", num_heads),
            head_dim=ttt_cfg.get("head_dim", head_dim),
            expand_v=ttt_cfg.get("expand_v", 1),
            chunk_size=ttt_cfg.get("chunk_size", 64),
            base_lr=ttt_cfg.get("base_lr", 1.0),
            norm_eps=ttt_cfg.get("norm_eps", 1e-5),
            use_gate=ttt_cfg.get("use_gate", True),
            mode=ttt_cfg.get("mode", "chunk"),
            layer_idx=getattr(attn_module, "layer_idx", None),
        )

    if not hasattr(attn_module, "_em_ttt_mag_fusion"):
        attn_module._em_ttt_mag_fusion = MAGFusion(
            hidden_size=hidden_size,
            norm_eps=ttt_cfg.get("norm_eps", 1e-5),
            baseline_safe_init=ttt_cfg.get("baseline_safe_init", True),
        )


def em_llm_ttt_mag_attn_forward(
    model,
    n_local,
    n_init,
    max_block_size,
    max_cached_block,
    exc_block_size,
    repr_topk: int = 1,
    surprisal_threshold_gamma: float = 1.1,
    async_global_stream=True,
    pin_memory=False,
    perhead=False,
    n_mem: int = 2048,
    min_block_size: int = 1,
    block_similarity_topk: bool = False,
    similarity_refinement_kwargs: dict | None = None,
    contiguity_buffer_kwargs: dict | None = None,
    random_topk_blocks=False,
    infini_attention=False,
    uniform_blocks=False,
    ttt_mag: dict | None = None,
    *args,
    **kwargs,
):
    ttt_cfg = ttt_mag or {}
    similarity_refinement_kwargs = similarity_refinement_kwargs or {}
    contiguity_buffer_kwargs = contiguity_buffer_kwargs or {}

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        use_cache: bool,
        past_key_value,
        project_q,
        project_k,
        project_v,
        attention_out,
        dim_head,
        num_heads,
        num_heads_kv,
    ):
        assert use_cache

        batch_size = query.size(0)
        len_q = query.size(1)

        h_q, h_k, h_v = _project_qkv(
            query, key_value, project_q, project_k, project_v, dim_head, num_heads, num_heads_kv
        )

        if past_key_value is None:
            context_manager = ContextManager(
                layer_idx=self.layer_idx,
                position_embedding=position_bias,
                n_init=n_init,
                n_local=n_local,
                max_block_size=max_block_size,
                max_cached_block=max_cached_block,
                exc_block_size=exc_block_size,
                min_block_size=min_block_size,
                async_global_stream=async_global_stream,
                pin_memory=pin_memory,
                perhead=perhead,
                repr_topk=repr_topk,
                surprisal_threshold_gamma=surprisal_threshold_gamma,
                n_mem=n_mem,
                block_similarity_topk=block_similarity_topk,
                uniform_blocks=uniform_blocks,
                random_topk_blocks=random_topk_blocks,
                infini_attention=infini_attention,
                **similarity_refinement_kwargs,
                **contiguity_buffer_kwargs,
                **kwargs,
            )
            layer_state = HybridLayerState(episodic_state=context_manager)
        elif isinstance(past_key_value, HybridLayerState):
            layer_state = past_key_value
        else:
            layer_state = HybridLayerState(episodic_state=past_key_value)

        episodic_out = layer_state.episodic_state.append(h_q, h_k, h_v, h_q, h_k, h_v)
        episodic_out = episodic_out.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        episodic_out = episodic_out.reshape(batch_size, len_q, dim_head * num_heads)
        episodic_out = attention_out(episodic_out)

        if not ttt_cfg.get("enabled", True):
            return episodic_out, None, layer_state

        if not hasattr(self, "_em_ttt_linear_sgd") or not hasattr(self, "_em_ttt_mag_fusion"):
            raise RuntimeError(
                "Hybrid modules are not attached. Ensure patch_hf() calls attach_hybrid_modules for em-llm-ttt-mag."
            )

        recurrent_module = self._em_ttt_linear_sgd.to(device=query.device, dtype=query.dtype)

        def qkv_fn(x):
            if project_k is not None:
                q = project_q(x)
                k = project_k(x)
                v = project_v(x)
            else:
                qkv = project_q(x)
                q_pos = num_heads * dim_head
                q = qkv[..., :q_pos]
                k = qkv[..., q_pos:q_pos + num_heads_kv * dim_head]
                v = qkv[..., q_pos + num_heads_kv * dim_head:]
            return q, k, v

        recurrent_module.set_qkv_callable(qkv_fn)
        ttt_out, recurrent_state = recurrent_module(query, past_key_values=layer_state.recurrent_state)

        fusion_module = self._em_ttt_mag_fusion.to(device=query.device, dtype=query.dtype)
        fused_out = fusion_module(episodic_out, ttt_out)
        layer_state.recurrent_state = recurrent_state

        return fused_out, None, layer_state

    return forward


def em_llm_ttt_mag_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    em_labels: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    past_key_values = outputs[1]

    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        if labels.shape[-1] != logits.shape[-2]:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

    if em_labels is not None:
        if em_labels.dtype != torch.bool:
            prob = torch.softmax(logits, dim=-1)
            surprisal = -torch.log(torch.gather(prob, dim=-1, index=em_labels.unsqueeze(-1))).squeeze(-1)
        else:
            surprisal = em_labels
    else:
        surprisal = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    surprisal_values = None
    first_state = past_key_values[0]
    first_pkv = first_state.episodic_state if isinstance(first_state, HybridLayerState) else first_state

    if surprisal is not None and em_labels.dtype != torch.bool and not first_pkv.uniform_blocks:
        assert surprisal.shape == (first_pkv.batch_size, input_ids.shape[-1])
        if first_pkv.similarity_refinement:
            exc_length = input_ids.shape[-1]
            global_remainder_ed = first_pkv._global_remainder_ed + exc_length
            global_remainder_len = global_remainder_ed - first_pkv._global_remainder_st

            if global_remainder_ed <= exc_length:
                divide = surprisal > first_pkv.surprisal_threshold_gamma * torch.std(surprisal, dim=-1) + torch.mean(surprisal, dim=-1)
            else:
                avg_st = max(global_remainder_ed - first_pkv.n_local, 0)
                avg_ed = max(global_remainder_ed - exc_length, first_pkv.n_init)
                divide = surprisal > first_pkv.surprisal_threshold_gamma * torch.std(first_pkv.global_remainder_surprisal[:, avg_st:avg_ed], dim=-1) + torch.mean(first_pkv.global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
            if global_remainder_len >= 2 * exc_length and first_pkv.refine_with_buffer:
                last_divide = torch.zeros(first_pkv.batch_size)
                for u in range(first_pkv.batch_size):
                    last_events = torch.where(first_pkv.global_block_divide[u, global_remainder_ed - 2 * exc_length:global_remainder_ed - exc_length] > 0)[0]
                    if len(last_events) == 0:
                        last_divide[u] = exc_length
                    else:
                        last_divide[u] = last_events[-1]
                offsets = exc_length - last_divide.int()
                max_offset = torch.max(offsets)
            else:
                offsets = torch.zeros(first_pkv.batch_size).int()
                max_offset = 0
            st_layer = first_pkv.refine_from_layer
            assert len(past_key_values) >= st_layer + 1

            def _episodic(i):
                s = past_key_values[i]
                return s.episodic_state if isinstance(s, HybridLayerState) else s

            K = torch.clone(_episodic(st_layer).global_remainder[0][:, :, global_remainder_ed - exc_length - max_offset:global_remainder_ed, :]).unsqueeze(dim=1).to(torch.cuda.current_device())
            for l in range(st_layer + 1, len(past_key_values)):
                K = torch.cat((K, torch.clone(_episodic(l).global_remainder[0][:, :, global_remainder_ed - exc_length - max_offset:global_remainder_ed, :]).unsqueeze(dim=1).to(torch.cuda.current_device())), dim=1)
            stacked_A = torch.einsum('blhtd,blhTd->btT', K, K).detach()
            del K
            try:
                stacked_A = stacked_A.to(first_pkv.global_remainder[0].device)
            except Exception:
                pass
            for u in range(first_pkv.batch_size):
                events_sur = torch.where(divide[u] > 0)[0]
                if len(events_sur) > 0:
                    events_sur_mod = events_with_similarity_adjustment(
                        events_sur,
                        stacked_A[u][max_offset - offsets[u]:, :][:, max_offset - offsets[u]:],
                        similarity_metric=first_pkv.similarity_metric,
                        min_size=first_pkv.min_block_size,
                        offset=offsets[u],
                    )
                    divide[u] = torch.zeros_like(divide[u])
                    divide[u][events_sur_mod] = True
            del stacked_A
            surprisal_values = torch.clone(surprisal)
            surprisal = divide

    for pkv in past_key_values:
        episodic = pkv.episodic_state if isinstance(pkv, HybridLayerState) else pkv
        episodic.update_memory(input_ids.shape[-1], surprisal, surprisal_values=surprisal_values)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
