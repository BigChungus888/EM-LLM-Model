# Rebase `ttt-mag` on `TitansMAGForCausalLM`

## Summary
Replace the current HF-patched meaning of `model.type: ttt-mag` with a direct `TitansMAGForCausalLM` runtime. The new `ttt-mag` path loads a TitansMAG checkpoint as the only model, keeps TitansMAG’s native `MemoryAsGate` / local FLA attention branch unchanged, and layers EM-LLM episodic retrieval on top of each TitansMAG block. First implementation targets single-GPU loading only.

## Implementation Changes
- **Loader/runtime**
  - Change `benchmark/pred.py` so `model.type: ttt-mag` no longer goes through `AutoModelForCausalLM + patch_hf`.
  - Add a TitansMAG-specific loader branch that imports the donor package, loads `TitansMAGForCausalLM` directly from `model.path`, and uses `tokenizer_path` if provided.
  - Keep the existing HF patch path only for `model.type: em-llm`; do not keep the old HF-based hybrid under `ttt-mag`.
  - Document in config/comments that `ttt-mag` is now TitansMAG-native and single-GPU only for v1.

- **TitansMAG + EM-LLM integration**
  - Add a TitansMAG-specific hybrid forward path by wrapping or patching `TitansMAGBlock.forward` instead of HF `self_attn`.
  - Keep the block’s native `self.attn` call intact so TitansMAG’s local FLA attention and recurrent cache semantics remain the source of truth.
  - Add EM-LLM episodic state alongside each block’s native TitansMAG layer state using a wrapper cache object similar to `HybridLayerState`, but adapted for TitansMAG block state.
  - After the block’s native attention output is produced, compute an EM-LLM episodic output from the same normalized hidden states and fuse the two before the MLP/residual path.
  - Do not replace `ContextManager`; adapt the TitansMAG block wrapper so `ContextManager.append(...)` receives the required `q/k/v` tensors and its state is updated after the LM forward from `em_labels` / surprisal, mirroring current EM-LLM behavior.

- **Config/API**
  - Keep public selection as `model.type: ttt-mag`.
  - Update `config/ttt_mag.yaml` to target a TitansMAG checkpoint and include only fields needed by the new native loader plus EM-LLM retrieval settings.
  - Treat TitansMAG config as the source of truth for hidden size, heads, chunk size, local attention settings, and recurrent-memory parameters; do not duplicate those under a nested `ttt_mag` override block unless a field is truly runtime-only.
  - Preserve EM-LLM runtime fields like `n_local`, `n_mem`, `repr_topk`, `max_cached_block`, segmentation/refinement flags, and offload settings where they still apply to episodic memory.

- **Generation/search**
  - Update `GreedySearch` and any benchmark assumptions so `ttt-mag` accepts TitansMAG-native `past_key_values` plus episodic state, rather than HF-patched layer caches.
  - Keep greedy chunked ingestion and token-by-token decode behavior, but use TitansMAG’s own model forward/generation contract instead of `patch_hf`.
  - Ensure `return_block_size` and similar reporting unwrap episodic state from the new cache wrapper.

## Test Plan
- Add unit tests for direct `ttt-mag` loader selection in `benchmark/pred.py`, confirming TitansMAG path is chosen and `patch_hf` is not called.
- Add block-level hybrid tests that verify:
  - TitansMAG native attention/recurrent state still runs,
  - EM episodic state is created and carried across calls,
  - fused output shape and cache shape are stable across prefill and decode.
- Add causal-LM tests for:
  - chunked prompt continuation with mixed TitansMAG + episodic cache,
  - `em_labels` / surprisal updating episodic memory after forward,
  - `return_block_size` and benchmark reporting on the new wrapper state.
- Add a regression test that fails if `ttt-mag` accidentally routes back through the HF patch path.
- Acceptance check: `model.type: ttt-mag` loads a TitansMAG checkpoint on a single GPU, performs chunked generation, and updates episodic memory without loading Llama/Mistral/Qwen/Phi classes.

## Assumptions
- `ttt-mag` is repurposed to mean TitansMAG-native runtime; the old HF-patched hybrid meaning is retired.
- First implementation is single-GPU only; no `use_hf_acc` / multi-GPU dispatch parity in v1.
- TitansMAG’s native `MemoryAsGate` / FLA local attention remains untouched and is not replaced by EM-LLM’s local attention backend.
- A loadable TitansMAG checkpoint path will be provided via `model.path`; if checkpoint packaging requires import registration, the loader will explicitly import the donor package before loading.
