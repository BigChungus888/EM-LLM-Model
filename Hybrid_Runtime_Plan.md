# Hybrid Runtime Plan: `em-llm-ttt-mag`

## Summary

Write this plan into a new markdown file at `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/HYBRID_MEMORY_MAG_RUNTIME_PLAN.md`.

Implement a new runtime path that explicitly combines, in one patched attention forward:

1. EM-LLM episodic retrieval output from `ContextManager`
2. vendored TTT recurrent output from `em_llm/attention/vendor_ttt_mag`
3. MAG-style fusion with a baseline-safe gate/output initialization

This is a new `model.type`, not a modification of baseline `em-llm`. Retrieval and segmentation stay exactly as they are in EM-LLM for this step. The hybrid runs in all patched attention layers. If vendored TTT dependencies are missing, the new path fails fast with a clear error; baseline EM-LLM remains usable.

## Key Changes

### 1. New hybrid runtime path

Add a new attention type and model type named `em-llm-ttt-mag`.

Behavior in each patched attention layer:

1. project the incoming hidden states exactly as the current EM-LLM path does
2. run the existing EM-LLM retrieval branch through `ContextManager.append(...)` to produce the episodic retrieval-aware attention output
3. run the vendored TTT recurrent branch on the same layer input hidden states, using the vendored `TTT` or vendored `LinearSGD` state path
4. fuse the two branch outputs with MAG-style gating
5. return the fused hidden states and a hybrid per-layer cache object

Keep the existing EM-LLM retrieval policy unchanged:
- same block creation
- same representative-key ranking
- same contiguity buffer
- same surprisal-driven update path

Do not add recurrent-state-driven retrieval or segmentation in this step.

### 2. New cache interface

Introduce a per-layer hybrid cache object, for example `HybridLayerState`, with these fields:

- `episodic_state`: the existing `ContextManager`
- `recurrent_state`: vendored TTT fast weights, shaped exactly like the vendored `LinearSGD` return payload
- `aux_state`: reserved field for future recurrent metrics or extra MAG data, default `None`

`patch_hf` and the patched attention forward should treat `past_key_values[i]` as a `HybridLayerState` for `em-llm-ttt-mag`, while keeping the existing `ContextManager` behavior unchanged for `em-llm`.

### 3. MAG fusion contract

Create a fusion module that takes:
- `episodic_out`
- `ttt_out`

and computes:
- normalized episodic branch
- normalized recurrent branch
- sigmoid gate from the episodic branch
- fused output projection

Use a baseline-safe initialization:
- initialize the fusion so the first hybrid behavior stays close to EM-LLM
- practical default: zero-init the gate projection output path or zero-init the final recurrent contribution path so the recurrent branch starts with minimal effect

The fusion contract should be fixed and explicit:
`fused = out_proj(norm_ep + gate(norm_ep) * norm_ttt)`

Do not let the recurrent branch bypass the fusion module.

### 4. File-level implementation decisions

Use these implementation entrypoints:

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/em_llm/attention/__init__.py`
  Register `em-llm-ttt-mag` in both `ATTN_FORWARD` and `CAUSAL_LM_FORWARD`.

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/em_llm/attention/em_llm_ttt_mag.py`
  Add the hybrid patched attention forward and a causal LM forward that reuses the current EM-LLM loss/surprisal behavior.

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/em_llm/attention/hybrid_state.py`
  Define `HybridLayerState`.

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/em_llm/attention/mag_fusion.py`
  Define the MAG fusion module with baseline-safe initialization.

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/em_llm/utils/patch_hf.py`
  Reuse the same patched HF attention path, but route the new attention type to the hybrid forward factory.

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/em_llm/utils/greedy_search.py`
  Accept both `em-llm` and `em-llm-ttt-mag`. Preserve current chunked ingestion and decode behavior.

- `/Users/jameschen/PycharmProjects/BachelorThesis/EM-LLM-model/benchmark/pred.py`
  Allow the new `model.type` to load through the same benchmark path.

Do not modify the donor repository under `sparse-linear-attention/`.

### 5. Dependency and import behavior

The new hybrid path may lazily import vendored TTT/MAG modules so baseline EM-LLM does not fail in environments without those dependencies.

Rules:
- baseline `em-llm` import path must remain unaffected
- selecting `em-llm-ttt-mag` without required runtime dependencies must raise a clear, immediate error
- no silent fallback to baseline EM-LLM
- no alternate simplified recurrent implementation in this step

## Test Plan

### Unit tests

Add tests for:

1. `HybridLayerState` creation and reuse across multiple forward calls
2. hybrid attention forward returning fused output plus hybrid cache
3. baseline-safe gate init making initial hybrid output numerically close to episodic-only output on the same inputs
4. vendored recurrent state continuing across chunked calls
5. fail-fast error path when hybrid runtime dependencies are unavailable

### Integration tests

Add or update a lightweight inference test so that:

1. `model.type: em-llm-ttt-mag` patches successfully
2. prompt chunking works with `past_key_values` carrying both episodic and recurrent state
3. single-token decode works after chunk ingestion
4. episodic retrieval is still exercised through `ContextManager.append(...)`
5. fused hidden states are produced in all layers, not only the first layer

### Acceptance scenarios

A correct implementation should satisfy all of these:

1. `em-llm` behavior is unchanged
2. `em-llm-ttt-mag` uses episodic retrieval and vendored TTT in the same forward pass
3. hybrid cache survives prompt chunking and decode continuation
4. no code under `sparse-linear-attention/` is modified
5. the new path fails fast, not silently, if required vendored runtime deps are missing

## Assumptions And Defaults

- Hybrid exposure: new `model.type` named `em-llm-ttt-mag`
- Layer coverage: all patched attention layers
- Retrieval scope: unchanged EM-LLM retrieval and segmentation for this implementation step
- Fusion init: baseline-safe
- Missing dependencies: fail fast only when the hybrid path is selected
- Vendored TTT source of truth: the copied files under `em_llm/attention/vendor_ttt_mag/`
- No recurrent-state-based retrieval steering in this step
- No recurrent-state-based segmentation changes in this step
