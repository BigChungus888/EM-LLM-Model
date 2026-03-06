# TTT MAG 340M to EM-LLM Implementation Plan

## Goal

Integrate a TTT-backed MAG memory branch into EM-LLM with the smallest possible runtime disruption.

The primary goal is **not** to replace EM-LLM with the standalone Titans model stack. The primary goal is to:

1. keep EM-LLM's chunked generation, episodic block formation, retrieval, and offloading machinery,
2. add a recurrent TTT memory path plus MAG-style fusion inside the existing patched attention flow,
3. leave the current benchmark and inference path as intact as possible.

This is the lowest-churn path that still gives a real EM-LLM + TTT-MAG hybrid.

## Implementation Rule

Treat `sparse-linear-attention` as a **read-only donor repository**.

Implementation rule:

1. copy the minimal TTT/MAG code needed from `sparse-linear-attention` into EM-LLM,
2. apply all fixes and hybrid-specific changes only to the copied files inside EM-LLM,
3. leave the original `sparse-linear-attention` files unchanged,
4. keep a short provenance note in each copied file header so the source of the code is still clear.

This matches the goal of preserving the previous versions while giving EM-LLM its own isolated hybrid implementation surface.

## Recommendation

Use **EM-LLM as the host runtime** and copy only the reusable Titans/TTT components needed for memory and gating into the EM-LLM tree.

Do **not** try to plug `TitansMAGForCausalLM` directly into the current EM-LLM benchmark path in phase 1.

Reason:

1. `EM-LLM-model/em_llm/utils/patch_hf.py` is built to patch Hugging Face attention modules from Llama/Mistral/Qwen/Phi-style models.
2. `EM-LLM-model/em_llm/utils/greedy_search.py` currently assumes `model_type == "em-llm"` and a patched HF forward path.
3. `EM-LLM-model/em_llm/attention/context_manager.py` expects per-layer episodic-memory state, while Titans/TTT expects per-layer recurrent state in `past_key_values`.
4. The existing `sparse-linear-attention` Titans models are standalone causal LM implementations, not drop-in replacements for the current patch-based EM-LLM runtime.

So the practical path is:

- keep the current EM-LLM patching and retrieval pipeline,
- copy the minimal TTT/MAG implementation into EM-LLM,
- add a second recurrent state alongside each `ContextManager`,
- fuse EM-LLM attention output with TTT output using MAG-style gating.

## Important Constraint Found During Code Review

Before integrating TTT into EM-LLM, fix the linear recurrent-state carry-over bug found in the source implementation of:

- `sparse-linear-attention/sparse_linear_attention/modules/linear_sgd.py`

Current issue:

- `LinearSGD.forward(...)` accepts `past_key_values` but does not actually initialize from the incoming recurrent state.
- It always starts `dynamic_weights` from zeros inside `chunk_linear_rule(...)`.

Why this matters:

- EM-LLM processes long prompts in chunks and then decodes token by token.
- If the TTT linear state is not carried across calls, the recurrent memory resets every chunk/step.
- That would make the hybrid design incorrect at inference time even if the fusion code is otherwise correct.

This fix should be applied to the **copied EM-LLM version** of `linear_sgd.py`, not to the original source file.

This is phase 0 and should be treated as a prerequisite, not a nice-to-have.

## Minimal-Change Target Architecture

At each layer, maintain a hybrid cache object:

- `context_manager`: the existing EM-LLM episodic memory state
- `recurrent_state`: the TTT fast weights / dynamic weights
- `aux_stats`: optional recurrent-state metrics used for segmentation or retrieval experiments

At each chunk or decode step:

1. Run the current EM-LLM attention path exactly as today to produce retrieval-augmented attention output.
2. Run a TTT memory block on the same layer input hidden states.
3. Fuse the two outputs using MAG-style gating.
4. Return the fused hidden states and the updated hybrid cache.

Recommended fusion rule:

```text
em_out   = existing EM-LLM attention output
ttt_out  = TTT memory output
gate     = sigmoid(Wg(norm(em_out)))
fused    = Wo(norm(em_out) + gate * norm(ttt_out))
```

This reuses EM-LLM's attention branch as the MAG attention branch.

## Why This Is Better Than Directly Reusing Standalone TitansMAG

The shipped `titans_mag_linear_sgd_340M.json` sets `use_attention=false`, which means the standalone 340M config is memory-only in its current checked-in form.

For EM-LLM integration, that is actually helpful:

- EM-LLM already provides the retrieval-aware attention branch.
- Recomputing a second local attention branch inside Titans MAG would add cost and duplicate functionality.
- Reusing EM-LLM's output as the "attention side" of MAG is the correct adaptation if the goal is to extend EM-LLM rather than reproduce the standalone model byte-for-byte.

This means the recommended hybrid is:

- **TTT memory path from Titans/TTT**
- **attention branch from EM-LLM**
- **gate/fusion logic from MAG**

## Phase Plan

## Phase 0: Create an isolated vendored TTT/MAG subset

### Objective

Create a self-contained copy of the required TTT/MAG code inside EM-LLM so all future hybrid work happens there.

### Required changes

Create a new vendored subtree, for example:

- `EM-LLM-model/em_llm/attention/vendor_ttt_mag/`

Copy only the files needed for the linear 340M scope:

1. `sparse-linear-attention/sparse_linear_attention/modules/linear_sgd.py`
2. `sparse-linear-attention/sparse_linear_attention/layers/ttt.py`
3. `sparse-linear-attention/sparse_linear_attention/layers/titans_mag.py`
4. `sparse-linear-attention/sparse_linear_attention/ops/titans/utils.py`

Optional copy only if later needed:

1. `sparse-linear-attention/sparse_linear_attention/modules/mlp_sgd.py`

Rules for this phase:

1. do not edit anything under `sparse-linear-attention/`,
2. do not import the sibling repo dynamically in the final hybrid path,
3. do keep copied modules minimal and scoped to the hybrid.

### Required tests

Create EM-LLM-local tests rather than editing source-repo tests.

Suggested location:

- `EM-LLM-model/tests/test_ttt_mag_vendor.py`

Use `sparse-linear-attention/tests/layers/test_ttt.py` as reference, but do not modify it.

### Exit criteria

EM-LLM contains an isolated vendored TTT/MAG subset and no hybrid implementation work depends on editing the source repo.

## Phase 1: Fix copied `LinearSGD` cache semantics

### Objective

Make the copied `LinearSGD` behave like a true recurrent module across chunked calls.

### Required changes

1. Update the copied `linear_sgd.py` so `forward(...)` initializes from incoming recurrent state when available.
2. Change the copied chunk kernel wrapper to accept an initial weight tensor instead of hard-coding zeros.
3. Return the updated recurrent state exactly once per call.

### Required tests

Add tests beside the new EM-LLM vendored test file:

1. A cache round-trip test:
   one forward pass, then continue from returned state, and confirm it matches a single longer forward pass.
2. A chunked decode equivalence test:
   prompt split across multiple calls should match a one-shot pass within tolerance.

### Exit criteria

The copied TTT linear path preserves state across calls and remains numerically stable in both chunk and decode scenarios.

## Phase 2: Add a new EM-LLM hybrid attention type

### Objective

Add a new attention mode without disturbing the existing `"em-llm"` path.

### New files

1. `EM-LLM-model/em_llm/attention/em_llm_ttt_mag.py`
2. `EM-LLM-model/em_llm/attention/hybrid_state.py`

### Existing files to update

1. `EM-LLM-model/em_llm/attention/__init__.py`
2. `EM-LLM-model/em_llm/utils/patch_hf.py`
3. `EM-LLM-model/em_llm/utils/greedy_search.py`
4. `EM-LLM-model/benchmark/pred.py`

### Design

Define a hybrid per-layer cache such as:

```python
HybridLayerState(
    context_manager=ContextManager(...),
    recurrent_state=(dynamic_weights, aux_state),
    metrics={}
)
```

The new forward wrapper should:

1. keep the current Q/K/V projection logic from `em_llm_attn_forward(...)`,
2. initialize or reuse `HybridLayerState`,
3. call `ContextManager.append(...)` to get the EM-LLM branch output,
4. call a TTT memory module to get the recurrent branch output,
5. fuse the two branches with MAG gating,
6. return the fused output and updated hybrid state.

### Key implementation choice

Do not rewrite `ContextManager`.

Wrap it.

That keeps:

- block storage,
- top-k retrieval,
- offloading,
- contiguity buffer,
- repr-key scoring,
- chunk/update lifecycle

all untouched for the first integration milestone.

## Phase 3: Wire the copied TTT/MAG core into the hybrid path

### Objective

Use the vendored TTT/MAG subset from phase 0 rather than importing or editing the original repository.

### Recommended scope

Use the copied logic derived from:

1. `sparse-linear-attention/sparse_linear_attention/modules/linear_sgd.py`
2. `sparse-linear-attention/sparse_linear_attention/layers/ttt.py`
3. `sparse-linear-attention/sparse_linear_attention/layers/titans_mag.py`
4. `sparse-linear-attention/sparse_linear_attention/ops/titans/utils.py`

### Recommended packaging choice

For this project, prefer **vendoring the minimal subset into EM-LLM** rather than importing the sibling repository at runtime.

Reason:

1. The current EM-LLM package is self-contained.
2. Runtime imports from a sibling repo would introduce fragile path assumptions.
3. The benchmark launcher should not need a special `PYTHONPATH` setup just to run the hybrid.

Suggested destination:

- `EM-LLM-model/em_llm/attention/vendor_ttt_mag/linear_sgd.py`
- `EM-LLM-model/em_llm/attention/vendor_ttt_mag/ttt.py`
- `EM-LLM-model/em_llm/attention/vendor_ttt_mag/titans_mag.py`
- `EM-LLM-model/em_llm/attention/vendor_ttt_mag/titans_utils.py`
- `EM-LLM-model/em_llm/attention/mag_fusion.py`

The vendored code should be narrowed to:

- the recurrent memory update path,
- the projection/convolution wrapper if still needed,
- the gate and normalization logic.

Do not copy the full Titans model classes into EM-LLM for the first pass, and do not apply hybrid patches back to the original repo.

## Phase 4: Add config support

### Objective

Make the hybrid selectable from the same benchmark/config flow already used by EM-LLM.

### New config file

Add a new config under:

- `EM-LLM-model/config/titans_mag_linear_sgd_340M.yaml`

### Required config fields

Keep the current EM-LLM fields and add a new `ttt_mag` section such as:

```yaml
model:
  type: em-llm-ttt-mag
  ...
  ttt_mag:
    enabled: true
    hidden_size: 1024
    num_heads: 8
    head_dim: 128
    expand_v: 1
    chunk_size: 64
    conv_size: 4
    base_lr: 1.0
    norm_eps: 1.0e-5
    use_gate: true
    fuse_mode: mag
    recurrent_state_based_segmentation: false
    recurrent_state_based_retrieval: false
```

### Notes

1. Keep the original EM-LLM config shape intact where possible.
2. The hybrid should be selectable by `model.type` only, not by a separate launcher.
3. The original `"em-llm"` path must keep working unchanged.

## Phase 5: Keep phase 1 behavior baseline-safe

### Objective

Ensure the hybrid can be enabled gradually.

### Rules

1. If `ttt_mag.enabled == false`, behavior must reduce to the current EM-LLM path.
2. If the MAG gate projection is zero-initialized, the first run should stay close to baseline EM-LLM.
3. If recurrent-state-driven segmentation/retrieval flags are off, segmentation and retrieval should remain exactly EM-LLM-style.

### Why this matters

This gives a clean A/B comparison and avoids mixing architecture changes with memory-policy changes too early.

## Phase 6: Add recurrent-state-based segmentation

### Objective

Use TTT memory dynamics as an optional extra boundary signal, without replacing the existing surprisal pipeline.

### Minimal-change integration point

Extend the logic in:

- `EM-LLM-model/em_llm/attention/em_llm.py`
- `EM-LLM-model/em_llm/attention/context_manager.py`

### Recommended approach

Keep existing surprisal as the primary signal and add a second optional term based on recurrent-state change magnitude.

Example:

```text
boundary_score =
    zscore(existing_surprisal)
    + lambda_mem * zscore(recurrent_state_delta)
```

Do not hard-reset the TTT memory at boundaries.

Only use the signal to decide when an EM-LLM block should be finalized.

### Stored diagnostics

For each layer or a selected layer subset, store:

- recurrent-state delta norm
- optional prediction loss proxy
- optional EMA of recurrent-state surprise

For the first pass, use only one layer or a small subset of upper layers. Using all layers will add noise and complexity immediately.

## Phase 7: Add recurrent-state-based retrieval

### Objective

Steer EM-LLM's block ranking with a TTT-derived controller signal while keeping repr-key retrieval intact.

### Minimal-change integration point

Extend:

- `EM-LLM-model/em_llm/attention/context_manager.py`

Current ranking path:

- `block_repr_k[u].sort_by_similarity(global_q[u])`

Recommended hybrid ranking:

```text
score(B) =
    alpha * sim(global_query, repr_key_B)
    + beta * sim(recurrent_controller, event_repr_B)
    + rho * recency(B)
```

### Additional stored state per block

Add a compact event representation per block:

- mean pooled recurrent branch output over the block, or
- a projected summary of recurrent-state deltas across the block

This should be stored alongside the existing representative key vectors, not instead of them.

### Rule

Do not replace EM-LLM retrieval with recurrent retrieval in the first experimental version.

Use a blended score so the existing retrieval path remains a strong fallback.

## File-Level Implementation Map

## EM-LLM host changes

1. `EM-LLM-model/em_llm/attention/__init__.py`
   add a new `ATTN_FORWARD` entry such as `"em-llm-ttt-mag"`.
2. `EM-LLM-model/em_llm/utils/patch_hf.py`
   route the new attention type through the same patch-based HF integration path.
3. `EM-LLM-model/em_llm/attention/em_llm_ttt_mag.py`
   implement the new attention forward wrapper and, if needed, a hybrid LM forward wrapper.
4. `EM-LLM-model/em_llm/attention/hybrid_state.py`
   define the hybrid cache object.
5. `EM-LLM-model/em_llm/utils/greedy_search.py`
   allow the new model type to reuse the current generation path.
6. `EM-LLM-model/benchmark/pred.py`
   allow config-driven loading of the new hybrid type without changing the benchmark interface.

## Vendored TTT/MAG changes

1. `EM-LLM-model/em_llm/attention/vendor_ttt_mag/linear_sgd.py`
   copied from the donor repo, then patched for recurrent cache carry-over.
2. `EM-LLM-model/em_llm/attention/vendor_ttt_mag/ttt.py`
   copied from the donor repo, then trimmed or adapted only as needed.
3. `EM-LLM-model/em_llm/attention/vendor_ttt_mag/titans_mag.py`
   copied from the donor repo, then reused for MAG-style gating only.
4. `EM-LLM-model/em_llm/attention/vendor_ttt_mag/titans_utils.py`
   copied helper utilities needed by the gating path.
5. `EM-LLM-model/tests/test_ttt_mag_vendor.py`
   EM-LLM-local continuity and equivalence tests.

## Preservation rule

Do not edit these source files during implementation:

1. `sparse-linear-attention/sparse_linear_attention/modules/linear_sgd.py`
2. `sparse-linear-attention/sparse_linear_attention/layers/ttt.py`
3. `sparse-linear-attention/sparse_linear_attention/layers/titans_mag.py`
4. `sparse-linear-attention/sparse_linear_attention/ops/titans/utils.py`
5. `sparse-linear-attention/tests/layers/test_ttt.py`

They are references only.

## Validation Plan

## Unit tests

1. `LinearSGD` state carry-over works across repeated calls.
2. Hybrid cache object survives chunked prompt ingestion and single-token decoding.
3. Gate-disabled hybrid matches baseline EM-LLM within tolerance.

## Integration tests

1. Long prompt ingestion still creates memory blocks.
2. Retrieval still works with offloading enabled.
3. Single-token decode continues to update both episodic and recurrent states.

## Benchmark ladder

Run in this order:

1. Baseline EM-LLM
2. EM-LLM + TTT branch only
3. EM-LLM + TTT branch + MAG fusion
4. EM-LLM + MAG fusion + recurrent segmentation
5. EM-LLM + MAG fusion + recurrent segmentation + recurrent retrieval

Suggested datasets:

1. InfiniteBench tasks already used by EM-LLM
2. RULER / NIAH style retrieval tasks
3. WikiText-103 or another LM sanity benchmark for perplexity drift

## What Should Not Change in the First Pass

To keep the integration seamless, do not change these in phase 1:

1. `ContextManager` block storage format
2. EM-LLM offload behavior
3. benchmark prompt formatting
4. greedy decoding loop structure
5. Hugging Face patching model families already supported by EM-LLM

This keeps the blast radius small and makes failures easier to isolate.

## Exact 340M Parity: Separate Track

If the actual requirement is not just "TTT-MAG ideas inside EM-LLM" but strict compatibility with the standalone `titans_mag_linear_sgd_340M.json` model/checkpoint, that is a separate track.

That path requires:

1. a new model-loading branch in `EM-LLM-model/benchmark/pred.py`,
2. new generation/runtime support beyond the current `patch_hf(...)` path,
3. a dedicated adapter between Titans layer state and EM-LLM episodic memory state,
4. likely separate tokenizer/config handling.

This is not the seamless path.

Recommendation:

- complete the EM-LLM-hosted hybrid first,
- only then decide whether exact 340M checkpoint parity is worth the additional runtime fork.

## Suggested Execution Order

1. Copy the minimal TTT/MAG files into `EM-LLM-model/em_llm/attention/vendor_ttt_mag/`.
2. Fix the copied `LinearSGD` cache semantics.
3. Add the new hybrid attention type with gate disabled by default.
4. Prove that chunked generation still works end to end.
5. Enable MAG fusion.
6. Add recurrent-state segmentation as an experiment flag.
7. Add recurrent-state retrieval as an experiment flag.
8. Only after that, evaluate whether a standalone 340M parity path is still needed.

## Bottom Line

The cleanest implementation is:

- **EM-LLM stays the host**
- **TTT/MAG code is copied into EM-LLM first**
- **TTT provides the recurrent memory branch**
- **MAG provides the fusion rule**
- **EM-LLM retrieval stays the external episodic memory system**
- **the original `sparse-linear-attention` code remains untouched**

That gives a real hybrid with minimal changes, preserves the current EM-LLM runtime, and avoids an unnecessary full-model rewrite.
