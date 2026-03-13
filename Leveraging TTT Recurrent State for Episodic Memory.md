# Leveraging TTT Recurrent State for Episodic Memory Management in EM-LLM

# Introduction

## Existing Works

### Test-Time Training ([TTT](https://arxiv.org/abs/2407.04620))

An RNN architecture that generalises linear attention by replacing the fixed update rule with a step of self-supervised gradient descent that trains an expressive recurrent state on the input sequence at test time.

### Episodic Memory LLM ([EM-LLM](https://arxiv.org/abs/2407.09450))

A human-inspired episodic memory framework designed to enable Large Language Models (LLMs) to handle infinite context by organising input sequences into coherent events based on surprise and retrieving them via a two-stage process of similarity and temporal contiguity.

### [Titans](https://arxiv.org/abs/2501.00663)

Titans has a recurrent neural memory state just like in TTT. It then integrates this recurrent memory into three different architectures, Memory as Context (MAC), Memory as Gate (MAG) and Memory as Layer (MAL).

## Motivation

LLMs still struggle with very long contexts because full attention is expensive and important information can be diluted or missed. Two common directions address this in different ways. External episodic memory banks store and retrieve past segments, while recurrent or hybrid models carry information forward in a compact recurrent state/fast weights. We aim to combine both by using a recurrent or linear-attention backbone together with an external episodic memory bank, and guiding memory formation and retrieval using the model’s hidden-state dynamics. In the end, we adopt the memory implementations from Titans to generate the final output.  

# Research Directions

Integrate an external episodic memory bank into a hybrid recurrent/linear-attention backbone, where episodic memory formation leverages the model’s recurrent state.

## Hidden-State-Based Memory Segmentation

Investigates the feasibility of TTT’s gradient-based surprise as an alternative online event-boundary signal for EM-LLM, by triggering segmentation when the memory-loss update magnitude exceeds an adaptive threshold.

### Next-token Surprisal (Negative Log-probability)

The EM-LLM paper defines surprise as the negative log-probability the model assigns to the next token given its context and weights. The probability is computed with attention on the local sliding window. 

$$
- \log P(x_t | x_1, \dots, x_{t-1}; \theta)
$$

### Surprise Based on the Gradient of the Recurrent State

The Titans paper defines surprise as the gradient of the memory loss. In the TTT paper, the formula of this surprise is:

$$
\ell(W; x_t) = \| TTT(\theta_K x_t; W) - \theta_V x_t \|^2
$$

Since EM-LLM is meant for infinity context, we can add a controlled/adaptive natural decay for each TTT recurrent state update to avoid reaching compression bottleneck. 

### Segment Boundary Condition

The adaptive threshold is calculated on the surprisal-threshold window. 

$$
T = \mu_{t-\tau:t} + \gamma \sigma_{t-\tau:t}
$$

$$
S(x_t)>T
$$

### Possible Extensions

1. Aggregate token-level surprise over an entire sentence to form a sentence-level cumulative surprise score, instead of triggering boundaries from single-token spikes.
2. Replacing the current loss function from mean square error to cross entropy loss ([TTT-E2E](https://arxiv.org/abs/2512.23675)). 

### Potential Benefits and Caveats

1. If TTT updates are already performed online, the gradient used to update $W_t$ can be reused as the surprise signal with minimal extra cost.
2. Gradient-based surprise reflects how strongly the recurrent state must change to accommodate the new token, so boundaries are triggered by distribution shifts rather than isolated low-probability tokens, potentially yielding more precise event segments.
3. If the TTT recurrent state is not being utilised elsewhere, it would be additional computation effort compared to the original surprise. 
4. The quality of the boundary signal is heavily dependent on the quality of the loss function.  

## Hidden-State-Based Memory Segment Retrieval

### KNN-Based Memory Segment Retrieval

Each memory segment has a representative key vector. The top $k$ segments with the highest similarity score is retrieved. The similarity score is the dot product between the current query vector and the representative key vector. 

$$
\text{sim}(\mathbf{q}_t, \mathbf{r}_i) = \mathbf{q}_t^\top \mathbf{r}_i
$$

### Memory Segment Retrieval Using the Recurrent State

We store the difference between the recurrent state at the end of the previous segment and the current recurrent state to represent each segments instead of a single key vector. Upon retrieval, the gradient-based surprise can be used, with a lower surprise indicating the segment being suitable for the current query. 

$$
\ell(\Delta W; x_t) = \| TTT(\theta_K x_t; \Delta W) - \theta_V x_t \|^2
$$

### Storing the Change in the Recurrent State

Using lower rank projection to compress the per event change of the recurrent state, making it less expensive to store. We can use [LoRA](https://arxiv.org/abs/2106.09685) to find the upper projection $A$ and the lower projection $B$. 

$$
\Delta W \approx AB, \text{ with } A \in \mathbb{R}^{V \times r}, B \in \mathbb{R}^{r \times K}, r \ll \min(V, K)
$$

### Potential Benefits and Caveats

1. Reduce the retrieval of semantically similar but unhelpful memory segments by leveraging the recurrent state for a more comprehensive retrieval. 
2. Extra computation at retrieval. Needs to reconstruct the recurrent state change and calculate the gradient loss. 

### Open Questions

Should we keep the representative key vectors for similarity score as well, then come up with a combined metric with both the similarity and the surprise? If so, how should we weight them?

## High-Level Implementation Plan

A hybrid design that uses a native `TitansMAGForCausalLM` checkpoint as the base model and adds EM-LLM episodic retrieval on top of each TitansMAG block.

### Titans-Inspired Memory Implementation in EM-LLM

The current runtime does not replace TitansMAG with an EM-LLM-only attention stack. Instead, it keeps each TitansMAG block intact and adds a second, episodic retrieval branch beside it. The native TitansMAG branch still handles the block-local computation, including the recurrent memory update and, when enabled by the checkpoint, the donor `MemoryAsGate` local FLA attention path. EM-LLM then computes an episodic output from the same normalized hidden states, using retrieved memory segments as extra context, and fuses that episodic output with the native TitansMAG output before the MLP/residual path.

$$
\tilde{x} = [p_1,...,p_{N_p}] \parallel [s_1,...,s_k] \parallel SW(x)
$$

$$
y_t^{\text{epi}} = \text{EMAttn}(\tilde{x})
$$

$$
o_t^{\text{titans}} = \text{TitansMAGBlock}(x_t, W_{t-1})
$$

$$
o_t = \text{Fuse}(o_t^{\text{titans}}, y_t^{\text{epi}})
$$

### The Attention Branch

The attention branch is now the additional EM-LLM episodic branch, not the main TitansMAG branch. For each block, we first run `attn_norm`, then reuse the native TitansMAG projection path via `self.attn.get_qkv(...)` to build the queries, keys, and values for episodic retrieval. These tensors are passed to `ContextManager.append(...)`, which attends over the retrieved episodic blocks together with the current local chunk. The resulting episodic output is projected back to hidden size and fused with the native TitansMAG output.

Two details matter here. First, this branch no longer depends on a patched Hugging Face decoder attention module. It is attached directly to `TitansMAGBlock.forward`, so the donor block remains the source of truth for the base model execution. Second, it does not replace TitansMAG's local attention. If the checkpoint uses `MemoryAsGate` with local FLA attention, that logic still runs inside the native branch exactly as the donor model defines it. The EM-LLM branch only adds retrieval from external episodic memory.

### The Recurrent Branch

The recurrent branch is now whatever recurrent state mechanism the loaded TitansMAG checkpoint already uses. In practice, that means the native block state returned by `self.attn(...)` is carried forward unchanged across prompt chunks and decode steps. If the checkpoint uses linear TTT memory, the carried state is the corresponding fast-weight state. If it uses the donor `MemoryAsGate` path, the carried state follows that module's own cache contract. EM-LLM wraps this native Titans state together with its episodic state in one per-layer cache object, but it does not reinterpret or overwrite the Titans state itself.

This is an important change from the earlier prototype. There is no longer a separate EM-managed recurrent module sitting beside a host Hugging Face attention layer. The native TitansMAG branch runs first, produces the block output, and returns the native recurrent cache. The episodic branch is then fused on top. As a result, TitansMAG remains responsible for recurrent-memory semantics, while EM-LLM remains responsible for long-range episodic retrieval.

### Training

Titans MAG is still trained jointly. The recurrent memory state is updated online in the inner-loop during each forward pass, while the persistent model parameters are optimized in the outer-loop with the standard next-token language-model objective.

For pretraining, we follow the setup summarized in [Model Training](Model Training.md), which is aligned with ATLAS and DeltaNet, except that we use FineWeb-Edu instead of SlimPajama. The recipe uses the Llama 2 tokenizer with a 32K vocabulary, sequence length `4K`, global batch size `0.5M` tokens, peak learning rate `4e-4`, weight decay `0.1`, gradient clipping `1.0`, and cosine annealing. We focus on two scales: a `340M` model trained on `15B` tokens with `0.5B` warm-up tokens, and a `1.3B` model trained on `100B` tokens with `1B` warm-up tokens. The existing `train_titans_mag.sh` and `train.sh` pipeline already matches this joint-training workflow, so adopting this pretraining recipe should mainly require configuration changes rather than a new training implementation.

### Adapting to EM-LLM

The current implementation already patches `TitansMAGBlock.forward` directly and keeps the donor attention/recurrent path intact. What remains open is the research direction proposed above: using the recurrent state itself as an additional signal for segmentation and retrieval. That part is still separate from the runtime integration described here.

## Experiments

### Models

1. TTT-MAG-340M (Baseline)
2. TTT-MAG-340M EM-LLM
3. TTT-MAG-340M EM-LLM with Recurrent State Based Segmentation
4. TTT-MAG-340M EM-LLM with Recurrent State Based Retrieval
5. TTT-MAG-340M EM-LLM with Recurrent State Based Segmentation and Retrieval

### Evaluation

1. Word-Level Language Capabilities
    1. [WikiText-103](https://arxiv.org/abs/1803.08240)
    2. Perplexity for next-word prediction.
2. Long-Context Capabilities
    1. Needle in a Haystack (NIAH) Tasks from [RULER](https://arxiv.org/abs/2404.06654)
    2. Accuracy of long-context retrieval. 
3. Efficiency
    1. Token throughput.
    2. CPU memory usage vs. context length. 
        1. Mainly between model with the original memory segment storage and the one that also stores the recurrent state.
