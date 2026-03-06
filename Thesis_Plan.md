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

LLMs still struggle with very long contexts because full attention is expensive and important information can be diluted or missed. Two common directions address this in different ways. External episodic memory banks store and retrieve past segments, while recurrent or hybrid models carry information forward in a compact recurrent state/fast weights. We aim to combine both by using a recurrent or linear-attention backbone together with an external episodic memory bank, and guiding memory formation and retrieval using the model’s hidden-state dynamics. In the end, we keep EM-LLM as the host runtime and adapt Titans/TTT memory components as an additional recurrent branch plus fusion module, rather than replacing the existing EM-LLM pipeline outright.  

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

A hybrid design that uses TTT-340M as the base model and adds a Titans-style attention branch inside EM-LLM.

### Titans-Inspired Memory Implementation in EM-LLM

We apply attention over the persistent tokens $p_1,...,p_{N_p}$, the retrieved EM-LLM memory segments $s_1,...,s_k$, and local recent tokens $SW(x)$, then fuse this attention output with the TTT branch output. Since EM-LLM already retrieves past segments into the context window, we omit Titans’ explicit long-term retrieval $h_t$. As a result, the fixed-size segmentation used in the original Titans MAC formulation is unnecessary, and we replace the current segment $S^{(t)}$ with a sliding local window $SW(x)$. Overall, this gives us a hybrid design: we add retrieval provided by EM-LLM as additional context (an adaptation of Titans MAC), while using Titans MAG-style gating for the fused final output.

$$
\tilde{x} = [p_1,...,p_{N_p}] \parallel [s_1,...,s_k] \parallel SW(x)
$$

$$
y_t = \text{Attn}(\tilde{x})
$$

$$
W_t = TTTUpdate(W_{t-1},y_t)
$$

$$
o_t = y_t \otimes TTT(y_t; W_t)
$$

### The Attention Branch

Implement the attention branch with PyTorch SDPA or flash-linear-attention for speed.

### Training

In Titans, training is joint: attention and gating parameters are optimised in the outer-loop, while the TTT recurrent state/fast weights $W_t$ are updated online in the inner-loop. For EM-LLM integration, we should not assume the existing MAG training stack can be reused unchanged; adapter work and runtime-specific training glue may be required after inference-path validation. 

### Adapting to EM-LLM

Add separate forward pass patching logic to run TitansMAG-340M with TTT linear SGD as the base model (with attention on the additional retrieved tokens from the episodic memory segments). Use the recurrent state from TTT linear for potentially more precise memory segmentation and retrieval. Add `recurrent_state_based_segmentation` and `recurrent_state_based_retrieval` as additional configuration parameters for `titans_mag_linear_sgd_340M.yaml`.

## Experiments

### Models

1. TitansMAG-340M (Baseline)
2. TitansMAG-340M EM-LLM
3. TitansMAG-340M EM-LLM with Recurrent State Based Segmentation
4. TitansMAG-340M EM-LLM with Recurrent State Based Retrieval
5. TitansMAG-340M EM-LLM with Recurrent State Based Segmentation and Retrieval

### Benchmarks

1. [WikiText-103](https://arxiv.org/abs/1803.08240)
2. Needle in a Haystack (NIAH) Tasks from [RULER](https://arxiv.org/abs/2404.06654)