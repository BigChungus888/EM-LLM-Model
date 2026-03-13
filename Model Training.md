# Model pretraining

https://wandb.ai/lin-attn/fla/workspace

This setup is suggested in both [ATLAS](https://arxiv.org/pdf/2505.23735) and [DeltaNet](https://arxiv.org/pdf/2406.06484) papers. Note that in DeltaNet, they use [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B), and we use [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) instead, as in ATLAS, since we also want to compare against their more recent models. Below you can find the table summary of training hyperparameters.

| Parameter | Value |
| --- | --- |
| Peak LR | 4e-4 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Schedule | Cosine annealing |
| Batch size | 0.5M tokens |
| Sequence length | 4K tokens |

Both works use LLama 2 tokenizer with a vocabulary size of 32K.

For now, we will focus on two scales of pretraining: 340M model on 15B tokens with warm-up period of 0.5B tokens and 1.3B model on 100B tokens with warm-up period of 1B tokens.