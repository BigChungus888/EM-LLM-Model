# Sparse + linear architectures for sequence modeling

Following recent works on modern RNN architectures, specifically [Titans](https://arxiv.org/abs/2501.00663v1), which is
based on [Test-Time Training](https://arxiv.org/abs/2407.04620), we seek to extend the existing ideas in these models and show
a measurable improvement in language modeling and other tasks.

## Usages

The code is developed upon [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) and [Flame](https://github.com/fla-org/flame), so please refer to their repositories for general instructions. 
As for installation, please run
```
pip install --no-build-isolation -r requirements.txt
pip install flash-attn --no-build-isolation
pip install causal-conv1d --no-build-isolation
pip install git+https://github.com/fla-org/flame@f5c618c6 --no-deps
pip install flash-linear-attention==0.4.1
```
Note the required versions of `torch` and `transformers` to avoid any incompatibility. 
If you have already installed `flash-linear-attention`, set this repo to be prior in your `PYTHONPATH` to ensure 
our implementations of hybrid attention and LTE are used.

## Relevant works

- [hybrid-linear-sparse-attention](https://github.com/idiap/hybrid-linear-sparse-attention/tree/main) - their work is also based on FLA, 
focusing on sparse attention architectures. We can compare their results with ours.