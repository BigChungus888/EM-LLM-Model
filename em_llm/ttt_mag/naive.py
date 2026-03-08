"""copied from sparse-linear-attention and adapted for EM-LLM vendoring."""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

# use the vendored helper so this copied kernel has no dependency on the donor repo.
from .sgd_utils import gelu_bwd


def linear_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eta: torch.Tensor, theta: torch.Tensor,
                 initial_weights: Optional[Tensor], B: int, H: int, K: int, T: int, V: int, BT: int):
    # accept an optional starting state so the naive path matches the vendored
    # chunked path when we test recurrent-state carry-over.
    S = torch.zeros(B, H, V, K, device=q.device, dtype=q.dtype) if initial_weights is None else initial_weights
    S_step = S.clone()
    outputs = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    q = q * K ** (-1 / 2)
    for i in range(T):
        e = S @ k[..., i, :, None] - v[..., i, :, None]
        S_step = S_step * eta[..., i, None, None] - theta[..., i, None, None] * e @ k[..., i, :, None].movedim(-2, -1)
        outputs[..., i, :] = (S_step @ q[..., i, :, None]).squeeze()
        if -i % BT == 1:
            S = S_step
    return S_step, outputs.movedim(-3, -2)


def mlp_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eta: torch.Tensor, theta: torch.Tensor,
              initial_weights: Optional[Tuple[Tensor, Tensor]], init_std: float, ratio: int, B: int, H: int, K: int,
              T: int, V: int, BT: int):
    if initial_weights is None:
        W1 = init_std * torch.randn(B, H, ratio * K, K, device=q.device, dtype=q.dtype)
        W2 = init_std * torch.randn(B, H, V, ratio * K, device=q.device, dtype=q.dtype)
    else:
        W1, W2 = initial_weights
    W1_step = W1.clone()
    W2_step = W2.clone()
    outputs = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    q = q * K ** (-1 / 2)
    for i in range(T):
        x1 = k[..., i, :, None]
        z1 = W1 @ x1
        x2 = F.gelu(z1, approximate="tanh")
        z2 = W2 @ x2
        e = z2 - v[..., i, :, None]
        eta_i = eta[..., i, None, None]
        theta_i = theta[..., i, None, None]
        W2_step = W2_step * eta_i - theta_i * e @ x2.movedim(-2, -1)
        W1_step = W1_step * eta_i - theta_i * (W2.movedim(-2, -1) @ e * gelu_bwd(z1)) @ x1.movedim(-2, -1)
        outputs[..., i, :] = (W2_step @ F.gelu(W1_step @ q[..., i, :, None], approximate="tanh")).squeeze()
        if -i % BT == 1:
            W1 = W1_step
            W2 = W2_step
    return (W1_step, W2_step), outputs.movedim(-3, -2)
