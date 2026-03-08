from typing import Tuple, Optional

import torch
from fla.utils import input_guard, autocast_custom_bwd, autocast_custom_fwd
from torch import Tensor
from torch.nn import functional as F

from sparse_linear_attention.ops.sgd.utils import gelu_bwd, gelu_bwd_bwd


def compute_temporal_gradient_weights(eta: Tensor) -> Tuple[Tensor, Tensor]:
    eta_cumsum = eta.clamp(min=1e-8).unsqueeze(-1).log()
    eta_cumsum = eta_cumsum.cumsum(dim=-2)
    beta = (eta_cumsum - eta_cumsum.movedim(-2, -1)).clamp(max=0).exp()
    return torch.tril(beta), eta_cumsum.exp()


def compute_temporal_gradient_weights_backward(
    dbeta: Tensor,
    deta_cumsum_exp: Tensor,
    eta: Tensor,
    eta_cumsum_exp: Tensor,
) -> Tensor:
    """
    Shapes:
        eta: [B, H, BT]
        beta: [B, H, BT, BT]
        eta_cumsum_exp: [B, H, BT, 1]
        dbeta: [B, H, BT, BT]
        deta_cumsum_exp: [B, H, BT, 1]

    Returns:
        deta: [B, H, BT]
    """

    deta_cumsum = torch.zeros_like(eta_cumsum_exp)
    deta_cumsum += deta_cumsum_exp * eta_cumsum_exp
    dbeta_full = torch.tril(dbeta)

    eta_clamped = eta.clamp(min=1e-8)
    log_eta = eta_clamped.unsqueeze(-1).log()
    eta_cumsum = log_eta.cumsum(dim=-2)

    diff = eta_cumsum - eta_cumsum.movedim(-2, -1)
    clamped_diff = diff.clamp(max=0)
    beta_full = clamped_diff.exp()

    d_clamped_diff = dbeta_full * beta_full

    mask = (diff <= 0).float()
    d_diff = d_clamped_diff * mask

    deta_cumsum += d_diff.sum(dim=-1, keepdim=True)
    deta_cumsum += -d_diff.sum(dim=-2, keepdim=True).movedim(-2, -1)

    d_log_eta = deta_cumsum.flip(dims=[-2]).cumsum(dim=-2).flip(dims=[-2])
    d_eta_clamped_unsqueezed = d_log_eta / eta_clamped.unsqueeze(-1)
    d_eta_clamped = d_eta_clamped_unsqueezed.squeeze(-1)
    mask_clamp = (eta >= 1e-8).float()
    deta = d_eta_clamped * mask_clamp
    return deta


def update_weights_forward(
    w1: Tensor, w2: Tensor, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, eta_cumsum_exp: Tensor, theta: Tensor,
    return_cache: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tuple]]:
    x1 = k
    z1 = x1 @ w1.movedim(-2, -1)
    x2 = F.gelu(z1, approximate="tanh")
    z2 = x2 @ w2.movedim(-2, -1)

    dz2 = z2 - v
    dz1 = dz2 @ w2 * gelu_bwd(z1)

    theta_dz1 = theta[..., None] * dz1
    theta_dz2 = theta[..., None] * dz2

    attn1 = q @ x1.movedim(-2, -1)

    z1_bar = (q * eta_cumsum_exp) @ w1.movedim(-2, -1) - (attn1 * beta) @ theta_dz1
    x2_bar = F.gelu(z1_bar, approximate="tanh")

    attn2 = x2_bar @ x2.movedim(-2, -1)
    z2_bar = (x2_bar * eta_cumsum_exp) @ w2.movedim(-2, -1) - (attn2 * beta) @ theta_dz2

    beta_last = beta[:, :, -1:, :].movedim(-2, -1)

    w1_new = w1 * eta_cumsum_exp[..., -1:, :] - theta_dz1.movedim(-2, -1) @ (x1 * beta_last)
    w2_new = w2 * eta_cumsum_exp[..., -1:, :] - theta_dz2.movedim(-2, -1) @ (x2 * beta_last)

    if return_cache:
        cache = (w1, w2, z1, z1_bar, dz1, dz2)
        return w1_new, w2_new, z2_bar, cache
    else:
        return w1_new, w2_new, z2_bar, None


@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)
def chunk_linear_rule(q: Tensor, k: Tensor, v: Tensor, eta: Tensor, theta: Tensor,
                      initial_weights: Tuple[Tensor, Tensor], chunk_size: int):
    B, H, T, K = q.shape
    BT = chunk_size
    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        eta = F.pad(eta, (0, pad_len), value=0)
        q = F.pad(q, (0, 0, 0, pad_len), value=0)
        k = F.pad(k, (0, 0, 0, pad_len), value=0)
        v = F.pad(v, (0, 0, 0, pad_len), value=0)
        theta = F.pad(theta, (0, pad_len), value=0)
        T_padded = T + pad_len
    else:
        T_padded = T

    NT = T_padded // BT
    outputs = torch.empty_like(v)
    q = q * K ** (-1 / 2)
    w1, w2 = initial_weights

    beta, eta_cumsum = compute_temporal_gradient_weights(eta.reshape(B, H, NT, BT))

    for idx in range(NT):
        chunk_start = idx * BT
        chunk_end = min(chunk_start + BT, T)
        actual_chunk_size = chunk_end - chunk_start

        if actual_chunk_size == 0:
            continue

        chunk_q = q[:, :, chunk_start:chunk_end]  # [B, H, L, D]
        chunk_k = k[:, :, chunk_start:chunk_end]  # [B, H, L, D]
        chunk_v = v[:, :, chunk_start:chunk_end]  # [B, H, L, D]
        chunk_beta = beta[:, :, idx]
        chunk_eta_cumsum = eta_cumsum[:, :, idx]  # [B, H, L, 1]
        chunk_theta = theta[:, :, chunk_start:chunk_end]  # [B, H, L]
        w1, w2, chunk_output, _ = update_weights_forward(w1, w2, chunk_q, chunk_k, chunk_v, chunk_beta,
                                                         chunk_eta_cumsum, chunk_theta)
        outputs[:, :, chunk_start:chunk_end] = chunk_output

    if pad_len > 0:
        outputs = outputs[:, :, :T]
    return (w1, w2), outputs.movedim(-3, -2)


def update_weights_backward(
    dz2_bar: Tensor,
    dw1_new: Tensor,
    dw2_new: Tensor,
    q: Tensor,
    k: Tensor,
    beta: Tensor,
    eta_cumsum_exp: Tensor,
    theta: Tensor,
    cache: tuple,
):
    """
    Manual backward pass for update_weights_forward.

    Shapes:
        q, k: [B, H, BT, head_dim]
        v: [B, H, BT, 2 * head_dim]
        w1: [B, H, hidden_ratio * head_dim, head_dim]
        w2: [B, H, 2 * head_dim, hidden_ratio * head_dim]
        beta: [B, H, BT, BT]
        eta_cumsum_exp: [B, H, BT, 1]
        theta: [B, H, BT] = [2, 8, 64]
        dz2_bar: [B, H, BT, 2 * head_dim]
        dw1_new: [B, H, hidden_ratio * head_dim, head_dim]
        dw2_new: [B, H, 2 * head_dim, hidden_ratio * head_dim]
    """
    w1, w2, z1, z1_bar, dz1, dz2 = cache

    x1 = k
    x2 = F.gelu(z1, approximate="tanh")
    x2_bar = F.gelu(z1_bar, approximate="tanh")

    theta_dz1 = theta[..., None] * dz1
    theta_dz2 = theta[..., None] * dz2

    attn1 = q @ x1.movedim(-2, -1)
    attn2 = x2_bar @ x2.movedim(-2, -1)

    beta_last = beta[:, :, -1:, :].movedim(-2, -1)
    eta_exp_last = eta_cumsum_exp[..., -1:, :]

    deta_exp = torch.zeros_like(eta_cumsum_exp)
    dbeta = torch.zeros_like(beta)

    dw2 = dw2_new * eta_exp_last
    deta_exp[..., -1:, :] += (dw2_new * w2).sum(dim=(-2, -1), keepdim=True)

    temp = x2 * beta_last
    dtheta_dz2 = -(temp @ dw2_new.movedim(-2, -1))
    dx2 = -(theta_dz2 @ dw2_new) * beta_last

    dw1 = dw1_new * eta_exp_last
    deta_exp[..., -1:, :] += (dw1_new * w1).sum(dim=(-2, -1), keepdim=True)

    temp = x1 * beta_last
    dtheta_dz1 = -(temp @ dw1_new.movedim(-2, -1))
    dk = -(theta_dz1 @ dw1_new) * beta_last

    d_x1_beta_last = -theta_dz1 @ dw1_new
    dbeta_last_from_w1 = (d_x1_beta_last * x1).sum(dim=-1, keepdim=True)

    d_x2_beta_last = -theta_dz2 @ dw2_new
    dbeta_last_from_w2 = (d_x2_beta_last * x2).sum(dim=-1, keepdim=True)

    dbeta_last = dbeta_last_from_w1 + dbeta_last_from_w2
    dbeta[:, :, -1:, :] += dbeta_last.movedim(-2, -1)

    dx2_bar = (dz2_bar @ w2) * eta_cumsum_exp
    dw2 += ((x2_bar * eta_cumsum_exp).movedim(-2, -1) @ dz2_bar).movedim(-2, -1)
    deta_exp += ((dz2_bar @ w2) * x2_bar).sum(dim=-1, keepdim=True)

    dtheta_dz2 += -(attn2 * beta).movedim(-2, -1) @ dz2_bar
    dattn2 = -(dz2_bar @ theta_dz2.movedim(-2, -1)) * beta
    dbeta += -(dz2_bar @ theta_dz2.movedim(-2, -1)) * attn2

    dx2_bar += dattn2 @ x2
    dx2 += dattn2.movedim(-2, -1) @ x2_bar

    dz1_bar = dx2_bar * gelu_bwd(z1_bar)

    dq = (dz1_bar @ w1) * eta_cumsum_exp
    dw1 += ((q * eta_cumsum_exp).movedim(-2, -1) @ dz1_bar).movedim(-2, -1)
    deta_exp += ((dz1_bar @ w1) * q).sum(dim=-1, keepdim=True)

    dtheta_dz1 += -(attn1 * beta).movedim(-2, -1) @ dz1_bar
    dattn1 = -(dz1_bar @ theta_dz1.movedim(-2, -1)) * beta
    dbeta += -(dz1_bar @ theta_dz1.movedim(-2, -1)) * attn1

    dq += dattn1 @ x1
    dk += dattn1.movedim(-2, -1) @ q

    dtheta = (dtheta_dz2 * dz2).sum(dim=-1)
    ddz2 = dtheta_dz2 * theta[..., None]

    dtheta += (dtheta_dz1 * dz1).sum(dim=-1)
    ddz1 = dtheta_dz1 * theta[..., None]

    gelu_bwd_z1 = gelu_bwd(z1)
    ddz2 += (ddz1 * gelu_bwd_z1) @ w2.movedim(-2, -1)
    dw2 += dz2.movedim(-2, -1) @ (ddz1 * gelu_bwd_z1)
    dz1_from_dz1 = ddz1 * (dz2 @ w2) * gelu_bwd_bwd(z1)

    dx2 += ddz2 @ w2
    dw2 += (x2.movedim(-2, -1) @ ddz2).movedim(-2, -1)
    dv = -ddz2

    dz1 = dx2 * gelu_bwd(z1) + dz1_from_dz1

    dk += dz1 @ w1
    dw1 += (x1.movedim(-2, -1) @ dz1).movedim(-2, -1)

    return dq, dk, dv, dbeta, deta_exp, dtheta, dw1, dw2


@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)
def chunk_linear_rule_backward(q, k, v, eta, theta, initial_weights, chunk_size, doutputs):
    B, H, T, K = q.shape
    BT = chunk_size

    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        eta = F.pad(eta, (0, pad_len))
        theta = F.pad(theta, (0, pad_len))
        doutputs = F.pad(doutputs, (0, 0, 0, pad_len))
        T += pad_len

    NT = T // BT
    eta = eta.reshape(B, H, NT, BT)
    beta, eta_cumsum = compute_temporal_gradient_weights(eta)

    q = q * K ** (-1 / 2)
    w1, w2 = initial_weights
    caches = []

    # ---- forward replay ----
    for idx in range(NT):
        s, e = idx * BT, (idx + 1) * BT
        w1, w2, _, cache = update_weights_forward(
            w1, w2,
            q[:, :, s:e],
            k[:, :, s:e],
            v[:, :, s:e],
            beta[:, :, idx],
            eta_cumsum[:, :, idx],
            theta[:, :, s:e],
            return_cache=True,
        )
        caches.append(cache)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    deta_cumsum = torch.empty_like(eta_cumsum)
    dtheta = torch.empty_like(theta)

    d_w1 = torch.zeros_like(w1)
    d_w2 = torch.zeros_like(w2)
    doutputs = doutputs.movedim(-3, -2)
    for idx in reversed(range(NT)):
        s, e = idx * BT, (idx + 1) * BT
        dq_i, dk_i, dv_i, dbeta_i, deta_cumsum_i, dtheta_i, d_w1, d_w2 = update_weights_backward(
            doutputs[:, :, s:e],
            d_w1,
            d_w2,
            q[:, :, s:e],
            k[:, :, s:e],
            beta[:, :, idx],
            eta_cumsum[:, :, idx],
            theta[:, :, s:e],
            caches[idx],
        )
        dq[:, :, s:e] = dq_i
        dk[:, :, s:e] = dk_i
        dv[:, :, s:e] = dv_i
        dbeta[:, :, idx] = dbeta_i
        deta_cumsum[:, :, idx] = deta_cumsum_i
        dtheta[:, :, s:e] = dtheta_i
    deta = compute_temporal_gradient_weights_backward(dbeta, deta_cumsum, eta, eta_cumsum)
    dq *= K ** (-1 / 2)
    if pad_len > 0:
        dq = dq[:, :, :T - pad_len]
        dk = dk[:, :, :T - pad_len]
        dv = dv[:, :, :T - pad_len]
        deta = deta[:, :, :T - pad_len]
        dtheta = dtheta[:, :, :T - pad_len]

    return dq, dk, dv, deta, dtheta


class ChunkMLPUpdate(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, eta, theta, initial_weights, chunk_size):
        with torch.no_grad():
            final_weights, outputs = chunk_linear_rule(q, k, v, eta, theta, initial_weights, chunk_size)

        ctx.save_for_backward(q, k, v, eta, theta)
        ctx.chunk_size = chunk_size
        ctx.initial_weights = initial_weights

        return final_weights, outputs

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dw, doutputs):
        q, k, v, eta, theta = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        initial_weights = ctx.initial_weights

        dq, dk, dv, deta, dtheta = chunk_linear_rule_backward(q, k, v, eta, theta, initial_weights, chunk_size,
                                                              doutputs)
        return dq, dk, dv, deta, dtheta, None, None
