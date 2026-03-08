from __future__ import annotations

import weakref
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from fla.models.utils import Cache
from fla.modules import RMSNorm, FusedRMSNormGated
from sparse_linear_attention.ops.sgd.naive import linear_chunk


class LinearSGD(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim: int = 128,
        expand_v: float = 1.0,
        get_qkv_fn: Callable = None,
        base_lr: float = 1e-1,
        norm_eps: float = 1e-5,
        chunk_size: int = 16,
        use_gate: bool = True,
        mode: str = "chunk",
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(num_heads * self.head_k_dim)
        self.value_dim = int(num_heads * self.head_v_dim)
        self.base_lr = base_lr
        self.norm_eps = norm_eps
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.to_eta = nn.Linear(hidden_size, num_heads, bias=False)
        self.to_theta = nn.Linear(hidden_size, num_heads, bias=False)
        self.use_gate = use_gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.mode = mode

        self._qkv_owner_ref = None
        self.qkv_name = None
        self.qkv_callable = None
        if get_qkv_fn is not None:
            self.set_qkv_callable(get_qkv_fn)

    def set_qkv_callable(self, get_qkv_fn: Callable):
        """
        Set the QKV callable function after initialization.

        Args:
            get_qkv_fn: A callable function to compute Q, K, V tensors.
                         Expected to return (q, k, v) each of shape (B, T, H*D_head).
        """
        owner = get_qkv_fn.__self__ if hasattr(get_qkv_fn, "__self__") else None
        self._qkv_owner_ref = weakref.ref(owner) if owner is not None else None
        self.qkv_name = get_qkv_fn.__name__ if hasattr(get_qkv_fn, "__name__") else None
        self.qkv_callable = None if owner is not None else get_qkv_fn

    def _get_qkv(self, inputs: torch.Tensor):
        """
        Get Q, K, V tensors using the configured qkv_callable.

        Args:
            inputs: Input tensor of shape (B, T, D)

        Returns:
            Tuple of (q, k, v) tensors
        """
        if self._qkv_owner_ref is not None:
            owner = self._qkv_owner_ref()
            if owner is not None:
                return getattr(owner, self.qkv_name)(inputs)
        elif self.qkv_callable is not None:
            return self.qkv_callable(inputs)
        else:
            raise ValueError("qkv_callable not set. Call set_qkv_callable() first.")

    def compute_weight_decay(self, inputs: torch.Tensor):
        return torch.sigmoid(self.to_eta(inputs)).permute(0, 2, 1)

    def compute_learning_rate(self, inputs: torch.Tensor):
        return self.base_lr * torch.sigmoid(self.to_theta(inputs)).permute(0, 2, 1)

    def compute_temporal_gradient_weights(self, eta: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        eta_cumsum = eta.clamp(min=1e-8).unsqueeze(-1).log()
        eta_cumsum = eta_cumsum.cumsum(dim=-2)
        beta = (eta_cumsum - eta_cumsum.movedim(-2, -1)).clamp(max=0).exp()
        return torch.tril(beta), eta_cumsum

    def update_weights(self, weights: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       eta: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = k @ weights.movedim(-2, -1) - v
        beta, eta_cumsum = self.compute_temporal_gradient_weights(eta)
        qk = q @ k.movedim(-2, -1)
        theta_e = theta[..., None] * e
        z = (q * eta_cumsum.exp()) @ weights.movedim(-2, -1) - (qk * beta) @ theta_e
        e_decay = beta[:, :, -1:, :].movedim(-2, -1) * theta_e
        dw = e_decay.movedim(-2, -1) @ k
        weights = weights * eta_cumsum[..., -1:, :].exp() - dw
        return weights, z

    def forward(
        self,
        inputs: torch.Tensor,
        past_key_values: Cache | None = None
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        BT = self.chunk_size
        B, T, _ = inputs.shape
        H = self.num_heads
        K = self.head_k_dim
        V = self.head_v_dim
        q, k, v = self._get_qkv(inputs)  # [B, T, internal_dim]
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        eta = self.compute_weight_decay(inputs)  # [B, H, T]
        theta = self.compute_learning_rate(inputs)  # [B, H, T]
        if self.mode == "chunk":
            dynamic_weights, outputs = self.chunk_linear_rule(q, k, v, eta, theta, B, H, K, T, V, BT)
        else:
            dynamic_weights, outputs = linear_chunk(q, k, v, eta, theta, B, H, K, T, V, BT)
        if self.use_gate:
            g = rearrange(self.g_proj(inputs), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(outputs, g)
        else:
            o = self.o_norm(outputs)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        return o, (dynamic_weights, None)

    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def chunk_linear_rule(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eta: torch.Tensor,
                          theta: torch.Tensor, B: int, H: int, K: int, T: int, V: int, BT: int):
        NT = (T + BT - 1) // BT
        dynamic_weights = torch.zeros(B, H, V, K, device=q.device, dtype=q.dtype)
        outputs = torch.empty(v.shape, device=q.device, dtype=q.dtype)
        q = q * K ** (-1 / 2)

        for idx in range(NT):
            seg_start = idx * BT
            seg_end = min(seg_start + BT, T)
            actual_seg_len = seg_end - seg_start

            if actual_seg_len == 0:
                continue

            seg_q = q[:, :, seg_start:seg_end]  # [B, H, L, D]
            seg_k = k[:, :, seg_start:seg_end]  # [B, H, L, D]
            seg_v = v[:, :, seg_start:seg_end]  # [B, H, L, D]
            seg_eta = eta[:, :, seg_start:seg_end]  # [B, H, L]
            seg_theta = theta[:, :, seg_start:seg_end]  # [B, H, L]
            dynamic_weights, seg_output = self.update_weights(
                weights=dynamic_weights,
                q=seg_q,
                k=seg_k,
                v=seg_v,
                eta=seg_eta,
                theta=seg_theta
            )
            outputs[:, :, seg_start:seg_end] = seg_output
        return dynamic_weights, outputs.movedim(-3, -2)
