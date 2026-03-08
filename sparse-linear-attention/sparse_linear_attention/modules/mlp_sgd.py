from __future__ import annotations

import weakref
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from fla.models.utils import Cache
from fla.modules import RMSNorm, FusedRMSNormGated

from sparse_linear_attention.ops.sgd.chunk import ChunkMLPUpdate
from sparse_linear_attention.ops.sgd.naive import mlp_chunk


class SwiGLUSGD(nn.Module):
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
        init_std: float = 0.05,
        hidden_ratio: int = 4,
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
        self.init_std = init_std
        self.hidden_ratio = hidden_ratio
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

    def forward(
        self,
        inputs: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        BT = self.chunk_size
        B, T, _ = inputs.shape
        H = self.num_heads
        K = self.head_k_dim
        V = self.head_v_dim
        q, k, v = self._get_qkv(inputs)  # [B, T, internal_dim]
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        eta = self.compute_weight_decay(inputs)  # [B, H, T]
        eta = torch.ones_like(eta)
        theta = self.compute_learning_rate(inputs)  # [B, H, T]
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        initial_weights = last_state['recurrent_state'] if last_state is not None else None
        if self.mode == "chunk":
            if initial_weights is None:
                w1 = self.init_std * torch.randn(B, H, self.hidden_ratio * K, K, device=q.device, dtype=q.dtype)
                w2 = self.init_std * torch.randn(B, H, V, self.hidden_ratio * K, device=q.device, dtype=q.dtype)
                initial_weights = (w1, w2)
            dynamic_weights, outputs = ChunkMLPUpdate.apply(q, k, v, eta, theta, initial_weights, BT)
        else:
            dynamic_weights, outputs = mlp_chunk(q, k, v, eta, theta, initial_weights, self.init_std, self.hidden_ratio,
                                                 B, H, K, T, V, BT)
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=initial_weights,
                layer_idx=self.layer_idx,
            )
        if self.use_gate:
            g = rearrange(self.g_proj(inputs), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(outputs, g)
        else:
            o = self.o_norm(outputs)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        return o, (dynamic_weights, None)
