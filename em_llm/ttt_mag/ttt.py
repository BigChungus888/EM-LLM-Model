"""copied from sparse-linear-attention and adapted for EM-LLM vendoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional

import torch
import torch.nn as nn
from fla.modules import ShortConvolution
from torch.nn import functional as F

if TYPE_CHECKING:
    from fla.models.utils import Cache


class TTT(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        memory_model: nn.Module,
        head_dim: int = None,
        num_heads: int = 8,
        expand_v: int = 1,
        chunk_size: int = 64,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if head_dim is None:
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * expand_v)

        self.expand_v = expand_v
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=use_bias)

        k_channels = self.key_dim
        v_channels = self.value_dim
        self.input_conv = ShortConvolution(
            hidden_size=hidden_size,
            kernel_size=conv_size,
            bias=conv_bias,
            activation='silu'
        )
        self.q_conv1d = ShortConvolution(
            hidden_size=k_channels,
            kernel_size=conv_size,
            bias=conv_bias,
            activation='silu'
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=k_channels,
            kernel_size=conv_size,
            bias=conv_bias,
            activation='silu'
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=v_channels,
            kernel_size=conv_size,
            bias=conv_bias,
            activation='silu'
        )

        self.M = memory_model
        self.converted = False
        # keep the donor callback contract intact so vendored `LinearSGD` can ask
        # this wrapper for q/k/v without knowing about EM-LLM internals.
        self.M.set_qkv_callable(self.get_qkv)

    def _project_upcast_weights(self, proj_module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == proj_module.weight.dtype:
            return proj_module(x)

        weight_casted = proj_module.weight.to(x.dtype)
        bias_casted = proj_module.bias.to(x.dtype) if proj_module.bias is not None else None
        return F.linear(x, weight_casted, bias_casted)

    def get_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        needs_squeeze = False
        if x.ndim == 2:
            needs_squeeze = True
            x = x.unsqueeze(0)

        hidden_states_conv, _ = self.input_conv(x)
        hidden_states = x + hidden_states_conv
        q = self.q_proj(hidden_states)
        q_conv, conv_state_q = self.q_conv1d(q)
        q = q + q_conv
        k = self.k_proj(hidden_states)
        k_conv, conv_state_k = self.k_conv1d(k)
        k = k + k_conv
        v = self.v_proj(hidden_states)
        v_conv, conv_state_v = self.v_conv1d(v)
        v = v + v_conv

        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        if needs_squeeze:
            q, k, v = map(lambda t: t.squeeze(0), (q, k, v))

        return q.to(x.dtype), k.to(x.dtype), v.to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional['Cache']]:

        assert not hidden_states.isnan().any(), "hidden_states contains NaNs"
        use_cache = kwargs.get('use_cache', False)
        past_key_values = kwargs.get('past_key_values', None)

        x = hidden_states  # [B, L, D]

        atlas_out, new_state = self.M(x, past_key_values)
        return atlas_out, None, new_state if use_cache else None
