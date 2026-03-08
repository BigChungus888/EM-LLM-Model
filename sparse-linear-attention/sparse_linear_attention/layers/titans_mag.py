from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from fla.layers.attn import Attention
from fla.modules import ShortConvolution
from sparse_linear_attention.ops.titans.utils import create_mag_block_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache


class MemoryAsGate(nn.Module):
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
        diff_projections: bool = True,
        qk_rmsnorm: bool = False,
        window_size: int = 64,
        use_attention: bool = True,
        layer_idx: int = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if use_attention:
            assert hidden_size % num_heads == 0, "d_model must be divisible by heads"
        if head_dim is None:
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * expand_v)

        self.expand_v = expand_v
        self.chunk_size = chunk_size
        self.diff_projections = diff_projections
        self.conv_size = conv_size
        self.window_size = window_size
        self.use_attention = use_attention
        self.layer_idx = layer_idx
        if use_attention:
            self.attention = Attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                qk_norm=qk_rmsnorm,
                max_position_embeddings=4096,
                window_size=window_size,
            )

        if use_attention and not diff_projections:
            if expand_v != 1:
                raise ValueError(
                    "diff_projections=False requires num_variants==1 and expand_v==1"
                )
            k_channels = self.attention.kv_dim
            v_channels = k_channels

        else:
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
        if use_attention:
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.gate_proj = nn.Linear(hidden_size, hidden_size)
            self.atlas_norm = nn.RMSNorm(hidden_size)
            self.swa_norm = nn.RMSNorm(hidden_size)
        self.converted = False
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

        if self.use_attention and not self.diff_projections:
            q_proj_layer = self.attention.q_proj.to(x.dtype)
            k_proj_layer = self.attention.k_proj.to(x.dtype)
            v_proj_layer = self.attention.v_proj.to(x.dtype)
        else:
            q_proj_layer = self.q_proj
            k_proj_layer = self.k_proj
            v_proj_layer = self.v_proj

        hidden_states_conv, _ = self.input_conv(x)
        hidden_states = x + hidden_states_conv
        q = q_proj_layer(hidden_states)
        q_conv, conv_state_q = self.q_conv1d(q)
        q = q + q_conv
        k = k_proj_layer(hidden_states)
        k_conv, conv_state_k = self.k_conv1d(k)
        k = k + k_conv
        v = v_proj_layer(hidden_states)
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

        use_cache = kwargs.get('use_cache', False)
        past_key_values = kwargs.get('past_key_values', None)

        x = hidden_states  # [B, L, D]

        atlas_out, new_state = self.M(x, past_key_values)
        if not self.use_attention:
            return atlas_out, None, new_state if use_cache else None

        seq_len = x.shape[1]
        attn_mask = create_mag_block_mask(
            seq_len,
            self.window_size,
            0,
            True
        )

        x_attn_in = x
        target_dtype = torch.bfloat16

        if x_attn_in.dtype == torch.float32:
            x_attn_in = x_attn_in.to(target_dtype)

        att_param_dtype = next(self.attention.parameters()).dtype
        if att_param_dtype == torch.float32:
            self.attention = self.attention.to(target_dtype)

        swa_out = self.attention(x_attn_in, attn_mask=attn_mask)[0]
        swa_out = swa_out.to(x.dtype)

        atlas_normed = self.atlas_norm(atlas_out)
        swa_normed = self.swa_norm(swa_out)

        gate = torch.sigmoid(self.gate_proj(swa_normed))
        gated_out = swa_normed + gate * atlas_normed
        out = self.out_proj(gated_out)

        return out, None, new_state if use_cache else None
