"""copied from sparse-linear-attention and adapted for EM-LLM vendoring."""

# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class TitansMAGConfig(PretrainedConfig):
    model_type = 'titans_mag'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 8,
        head_dim: int = None,
        expand_v: int = 1,
        chunk_size: int = 64,
        conv_size: int = 4,
        use_bias: bool = False,
        diff_projections: bool = True,
        qk_rmsnorm: bool = False,
        window_size: int = 64,
        use_attention: bool = True,
        use_deltanet_qkv_order: bool = False,
        hidden_act: str = "swish",
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 21,
        attn: Optional[Dict] = None,
        memory_model: str = 'linear',
        memory_model_config: Optional[Dict] = None,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        norm_eps: float = 1e-5,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.use_bias = use_bias
        self.diff_projections = diff_projections
        self.qk_rmsnorm = qk_rmsnorm
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_deltanet_qkv_order = use_deltanet_qkv_order
        self.norm_eps = norm_eps

        self.hidden_ratio = hidden_ratio
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.attn = attn
        self.memory_model = memory_model
        self.memory_model_config = self._parse_memory_model_config(memory_model, memory_model_config)
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _parse_memory_model_config(self, memory_model: str, config_dict: Dict) -> Optional[PretrainedConfig]:
        if config_dict is None:
            return None
        if memory_model == 'linear':
            return TitansMAGLinearConfig(**config_dict)
        elif memory_model == 'mlp':
            return TitantMAGMLPConfig(**config_dict)

        else:
            raise ValueError(f"Unknown memory model: {memory_model}")


class TitansMAGLinearConfig(PretrainedConfig):
    def __init__(
        self,
        norm_eps: float = 1e-5,
        use_gate: bool = True,
        base_lr: float = 1e-1,
        **kwargs
    ):
        self.norm_eps = norm_eps
        self.use_gate = use_gate
        self.base_lr = base_lr
        super().__init__(**kwargs)


class TitantMAGMLPConfig(PretrainedConfig):
    def __init__(
        self,
        base_lr: float = 1e-2,
        use_gate: bool = True,
        norm_eps: float = 1e-5,
        init_std: float = 0.02,
        hidden_ratio: int = 4,
        **kwargs
    ):
        self.base_lr = base_lr
        self.use_gate = use_gate
        self.norm_eps = norm_eps
        self.init_std = init_std
        self.hidden_ratio = hidden_ratio
        super().__init__(**kwargs)
