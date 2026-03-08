# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from sparse_linear_attention.models.ttt.configuration_ttt import TTTConfig
from sparse_linear_attention.models.ttt.modeling_ttt import TTTModel, TTTForCausalLM
from sparse_linear_attention.models.titans_mag.configuration_titans_mag import TitansMAGConfig
from sparse_linear_attention.models.titans_mag.modeling_titans_mag import TitansMAGModel, TitansMAGForCausalLM

AutoConfig.register(TTTConfig.model_type, TTTConfig, exist_ok=True)
AutoModel.register(TTTConfig, TTTModel, exist_ok=True)
AutoModelForCausalLM.register(TTTConfig, TTTForCausalLM, exist_ok=True)
AutoConfig.register(TitansMAGConfig.model_type, TitansMAGConfig, exist_ok=True)
AutoModel.register(TitansMAGConfig, TitansMAGModel, exist_ok=True)
AutoModelForCausalLM.register(TitansMAGConfig, TitansMAGForCausalLM, exist_ok=True)

__all__ = ['TTTConfig', 'TTTForCausalLM', 'TTTModel', 'TitansMAGConfig', 'TitansMAGForCausalLM', 'TitansMAGModel']
