# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from sparse_linear_attention.models.titans_mag.configuration_titans_mag import TitansMAGConfig
from sparse_linear_attention.models.titans_mag.modeling_titans_mag import TitansMAGModel, TitansMAGForCausalLM

AutoConfig.register(TitansMAGConfig.model_type, TitansMAGConfig, exist_ok=True)
AutoModel.register(TitansMAGConfig, TitansMAGModel, exist_ok=True)
AutoModelForCausalLM.register(TitansMAGConfig, TitansMAGForCausalLM, exist_ok=True)

__all__ = ['TitansMAGConfig', 'TitansMAGForCausalLM', 'TitansMAGModel']
