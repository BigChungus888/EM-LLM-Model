from sparse_linear_attention.layers import (
    TTT, MemoryAsGate
)

from sparse_linear_attention.layers import TTT, MemoryAsGate
from sparse_linear_attention.models import (
    TTTForCausalLM, TTTModel, TitansMAGForCausalLM, TitansMAGModel
)

__all__ = [
    'TTT', 'TTTForCausalLM', 'TTTModel', 'MemoryAsGate', 'TitansMAGForCausalLM', 'TitansMAGModel'
]

__version__ = '0.4.0'