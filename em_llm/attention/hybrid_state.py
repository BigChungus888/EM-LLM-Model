from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HybridLayerState:
    episodic_state: Any
    recurrent_state: Any = None
    aux_state: Optional[Any] = None
