from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HybridLayerState:
    episodic_state: Any
    recurrent_state: Any = None
    aux_state: Optional[Any] = None


@dataclass
class TitansHybridLayerState:
    episodic_state: Any
    titans_state: Any = None
    aux_state: Optional[Any] = None

    @property
    def recurrent_state(self):
        return self.titans_state

    @recurrent_state.setter
    def recurrent_state(self, value):
        self.titans_state = value


def get_episodic_state(layer_state):
    if isinstance(layer_state, (HybridLayerState, TitansHybridLayerState)):
        return layer_state.episodic_state
    return layer_state


def get_recurrent_state(layer_state):
    if isinstance(layer_state, TitansHybridLayerState):
        return layer_state.titans_state
    if isinstance(layer_state, HybridLayerState):
        return layer_state.recurrent_state
    return layer_state
