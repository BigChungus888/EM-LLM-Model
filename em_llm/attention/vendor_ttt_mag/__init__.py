"""Compatibility wrapper so hybrid attention can import vendored modules under em_llm.attention."""

from em_llm.vendor_ttt_mag.linear_sgd import LinearSGD
from em_llm.vendor_ttt_mag.ttt import TTT
from em_llm.vendor_ttt_mag.titans_mag import MemoryAsGate

__all__ = ["LinearSGD", "TTT", "MemoryAsGate"]
