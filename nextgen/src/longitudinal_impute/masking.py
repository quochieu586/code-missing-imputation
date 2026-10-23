"""Trusted masking boundary. O=observed, H=hidden subset of O, C=O & ~H.

No temporal causal mask in the primary offline protocol. Hide H globally by
row/feature identity before forming windows, donors, scalers or derived Other.
"""
from typing import Protocol
from .contracts import Array, ConditioningPanel, EvaluationLabels, RawPanel


class MaskBuilder(Protocol):
    def build(self, panel: RawPanel, *, scenario: str, seed: int) -> Array: ...


class ConditioningBoundary(Protocol):
    def hide(self, panel: RawPanel, hidden_mask: Array
             ) -> tuple[ConditioningPanel, EvaluationLabels]: ...
