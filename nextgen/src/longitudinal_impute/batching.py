"""Windows are formed only after a global hidden-cell identity is established.

Full panels, length buckets and overlapping windows are implementation choices.
Padding is excluded from attention keys/loss; overlap predictions need a
predeclared merge rule and each target must be scored exactly once.
"""
from typing import Protocol, Sequence
from .contracts import ConditioningPanel, InitializedPanel, ModelBatch


class BatchBuilder(Protocol):
    def build(self, panel: ConditioningPanel, initialized: InitializedPanel
              ) -> Sequence[ModelBatch]: ...
