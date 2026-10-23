"""Donor pools and caches are scoped to dataset/split/mask/config hashes."""
from typing import Protocol
from .contracts import ConditioningPanel, FitScope, InitializedPanel


class Initializer(Protocol):
    def fit(self, panel: ConditioningPanel, scope: FitScope) -> None: ...

    def transform(self, panel: ConditioningPanel) -> InitializedPanel: ...
