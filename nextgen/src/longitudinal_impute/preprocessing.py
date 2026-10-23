"""Train-fitted zero policy and metadata encoding; never overwrite raw counts."""
from typing import Protocol
from .contracts import ConditioningPanel, FitScope


class Preprocessor(Protocol):
    def fit(self, panel: ConditioningPanel, scope: FitScope) -> None: ...

    def transform(self, panel: ConditioningPanel) -> ConditioningPanel: ...
