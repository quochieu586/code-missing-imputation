"""Offline temporal + feature encoder contract; padding needs explicit handling."""
from typing import Protocol
from ..contracts import Array, ModelBatch


class Predictor(Protocol):
    def predict(self, batch: ModelBatch) -> Array: ...
