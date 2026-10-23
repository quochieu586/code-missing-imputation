"""Loss uses only valid labels. Dataset-wide masks precede windowing.

No pseudolabel is promoted to observation. Validation selects checkpoints;
test labels never choose epochs, hyperparameters or an outer-loop stop point.
"""
from typing import Protocol, Sequence
from .contracts import EvaluationLabels, ModelBatch
from .models.interfaces import Predictor


class Trainer(Protocol):
    def fit(self, model: Predictor,
            train: Sequence[tuple[ModelBatch, EvaluationLabels]],
            validation: Sequence[tuple[ModelBatch, EvaluationLabels]]) -> Predictor: ...
