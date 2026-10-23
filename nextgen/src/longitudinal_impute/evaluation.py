"""Score artificial targets; native missing values have no accuracy labels.

Counts/relative errors primary; zero-aware compositional, temporal and feature
relationship metrics secondary. Report support sizes and location-level CIs.
"""
from typing import Any, Mapping, Protocol
from .contracts import EvaluationLabels, ImputationResult


class Evaluator(Protocol):
    def score(self, result: ImputationResult, labels: EvaluationLabels
              ) -> Mapping[str, Any]: ...
