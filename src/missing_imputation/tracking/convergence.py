"""Convergence tracking for iterative refinement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConvergenceState:
    """Tracks convergence of iterative refinement."""

    history: list[dict] = field(default_factory=list)
    patience: int = 5
    min_delta: float = 1e-4
    max_rounds: int = 10

    def record(self, round_num: int, metrics: dict) -> None:
        self.history.append({"round": round_num, **metrics})

    def should_stop(self) -> tuple[bool, str]:
        """Check if iteration should stop.

        Returns:
            (should_stop, reason).
        """
        if len(self.history) >= self.max_rounds:
            return True, "max_rounds_reached"

        if len(self.history) < self.patience + 1:
            return False, "not_enough_history"

        recent = self.history[-self.patience:]
        jsd_values = [h.get("validation_jsd", np.inf) for h in recent]

        if len(jsd_values) >= 2:
            improvements = [
                jsd_values[i] - jsd_values[i + 1]
                for i in range(len(jsd_values) - 1)
            ]
            if all(abs(imp) < self.min_delta for imp in improvements):
                return True, "no_improvement_patience"

        return False, "continue"

    def get_best_round(self, metric: str = "validation_jsd") -> int:
        """Get the round with best metric value."""
        if not self.history:
            return 0
        best_idx = min(
            range(len(self.history)),
            key=lambda i: self.history[i].get(metric, np.inf),
        )
        return self.history[best_idx]["round"]