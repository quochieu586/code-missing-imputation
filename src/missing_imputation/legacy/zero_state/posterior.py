from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .zinb import PosteriorProbs


@dataclass
class ZeroStatePosterior:
    variant_posteriors: dict[str, PosteriorProbs]
    location_index: pd.Index
    time_index: pd.DatetimeIndex
    feature_names: list[str]
    quality_flags: dict[str, bool] = field(default_factory=dict)

    def get_variant_posterior(self, variant_name: str) -> PosteriorProbs:
        return self.variant_posteriors.get(variant_name)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "location_index": self.location_index.to_numpy(),
            "time_index": self.time_index.to_numpy(),
            "feature_names": np.array(self.feature_names, dtype=object),
            "quality_flags": np.array(
                [(k, v) for k, v in self.quality_flags.items()], dtype=object
            ),
        }
        for variant, post in self.variant_posteriors.items():
            save_dict[f"{variant}_p_structural"] = post.p_structural
            save_dict[f"{variant}_p_sampling"] = post.p_sampling
            save_dict[f"{variant}_p_nonzero"] = post.p_nonzero
            save_dict[f"{variant}_target_indices"] = post.target_indices
        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: str | Path) -> "ZeroStatePosterior":
        data = np.load(path, allow_pickle=True)
        variant_posteriors = {}
        feature_names = data["feature_names"].tolist()

        for variant in feature_names:
            p_struct = data.get(f"{variant}_p_structural")
            p_samp = data.get(f"{variant}_p_sampling")
            p_nonzero = data.get(f"{variant}_p_nonzero")
            targ_idx = data.get(f"{variant}_target_indices")
            if p_struct is not None:
                variant_posteriors[variant] = PosteriorProbs(
                    p_structural=p_struct,
                    p_sampling=p_samp,
                    p_nonzero=p_nonzero,
                    target_indices=targ_idx,
                )

        quality_flags = {}
        if "quality_flags" in data.files:
            for pair in data["quality_flags"]:
                quality_flags[str(pair[0])] = bool(pair[1])

        return cls(
            variant_posteriors=variant_posteriors,
            location_index=pd.Index(data["location_index"]),
            time_index=pd.DatetimeIndex(data["time_index"]),
            feature_names=feature_names,
            quality_flags=quality_flags,
        )

    def verify_probabilities_sum_to_one(self, atol: float = 1e-6) -> tuple[bool, int]:
        violations = 0
        for variant, post in self.variant_posteriors.items():
            total = post.p_structural + post.p_sampling + post.p_nonzero
            violations += int(np.sum(np.abs(total - 1.0) > atol))
        return violations == 0, violations


def compute_global_day_indices(
    panel, loc_indices: np.ndarray, time_indices: np.ndarray
) -> np.ndarray:
    if len(panel.time_index) == 0:
        return np.zeros(len(loc_indices), dtype=np.float64)
    t0 = panel.time_index[0]
    return np.array(
        [(panel.time_index[int(t)] - t0).days for t in time_indices], dtype=np.float64
    )


def build_zero_state_posterior(
    fitted_models: dict,
    panel,
    M_target: np.ndarray,
    quality_flags: dict[str, bool],
    M_observed: np.ndarray,
) -> ZeroStatePosterior:
    variant_posteriors = {}

    for variant_name in panel.feature_names:
        variant_idx = panel.feature_names.index(variant_name)
        target_locs, target_times = np.where(M_target[:, :, variant_idx] == 1)
        n_target = len(target_locs)
        target_indices = np.column_stack([
            target_locs,
            target_times,
            np.full(n_target, variant_idx, dtype=int),
        ])

        entry = fitted_models.get(variant_name)
        if entry is None:
            variant_posteriors[variant_name] = PosteriorProbs(
                p_structural=np.full(n_target, np.nan),
                p_sampling=np.full(n_target, np.nan),
                p_nonzero=np.full(n_target, np.nan),
                target_indices=target_indices,
            )
            continue

        model = entry["model"]
        fit_result = entry["fit_result"]

        if fit_result is None or fit_result.model_type.value != "zinb" or n_target == 0:
            variant_posteriors[variant_name] = PosteriorProbs(
                p_structural=np.full(n_target, np.nan),
                p_sampling=np.full(n_target, np.nan),
                p_nonzero=np.full(n_target, np.nan),
                target_indices=target_indices,
            )
            continue

        total_seq = panel.total_sequence[target_locs, target_times].astype(np.float64)
        day_index = compute_global_day_indices(panel, target_locs, target_times)

        counts_grid = panel.counts[:, :, variant_idx].astype(np.float64)
        lag_counts, lag_avail, lead_counts, lead_avail = _compute_lag_lead_for_cells(
            counts_grid, M_observed[:, :, variant_idx], target_locs, target_times
        )

        if getattr(model, "variant_names", None) is not None:
            variant_code = model.variant_names.index(variant_name)
            post = model.predict_proba_variant(
                variant_code=variant_code,
                total_seq=total_seq,
                day_index=day_index,
                lag_counts=lag_counts,
                lag_avail=lag_avail,
                lead_counts=lead_counts,
                lead_avail=lead_avail,
            )
        else:
            post = model.predict_proba(
                total_seq=total_seq,
                day_index=day_index,
                lag_counts=lag_counts,
                lag_avail=lag_avail,
                lead_counts=lead_counts,
                lead_avail=lead_avail,
            )
        variant_posteriors[variant_name] = PosteriorProbs(
            p_structural=post.p_structural,
            p_sampling=post.p_sampling,
            p_nonzero=post.p_nonzero,
            target_indices=target_indices,
        )

    return ZeroStatePosterior(
        variant_posteriors=variant_posteriors,
        location_index=panel.location_index,
        time_index=panel.time_index,
        feature_names=panel.feature_names,
        quality_flags=dict(quality_flags),
    )


def _compute_lag_lead_for_cells(
    counts_grid: np.ndarray,
    M_observed_variant: np.ndarray,
    loc_indices: np.ndarray,
    time_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(loc_indices)
    lag_counts = np.zeros(n, dtype=np.float64)
    lag_avail = np.zeros(n, dtype=np.float64)
    lead_counts = np.zeros(n, dtype=np.float64)
    lead_avail = np.zeros(n, dtype=np.float64)
    n_times = counts_grid.shape[1]

    for i in range(n):
        loc, t = int(loc_indices[i]), int(time_indices[i])
        if t - 1 >= 0 and M_observed_variant[loc, t - 1] == 1:
            lag_counts[i] = counts_grid[loc, t - 1]
            lag_avail[i] = 1.0
        if t + 1 < n_times and M_observed_variant[loc, t + 1] == 1:
            lead_counts[i] = counts_grid[loc, t + 1]
            lead_avail[i] = 1.0

    return lag_counts, lag_avail, lead_counts, lead_avail