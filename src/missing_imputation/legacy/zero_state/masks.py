from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .sampler import (
    SampledState,
    STATE_STRUCTURAL_ZERO,
    STATE_SAMPLING_ZERO,
    STATE_NONZERO,
    STATE_UNCERTAIN,
)


@dataclass
class ZeroStateMasks:
    M_target_zero: np.ndarray
    M_target_nonzero: np.ndarray
    M_target_uncertain: np.ndarray
    M_target: np.ndarray
    location_index: pd.Index
    time_index: pd.DatetimeIndex
    feature_names: list[str]
    seed: int
    variant_name: str
    quality_flags: dict[str, bool] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            M_target_zero=self.M_target_zero,
            M_target_nonzero=self.M_target_nonzero,
            M_target_uncertain=self.M_target_uncertain,
            M_target=self.M_target,
            location_index=self.location_index.to_numpy(),
            time_index=self.time_index.to_numpy(),
            feature_names=np.array(self.feature_names, dtype=object),
            seed=self.seed,
            variant_name=self.variant_name,
            quality_flags=np.array(
                [(k, v) for k, v in self.quality_flags.items()], dtype=object
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ZeroStateMasks":
        data = np.load(path, allow_pickle=True)
        quality_flags = {}
        if "quality_flags" in data.files:
            for pair in data["quality_flags"]:
                quality_flags[str(pair[0])] = bool(pair[1])
        return cls(
            M_target_zero=data["M_target_zero"],
            M_target_nonzero=data["M_target_nonzero"],
            M_target_uncertain=data["M_target_uncertain"],
            M_target=data["M_target"],
            location_index=pd.Index(data["location_index"]),
            time_index=pd.DatetimeIndex(data["time_index"]),
            feature_names=data["feature_names"].tolist(),
            seed=int(data["seed"]),
            variant_name=str(data["variant_name"]),
            quality_flags=quality_flags,
        )

    def verify_partition(self) -> tuple[bool, int]:
        overlap_zn = (self.M_target_zero & self.M_target_nonzero).sum()
        overlap_zu = (self.M_target_zero & self.M_target_uncertain).sum()
        overlap_nu = (self.M_target_nonzero & self.M_target_uncertain).sum()

        union = self.M_target_zero | self.M_target_nonzero | self.M_target_uncertain
        missing = (union != self.M_target).sum()

        violations = int(overlap_zn + overlap_zu + overlap_nu + missing)
        return violations == 0, violations


def create_target_masks_from_states(
    sampled_states: dict[str, SampledState],
    panel,
    M_target: np.ndarray,
    seed: int,
    quality_flags: dict[str, bool],
    use_feasible: bool = True,
) -> dict[str, ZeroStateMasks]:
    n_locations = panel.n_locations
    n_times = panel.n_times
    n_features = panel.n_features

    masks = {}

    for variant_name, state in sampled_states.items():
        variant_idx = panel.feature_names.index(variant_name)

        M_target_zero = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
        M_target_nonzero = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
        M_target_uncertain = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)

        variant_target = M_target[:, :, variant_idx].astype(bool)
        target_locs, target_times = np.where(variant_target)
        n_target = len(target_locs)

        states = state.sampled_state_feasible if use_feasible else state.sampled_state_raw
        assert len(states) == n_target, (
            f"{variant_name}: sampled states length {len(states)} != per-variant target count {n_target}"
        )

        for i in range(n_target):
            loc_idx = target_locs[i]
            time_idx = target_times[i]
            s = states[i]
            if s == STATE_NONZERO:
                M_target_nonzero[loc_idx, time_idx, variant_idx] = 1
            elif s in (STATE_STRUCTURAL_ZERO, STATE_SAMPLING_ZERO):
                M_target_zero[loc_idx, time_idx, variant_idx] = 1
            else:
                M_target_uncertain[loc_idx, time_idx, variant_idx] = 1

        M_target_variant_mask = (
            M_target_zero[:, :, variant_idx]
            | M_target_nonzero[:, :, variant_idx]
            | M_target_uncertain[:, :, variant_idx]
        )
        combined = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
        combined[:, :, variant_idx] = M_target_variant_mask

        masks[variant_name] = ZeroStateMasks(
            M_target_zero=M_target_zero,
            M_target_nonzero=M_target_nonzero,
            M_target_uncertain=M_target_uncertain,
            M_target=combined,
            location_index=panel.location_index,
            time_index=panel.time_index,
            feature_names=panel.feature_names,
            seed=seed,
            variant_name=variant_name,
            quality_flags=dict(quality_flags),
        )

    return masks


def combine_variant_masks(
    variant_masks: dict[str, ZeroStateMasks],
    M_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not variant_masks:
        raise ValueError("No variant masks provided")

    first_mask = next(iter(variant_masks.values()))
    n_locations, n_times, n_features = first_mask.M_target_zero.shape

    M_target_zero = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
    M_target_nonzero = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
    M_target_uncertain = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)

    for variant_name, mask in variant_masks.items():
        variant_idx = mask.feature_names.index(variant_name)
        M_target_zero[:, :, variant_idx] = mask.M_target_zero[:, :, variant_idx]
        M_target_nonzero[:, :, variant_idx] = mask.M_target_nonzero[:, :, variant_idx]
        M_target_uncertain[:, :, variant_idx] = mask.M_target_uncertain[:, :, variant_idx]

    union = M_target_zero | M_target_nonzero | M_target_uncertain
    assert np.array_equal(union, M_target.astype(np.uint8)), (
        f"Combined target masks do not equal M_target: union={int(union.sum())}, "
        f"M_target={int(M_target.sum())}"
    )

    return M_target_zero, M_target_nonzero, M_target_uncertain, M_target.astype(np.uint8)


def save_combined_masks(
    M_target_zero: np.ndarray,
    M_target_nonzero: np.ndarray,
    M_target_uncertain: np.ndarray,
    M_target: np.ndarray,
    location_index: pd.Index,
    time_index: pd.DatetimeIndex,
    feature_names: list[str],
    seed: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        M_target_zero=M_target_zero,
        M_target_nonzero=M_target_nonzero,
        M_target_uncertain=M_target_uncertain,
        M_target=M_target,
        location_index=location_index.to_numpy(),
        time_index=time_index.to_numpy(),
        feature_names=np.array(feature_names, dtype=object),
        seed=seed,
    )