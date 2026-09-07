from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .zinb import PosteriorProbs


STATE_STRUCTURAL_ZERO = 0
STATE_SAMPLING_ZERO = 1
STATE_NONZERO = 2
STATE_UNCERTAIN = 3


@dataclass
class SampledState:
    sampled_state_raw: np.ndarray
    sampled_state_feasible: np.ndarray
    feasibility_overrides: list[dict] = field(default_factory=list)
    seed: int = 42
    n_target: int = 0
    n_nonzero_raw: int = 0
    n_nonzero_feasible: int = 0
    n_uncertain: int = 0
    n_overridden: int = 0
    target_locs: np.ndarray | None = None
    target_times: np.ndarray | None = None


def sample_states_three_way(
    posterior: PosteriorProbs,
    seed: int,
    threshold_nonzero: float,
    threshold_zero: float,
    quality_pass: bool = True,
) -> np.ndarray:
    n = len(posterior.p_nonzero)
    states = np.full(n, STATE_UNCERTAIN, dtype=np.uint8)

    if not quality_pass:
        return states

    p_nonzero = posterior.p_nonzero
    p_zero = posterior.p_structural + posterior.p_sampling

    nonzero_mask = p_nonzero >= threshold_nonzero
    zero_mask = (~nonzero_mask) & (p_zero >= threshold_zero)

    states[nonzero_mask] = STATE_NONZERO
    states[zero_mask] = STATE_STRUCTURAL_ZERO

    return states


def apply_feasibility_projection_uncertain(
    states: np.ndarray,
    p_nonzero: np.ndarray,
    total_seq: np.ndarray,
    other_counts: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    n = len(states)
    feasible_states = states.copy()
    overrides: list[dict] = []

    budget = np.maximum(np.asarray(total_seq, dtype=np.float64) - np.asarray(other_counts, dtype=np.float64), 0)

    nonzero_indices = np.where(feasible_states == STATE_NONZERO)[0]
    if len(nonzero_indices) == 0:
        return feasible_states, overrides

    over_budget = nonzero_indices[budget[nonzero_indices] < 1]

    if len(over_budget) > 0:
        over_budget_ranked = over_budget[np.argsort(p_nonzero[over_budget])]
        for idx in over_budget_ranked:
            old_state = int(feasible_states[idx])
            feasible_states[idx] = STATE_UNCERTAIN
            overrides.append({
                "cell_index": int(idx),
                "old_state": old_state,
                "new_state": STATE_UNCERTAIN,
                "reason": "budget_exceeded_uncertain",
                "budget": float(budget[idx]),
                "p_nonzero": float(p_nonzero[idx]),
            })

    remaining_nonzero = np.where(feasible_states == STATE_NONZERO)[0]
    if len(remaining_nonzero) > 0:
        total_budget = budget[remaining_nonzero].sum()
        if len(remaining_nonzero) > total_budget:
            in_budget_mask = budget[remaining_nonzero] > 0
            demotable = remaining_nonzero[in_budget_mask]
            if len(demotable) > 0:
                n_to_demote = len(remaining_nonzero) - int(total_budget)
                demotable_ranked = demotable[np.argsort(p_nonzero[demotable])]
                to_demote = demotable_ranked[:n_to_demote]
                for idx in to_demote:
                    old_state = int(feasible_states[idx])
                    feasible_states[idx] = STATE_UNCERTAIN
                    overrides.append({
                        "cell_index": int(idx),
                        "old_state": old_state,
                        "new_state": STATE_UNCERTAIN,
                        "reason": "total_budget_exceeded_uncertain",
                        "budget": float(budget[idx]),
                        "p_nonzero": float(p_nonzero[idx]),
                    })

    return feasible_states, overrides


def run_seeded_sampling_three_way(
    posteriors: dict[str, PosteriorProbs],
    quality_flags: dict[str, bool],
    panel,
    M_target: np.ndarray,
    seeds: list[int],
    thresholds: dict[str, float],
    default_threshold: float = 0.5,
) -> dict[int, dict[str, SampledState]]:
    results = {}
    n_loc, n_time = panel.shape[:2]

    for seed in seeds:
        raw_states_by_variant: dict[str, np.ndarray] = {}
        target_coords: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for variant_name, post in posteriors.items():
            variant_idx = panel.feature_names.index(variant_name)
            locs, times = np.where(M_target[:, :, variant_idx] == 1)
            target_coords[variant_name] = (locs, times)

            quality_pass = quality_flags.get(variant_name, False)
            threshold = thresholds.get(variant_name, default_threshold)

            states = sample_states_three_way(
                posterior=post,
                seed=seed,
                threshold_nonzero=threshold,
                threshold_zero=0.5,
                quality_pass=quality_pass,
            )
            raw_states_by_variant[variant_name] = states

        state_grids: dict[str, np.ndarray] = {}
        for variant_name, states in raw_states_by_variant.items():
            locs, times = target_coords[variant_name]
            grid = np.full((n_loc, n_time), STATE_UNCERTAIN, dtype=np.uint8)
            grid[locs, times] = states
            state_grids[variant_name] = grid

        seed_results = {}
        for variant_name, post in posteriors.items():
            variant_idx = panel.feature_names.index(variant_name)
            locs, times = target_coords[variant_name]
            n_target = len(locs)
            raw_states = raw_states_by_variant[variant_name]

            other_counts = _compute_other_counts(
                panel, locs, times, variant_idx, state_grids, variant_name,
            )

            feasible_states, overrides = apply_feasibility_projection_uncertain(
                raw_states,
                post.p_nonzero,
                panel.total_sequence[locs, times].astype(np.float64),
                other_counts.astype(np.float64),
            )

            seed_results[variant_name] = SampledState(
                sampled_state_raw=raw_states,
                sampled_state_feasible=feasible_states,
                feasibility_overrides=overrides,
                seed=seed,
                n_target=n_target,
                n_nonzero_raw=int(np.sum(raw_states == STATE_NONZERO)),
                n_nonzero_feasible=int(np.sum(feasible_states == STATE_NONZERO)),
                n_uncertain=int(np.sum(feasible_states == STATE_UNCERTAIN)),
                n_overridden=len(overrides),
                target_locs=locs,
                target_times=times,
            )
        results[seed] = seed_results

    return results


def _compute_other_counts(
    panel,
    loc_indices: np.ndarray,
    time_indices: np.ndarray,
    exclude_variant_idx: int,
    state_grids: dict[str, np.ndarray],
    exclude_variant_name: str,
) -> np.ndarray:
    n = len(loc_indices)
    other = np.zeros(n, dtype=np.int64)
    variant_names = list(panel.feature_names)

    for i in range(n):
        loc_idx, time_idx = int(loc_indices[i]), int(time_indices[i])
        for f_idx in range(panel.n_features):
            if f_idx == exclude_variant_idx:
                continue
            if panel.M_observed[loc_idx, time_idx, f_idx] == 1:
                other[i] += panel.counts[loc_idx, time_idx, f_idx]
            else:
                other_variant = variant_names[f_idx]
                grid = state_grids.get(other_variant)
                if grid is not None and grid[loc_idx, time_idx] == STATE_NONZERO:
                    other[i] += 1
    return other


def save_sampled_states(
    results: dict[int, dict[str, SampledState]],
    path,
) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for seed, variant_results in results.items():
        for variant, state in variant_results.items():
            prefix = f"seed{seed}_{variant}"
            save_dict[f"{prefix}_raw"] = state.sampled_state_raw
            save_dict[f"{prefix}_feasible"] = state.sampled_state_feasible
            save_dict[f"{prefix}_seed"] = np.array([state.seed])
            save_dict[f"{prefix}_n_target"] = np.array([state.n_target])
            save_dict[f"{prefix}_n_nonzero_raw"] = np.array([state.n_nonzero_raw])
            save_dict[f"{prefix}_n_nonzero_feasible"] = np.array([state.n_nonzero_feasible])
            save_dict[f"{prefix}_n_uncertain"] = np.array([state.n_uncertain])
            save_dict[f"{prefix}_n_overridden"] = np.array([state.n_overridden])
    np.savez_compressed(path, **save_dict)