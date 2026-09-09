"""Run all three Tsagris baselines with masked CV, select champion, produce dataset_0.

Pipeline (Section 4.3):
1. Load and validate data.
2. Build complete-row neighbor pool (516 rows).
3. Repeated masked CV simulating empirical missingness patterns.
4. Time-block evaluation on complete rows.
5. Select champion per Section 6.3 rules.
6. Impute all incomplete rows, integerize with largest-remainder, enforce invariants.
7. Write dataset_0.csv, original_mask.npz, metrics, manifest, selection report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..baselines.adaptive_jsd_alpha_knn import AdaptiveJSDAlphaKNN
from ..baselines.frechet import frechet_mean
from ..baselines.jsd import jsd_one_vs_many, partial_jsd_one_vs_many
from ..data.closure import (
    build_full_composition,
    proportions_to_counts_largest_remainder,
    validate_closure,
)
from ..data.loader import load_config, load_covariants
from ..data.masks import extract_patterns
from ..tracking.manifest import create_run_manifest


def load_split_ids(splits_dir: str | Path) -> dict:
    """Load split IDs from configs/splits/ directory."""
    splits_dir = Path(splits_dir)
    splits = {}
    for name in ["random_cell", "empirical_pattern", "time_block", "country_holdout"]:
        path = splits_dir / f"{name}.json"
        with path.open(encoding="utf-8") as f:
            splits[name] = json.load(f)
    return splits


def load_baseline_pool_split(splits_dir: str | Path, n_complete: int, n_parts: int) -> dict:
    """Load the shared complete-pool split used by all three Tsagris baselines.

    Regenerate with `python scripts/generate_split_ids.py` if this is missing or
    stale. Indices are positions inside the complete-row pool, which is the index
    space the benchmark actually evaluates in.
    """
    path = Path(splits_dir) / "baseline_pool.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python scripts/generate_split_ids.py"
        )
    with path.open(encoding="utf-8") as f:
        split = json.load(f)
    if split["n_complete_rows"] != n_complete:
        raise ValueError(
            f"baseline_pool.json has {split['n_complete_rows']} complete rows, "
            f"data has {n_complete}; regenerate the split IDs"
        )
    if split.get("n_composition_parts") != n_parts:
        raise ValueError(
            f"baseline_pool.json is for {split.get('n_composition_parts')} composition "
            f"parts, data has {n_parts}; regenerate the split IDs"
        )
    return split


def _mse_proportions(
    true_props: np.ndarray,
    imputed_props: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute MSE on proportions at masked positions."""
    if not mask.any():
        return float(np.log(2))  # or np.inf
    return float(np.mean((true_props[mask] - imputed_props[mask]) ** 2))

K_GRID_DEFAULT = [3, 5, 7, 10, 15]
ALPHA_GRID_DEFAULT = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEEDS_DEFAULT = [42, 123, 456, 789, 999]
MIN_PATTERN_SUPPORT_DEFAULT = 30


@dataclass
class MethodScore:
    method: str
    params: dict
    mse_empirical: float
    mse_timeblock: float
    jsd_empirical: float
    jsd_timeblock: float
    runtime_seconds: float
    invariants_ok: bool = True
    notes: str = ""

    @property
    def mse_mean(self) -> float:
        return 0.5 * (self.mse_empirical + self.mse_timeblock)

    @property
    def jsd_mean(self) -> float:
        return 0.5 * (self.jsd_empirical + self.jsd_timeblock)


def _load_arrays(config_path: str | Path):
    """Load the data as a closed (variants + other) composition.

    The residual is part of the composition so the neighbour pool sums to 1 and
    kNN imputes `other` instead of it being forced to zero.
    """
    config = load_config(config_path)
    df = load_covariants(config.path, config)
    counts, observed_mask, proportions, total_seq = build_full_composition(
        df, config.variant_components, config.total_sequence_col
    )
    return config, df, observed_mask, counts, total_seq, proportions


def _impute_rows(
    pool_props: np.ndarray,
    query_props: np.ndarray,
    query_obs: np.ndarray,
    k: int,
    alpha: float,
) -> np.ndarray:
    """Impute missing cells of query rows using kNN + Frechet mean from pool."""
    n, d = query_props.shape
    result = np.zeros((n, d), dtype=np.float64)
    pool_mean = pool_props.mean(axis=0)

    for i in range(n):
        obs_idx = np.where(query_obs[i])[0]

        if len(obs_idx) == 0:
            result[i] = pool_mean
            continue

        distances = partial_jsd_one_vs_many(query_props[i, obs_idx], pool_props[:, obs_idx])
        k_actual = min(k, pool_props.shape[0])
        neighbor_idx = np.argpartition(distances, k_actual - 1)[:k_actual]
        neighbors = pool_props[neighbor_idx]
        mean_comp = frechet_mean(neighbors, alpha=alpha)

        row = query_props[i].copy()
        missing_idx = np.where(~query_obs[i])[0]
        row[missing_idx] = mean_comp[missing_idx]

        total = row.sum()
        result[i] = row / total if total > 0 else pool_mean

    return result


def _fixed_imputer(k: int, alpha: float):
    """Imputer closure with constant (k, alpha)."""

    def fn(pool: np.ndarray, query: np.ndarray, obs: np.ndarray) -> np.ndarray:
        return _impute_rows(pool, query[np.newaxis, :], obs[np.newaxis, :], k, alpha)[0]

    return fn


def _adaptive_imputer(pattern_params: dict, global_k: int, global_alpha: float):
    """Imputer closure that looks (k, alpha) up by missingness pattern."""

    def fn(pool: np.ndarray, query: np.ndarray, obs: np.ndarray) -> np.ndarray:
        alpha, k = pattern_params.get(tuple((~obs).tolist()), (global_alpha, global_k))
        return _impute_rows(pool, query[np.newaxis, :], obs[np.newaxis, :], k, alpha)[0]

    return fn


def _pattern_masks_from_split(split: dict) -> list[np.ndarray]:
    """Per complete-pool row, the observed mask of its assigned missingness pattern."""
    table = [np.array(p, dtype=bool) for p in split["pattern_table"]]
    return [~table[i] for i in split["pattern_index_of_row"]]


def _empirical_pattern_eval(
    complete_props: np.ndarray,
    split: dict,
    impute_fn,
) -> tuple[float, float]:
    """K-fold masked CV on the complete pool using the shared split IDs.

    Every method sees the identical folds and the identical per-row missingness
    pattern, which is what plan S4.5 asks for. Scoring covers ALL artificially
    masked cells, including the ones whose true proportion is 0: 78% of cells in
    the pool are exact zeros, and excluding them made the metric blind to
    precisely the behaviour the zero-preservation work is about.

    Returns:
        (mean_jsd, mean_mse) over held-out rows.
    """
    obs_masks = _pattern_masks_from_split(split)
    fold_of_row = np.array(split["fold_of_row"])
    n_folds = int(split["n_folds"])

    jsd_values: list[float] = []
    mse_values: list[float] = []

    for fold in range(n_folds):
        test_rows = np.where(fold_of_row == fold)[0]
        pool = complete_props[fold_of_row != fold]
        if len(test_rows) == 0 or len(pool) == 0:
            continue
        for vi in test_rows:
            obs_mask = obs_masks[vi]
            if not obs_mask.any():
                continue
            true_row = complete_props[vi]
            query = true_row.copy()
            query[~obs_mask] = 0.0

            imputed = impute_fn(pool, query, obs_mask)

            mask = ~obs_mask
            jsd_values.append(float(jsd_one_vs_many(true_row, imputed[np.newaxis, :])[0]))
            mse_values.append(_mse_proportions(true_row, imputed, mask))

    return (
        float(np.mean(jsd_values)) if jsd_values else float(np.log(2)),
        float(np.mean(mse_values)) if mse_values else float(np.log(2)),
    )


def _time_block_eval(
    complete_props: np.ndarray,
    split: dict,
    impute_fn,
    mask_fraction: float = 0.3,
) -> tuple[float, float]:
    """Contiguous time blocks per location held out, cells masked inside them.

    The held-out positions come from the shared split file and are already in
    complete-pool index space. The masking RNG is seeded once from the split seed
    instead of being re-created per row, which previously handed every row the
    identical mask and ignored the seed list entirely.

    Returns:
        (mean_jsd, mean_mse) over the artificially masked cells.
    """
    held_out = np.array(split["time_block_test_rows"], dtype=np.int64)
    if len(held_out) == 0:
        return float(np.log(2)), float(np.log(2))

    pool_mask = np.ones(complete_props.shape[0], dtype=bool)
    pool_mask[held_out] = False
    pool = complete_props[pool_mask]
    if len(pool) == 0:
        return float(np.log(2)), float(np.log(2))

    rng = np.random.default_rng(int(split["seed"]))
    n_parts = complete_props.shape[1]

    jsd_values: list[float] = []
    mse_values: list[float] = []

    for local in held_out:
        true_row = complete_props[local]
        mask = rng.random(n_parts) < mask_fraction
        if not mask.any() or mask.all():
            continue
        obs_mask = ~mask
        query = true_row.copy()
        query[mask] = 0.0

        imputed = impute_fn(pool, query, obs_mask)

        jsd_values.append(float(jsd_one_vs_many(true_row, imputed[np.newaxis, :])[0]))
        mse_values.append(_mse_proportions(true_row, imputed, mask))

    return (
        float(np.mean(jsd_values)) if jsd_values else float(np.log(2)),
        float(np.mean(mse_values)) if mse_values else float(np.log(2)),
    )


def _tune_grid(
    complete_props: np.ndarray,
    split: dict,
    k_grid: list[int],
    alpha_grid: list[float],
) -> tuple[float, dict]:
    """Grid search (k, alpha) on the empirical-pattern recipe.

    Selection is by MSE, the primary metric of plan S4.6/S9.1. It previously
    ranked by JSD while the champion rule compared MSE.
    """
    best_mse = np.inf
    best = {"k": k_grid[0], "alpha": alpha_grid[0]}

    for k in k_grid:
        for alpha in alpha_grid:
            _, mse = _empirical_pattern_eval(
                complete_props, split, _fixed_imputer(k, alpha)
            )
            if mse < best_mse:
                best_mse = mse
                best = {"k": k, "alpha": float(alpha)}

    return best_mse, best


def run_baselines(
    data_path: str | Path | None = None,
    config_dir: str | Path = None,
    output_dir: str | Path = None,
    seeds: list[int] | None = None,
    splits_dir: str | Path | None = None,
) -> dict:
    """Full baseline pipeline: tune, evaluate, select, generate dataset_0."""
    project_root = Path(__file__).parents[3]
    config_dir = Path(config_dir) if config_dir else project_root / "configs" / "baseline"
    output_dir = Path(output_dir) if output_dir else project_root / "artifacts"
    seeds = seeds or SEEDS_DEFAULT
    splits_dir = splits_dir or project_root / "configs" / "splits"

    config, df, observed_mask, counts, total_seq, proportions = _load_arrays(
        project_root / "configs" / "data.yaml"
    )
    variant_cols = config.variant_components

    complete_mask = observed_mask.all(axis=1)
    complete_idx = np.where(complete_mask)[0]
    incomplete_idx = np.where(~complete_mask)[0]
    complete_props = proportions[complete_idx]
    incomplete_props = proportions[incomplete_idx]
    incomplete_obs = observed_mask[incomplete_idx]

    patterns = [p for p in extract_patterns(incomplete_obs) if p.n_observed >= 1]

    # Shared complete-pool folds and per-row patterns; all three methods use the
    # same ones so the comparison is like-for-like (plan S4.5).
    pool_split = load_baseline_pool_split(
        splits_dir, n_complete=len(complete_idx), n_parts=proportions.shape[1]
    )
    if list(pool_split["complete_row_df_indices"]) != [int(i) for i in complete_idx]:
        raise ValueError(
            "baseline_pool.json complete-row indices do not match the data; "
            "regenerate with python scripts/generate_split_ids.py"
        )

    k_grid = K_GRID_DEFAULT
    alpha_grid = ALPHA_GRID_DEFAULT
    min_support = MIN_PATTERN_SUPPORT_DEFAULT

    for cfg_file in ["jsd_alpha_knn.yaml", "adaptive_jsd_alpha_knn.yaml"]:
        cfg_path = Path(config_dir) / cfg_file
        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            hp = cfg.get("hyperparameters", {})
            k_grid = hp.get("k_grid", k_grid)
            alpha_grid = hp.get("alpha_grid", alpha_grid)
            min_support = hp.get("min_pattern_support", min_support)
            break

    scores: list[MethodScore] = []

    # --- 1. JSD-kNN: tune k only (alpha fixed at 1.0) --------------------------
    start = time.time()
    best_k = k_grid[0]
    best_k_mse = np.inf
    for k in k_grid:
        _, mse = _empirical_pattern_eval(complete_props, pool_split, _fixed_imputer(k, 1.0))
        if mse < best_k_mse:
            best_k_mse = mse
            best_k = k

    knn_fn = _fixed_imputer(best_k, 1.0)
    knn_emp_jsd, knn_emp_mse = _empirical_pattern_eval(complete_props, pool_split, knn_fn)
    knn_tb_jsd, knn_tb_mse = _time_block_eval(complete_props, pool_split, knn_fn)
    scores.append(
        MethodScore(
            method="jsd_knn",
            params={"k": best_k, "alpha": 1.0},
            mse_empirical=knn_emp_mse,
            mse_timeblock=knn_tb_mse,
            jsd_empirical=knn_emp_jsd,
            jsd_timeblock=knn_tb_jsd,
            runtime_seconds=time.time() - start,
        )
    )

    # --- 2. JSD-alpha-kNN: tune (k, alpha) ------------------------------------
    start = time.time()
    _, chosen_params = _tune_grid(complete_props, pool_split, k_grid, alpha_grid)
    global_k = int(chosen_params["k"])
    global_alpha = float(chosen_params["alpha"])

    alpha_fn = _fixed_imputer(global_k, global_alpha)
    alpha_emp_jsd, alpha_emp_mse = _empirical_pattern_eval(complete_props, pool_split, alpha_fn)
    alpha_tb_jsd, alpha_tb_mse = _time_block_eval(complete_props, pool_split, alpha_fn)
    scores.append(
        MethodScore(
            method="jsd_alpha_knn",
            params=chosen_params,
            mse_empirical=alpha_emp_mse,
            mse_timeblock=alpha_tb_mse,
            jsd_empirical=alpha_emp_jsd,
            jsd_timeblock=alpha_tb_jsd,
            runtime_seconds=time.time() - start,
        )
    )

    # --- 3. Adaptive: per-pattern (k, alpha) ----------------------------------
    # Uses the same AdaptiveJSDAlphaKNN class that generate-dataset0 runs, so the
    # benchmark measures the method the pipeline actually ships. The previous
    # inline re-implementation scored a different procedure than the one used to
    # build dataset_00_tsagris.
    start = time.time()
    adaptive = AdaptiveJSDAlphaKNN(
        k_grid=k_grid,
        alpha_grid=alpha_grid,
        min_pattern_support=min_support,
        global_k=global_k,
        global_alpha=global_alpha,
    )
    adaptive.fit(complete_props)
    pattern_params = adaptive.tune_patterns(complete_props, patterns, seed=seeds[0])
    n_tuned = sum(1 for p in patterns if p.n_rows >= min_support)

    adaptive_fn = _adaptive_imputer(pattern_params, global_k, global_alpha)
    ad_emp_jsd, ad_emp_mse = _empirical_pattern_eval(complete_props, pool_split, adaptive_fn)
    ad_tb_jsd, ad_tb_mse = _time_block_eval(complete_props, pool_split, adaptive_fn)
    scores.append(
        MethodScore(
            method="adaptive_jsd_alpha_knn",
            params={
                "global_k": global_k,
                "global_alpha": global_alpha,
                "min_pattern_support": min_support,
            },
            mse_empirical=ad_emp_mse,
            mse_timeblock=ad_tb_mse,
            jsd_empirical=ad_emp_jsd,
            jsd_timeblock=ad_tb_jsd,
            runtime_seconds=time.time() - start,
            notes=(
                f"{n_tuned}/{len(patterns)} patterns tuned per-pattern, "
                f"{len(patterns) - n_tuned} fell back to global"
            ),
        )
    )

    champion = _select_champion(scores)

    imputed_props = _final_impute(
        complete_props, incomplete_props, incomplete_obs, patterns, champion, pattern_params,
        global_k, global_alpha,
    )

    artifacts = _write_outputs(
        output_dir, df, config, variant_cols, observed_mask, counts, total_seq,
        proportions, incomplete_idx, imputed_props, scores, champion, seeds,
    )

    return {
        "champion": champion.method,
        "champion_params": champion.params,
        "scores": [
            {
                "method": s.method,
                "mse_empirical": s.mse_empirical,
                "mse_timeblock": s.mse_timeblock,
                "mse_mean": s.mse_mean,
                "jsd_empirical": s.jsd_empirical,
                "jsd_timeblock": s.jsd_timeblock,
                "jsd_mean": s.jsd_mean,
                "params": s.params,
                "runtime_seconds": s.runtime_seconds,
                "notes": s.notes,
            }
            for s in scores
        ],
        "artifacts": artifacts,
    }


def _select_champion(scores: list[MethodScore]) -> MethodScore:
    valid = [s for s in scores if s.invariants_ok]
    if not valid:
        raise ValueError("All baselines violated invariants")

    # Primary: MSE mean (lower is better)
    ranked = sorted(valid, key=lambda s: s.mse_mean)
    best = ranked[0]

    simplicity = {"jsd_knn": 0, "jsd_alpha_knn": 1, "adaptive_jsd_alpha_knn": 2}
    tolerance = 0.005  # MSE tolerance for tie-breaking
    for s in ranked[1:]:
        diff = s.mse_mean - best.mse_mean
        if diff <= tolerance and simplicity.get(s.method, 9) < simplicity.get(best.method, 9):
            best = s

    # Adaptive never selected if worse on time-block MSE
    if best.method == "adaptive_jsd_alpha_knn":
        non_adaptive = [s for s in ranked if s.method != "adaptive_jsd_alpha_knn"]
        if non_adaptive and best.mse_timeblock > non_adaptive[0].mse_timeblock:
            best = non_adaptive[0]

    return best


def _final_impute(
    complete_props: np.ndarray,
    incomplete_props: np.ndarray,
    incomplete_obs: np.ndarray,
    patterns: list,
    champion: MethodScore,
    pattern_params: dict,
    global_k: int,
    global_alpha: float,
) -> np.ndarray:
    if champion.method == "adaptive_jsd_alpha_knn":
        n, d = incomplete_props.shape
        result = np.zeros((n, d), dtype=np.float64)
        pattern_to_rows: dict[tuple, list[int]] = {}
        for i in range(n):
            key = tuple((~incomplete_obs[i]).tolist())
            pattern_to_rows.setdefault(key, []).append(i)

        for key, rows in pattern_to_rows.items():
            a, kk = pattern_params.get(key, (global_alpha, global_k))
            rows_arr = np.array(rows)
            result[rows_arr] = _impute_rows(
                complete_props, incomplete_props[rows_arr], incomplete_obs[rows_arr], kk, a
            )
        return result

    k = int(champion.params.get("k", global_k))
    alpha = float(champion.params.get("alpha", global_alpha))
    return _impute_rows(complete_props, incomplete_props, incomplete_obs, k, alpha)


def _write_outputs(
    output_dir: Path,
    df: pd.DataFrame,
    config,
    variant_cols: list[str],
    observed_mask: np.ndarray,
    counts: np.ndarray,
    total_seq: np.ndarray,
    proportions: np.ndarray,
    incomplete_idx: np.ndarray,
    imputed_props: np.ndarray,
    scores: list[MethodScore],
    champion: MethodScore,
    seeds: list[int],
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_vars = len(variant_cols)

    full_props = proportions.copy()
    full_props[incomplete_idx] = imputed_props

    # Composition includes the residual, so integerisation closes over all parts
    # and `other` is the last allocated column rather than a leftover.
    allocated = proportions_to_counts_largest_remainder(
        full_props, total_seq, observed_mask, counts
    )
    new_counts = allocated[:, :n_vars]
    other = allocated[:, n_vars]
    closure_ok, n_viol = validate_closure(new_counts, total_seq, other)

    observed_unchanged = bool(
        np.array_equal(allocated[observed_mask], counts[observed_mask])
    )
    no_nan = bool(not np.any(np.isnan(allocated)))
    non_negative = bool(np.all(allocated >= 0))
    invariants_ok = closure_ok and observed_unchanged and no_nan and non_negative

    df_out = df.copy()
    for j, col in enumerate(variant_cols):
        df_out[col] = new_counts[:, j]
    df_out[config.other_col] = other

    # Per plan Section 4.3: output dataset_00_tsagris.csv
    dataset00_path = output_dir / "dataset_00_tsagris.csv"
    df_out.to_csv(dataset00_path, index=False)

    # Masks are over the 17 variants only: downstream stages (occurrence, fusion)
    # index the variant grid, not the composition with the residual appended.
    variant_observed = observed_mask[:, :n_vars]
    m_observed_path = output_dir / "M_observed.npz"
    m_target_path = output_dir / "M_target.npz"
    np.savez_compressed(m_observed_path, M_observed=variant_observed.astype(np.uint8))
    np.savez_compressed(
        m_target_path,
        M_target=(~variant_observed).astype(np.uint8),
    )

    # Also save the combined original_mask.npz for backward compatibility
    mask_path = output_dir / "original_mask.npz"
    np.savez_compressed(
        mask_path,
        observed_mask=variant_observed,
        original_counts=counts[:, :n_vars],
        total_sequence=total_seq,
        locations=df[config.location_col].to_numpy(),
        dates=df[config.date_col].astype(str).to_numpy(),
        variant_columns=np.array(variant_cols),
    )

    metrics_df = pd.DataFrame(
        [
            {
                "method": s.method,
                "mse_empirical": s.mse_empirical,
                "mse_timeblock": s.mse_timeblock,
                "mse_mean": s.mse_mean,
                "jsd_empirical": s.jsd_empirical,
                "jsd_timeblock": s.jsd_timeblock,
                "jsd_mean": s.jsd_mean,
                "runtime_seconds": s.runtime_seconds,
                "params": json.dumps(s.params),
            }
            for s in scores
        ]
    )
    metrics_path = output_dir / "baselines" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    report_lines = [
        "# Baseline Selection Report",
        "",
        "## Selection Rule (Section 6.3 / Nhánh A)",
        "",
        "1. Discard methods violating invariants.",
        "2. Rank by mean MSE on proportions at artificial masked observed cells (primary).",
        "3. Tie within tolerance -> prefer simpler/faster method (JSD as tie-breaker).",
        "5. Adaptive never selected if worse on time-block MSE.",
        "",
        "## Results",
        "",
        "| Method | MSE empirical | MSE time-block | MSE mean | JSD empirical | JSD time-block | JSD mean | Runtime (s) | Params |",
        "|--------|--------------|----------------|----------|---------------|----------------|----------|-------------|--------|",
    ]
    for s in scores:
        report_lines.append(
            f"| {s.method} | {s.mse_empirical:.6f} | {s.mse_timeblock:.6f} | "
            f"{s.mse_mean:.6f} | {s.jsd_empirical:.6f} | {s.jsd_timeblock:.6f} | "
            f"{s.jsd_mean:.6f} | {s.runtime_seconds:.1f} | {s.params} |"
        )
    report_lines.extend(
        [
            "",
            f"## Champion: `{champion.method}`",
            "",
            f"- Params: {champion.params}",
            f"- Invariants OK: {invariants_ok}",
            f"- Closure valid: {closure_ok} (violations: {n_viol})",
            f"- Observed cells unchanged: {observed_unchanged}",
            f"- Seeds: {seeds}",
            "",
        ]
    )
    report_path = Path(output_dir).parent / "reports" / "baseline_selection.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    manifest = create_run_manifest(
        run_id=f"baseline_{int(time.time())}",
        method=champion.method,
        params=champion.params,
        metrics={
            "mse_empirical": champion.mse_empirical,
            "mse_timeblock": champion.mse_timeblock,
            "mse_mean": champion.mse_mean,
            "jsd_empirical": champion.jsd_empirical,
            "jsd_timeblock": champion.jsd_timeblock,
        },
        data_path=config.path,
        config_paths=[Path("configs/data.yaml")],
        seeds=seeds,
        invariants_ok=invariants_ok,
        output_dir=output_dir / "manifests",
    )

    return {
        "dataset_00_tsagris": str(dataset00_path),
        "M_observed": str(m_observed_path),
        "M_target": str(m_target_path),
        "original_mask": str(mask_path),
        "metrics_csv": str(metrics_path),
        "report": str(report_path),
        "manifest": manifest["run_id"],
        "invariants_ok": invariants_ok,
    }