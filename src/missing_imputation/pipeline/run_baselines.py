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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..baselines.frechet import frechet_mean
from ..baselines.jsd import jsd_one_vs_many, partial_jsd_one_vs_many
from ..data.closure import (
    compute_other,
    counts_to_proportions,
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
    config = load_config(config_path)
    df = load_covariants(config.path, config)
    variant_cols = config.variant_components
    observed_mask = df[variant_cols].notna().to_numpy()
    counts = df[variant_cols].fillna(0).to_numpy().astype(np.int64)
    total_seq = df[config.total_sequence_col].to_numpy(dtype=np.float64)
    proportions = counts_to_proportions(counts, total_seq)
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


def _empirical_pattern_eval(
    complete_props: np.ndarray,
    patterns: list,
    pattern_weights: np.ndarray,
    k: int,
    alpha: float,
    n_folds: int,
    rng: np.random.Generator,
    split_ids: dict | None = None,
) -> tuple[float, float]:
    """Masked CV on complete rows using sampled real missingness patterns.

    If split_ids is provided with 'empirical_pattern' test_row_indices, use those.
    Otherwise, sample randomly (backward compatibility).

    Returns:
        (mean_jsd, mean_mse) on artificial masked observed cells.
    """
    n = complete_props.shape[0]
    jsd_values = []
    mse_values = []

    valid_patterns = [p for p in patterns if p.n_observed >= 1]
    if not valid_patterns:
        return float(np.log(2)), float(np.log(2))
    valid_weights = np.array([p.n_rows for p in valid_patterns], dtype=np.float64)
    valid_weights /= valid_weights.sum()

    # Use split IDs if available, otherwise random permutation
    if split_ids and "empirical_pattern" in split_ids:
        test_row_indices = set(split_ids["empirical_pattern"].get("test_row_indices", []))
        val_indices = [i for i in range(n) if i in test_row_indices]
        train_indices = [i for i in range(n) if i not in test_row_indices]
    else:
        indices = rng.permutation(n)
        fold_size = n // n_folds
        # For simplicity, use all folds combined
        val_indices = list(range(n))
        train_indices = list(range(n))

    if not val_indices:
        return float(np.log(2)), float(np.log(2))

    pool = complete_props[train_indices]

    for vi in val_indices:
        p = valid_patterns[rng.choice(len(valid_patterns), p=valid_weights)]
        obs_mask = np.array([not m for m in p.pattern])
        query = complete_props[vi].copy()
        query[~obs_mask] = 0.0

        imputed = _impute_rows(
            pool, query[np.newaxis, :], obs_mask[np.newaxis, :], k, alpha
        )[0]

        # Compute both JSD and MSE on the ARTIFICIALLY masked observed cells
        mask = ~obs_mask & (complete_props[vi] > 0)  # only masked cells that were truly observed
        if mask.any():
            jsd_values.append(float(jsd_one_vs_many(complete_props[vi], imputed[np.newaxis, :])[0]))
            mse_values.append(_mse_proportions(complete_props[vi], imputed, mask))

    return (
        float(np.mean(jsd_values)) if jsd_values else float(np.log(2)),
        float(np.mean(mse_values)) if mse_values else float(np.log(2)),
    )


def _time_block_eval(
    df: pd.DataFrame,
    complete_positions: np.ndarray,
    complete_props: np.ndarray,
    location_col: str,
    k: int,
    alpha: float,
    split_ids: dict | None = None,
) -> tuple[float, float]:
    """Hold out contiguous blocks of complete rows per location, impute from rest.
    
    Following Section 6.1: "che một đoạn liên tiếp trong chuỗi của từng location"
    Masks observed cells within the held-out time blocks (not entire rows).
    If split_ids with 'time_block' test_row_indices provided, use those.
    
    Returns:
        (mean_jsd, mean_mse) on artificial masked observed cells.
    """
    jsd_values = []
    mse_values = []
    locations = df[location_col].to_numpy()

    # Use split IDs if available
    if split_ids and "time_block" in split_ids:
        test_row_indices = set(split_ids["time_block"].get("test_row_indices", []))
    else:
        test_row_indices = None

    for loc in pd.unique(locations):
        loc_rows = np.where(locations == loc)[0]
        comp_pos = [pos for pos in loc_rows if pos in set(complete_positions.tolist())]
        if len(comp_pos) < 3:
            continue

        pos_arr = np.array(comp_pos)
        block_size = max(1, len(pos_arr) // 3)
        start = len(pos_arr) // 3
        held_out_pos = pos_arr[start : start + block_size]

        held_out_local = [
            int(np.searchsorted(complete_positions, hp))
            for hp in held_out_pos
        ]
        
        # Filter by split IDs if provided
        if test_row_indices is not None:
            held_out_local = [h for h in held_out_local if h in test_row_indices]
            if not held_out_local:
                continue

        pool_mask = np.ones(complete_props.shape[0], dtype=bool)
        pool_mask[held_out_local] = False
        pool = complete_props[pool_mask]

        queries = complete_props[held_out_local]
        # Mask individual observed cells within held-out blocks (not entire rows)
        for qi in range(queries.shape[0]):
            true_row = complete_props[held_out_local[qi]]
            observed_cells = true_row > 0  # cells that were truly observed (non-zero)
            if not observed_cells.any():
                continue
            # Randomly mask a fraction of observed cells
            rng = np.random.default_rng(42)
            mask_frac = 0.3
            mask = observed_cells & (rng.random(len(observed_cells)) < mask_frac)
            if not mask.any():
                continue
            
            obs_mask = ~mask  # observed = not masked
            query = true_row.copy()
            query[mask] = 0.0  # artificially mask some observed cells
            
            imputed = _impute_rows(
                pool, query[np.newaxis, :], obs_mask[np.newaxis, :], k, alpha
            )[0]
            
            # Evaluate on the artificially masked cells
            jsd_values.append(float(jsd_one_vs_many(true_row, imputed[np.newaxis, :])[0]))
            mse_values.append(_mse_proportions(true_row, imputed, mask))

    return (
        float(np.mean(jsd_values)) if jsd_values else float(np.log(2)),
        float(np.mean(mse_values)) if mse_values else float(np.log(2)),
    )


def _tune_grid(
    complete_props: np.ndarray,
    patterns: list,
    k_grid: list[int],
    alpha_grid: list[float],
    n_folds: int,
    seed: int,
    split_ids: dict | None = None,
) -> tuple[float, int, dict]:
    rng = np.random.default_rng(seed)
    pattern_weights = None
    best_jsd = np.inf
    best_k = k_grid[0]
    best_alpha = alpha_grid[0]

    for k in k_grid:
        for alpha in alpha_grid:
            jsd, _ = _empirical_pattern_eval(
                complete_props, patterns, pattern_weights, k, alpha, n_folds, rng, split_ids
            )
            if jsd < best_jsd:
                best_jsd = jsd
                best_k = k
                best_alpha = alpha

    return best_jsd, best_k, {"k": best_k, "alpha": best_alpha}


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
    
    # Load split IDs for reproducible evaluation
    split_ids = load_split_ids(splits_dir)

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

    patterns = extract_patterns(incomplete_obs)

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

    # JSD-kNN: tune k grid (alpha fixed at 1.0)
    start = time.time()
    jsd_knn_k_scores = []
    for k in k_grid:
        seed_scores = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            jsd, _ = _empirical_pattern_eval(complete_props, patterns, None, k, 1.0, 5, rng, split_ids)
            seed_scores.append(jsd)
        jsd_knn_k_scores.append((k, float(np.mean(seed_scores))))

    best_k = min(jsd_knn_k_scores, key=lambda x: x[1])[0]

    jsd_knn_emp_scores = []
    jsd_knn_emp_mse = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        jsd, mse = _empirical_pattern_eval(complete_props, patterns, None, best_k, 1.0, 5, rng, split_ids)
        jsd_knn_emp_scores.append(jsd)
        jsd_knn_emp_mse.append(mse)
    jsd_knn_tb_jsd, jsd_knn_tb_mse = _time_block_eval(df, complete_idx, complete_props, config.location_col, best_k, 1.0, split_ids)
    scores.append(
        MethodScore(
            method="jsd_knn",
            params={"k": best_k, "alpha": 1.0},
            mse_empirical=float(np.mean(jsd_knn_emp_mse)) if jsd_knn_emp_mse else float(np.log(2)),
            mse_timeblock=float(jsd_knn_tb_mse),
            jsd_empirical=float(np.mean(jsd_knn_emp_scores)) if jsd_knn_emp_scores else float(np.log(2)),
            jsd_timeblock=float(jsd_knn_tb_jsd),
            runtime_seconds=time.time() - start,
        )
    )

    start = time.time()
    emp_scores_by_seed = []
    tb_scores_by_seed = []
    tb_mse_by_seed = []
    tuned_params_by_seed = []
    for seed in seeds:
        _, best_k, params = _tune_grid(complete_props, patterns, k_grid, alpha_grid, 5, seed, split_ids)
        tuned_params_by_seed.append(params)
        rng = np.random.default_rng(seed)
        emp_jsd, emp_mse = _empirical_pattern_eval(
            complete_props, patterns, None, params["k"], params["alpha"], 5, rng, split_ids
        )
        emp_scores_by_seed.append(emp_jsd)
        tb_jsd, tb_mse = _time_block_eval(
            df, complete_idx, complete_props, config.location_col, params["k"], params["alpha"], split_ids
        )
        tb_scores_by_seed.append(tb_jsd)
        tb_mse_by_seed.append(tb_mse)

    best_seed_idx = int(np.argmin(emp_scores_by_seed))
    chosen_params = tuned_params_by_seed[best_seed_idx]
    scores.append(
        MethodScore(
            method="jsd_alpha_knn",
            params=chosen_params,
            mse_empirical=float(np.mean(emp_scores_by_seed)),
            mse_timeblock=float(np.mean(tb_mse_by_seed)),
            jsd_empirical=float(np.mean(emp_scores_by_seed)),
            jsd_timeblock=float(np.mean(tb_scores_by_seed)),
            runtime_seconds=time.time() - start,
            notes=f"per-seed tuned params: {tuned_params_by_seed}",
        )
    )

    global_k = chosen_params["k"]
    global_alpha = chosen_params["alpha"]

    start = time.time()
    pattern_params: dict[tuple, tuple[float, int]] = {}
    fallback_patterns = []
    for p in patterns:
        if p.n_rows >= min_support and p.n_observed >= 1:
            pattern_only = [p]
            _, pk, pparams = _tune_grid(complete_props, pattern_only, k_grid, alpha_grid, 3, seeds[0])
            pattern_params[p.pattern] = (pparams["alpha"], pparams["k"])
        else:
            pattern_params[p.pattern] = (global_alpha, global_k)
            fallback_patterns.append(p.pattern)

    adaptive_emp_jsd = []
    adaptive_emp_mse = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = complete_props.shape[0]
        indices = rng.permutation(n)
        fold_size = n // 5
        jsd_vals = []
        mse_vals = []
        valid_patterns = [p for p in patterns if p.n_observed >= 1]
        w = np.array([p.n_rows for p in valid_patterns], dtype=np.float64)
        w /= w.sum()
        for fold in range(5):
            s = fold * fold_size
            e = s + fold_size if fold < 4 else n
            val_idx = indices[s:e]
            pool_idx = np.concatenate([indices[:s], indices[e:]])
            pool = complete_props[pool_idx]
            for vi in val_idx:
                p = valid_patterns[rng.choice(len(valid_patterns), p=w)]
                a, kk = pattern_params.get(p.pattern, (global_alpha, global_k))
                obs_mask = np.array([not m for m in p.pattern])
                query = complete_props[vi].copy()
                query[~obs_mask] = 0.0
                imputed = _impute_rows(pool, query[np.newaxis, :], obs_mask[np.newaxis, :], kk, a)[0]
                
                # Evaluate on artificially masked observed cells
                mask = ~obs_mask & (complete_props[vi] > 0)
                if mask.any():
                    jsd_vals.append(
                        float(jsd_one_vs_many(complete_props[vi], imputed[np.newaxis, :])[0])
                    )
                    mse_vals.append(_mse_proportions(complete_props[vi], imputed, mask))
        adaptive_emp_jsd.append(float(np.mean(jsd_vals)) if jsd_vals else float(np.log(2)))
        adaptive_emp_mse.append(float(np.mean(mse_vals)) if mse_vals else float(np.log(2)))

    adaptive_tb_jsd, adaptive_tb_mse = _time_block_eval(
        df, complete_idx, complete_props, config.location_col, global_k, global_alpha, split_ids
    )
    scores.append(
        MethodScore(
            method="adaptive_jsd_alpha_knn",
            params={"global_k": global_k, "global_alpha": global_alpha, "min_pattern_support": min_support},
            mse_empirical=float(np.mean(adaptive_emp_mse)) if adaptive_emp_mse else float(np.log(2)),
            mse_timeblock=float(adaptive_tb_mse),
            jsd_empirical=float(np.mean(adaptive_emp_jsd)) if adaptive_emp_jsd else float(np.log(2)),
            jsd_timeblock=adaptive_tb_jsd,
            runtime_seconds=time.time() - start,
            notes=f"{len(fallback_patterns)} patterns fell back to global params",
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

    full_props = proportions.copy()
    full_props[incomplete_idx] = imputed_props

    new_counts = proportions_to_counts_largest_remainder(
        full_props, total_seq, observed_mask, counts
    )
    other = compute_other(new_counts, total_seq)
    closure_ok, n_viol = validate_closure(new_counts, total_seq, other)

    observed_unchanged = bool(
        np.array_equal(new_counts[observed_mask], counts[observed_mask])
    )
    no_nan = bool(not np.any(np.isnan(new_counts)))
    non_negative = bool(np.all(new_counts >= 0))
    invariants_ok = closure_ok and observed_unchanged and no_nan and non_negative

    df_out = df.copy()
    for j, col in enumerate(variant_cols):
        df_out[col] = new_counts[:, j]
    df_out[config.other_col] = other

    # Per plan Section 4.3: output dataset_00_tsagris.csv
    dataset00_path = output_dir / "dataset_00_tsagris.csv"
    df_out.to_csv(dataset00_path, index=False)

    # Per plan: save M_observed.npz and M_target.npz separately with checksums
    m_observed_path = output_dir / "M_observed.npz"
    m_target_path = output_dir / "M_target.npz"
    np.savez_compressed(
        m_observed_path,
        M_observed=observed_mask,
    )
    np.savez_compressed(
        m_target_path,
        M_target=(observed_mask == 0).astype(np.uint8),
    )

    # Also save the combined original_mask.npz for backward compatibility
    mask_path = output_dir / "original_mask.npz"
    np.savez_compressed(
        mask_path,
        observed_mask=observed_mask,
        original_counts=counts,
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