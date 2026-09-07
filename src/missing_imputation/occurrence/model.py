"""Pooled regularized logistic Occurrence Gate model with OOF evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class OccurrenceConfig:
    C_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    max_iter: int = 2000
    recipes: list[str] = field(
        default_factory=lambda: [
            "random-cell",
            "empirical-pattern",
            "time-block",
            "country-holdout",
        ]
    )
    n_folds: int = 5
    min_fold_coverage: float = 0.8
    min_valid_folds: int = 3
    seed: int = 42
    max_false_zero_rate: float = 0.05
    min_zero_gate_support: int = 50
    threshold_grid: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )
    ece_bins: int = 10
    collapse_std_min: float = 0.01


class PooledLogisticGate:
    """Pooled regularized logistic model on long table (row, variant)."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.model = LogisticRegression(
            penalty="l2", solver="lbfgs", C=C, max_iter=max_iter, random_state=seed
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    @property
    def coefficients(self) -> np.ndarray:
        return self.model.coef_[0]

    @property
    def intercept(self) -> float:
        return float(self.model.intercept_[0])


def tune_C(
    X: np.ndarray,
    y: np.ndarray,
    C_grid: list[float],
    validation_mask: np.ndarray,
    max_iter: int = 2000,
    seed: int = 42,
) -> float:
    train_mask = ~validation_mask
    best_C = C_grid[0]
    best_loss = np.inf
    for C in C_grid:
        try:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_mask])
            model = LogisticRegression(
                penalty="l2", solver="lbfgs", C=C, max_iter=max_iter, random_state=seed
            )
            model.fit(X_tr, y[train_mask])
            X_val = scaler.transform(X[validation_mask])
            p = model.predict_proba(X_val)[:, 1]
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y[validation_mask] * np.log(p) + (1 - y[validation_mask]) * np.log(1 - p))
            if loss < best_loss:
                best_loss = loss
                best_C = C
        except Exception:
            continue
    return best_C


@dataclass
class FoldResult:
    recipe: str
    fold: int
    status: str
    n_train: int = 0
    n_test: int = 0
    reason: str = ""


@dataclass
class OOFResult:
    recipe: str
    y_true: np.ndarray
    p_raw: np.ndarray
    p_calibrated: np.ndarray
    cell_indices: np.ndarray
    fold_ids: np.ndarray
    fold_results: list[FoldResult]
    coverage: float
    n_folds_valid: int

    @property
    def n_predictions(self) -> int:
        return len(self.y_true)


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        test_idx = perm[start:end]
        train_idx = np.concatenate([perm[:start], perm[end:]])
        folds.append((train_idx, test_idx))
    return folds


def run_oof(
    X: np.ndarray,
    y: np.ndarray,
    fold_splits: list[list[tuple[np.ndarray, np.ndarray]]],
    recipe_names: list[str],
    config: OccurrenceConfig,
) -> dict[str, OOFResult]:
    """Run OOF refit for each recipe. Each fold fits model from scratch."""
    results: dict[str, OOFResult] = {}

    for recipe, folds in zip(recipe_names, fold_splits):
        y_true_parts = []
        p_raw_parts = []
        p_cal_parts = []
        idx_parts = []
        fold_ids_parts = []
        fold_results = []
        n_total_test = 0
        n_valid = 0

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            if len(train_idx) == 0 or len(test_idx) == 0:
                fold_results.append(
                    FoldResult(recipe, fold_i, "skipped", reason="empty split")
                )
                continue
            if len(np.unique(y[train_idx])) < 2:
                fold_results.append(
                    FoldResult(recipe, fold_i, "abstain", reason="single class in train")
                )
                continue

            try:
                n_val = max(1, len(train_idx) // 5)
                rng = np.random.default_rng(config.seed + fold_i)
                val_pick = rng.choice(len(train_idx), size=n_val, replace=False)
                val_mask_local = np.zeros(len(train_idx), dtype=bool)
                val_mask_local[val_pick] = True

                best_C = tune_C(
                    X[train_idx], y[train_idx], config.C_grid, val_mask_local,
                    config.max_iter, config.seed,
                )

                gate = PooledLogisticGate(C=best_C, max_iter=config.max_iter, seed=config.seed)
                gate.fit(X[train_idx], y[train_idx])

                p_test = gate.predict_proba(X[test_idx])

                p_val = gate.predict_proba(X[train_idx[val_mask_local]])
                y_val = y[train_idx[val_mask_local]]
                cal_map = _fit_simple_calibration(y_val, p_val)
                p_test_cal = cal_map(p_test)

                y_true_parts.append(y[test_idx])
                p_raw_parts.append(p_test)
                p_cal_parts.append(p_test_cal)
                idx_parts.append(test_idx)
                fold_ids_parts.append(np.full(len(test_idx), fold_i, dtype=np.int64))
                n_total_test += len(test_idx)
                n_valid += 1
                fold_results.append(
                    FoldResult(recipe, fold_i, "ok", n_train=len(train_idx), n_test=len(test_idx))
                )
            except Exception as e:
                fold_results.append(
                    FoldResult(recipe, fold_i, "failed", reason=str(e)[:200])
                )

        n_expected = sum(len(t) for _, t in folds)
        coverage = n_total_test / max(n_expected, 1)

        if y_true_parts:
            results[recipe] = OOFResult(
                recipe=recipe,
                y_true=np.concatenate(y_true_parts),
                p_raw=np.concatenate(p_raw_parts),
                p_calibrated=np.concatenate(p_cal_parts),
                cell_indices=np.concatenate(idx_parts),
                fold_ids=np.concatenate(fold_ids_parts),
                fold_results=fold_results,
                coverage=coverage,
                n_folds_valid=n_valid,
            )
        else:
            results[recipe] = OOFResult(
                recipe=recipe,
                y_true=np.array([], dtype=np.int64),
                p_raw=np.array([], dtype=np.float64),
                p_calibrated=np.array([], dtype=np.float64),
                cell_indices=np.array([], dtype=np.int64),
                fold_ids=np.array([], dtype=np.int64),
                fold_results=fold_results,
                coverage=0.0,
                n_folds_valid=0,
            )

    return results


def _fit_simple_calibration(y_val: np.ndarray, p_val: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    if len(np.unique(y_val)) < 2:
        return lambda p: p
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_val, y_val)
    return ir.predict
