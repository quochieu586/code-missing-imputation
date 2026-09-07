import numpy as np
import pandas as pd
import pytest

from missing_imputation.evaluation.masking import (
    create_evaluation_masks,
    create_all_evaluation_masks,
    evaluate_with_masks,
)


def test_random_cell_mask():
    observed = np.array([[True, True, False], [True, False, True], [False, True, True]])
    values = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 6.0]])
    mask = create_evaluation_masks(observed, values, "random-cell", seed=42, test_fraction=0.5)
    assert mask.recipe == "random-cell"
    assert mask.test_mask.sum() > 0
    assert mask.train_mask.sum() > 0
    assert not np.any(mask.train_mask & mask.test_mask)


def test_empirical_pattern_mask():
    observed = np.array([[True, True, False], [True, False, True], [False, True, True]])
    values = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 6.0]])
    mask = create_evaluation_masks(observed, values, "empirical-pattern", seed=42)
    assert mask.recipe == "empirical-pattern"


def test_evaluate_with_masks():
    observed = np.array([[True, True, False], [True, False, True], [True, True, True]])
    values = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0], [1.0, 2.0, 3.0]])
    imputed = np.array([[1.1, 1.9, 0.5], [2.9, 0.5, 4.1], [1.05, 2.1, 2.9]])
    mask = create_evaluation_masks(observed, values, "random-cell", seed=42, test_fraction=0.4)

    if mask.test_mask.sum() > 0:
        def mse_fn(true, pred):
            return float(np.mean((true - pred) ** 2))

        result = evaluate_with_masks(values, imputed, mask, mse_fn)
        assert result >= 0


def test_create_all_masks():
    n_rows = 20
    observed = np.ones((n_rows, 3), dtype=bool)
    values = np.random.rand(n_rows, 3)
    total_seq = np.ones(n_rows) * 100
    df = pd.DataFrame({
        "location": [f"L{i % 5}" for i in range(n_rows)],
        "date": pd.date_range("2020-01-01", periods=n_rows, freq="14D"),
    })
    masks = create_all_evaluation_masks(observed, values, total_seq, df, seed=42)
    assert "random-cell" in masks
    assert "empirical-pattern" in masks
    assert "time-block" in masks
    assert "country-holdout" in masks


def test_mask_test_values_nan_outside():
    observed = np.array([[True, True, True], [True, True, True]])
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = create_evaluation_masks(observed, values, "random-cell", seed=42, test_fraction=0.5)
    nan_count = np.isnan(mask.test_values).sum()
    test_count = mask.test_mask.sum()
    assert nan_count == (values.size - test_count)