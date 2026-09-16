"""Preprocessing for compositional missing data imputation.

Follows §2 of pipeline_knn_diffusion_handover.md:
  P1: Filter low abundance (< 0.01%) -> 0
  P2: Record observation mask M0
  P3: Add pseudo-count to zeros
  P4: Closure (optional, clr is scale-invariant)
  P5: Assert all observed values > 0

Three types of 'no value' — must never be confused:
  Missing:         not measured          -> M0=0 (this is what we impute)
  Rounded zero:    present but below LOD -> pseudo-count, M0=1
  Structural zero: truly absent          -> pseudo-count, M0=1
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PreprocessResult:
    """Output of preprocessing pipeline."""
    X_pos: NDArray             # (n, K) strictly positive where observed, NaN at missing
    M0: NDArray                # (n, K) bool: True=observed, False=missing
    pseudo_count: float        # value added to zeros
    variant_cols: List[str]    # column names of the K compositional parts
    meta: dict                 # location, date, total_sequence arrays


def preprocess(df: pd.DataFrame,
               variant_cols: Optional[List[str]] = None,
               detect_thresh: float = 1e-4,
               closure_sum: Optional[float] = None) -> PreprocessResult:
    """Full preprocessing pipeline P1-P5.

    Parameters
    ----------
    df : DataFrame
        Raw data with columns: location, date, total_sequence, + variant columns.
    variant_cols : list of str, optional
        Which columns are compositional parts. If None, auto-detect
        (all columns except location, date, total_sequence).
    detect_thresh : float
        P1 threshold. Values where (value / total_sequence) < detect_thresh
        are set to 0. Set to 0 to disable.
    closure_sum : float, optional
        If provided, normalize rows to this sum after pseudo-count.
        None = no closure (recommended: clr is scale-invariant).

    Returns
    -------
    PreprocessResult
    """
    df = df.copy()

    # Auto-detect variant columns
    meta_cols = ['location', 'date', 'total_sequence']
    if variant_cols is None:
        variant_cols = [c for c in df.columns if c not in meta_cols]

    K = len(variant_cols)
    n = len(df)

    # Extract metadata
    meta = {
        'location': df['location'].values,
        'date': pd.to_datetime(df['date']).values,
        'total_sequence': df['total_sequence'].values,
    }

    # Extract variant matrix
    X = df[variant_cols].values.astype(np.float64)  # (n, K)

    # --- P1: Filter low relative abundance -> 0 ---
    if detect_thresh > 0:
        total_seq = meta['total_sequence'].reshape(-1, 1).astype(np.float64)
        valid = ~np.isnan(X) & (total_seq > 0)
        relative = np.where(valid, X / total_seq, np.nan)
        low_mask = valid & (relative < detect_thresh) & (relative >= 0)
        X[low_mask] = 0.0

    # --- P2: Record observation mask M0 ---
    # M0 = True means OBSERVED (not missing). M0 = False means MISSING.
    # Zero is NOT missing — it's a rounded/structural zero.
    M0 = ~np.isnan(X)  # (n, K) bool

    # --- P3: Pseudo-count for zeros ---
    # pseudo_count = min(non-zero observed value) / 2
    observed_nonzero = X[M0 & (X > 0)]
    if len(observed_nonzero) == 0:
        raise ValueError("No non-zero observed values found.")
    pseudo_count = float(np.min(observed_nonzero)) / 2.0

    # Add pseudo_count to all zeros (where observed but zero)
    zero_mask = M0 & (X == 0)
    X[zero_mask] = pseudo_count

    # --- P4: Optional closure ---
    if closure_sum is not None:
        full_rows = M0.all(axis=1)
        row_sums = np.nansum(X[full_rows], axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X[full_rows] = X[full_rows] / row_sums * closure_sum

    # --- P5: Assert all observed values > 0 ---
    observed_vals = X[M0]
    if np.any(observed_vals <= 0):
        n_bad = int(np.sum(observed_vals <= 0))
        raise ValueError(
            f"P5 failed: {n_bad} observed values are <= 0 after preprocessing."
        )
    if np.any(np.isnan(observed_vals)):
        raise ValueError("P5 failed: NaN found in observed cells.")

    return PreprocessResult(
        X_pos=X,
        M0=M0,
        pseudo_count=pseudo_count,
        variant_cols=variant_cols,
        meta=meta,
    )
