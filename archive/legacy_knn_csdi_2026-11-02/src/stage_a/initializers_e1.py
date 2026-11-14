"""Raw-count Hron-2a and Aitchison-complete initializers for E1.

The input remains raw counts with NaN for missing cells.  Zeros are observed
values.  A positivity delta is used only inside the CLR calculation needed for
the distance; it is never written into the raw query or donor arrays and is
never used for the median scale adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


METHODS = ("hron_2a", "aitchison_complete")


@dataclass
class E1FitInfo:
    method: str
    n_train_rows: int
    n_pool_rows: int
    n_complete_pool_rows: int
    distance_delta: float
    max_k: int


def distance_delta(train_x: np.ndarray) -> float:
    """Return a train-only delta for CLR distance, without changing raw data."""
    values = np.asarray(train_x, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("No positive train count exists for a CLR distance policy")
    delta = float(np.min(positive) / 2.0)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"Invalid distance positivity delta: {delta!r}")
    return delta


def _clr_positive(x: np.ndarray, delta: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("CLR distance received a non-finite or negative count")
    y = np.where(y > 0, y, delta)
    logs = np.log(y)
    return logs - logs.mean(axis=-1, keepdims=True)


def _a_distance(query: np.ndarray, candidates: np.ndarray, delta: float) -> np.ndarray:
    """Aitchison distance on one query subcomposition and many donors."""
    if query.size == 1:
        return np.zeros(len(candidates), dtype=np.float64)
    q_clr = _clr_positive(query[None, :], delta)[0]
    c_clr = _clr_positive(candidates, delta)
    return np.linalg.norm(c_clr - q_clr[None, :], axis=1)


def _scale_median(query: np.ndarray, donor: np.ndarray) -> tuple[float, bool]:
    """Hron formula (7), with an explicit degenerate-zero flag."""
    q_med = float(np.median(query))
    d_med = float(np.median(donor))
    if d_med > 0:
        factor = q_med / d_med
        if np.isfinite(factor) and factor >= 0:
            return float(factor), False
    # A zero denominator provides no scale anchor.  Preserve a non-negative
    # donor value without manufacturing a pseudo-count in the raw pipeline.
    return 1.0, True


def build_cache(
    query_x: np.ndarray,
    train_x: np.ndarray,
    train_ids: np.ndarray,
    method: str,
    max_k: int,
    query_ids: np.ndarray | None = None,
    delta: float | None = None,
) -> tuple[list[list[dict[str, Any]]], E1FitInfo]:
    """Build distance/donor rankings once so all k values use identical donors."""
    if method not in METHODS:
        raise ValueError(f"Unknown E1 method: {method}")
    if max_k < 1:
        raise ValueError("max_k must be positive")
    qx = np.asarray(query_x, dtype=np.float64)
    tx = np.asarray(train_x, dtype=np.float64)
    ids = np.asarray(train_ids, dtype=np.int64)
    if qx.ndim != 2 or tx.ndim != 2 or qx.shape[1] != tx.shape[1]:
        raise ValueError("query_x and train_x must be 2-D with equal feature count")
    if ids.shape != (len(tx),) or len(np.unique(ids)) != len(ids):
        raise ValueError("train_ids must be unique stable integer row IDs")
    if np.any(~np.isfinite(tx[~np.isnan(tx)])) or np.any(tx[~np.isnan(tx)] < 0):
        raise ValueError("Raw train counts must be finite and non-negative")
    if query_ids is not None:
        qids = np.asarray(query_ids, dtype=np.int64)
        if qids.shape != (len(qx),):
            raise ValueError("query_ids length mismatch")
    else:
        qids = None
    if delta is None:
        delta = distance_delta(tx)
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")

    train_observed = np.isfinite(tx)
    query_observed = np.isfinite(qx)
    complete_pool = train_observed.all(axis=1)
    pool_mask = complete_pool if method == "aitchison_complete" else np.ones(len(tx), dtype=bool)
    cache: list[list[dict[str, Any]]] = []
    for i in range(len(qx)):
        observed_idx = np.flatnonzero(query_observed[i])
        missing_idx = np.flatnonzero(~query_observed[i])
        row_cache: list[dict[str, Any]] = []
        for j in missing_idx:
            required = np.unique(np.concatenate((observed_idx, np.asarray([j], dtype=int))))
            valid = pool_mask & train_observed[:, required].all(axis=1)
            if qids is not None:
                valid &= ids != qids[i]
            candidates = np.flatnonzero(valid)
            record: dict[str, Any] = {
                "j": int(j),
                "observed_idx": observed_idx.copy(),
                "n_observed_anchor": int(len(observed_idx)),
                "loss_of_discriminability": bool(len(observed_idx) == 1),
                "no_observed_anchor": bool(len(observed_idx) == 0),
                "n_valid_donors": int(len(candidates)),
                "candidate_ids": np.empty(0, dtype=np.int64),
                "distances": np.empty(0, dtype=np.float64),
                "scales": np.empty(0, dtype=np.float64),
                "scaled_values": np.empty(0, dtype=np.float64),
                "scale_degenerate": np.empty(0, dtype=bool),
            }
            if len(candidates) == 0 or len(observed_idx) == 0:
                row_cache.append(record)
                continue
            distances = _a_distance(qx[i, observed_idx], tx[candidates][:, observed_idx], delta)
            order = np.lexsort((ids[candidates], distances))
            chosen = candidates[order[: min(max_k, len(order))]]
            chosen_ids = ids[chosen]
            chosen_distances = distances[order[: min(max_k, len(order))]]
            scales = np.empty(len(chosen), dtype=np.float64)
            degenerate = np.empty(len(chosen), dtype=bool)
            scaled_values = np.empty(len(chosen), dtype=np.float64)
            for n, donor_index in enumerate(chosen):
                scales[n], degenerate[n] = _scale_median(qx[i, observed_idx], tx[donor_index, observed_idx])
                scaled_values[n] = scales[n] * tx[donor_index, j]
            record.update(
                {
                    "candidate_ids": chosen_ids,
                    "distances": chosen_distances,
                    "scales": scales,
                    "scaled_values": scaled_values,
                    "scale_degenerate": degenerate,
                }
            )
            row_cache.append(record)
        cache.append(row_cache)
    info = E1FitInfo(
        method=method,
        n_train_rows=len(tx),
        n_pool_rows=int(pool_mask.sum()),
        n_complete_pool_rows=int(complete_pool.sum()),
        distance_delta=delta,
        max_k=max_k,
    )
    return cache, info


def materialize(
    query_x: np.ndarray,
    cache: list[list[dict[str, Any]]],
    fallback_values: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    """Materialize one k from a cache, preserving every finite query value."""
    if k < 1:
        raise ValueError("k must be positive")
    qx = np.asarray(query_x, dtype=np.float64)
    fallback_values_arr = np.asarray(fallback_values, dtype=np.float64)
    if fallback_values_arr.shape != (qx.shape[1],):
        raise ValueError("fallback_values shape mismatch")
    out = qx.copy()
    provenance: list[list[dict[str, Any]]] = []
    for i, row_cache in enumerate(cache):
        row_prov: list[dict[str, Any]] = []
        missing_by_j = {int(record["j"]): record for record in row_cache}
        for j in range(qx.shape[1]):
            if np.isfinite(qx[i, j]):
                row_prov.append(
                    {
                        "is_imputed": False,
                        "fallback": False,
                        "fallback_reason": None,
                        "donor_ids": [],
                        "effective_k": 0,
                        "distance": [],
                        "f_scale": [],
                        "n_valid_donors": None,
                        "n_observed_anchor": None,
                        "loss_of_discriminability": False,
                        "scale_degenerate": False,
                    }
                )
                continue
            record = missing_by_j[j]
            n_valid = int(record["n_valid_donors"])
            if n_valid == 0 or record["no_observed_anchor"]:
                reason = "no_observed_anchor" if record["no_observed_anchor"] else "zero_valid_donors"
                value = float(fallback_values_arr[j])
                out[i, j] = value
                row_prov.append(
                    {
                        "is_imputed": True,
                        "fallback": True,
                        "fallback_reason": reason,
                        "donor_ids": [],
                        "effective_k": 0,
                        "distance": [],
                        "f_scale": [],
                        "n_valid_donors": n_valid,
                        "n_observed_anchor": int(record["n_observed_anchor"]),
                        "loss_of_discriminability": bool(record["loss_of_discriminability"]),
                        "scale_degenerate": False,
                    }
                )
                continue
            effective = min(k, n_valid, len(record["scaled_values"]))
            values = record["scaled_values"][:effective]
            value = float(np.median(values))
            fallback_used = False
            reason = None
            if not np.isfinite(value) or value < 0:
                value = float(fallback_values_arr[j])
                fallback_used = True
                reason = "nonfinite_or_negative_scaled_median"
                effective = 0
            out[i, j] = value
            row_prov.append(
                {
                    "is_imputed": True,
                        "fallback": fallback_used,
                    "fallback_reason": reason,
                    "donor_ids": [int(x) for x in record["candidate_ids"][:effective]],
                    "effective_k": int(effective),
                    "distance": [float(x) for x in record["distances"][:effective]],
                    "f_scale": [float(x) for x in record["scales"][:effective]],
                    "n_valid_donors": n_valid,
                    "n_observed_anchor": int(record["n_observed_anchor"]),
                    "loss_of_discriminability": bool(record["loss_of_discriminability"]),
                    "scale_degenerate": bool(record["scale_degenerate"][:effective].any()),
                }
            )
        provenance.append(row_prov)
    if np.any(np.isfinite(qx) & ~np.isfinite(out)):
        raise AssertionError("Materialization changed the finite mask unexpectedly")
    return out, provenance


def train_fallback_values(train_x: np.ndarray) -> np.ndarray:
    tx = np.asarray(train_x, dtype=np.float64)
    result = np.empty(tx.shape[1], dtype=np.float64)
    for j in range(tx.shape[1]):
        values = tx[np.isfinite(tx[:, j]), j]
        result[j] = float(np.median(values)) if len(values) else 0.0
    return result
