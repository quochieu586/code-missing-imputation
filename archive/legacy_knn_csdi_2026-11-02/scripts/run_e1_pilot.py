"""Run the blocking E1 initializer pilot.

The script consumes the E0 bundle read-only and writes artifacts/e1_pilot.
It deliberately does not import the old preprocessing pipeline because that
pipeline applies a pseudo-count before initialization.  E1 works on raw counts
and applies a train-only positivity delta only inside CLR distance/diagnostic
metrics.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage_a.initializers_e1 import (
    METHODS,
    build_cache,
    distance_delta,
    materialize,
    train_fallback_values,
)


DATA_PATH = ROOT / "data" / "covariants.csv"
E0 = ROOT / "artifacts" / "e0_extended"
OUT = ROOT / "artifacts" / "e1_pilot"
EXPECTED_DATA_SHA = "bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab"
KS = tuple(range(2, 11))
MAX_K = max(KS)
INNER_SEVERITY = 0.30
OUTER_MASK_ID = "fold0_random_r0.30_seed42"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def random_hidden(n_rows: int, n_features: int, severity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_hide = min(int(np.floor(n_features * severity)), n_features - 3)
    hidden = np.zeros((n_rows, n_features), dtype=bool)
    for i in range(n_rows):
        hidden[i, rng.choice(n_features, size=n_hide, replace=False)] = True
    return hidden


def clr_positive(x: np.ndarray, delta: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = np.where(y > 0, y, float(delta))
    logs = np.log(y)
    return logs - logs.mean(axis=-1, keepdims=True)


def native_jsd_tsagris(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Tsagris J_T convention, evaluated on nonnegative closed rows."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_positive = p > 0
    q_positive = q > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p_term[p_positive] = p[p_positive] * np.log(p[p_positive] / m[p_positive])
        q_term[q_positive] = q[q_positive] * np.log(q[q_positive] / m[q_positive])
    return p_term.sum(axis=-1) + q_term.sum(axis=-1)


def metrics(truth: np.ndarray, prediction: np.ndarray, hidden: np.ndarray, delta: float) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    hidden = np.asarray(hidden, dtype=bool)
    truth_sum = truth.sum(axis=1)
    pred_sum = prediction.sum(axis=1)
    finite = np.isfinite(truth).all(axis=1) & np.isfinite(prediction).all(axis=1)
    eligible = finite & (truth_sum > 0) & (pred_sum > 0)
    z = clr_positive(truth, delta)
    zp = clr_positive(prediction, delta)
    error = np.abs(z - zp)
    cell = hidden & eligible[:, None]
    zero = truth == 0

    def mean_or_none(values: np.ndarray, mask: np.ndarray):
        return float(values[mask].mean()) if bool(mask.any()) else None

    affected = hidden.any(axis=1) & eligible
    d2 = ((z - zp) ** 2).sum(axis=1)
    jsd_rows = native_jsd_tsagris(truth[eligible], prediction[eligible]) if bool(eligible.any()) else np.array([])
    return {
        "clr_mae_all_hidden": mean_or_none(error, cell),
        "clr_mae_nonzero_hidden": mean_or_none(error, cell & (truth > 0)),
        "clr_mae_zero_hidden": mean_or_none(error, cell & zero),
        "m2_mean_squared_aitchison_affected_rows": float(d2[affected].mean()) if bool(affected.any()) else None,
        "native_jsd_tsagris_mean": float(jsd_rows.mean()) if len(jsd_rows) else None,
        "native_jsd_convention": "J_T = sum[p log(p/m) + q log(q/m)], natural log, closed rows",
        "n_hidden_cells": int(hidden.sum()),
        "n_hidden_nonzero": int((cell & (truth > 0)).sum()),
        "n_hidden_zero": int((cell & zero).sum()),
        "n_affected_rows": int(affected.sum()),
        "n_rows_truth_positive": int((finite & (truth_sum > 0)).sum()),
        "n_rows_excluded_truth_zero_sum": int((finite & (truth_sum <= 0)).sum()),
        "n_rows_excluded_prediction_zero_sum": int((finite & (truth_sum > 0) & (pred_sum <= 0)).sum()),
        "n_native_jsd_rows": int(len(jsd_rows)),
        "metric_distance_delta": float(delta),
    }


def attach_mask_flags(provenance, truth, hidden):
    out = []
    for i, row in enumerate(provenance):
        row_out = []
        for j, item in enumerate(row):
            enriched = dict(item)
            enriched["artificial_hidden"] = bool(hidden[i, j])
            enriched["natural_missing"] = bool(pd.isna(truth[i, j]))
            row_out.append(enriched)
        # Materialize emits all columns in order.
        if len(row_out) != truth.shape[1]:
            raise AssertionError("Provenance does not cover the full feature order")
        out.append(row_out)
    return out


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def donor_diagnostics(provenance, fit_info, selected_k: int, hidden: np.ndarray) -> dict:
    cells = [item for i, row in enumerate(provenance) for j, item in enumerate(row) if hidden[i, j]]
    eligible = [int(x["n_valid_donors"]) for x in cells if x["n_valid_donors"] is not None]
    overlap = [int(x["n_observed_anchor"]) for x in cells if x["n_observed_anchor"] is not None]
    effective = [int(x["effective_k"]) for x in cells]
    selected_distances = [float(d) for x in cells for d in x["distance"]]
    return {
        "method": fit_info.method,
        "selected_k": int(selected_k),
        "distance_delta": float(fit_info.distance_delta),
        "n_train_rows": int(fit_info.n_train_rows),
        "n_donor_pool_rows": int(fit_info.n_pool_rows),
        "n_complete_rows_in_train": int(fit_info.n_complete_pool_rows),
        "n_hidden_cells": int(len(cells)),
        "eligible_donor_count_distribution": distribution(eligible),
        "overlap_size_distribution": distribution(overlap),
        "effective_k_distribution": distribution(effective),
        "selected_distance_distribution": distribution(selected_distances),
        "n_cells_with_fewer_than_k_donors": int(sum(x["n_valid_donors"] is not None and x["n_valid_donors"] < selected_k for x in cells)),
        "n_fallback_cells": int(sum(bool(x["fallback"]) for x in cells)),
        "fallback_rate": float(sum(bool(x["fallback"]) for x in cells) / len(cells)) if cells else None,
        "n_loss_of_discriminability_cells": int(sum(bool(x["loss_of_discriminability"]) for x in cells)),
        "n_no_observed_anchor_cells": int(sum(x["fallback_reason"] == "no_observed_anchor" for x in cells)),
        "n_scale_degenerate_cells": int(sum(bool(x["scale_degenerate"]) for x in cells)),
        "raw_observed_values_preserved": True,
    }


def tune_k(method: str, df: pd.DataFrame, variants: list[str], outer_train_rows: list[int], complete_locations: list[str]) -> dict:
    records = []
    for inner_id, validation_location in enumerate(complete_locations):
        val_rows = np.flatnonzero(df["location"].eq(validation_location).to_numpy()).tolist()
        val_rows = [i for i in val_rows if i in set(outer_train_rows)]
        train_rows = [i for i in outer_train_rows if i not in set(val_rows)]
        truth = df.iloc[val_rows][variants].to_numpy(dtype=np.float64)
        hidden = random_hidden(len(val_rows), len(variants), INNER_SEVERITY, 41000 + inner_id)
        query = truth.copy()
        query[hidden] = np.nan
        train_x = df.iloc[train_rows][variants].to_numpy(dtype=np.float64)
        delta = distance_delta(train_x)
        cache, info = build_cache(query, train_x, np.asarray(train_rows), method, MAX_K, np.asarray(val_rows), delta)
        fallback_values = train_fallback_values(train_x)
        for k in KS:
            prediction, provenance = materialize(query, cache, fallback_values, k)
            score = metrics(truth, prediction, hidden, delta)
            records.append(
                {
                    "inner_fold_id": inner_id,
                    "validation_location": validation_location,
                    "seed": 41000 + inner_id,
                    "k": k,
                    "m2": score["m2_mean_squared_aitchison_affected_rows"],
                    "n_affected_rows": score["n_affected_rows"],
                    "n_hidden_cells": score["n_hidden_cells"],
                    "n_fallback_cells": int(sum(bool(x["fallback"]) for row in provenance for x in row if x["is_imputed"])),
                    "distance_delta": delta,
                    "n_train_rows": len(train_rows),
                }
            )
    summary = {}
    for k in KS:
        values = [x["m2"] for x in records if x["k"] == k and x["m2"] is not None]
        summary[str(k)] = {
            "mean_m2": float(np.mean(values)) if values else None,
            "n_inner_folds": len(values),
            "fold_m2": values,
        }
    valid = [(float(v["mean_m2"]), int(k)) for k, v in summary.items() if v["mean_m2"] is not None]
    selected_k = min(valid, key=lambda pair: (pair[0], pair[1]))[1] if valid else 8
    return {
        "method": method,
        "k_grid": list(KS),
        "selection_metric": "mean squared Aitchison distance M2 on inner affected rows",
        "selected_k": int(selected_k),
        "inherited_k8": 8,
        "inner_validation_locations": complete_locations,
        "inner_severity": INNER_SEVERITY,
        "summary_by_k": summary,
        "records": records,
    }


def row_prediction_records(df, row_ids, truth, query, prediction, hidden, provenance, method, selected_k):
    records = []
    for i, row_id in enumerate(row_ids):
        records.append(
            {
                "row_id": int(row_id),
                "location": str(df.iloc[row_id]["location"]),
                "date": str(df.iloc[row_id]["date"]),
                "method": method,
                "selected_k": int(selected_k),
                "input_raw_with_hidden_as_null": [None if not np.isfinite(x) else float(x) for x in query[i]],
                "truth_raw": [None if not np.isfinite(x) else float(x) for x in truth[i]],
                "prediction_raw": [None if not np.isfinite(x) else float(x) for x in prediction[i]],
                "hidden_eval_mask": hidden[i].astype(bool).tolist(),
                "natural_missing_mask": (~np.isfinite(truth[i])).astype(bool).tolist(),
                "cell_provenance": provenance[i],
            }
        )
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_hash = sha256(DATA_PATH)
    if data_hash != EXPECTED_DATA_SHA:
        raise AssertionError(f"Frozen data hash changed: {data_hash}")
    df = pd.read_csv(DATA_PATH)
    if not np.array_equal(df.index.to_numpy(), np.arange(len(df))):
        raise AssertionError("CSV row index is not the stable 0..N-1 row ID contract")
    e0_manifest = json.loads((E0 / "manifest.json").read_text(encoding="utf-8"))
    variants = e0_manifest["variant_columns"]
    if len(variants) != 17:
        raise AssertionError("E0 manifest must contain 17 variants")
    folds = json.loads((E0 / "cv_splits.json").read_text(encoding="utf-8"))
    fold = next(x for x in folds if x["fold_id"] == 0)
    if fold["test_location"] != "Canada" or len(fold["test_rows"]) != 109:
        raise AssertionError("E1 pilot requires fold 0 / Canada / 109 rows")
    mask_manifest = json.loads((E0 / "mask_manifest.json").read_text(encoding="utf-8"))
    mask_item = next(x for x in mask_manifest["random_cell"] if x["mask_id"] == OUTER_MASK_ID)
    with np.load(E0 / mask_item["array_file"]) as masks:
        hidden = masks[OUTER_MASK_ID].astype(bool)
    if hidden.shape != (len(fold["test_rows"]), len(variants)) or (~hidden).sum(axis=1).min() < 3:
        raise AssertionError("Frozen pilot mask violates shape or visible-part contract")
    if mask_item["row_ids"] != fold["test_rows"]:
        raise AssertionError("Frozen mask row IDs do not match frozen fold row order")

    train_rows = np.asarray(fold["train_rows"], dtype=np.int64)
    test_rows = np.asarray(fold["test_rows"], dtype=np.int64)
    train_x = df.iloc[train_rows][variants].to_numpy(dtype=np.float64)
    truth = df.iloc[test_rows][variants].to_numpy(dtype=np.float64)
    query = truth.copy()
    query[hidden] = np.nan
    outer_delta = distance_delta(train_x)
    complete_locations = [
        str(x)
        for x in sorted(df.iloc[train_rows].loc[df.iloc[train_rows][variants].notna().all(axis=1), "location"].unique())
    ]
    if len(complete_locations) != 4:
        raise AssertionError(f"Expected four complete inner-validation locations, got {complete_locations}")

    all_metrics = {
        "metadata": {
            "data_sha256": data_hash,
            "fold_id": 0,
            "test_location": "Canada",
            "outer_mask_id": OUTER_MASK_ID,
            "outer_mask_scenario": "random_cell",
            "outer_mask_severity": 0.30,
            "outer_mask_seed": 42,
            "feature_order": variants,
            "row_id_contract": "original CSV integer index",
            "raw_counts_only_before_initializer": True,
            "distance_positivity_policy": "train minimum positive raw count / 2, used only inside CLR distance and positive LR metrics",
            "outer_distance_delta": outer_delta,
            "protocol": "U; original JSD initializer blocked by E0",
        },
        "methods": {},
    }
    prediction_output = {
        "metadata": all_metrics["metadata"],
        "methods": {},
    }
    diagnostics_output = {
        "metadata": all_metrics["metadata"],
        "methods": {},
    }

    for method in METHODS:
        tuning = tune_k(method, df, variants, fold["train_rows"], complete_locations)
        selected_k = int(tuning["selected_k"])
        cache, fit_info = build_cache(query, train_x, train_rows, method, MAX_K, test_rows, outer_delta)
        fallback_values = train_fallback_values(train_x)
        prediction, provenance = materialize(query, cache, fallback_values, selected_k)
        if not np.array_equal(prediction[~hidden], truth[~hidden]):
            raise AssertionError(f"Observed values changed for {method}")
        provenance = attach_mask_flags(provenance, truth, hidden)
        method_metrics = metrics(truth, prediction, hidden, outer_delta)
        method_metrics["selected_k"] = selected_k
        method_metrics["tuning"] = tuning
        all_metrics["methods"][method] = method_metrics
        prediction_output["methods"][method] = row_prediction_records(
            df, test_rows.tolist(), truth, query, prediction, hidden, provenance, method, selected_k
        )
        diagnostics_output["methods"][method] = donor_diagnostics(provenance, fit_info, selected_k, hidden)

    (OUT / "predictions_fold0.json").write_text(json.dumps(jsonable(prediction_output), indent=2), encoding="utf-8")
    (OUT / "metrics_fold0.json").write_text(json.dumps(jsonable(all_metrics), indent=2), encoding="utf-8")
    (OUT / "donor_diagnostics_fold0.json").write_text(json.dumps(jsonable(diagnostics_output), indent=2), encoding="utf-8")

    hron = all_metrics["methods"]["hron_2a"]
    complete = all_metrics["methods"]["aitchison_complete"]
    report = [
        "# E1 pilot report: Hron-2a vs Aitchison-complete",
        "",
        "**Status:** completed on the frozen fold-0 random-cell pilot; no diffusion model was trained.",
        "",
        "## Frozen protocol",
        "",
        f"- Data hash: `{data_hash}`; fold 0 test location: **Canada**; rows: {len(test_rows)}.",
        f"- Mask: `{OUTER_MASK_ID}`; 0.30 requested severity; seed 42; hidden cells: {int(hidden.sum())}; minimum visible parts per row: {int((~hidden).sum(axis=1).min())}.",
        f"- Train rows: {len(train_rows)}; complete inner-validation locations: {', '.join(complete_locations)}.",
        f"- Raw counts were passed unchanged to both initializers. CLR distance used train-only delta `{outer_delta:.17g}`; this delta was not written into predictions or used for Hron scale adjustment.",
        "- The native JSD column is a metric only. Original JSD initialization remains blocked by E0 Protocol U.",
        "",
        "## Inner-CV k selection",
        "",
        "The k grid was 2–10, with four location-held-out inner validations inside outer training data. Selection minimized mean M2 (mean squared Aitchison distance on affected eligible rows); ties choose smaller k. `k=8` is retained as the inherited reference and is not called optimal.",
        "",
        "| Method | selected k | mean inner M2 at selected k | mean inner M2 at k=8 |",
        "|---|---:|---:|---:|",
        f"| Hron-2a | {hron['selected_k']} | {hron['tuning']['summary_by_k'][str(hron['selected_k'])]['mean_m2']:.6g} | {hron['tuning']['summary_by_k']['8']['mean_m2']:.6g} |",
        f"| Aitchison-complete | {complete['selected_k']} | {complete['tuning']['summary_by_k'][str(complete['selected_k'])]['mean_m2']:.6g} | {complete['tuning']['summary_by_k']['8']['mean_m2']:.6g} |",
        "",
        "## Outer pilot metrics",
        "",
        "Metrics use hidden evaluation cells only for CLR MAE. M2 is computed on affected rows with positive truth and prediction sums; native JSD uses the Tsagris `J_T` convention on closed nonnegative rows. Rows whose true 17-variant sum is zero are excluded from compositional metrics and counted explicitly.",
        "",
        "| Method | CLR MAE all | CLR MAE nonzero | CLR MAE zero | M2 | native JSD `J_T` | fallback rate | eligible donor median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Hron-2a", "hron_2a"), ("Aitchison-complete", "aitchison_complete")):
        met = all_metrics["methods"][key]
        diag = diagnostics_output["methods"][key]
        report.append(
            f"| {label} | {met['clr_mae_all_hidden']:.6g} | {met['clr_mae_nonzero_hidden']:.6g} | {met['clr_mae_zero_hidden']:.6g} | {met['m2_mean_squared_aitchison_affected_rows']:.6g} | {met['native_jsd_tsagris_mean']:.6g} | {diag['fallback_rate']:.2%} | {diag['eligible_donor_count_distribution']['median']:.1f} |"
        )
    report += [
        "",
        "## Donor coverage",
        "",
        "The full distributions are in `donor_diagnostics_fold0.json`. Hron-2a permits partial-overlap donors; Aitchison-complete restricts the pool to train rows with all 17 raw parts observed. Therefore their donor coverage difference is expected and is part of the factorial comparison, not a distance-only claim.",
        "",
        "Both methods preserve every unhidden observed raw value exactly, including zero. Fallbacks, insufficient donor counts, one-anchor loss of discriminability and degenerate median scales are recorded per cell in `predictions_fold0.json`.",
        "",
        f"- The raw median scale anchor was degenerate for {diagnostics_output['methods']['hron_2a']['n_scale_degenerate_cells']}/{diagnostics_output['methods']['hron_2a']['n_hidden_cells']} Hron cells and {diagnostics_output['methods']['aitchison_complete']['n_scale_degenerate_cells']}/{diagnostics_output['methods']['aitchison_complete']['n_hidden_cells']} Aitchison-complete cells. This is a zero-driven diagnostic, not a silent pseudo-count; the implementation used scale 1.0 for those cells and flags them.",
        f"- The outer metrics exclude {hron['n_rows_excluded_prediction_zero_sum']} Hron and {complete['n_rows_excluded_prediction_zero_sum']} Aitchison-complete rows whose predicted raw sum is zero while the truth sum is positive; this is reported explicitly because native closure/JSD is undefined there.",
        "",
        "## Interpretation gate",
        "",
        "This is one outer fold and one mask realization. It is a feasibility/coverage pilot, not evidence of general superiority. The next E1 gate is numerical parity and donor diagnostics across the remaining outer folds or a preregistered finalist decision; do not call k=8 optimal and do not re-enable JSD without a new closure decision.",
    ]
    (OUT / "e1_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    file_hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "file_hashes.json":
            file_hashes[path.name] = sha256(path)
    (OUT / "file_hashes.json").write_text(json.dumps(file_hashes, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(OUT), "methods": list(METHODS), "selected_k": {m: all_metrics["methods"][m]["selected_k"] for m in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
