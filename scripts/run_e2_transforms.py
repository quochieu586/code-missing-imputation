"""Run E2 transform roundtrip and fold-0 initializer output controls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "covariants.csv"
E0 = ROOT / "artifacts" / "e0_extended"
E1 = ROOT / "artifacts" / "e1_pilot"
OUT = ROOT / "artifacts" / "e2_transforms"
DATA_SHA = "bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab"
MASK_ID = "fold0_random_r0.30_seed42"
TOL = 1e-8


def load_transform_module():
    path = OUT / "transforms.py"
    spec = importlib.util.spec_from_file_location("e2_transform_artifact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def reference_sets(df: pd.DataFrame, variants: list[str], folds: list[dict]) -> dict:
    output = {}
    alphabetic = sorted(variants)
    for fold in folds:
        train = df.iloc[np.asarray(fold["train_rows"], dtype=np.int64)]
        rates = train[variants].isna().mean()
        actual = sorted(variants, key=lambda v: (float(rates[v]), v))[:5]
        high = sorted(variants, key=lambda v: (-float(rates[v]), v))[:5]
        random = {}
        for seed in (42, 43, 44):
            rng = np.random.default_rng(seed)
            random[str(seed)] = sorted(rng.choice(alphabetic, size=5, replace=False).tolist())
        output[str(fold["fold_id"])] = {
            "test_location": fold["test_location"],
            "train_rows": len(fold["train_rows"]),
            "missing_rate_by_variant": {v: float(rates[v]) for v in variants},
            "actual_lowest_missing": actual,
            "random": random,
            "high_highest_missing": high,
        }
    return output


def indices(names: list[str], variants: list[str]) -> tuple[int, ...]:
    return tuple(variants.index(name) for name in names)


def metric_view(x: np.ndarray, delta: float) -> np.ndarray:
    y = np.where(x > 0, x, delta)
    logs = np.log(y)
    return logs - logs.mean(axis=-1, keepdims=True)


def native_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    pterm = np.zeros_like(p)
    qterm = np.zeros_like(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = p > 0
        qm = q > 0
        pterm[pm] = p[pm] * np.log(p[pm] / m[pm])
        qterm[qm] = q[qm] * np.log(q[qm] / m[qm])
    return pterm.sum(axis=-1) + qterm.sum(axis=-1)


def evaluate(truth: np.ndarray, prediction: np.ndarray, hidden: np.ndarray, delta: float, restored_exact: bool, adjusted: np.ndarray, restore_info: dict) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    z = metric_view(truth, delta)
    zp = metric_view(pred, delta)
    err = np.abs(z - zp)
    positive_rows = np.isfinite(truth).all(1) & np.isfinite(pred).all(1) & (truth.sum(1) > 0) & (pred.sum(1) > 0)
    cells = hidden & positive_rows[:, None]
    zero = truth == 0
    affected = hidden.any(1) & positive_rows
    d2 = ((z - zp) ** 2).sum(1)
    jsd_rows = native_jsd(truth[positive_rows], pred[positive_rows]) if positive_rows.any() else np.array([])

    def avg(a, mask):
        return float(a[mask].mean()) if mask.any() else None

    return {
        "clr_mae_all_hidden": avg(err, cells),
        "clr_mae_nonzero_hidden": avg(err, cells & (truth > 0)),
        "clr_mae_zero_hidden": avg(err, cells & zero),
        "m2_mean_squared_aitchison_affected_rows": float(d2[affected].mean()) if affected.any() else None,
        "native_jsd_tsagris_mean": float(jsd_rows.mean()) if len(jsd_rows) else None,
        "native_jsd_convention": "J_T, natural log, row-wise closure",
        "n_hidden_cells": int(hidden.sum()),
        "n_hidden_nonzero": int((cells & (truth > 0)).sum()),
        "n_hidden_zero": int((cells & zero).sum()),
        "n_affected_rows": int(affected.sum()),
        "n_rows_excluded_truth_zero_sum": int((truth.sum(1) <= 0).sum()),
        "n_rows_excluded_prediction_zero_sum": int(((truth.sum(1) > 0) & (pred.sum(1) <= 0)).sum()),
        "n_adjusted_cells_in_forward": int(adjusted.sum()),
        "n_adjusted_hidden_cells": int((adjusted & hidden).sum()),
        "n_adjusted_observed_cells": int((adjusted & ~hidden).sum()),
        "inverse_sum_max_error": float(np.max(np.abs(pred.sum(1) - 1.0))),
        "observed_restoration_exact": bool(restored_exact),
        "restore_n_no_positive_visible_anchor": int(restore_info["n_no_positive_visible_anchor"]),
        "metric_distance_delta": float(delta),
    }


def roundtrip(transform, train_raw: np.ndarray, delta: float) -> dict:
    latent, adjusted = transform.forward(train_raw, return_adjustment=True)
    recovered = transform.inverse(latent)
    positive, _ = transform_module.positive_view(train_raw, delta)
    target = positive / positive.sum(axis=-1, keepdims=True)
    error = np.max(np.abs(recovered - target), axis=1)
    return {
        "transform": transform.name,
        "n_rows": int(len(train_raw)),
        "n_adjusted_cells": int(adjusted.sum()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "q99_abs_error": float(np.quantile(error, 0.99)),
        "tolerance": TOL,
        "pass": bool(error.max() < TOL),
        "basis_metadata": transform.metadata(),
    }


def main() -> None:
    global transform_module
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(DATA) != DATA_SHA:
        raise AssertionError("Frozen data hash changed")
    transform_module = load_transform_module()
    df = pd.read_csv(DATA)
    e0 = json.loads((E0 / "manifest.json").read_text(encoding="utf-8"))
    variants = e0["variant_columns"]
    folds = json.loads((E0 / "cv_splits.json").read_text(encoding="utf-8"))
    fold0 = next(f for f in folds if f["fold_id"] == 0)
    refs = reference_sets(df, variants, folds)
    fold0_refs = refs["0"]
    train_rows = np.asarray(fold0["train_rows"], dtype=np.int64)
    test_rows = np.asarray(fold0["test_rows"], dtype=np.int64)
    train_raw = df.iloc[train_rows][variants].to_numpy(dtype=np.float64)
    complete_train = df.iloc[train_rows][variants].notna().all(axis=1).to_numpy()
    train_complete_raw = train_raw[complete_train]
    delta = transform_module.fit_positivity_delta(train_raw)

    transform_specs = [
        ("clr16", None),
        ("clr17", None),
        ("ilr16", None),
        ("hkglr16_actual", indices(fold0_refs["actual_lowest_missing"], variants)),
        ("hkglr16_random_seed42", indices(fold0_refs["random"]["42"], variants)),
        ("hkglr16_random_seed43", indices(fold0_refs["random"]["43"], variants)),
        ("hkglr16_random_seed44", indices(fold0_refs["random"]["44"], variants)),
        ("hkglr16_high", indices(fold0_refs["high_highest_missing"], variants)),
    ]
    transform_objects = {name: transform_module.make_transform(name, len(variants), delta, refs_idx) for name, refs_idx in transform_specs}

    roundtrips = [roundtrip(transform, train_complete_raw, delta) for transform in transform_objects.values()]
    if not all(item["pass"] for item in roundtrips):
        raise AssertionError("At least one E2 transform failed the 1e-8 train roundtrip gate")
    train_adjusted_mask = train_complete_raw == 0
    adjustment_output = {
        "metadata": {
            "data_sha256": DATA_SHA,
            "fold_id": 0,
            "positivity_delta": delta,
            "policy": "cells equal to raw zero are replaced by delta only in LR positive views",
            "feature_order": variants,
        },
        "train_complete": {
            "row_ids": [int(x) for x in train_rows[complete_train]],
            "adjusted_cells_by_row": {
                str(int(row_id)): [variants[j] for j in np.flatnonzero(train_adjusted_mask[i])]
                for i, row_id in enumerate(train_rows[complete_train])
            },
        },
        "outer_predictions": {},
    }
    (OUT / "roundtrip_test.json").write_text(json.dumps(jsonable({
        "data_sha256": DATA_SHA,
        "fold_id": 0,
        "eligible_train_rows": int(len(train_complete_raw)),
        "positivity_delta": delta,
        "positive_train_rows_definition": "outer-train rows with all 17 raw variants observed; zeros replaced only for LR",
        "transforms": roundtrips,
    }), indent=2), encoding="utf-8")

    e1_predictions = json.loads((E1 / "predictions_fold0.json").read_text(encoding="utf-8"))
    e1_diagnostics = json.loads((E1 / "donor_diagnostics_fold0.json").read_text(encoding="utf-8"))
    if e1_predictions["metadata"]["outer_mask_id"] != MASK_ID:
        raise AssertionError("E1 predictions are not from the frozen E2 mask")

    metrics_output = {
        "metadata": {
            "data_sha256": DATA_SHA,
            "fold_id": 0,
            "test_location": fold0["test_location"],
            "mask_id": MASK_ID,
            "feature_order": variants,
            "positivity_delta": delta,
            "positivity_policy": "min positive finite raw count in outer train / 2, applied once inside every transform",
            "protocol": "U",
            "no_diffusion": True,
        },
        "reference_sets_all_folds": refs,
        "methods": {},
    }
    predictions_by_method = {}
    for method in ("hron_2a", "aitchison_complete"):
        rows = e1_predictions["methods"][method]
        truth = np.asarray([[np.nan if v is None else float(v) for v in row["truth_raw"]] for row in rows], dtype=np.float64)
        query = np.asarray([[np.nan if v is None else float(v) for v in row["input_raw_with_hidden_as_null"]] for row in rows], dtype=np.float64)
        base_prediction = np.asarray([[np.nan if v is None else float(v) for v in row["prediction_raw"]] for row in rows], dtype=np.float64)
        hidden = np.asarray([row["hidden_eval_mask"] for row in rows], dtype=bool)
        if not np.array_equal(np.asarray([row["row_id"] for row in rows]), test_rows):
            raise AssertionError("E1 row order is not the frozen fold-0 row order")
        predictions_by_method[method] = {}
        for name, transform in transform_objects.items():
            latent, adjusted = transform.forward(base_prediction, return_adjustment=True)
            composition = transform.inverse(latent)
            restored, restore_info = transform_module.restore_observed_counts(composition, query)
            if not np.array_equal(restored[np.isfinite(query)], query[np.isfinite(query)]):
                raise AssertionError(f"Observed restoration failed for {method}/{name}")
            restore_info["n_no_positive_visible_anchor"] = int(restore_info["n_no_positive_visible_anchor"])
            score = evaluate(truth, restored, hidden, delta, bool(restore_info["observed_exact"]), adjusted, restore_info)
            score.update({
                "initializer": method,
                "transform": name,
                "transform_metadata": transform.metadata(),
                "donor_diagnostics_from_e1": e1_diagnostics["methods"][method],
            })
            metrics_output["methods"][f"{method}__{name}"] = score
            adjustment_output["outer_predictions"].setdefault(method, {})[name] = {
                "adjusted_cells_by_row": {
                    str(int(row_id)): [variants[j] for j in np.flatnonzero(adjusted[i])]
                    for i, row_id in enumerate(test_rows)
                },
                "n_adjusted_cells": int(adjusted.sum()),
            }
            predictions_by_method[method][name] = {
                "max_pairwise_transform_composition_error": None,
                "observed_restoration_exact": bool(restore_info["observed_exact"]),
            }

    # Transform-only control: all algebraically equivalent representations must
    # produce the same inverse composition before raw observed restoration.
    transform_control = {}
    for method in ("hron_2a", "aitchison_complete"):
        base_rows = e1_predictions["methods"][method]
        base_prediction = np.asarray([[float(v) for v in row["prediction_raw"]] for row in base_rows], dtype=np.float64)
        base_comp = transform_objects["clr16"].inverse(transform_objects["clr16"].forward(base_prediction))
        for name, transform in transform_objects.items():
            comp = transform.inverse(transform.forward(base_prediction))
            transform_control[f"{method}__{name}"] = {
                "max_abs_composition_difference_vs_clr16": float(np.max(np.abs(comp - base_comp))),
                "mean_abs_composition_difference_vs_clr16": float(np.mean(np.abs(comp - base_comp))),
            }
    metrics_output["transform_only_control"] = transform_control
    (OUT / "e2_metrics_fold0.json").write_text(json.dumps(jsonable(metrics_output), indent=2), encoding="utf-8")
    (OUT / "positivity_adjustments_fold0.json").write_text(json.dumps(jsonable(adjustment_output), indent=2), encoding="utf-8")

    # Compact report with the full metrics retained in JSON.
    report = [
        "# E2 report: CLR, ILR and HKGLR transform controls",
        "",
        "**Status:** completed on the frozen fold-0 E1 pilot; no diffusion model was trained.",
        "",
        "## Protocol",
        "",
        f"- Data SHA-256: `{DATA_SHA}`; test location: **{fold0['test_location']}**; rows: {len(test_rows)}.",
        f"- Mask: `{MASK_ID}`; initializer predictions are read from E1; outer-train positivity delta: `{delta:.17g}`.",
        f"- Roundtrip rows: {len(train_complete_raw)} complete rows in outer train. Zero cells are adjusted only in the transform's positive view; raw observed zeros are restored exactly after inverse.",
        f"- The exact adjustment locations are recorded in `positivity_adjustments_fold0.json`: {int(train_adjusted_mask.sum())} complete-train cells and one per-method outer-prediction mask.",
        "- Because Protocol U leaves the 17-part total unidentified, the inverse composition is converted to a raw-count view using a scale estimated from visible positive counts, then all visible raw cells are written back exactly.",
        "",
        "## Frozen HKGLR references",
        "",
        "| Fold | Test location | Actual: 5 lowest missing | Random 42 | Random 43 | Random 44 | High: 5 highest missing |",
        "|---:|---|---|---|---|---|---|",
    ]
    for fold_id in sorted(refs, key=int):
        r = refs[fold_id]
        report.append(
            f"| {fold_id} | {r['test_location']} | {', '.join(r['actual_lowest_missing'])} | {', '.join(r['random']['42'])} | {', '.join(r['random']['43'])} | {', '.join(r['random']['44'])} | {', '.join(r['high_highest_missing'])} |"
        )
    report += [
        "",
        "All rates and random reference sets were calculated from outer-train rows before evaluation masking. Variant names are the alphabetical tie-break for actual/high ordering; random controls use seeds 42, 43 and 44 without replacement.",
        "",
        "## Capacity and basis controls",
        "",
        "| Transform | Latent dimension | Rank | Channels | Projection | Basis/reference |",
        "|---|---:|---:|---:|---|---|",
        "| CLR-16 | 17 | 16 | 16 | final sum-zero | CLR, projected at inverse |",
        "| CLR-17 | 17 | 16 | 17 | none | CLR, unconstrained capacity control |",
        "| ILR-16 | 16 | 16 | 16 | none | fixed Helmert basis |",
        "| HKGLR-16 | 17 | 16 | 16 | final mean_H-zero | five frozen references |",
        "",
        "The Helmert matrix has shape D×(D−1), with column j containing positive entries in rows 0..j and a negative entry at row j+1, as documented in `transforms.py`. It satisfies VᵀV=I and Vᵀ1=0. HKGLR projects by subtracting the mean of its five references before inverse.",
        "",
        "## Roundtrip gate",
        "",
        "| Transform | Max absolute error | Adjusted train cells | Pass (<1e-8) |",
        "|---|---:|---:|---|",
    ]
    for item in roundtrips:
        report.append(f"| {item['transform']} | {item['max_abs_error']:.3e} | {item['n_adjusted_cells']} | {'PASS' if item['pass'] else 'FAIL'} |")
    report += [
        "",
        "The roundtrip compares the recovered composition with the closure of the train-positive view. This is the correct scale-invariant identity test; comparing an inverse composition directly with unclosed raw counts would test an unidentified scale instead.",
        "",
        "## Fold-0 metrics by initializer and transform",
        "",
        "CLR MAE uses hidden cells only. M2 is mean squared Aitchison distance on affected rows with positive truth/prediction sums. Native JSD is included as a metric only and uses the Tsagris natural-log J_T convention.",
        "",
        "| Initializer | Transform | CLR MAE | Nonzero MAE | Zero MAE | M2 | Native JSD | Max transform-only diff vs CLR16 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, score in metrics_output["methods"].items():
        control = metrics_output["transform_only_control"][key]
        report.append(
            f"| {score['initializer']} | {score['transform']} | {score['clr_mae_all_hidden']:.6g} | {score['clr_mae_nonzero_hidden']:.6g} | {score['clr_mae_zero_hidden']:.6g} | {score['m2_mean_squared_aitchison_affected_rows']:.6g} | {score['native_jsd_tsagris_mean']:.6g} | {control['max_abs_composition_difference_vs_clr16']:.3e} |"
        )
    max_control = max(x["max_abs_composition_difference_vs_clr16"] for x in transform_control.values())
    report += [
        "",
        f"The maximum transform-only inverse-composition difference versus CLR16 across both initializers and all eight transform specs is `{max_control:.3e}`. Thus this E2 algebra control does not provide a transform-only accuracy gain; downstream differences must come from model capacity, noise/conditioning or optimization.",
        "",
        "Observed raw restoration was checked exactly for all 16 initializer×transform combinations. The complete donor diagnostics from E1 are embedded in `e2_metrics_fold0.json` for each combination.",
        "",
        "## Scope",
        "",
        "This is a transform/control phase on one outer fold and one mask. It validates basis, positivity provenance, reference freezing and inverse invariants; it is not evidence that CLR, ILR or HKGLR is superior in a learned diffusion model. The next phase may reuse these frozen specs as target branches without changing the reference sets or positivity delta.",
    ]
    (OUT / "e2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "file_hashes.json":
            hashes[path.name] = sha256(path)
    (OUT / "file_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(OUT), "transforms": list(transform_objects), "combinations": len(metrics_output["methods"])}, indent=2))


if __name__ == "__main__":
    main()
