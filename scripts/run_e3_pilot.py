"""Run the fold-0 E3 CSDI pilot and its no-init/mask-only controls."""

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
E2 = ROOT / "artifacts" / "e2_transforms"
OUT = ROOT / "artifacts" / "e3_csdi"
OUT.mkdir(parents=True, exist_ok=True)
DATA_SHA = "bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab"
MASK_ID = "fold0_random_r0.30_seed42"
TOL = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


csdi = load_module(OUT / "csdi_core.py", "e3_csdi_core")
transforms = load_module(E2 / "transforms.py", "e2_transform_artifact_e3")


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


def metric_view(x: np.ndarray, delta: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.where(x > 0.0, x, delta)
    logs = np.log(y)
    return logs - logs.mean(axis=-1, keepdims=True)


def native_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    pt = np.zeros_like(p)
    qt = np.zeros_like(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = p > 0
        qm = q > 0
        pt[pm] = p[pm] * np.log(p[pm] / m[pm])
        qt[qm] = q[qm] * np.log(q[qm] / m[qm])
    return pt.sum(axis=-1) + qt.sum(axis=-1)


def evaluate(truth: np.ndarray, pred: np.ndarray, hidden: np.ndarray,
             visible: np.ndarray, delta: float, restore_info: dict,
             method_type: str, initializer: str | None, transform_name: str | None,
             model_metadata: dict | None = None) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    hidden = np.asarray(hidden, dtype=bool)
    visible = np.asarray(visible, dtype=bool)
    if truth.shape != pred.shape or truth.shape != hidden.shape:
        raise ValueError("truth, prediction and hidden mask shape mismatch")
    z = metric_view(truth, delta)
    zp = metric_view(pred, delta)
    err = np.abs(z - zp)
    positive_rows = (np.isfinite(truth).all(1) & np.isfinite(pred).all(1)
                     & (truth.sum(1) > 0) & (pred.sum(1) > 0))
    cells = hidden & positive_rows[:, None]
    affected = hidden.any(1) & positive_rows
    d2 = ((z - zp) ** 2).sum(1)
    jsd = native_jsd(truth[positive_rows], pred[positive_rows]) if positive_rows.any() else np.array([])

    def avg(a, m):
        return float(a[m].mean()) if np.any(m) else None

    observed_diff = pred[visible] - truth[visible]
    exact = bool(np.array_equal(pred[visible], truth[visible]))
    result = {
        "method_type": method_type,
        "initializer": initializer,
        "transform": transform_name,
        "clr_mae_all_hidden": avg(err, cells),
        "clr_mae_nonzero_hidden": avg(err, cells & (truth > 0)),
        "clr_mae_zero_hidden": avg(err, cells & (truth == 0)),
        "m2_mean_squared_aitchison_affected_rows": float(d2[affected].mean()) if np.any(affected) else None,
        "native_jsd_tsagris_mean": float(jsd.mean()) if len(jsd) else None,
        "native_jsd_convention": "J_T, natural log, row-wise closure",
        "n_hidden_cells": int(hidden.sum()),
        "n_hidden_nonzero": int((cells & (truth > 0)).sum()),
        "n_hidden_zero": int((cells & (truth == 0)).sum()),
        "n_affected_rows": int(affected.sum()),
        "n_rows_excluded_truth_zero_sum": int((truth.sum(1) <= 0).sum()),
        "n_rows_excluded_prediction_zero_sum": int(((truth.sum(1) > 0) & (pred.sum(1) <= 0)).sum()),
        "observed_restoration_exact": exact,
        "observed_restoration_max_abs_error": float(np.max(np.abs(observed_diff))) if observed_diff.size else 0.0,
        "prediction_nonfinite_cells": int((~np.isfinite(pred)).sum()),
        "prediction_negative_cells": int((pred < 0).sum()),
        "prediction_hidden_min": float(np.min(pred[hidden])) if np.any(hidden) else None,
        "prediction_hidden_max": float(np.max(pred[hidden])) if np.any(hidden) else None,
        "n_no_positive_visible_anchor": int(restore_info.get("n_no_positive_visible_anchor", 0)),
        "scale_summary": {
            "min": float(np.min(restore_info["scales"])),
            "median": float(np.median(restore_info["scales"])),
            "max": float(np.max(restore_info["scales"])),
        },
    }
    if model_metadata is not None:
        result["model_metadata"] = model_metadata
    return result


def load_e1_arrays(e1: dict, method: str):
    rows = e1["methods"][method]
    truth = np.asarray([[float(v) for v in row["truth_raw"]] for row in rows], dtype=np.float64)
    query = np.asarray([[np.nan if v is None else float(v)
                         for v in row["input_raw_with_hidden_as_null"]] for row in rows], dtype=np.float64)
    init = np.asarray([[float(v) for v in row["prediction_raw"]] for row in rows], dtype=np.float64)
    hidden = np.asarray([row["hidden_eval_mask"] for row in rows], dtype=bool)
    return rows, truth, query, init, hidden


def make_reference_transforms(df: pd.DataFrame, variants: list[str], fold0: dict,
                              delta: float):
    folds = json.loads((E0 / "cv_splits.json").read_text(encoding="utf-8"))
    references = {}
    for fold in folds:
        train = df.iloc[np.asarray(fold["train_rows"], dtype=np.int64)]
        rates = train[variants].isna().mean()
        refs = sorted(variants, key=lambda v: (float(rates[v]), v))[:5]
        references[str(fold["fold_id"])] = refs
    refs_idx = tuple(variants.index(v) for v in references["0"])
    return {
        "clr16": transforms.make_transform("clr16", len(variants), delta),
        "ilr16": transforms.make_transform("ilr16", len(variants), delta),
        "hkglr16_actual": transforms.make_transform("hkglr16_actual", len(variants), delta, refs_idx),
    }, references


def main():
    if sha256(DATA) != DATA_SHA:
        raise AssertionError("Frozen data SHA-256 does not match the E0 manifest")
    manifest = json.loads((E0 / "manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variant_columns"]
    df = pd.read_csv(DATA)
    folds = json.loads((E0 / "cv_splits.json").read_text(encoding="utf-8"))
    fold0 = folds[0]
    if fold0["fold_id"] != 0 or fold0["test_location"] != "Canada":
        raise AssertionError("Frozen fold-0 contract changed")
    train_rows = np.asarray(fold0["train_rows"], dtype=np.int64)
    test_rows = np.asarray(fold0["test_rows"], dtype=np.int64)
    train_frame = df.iloc[train_rows]
    test_frame = df.iloc[test_rows]
    train_raw_all = train_frame[variants].to_numpy(dtype=np.float64)
    complete_train = np.isfinite(train_raw_all).all(axis=1)
    train_raw = train_raw_all[complete_train]
    train_ids = train_rows[complete_train]
    if len(test_rows) != 109 or len(train_raw) != 407:
        raise AssertionError("Frozen fold-0 row eligibility changed")
    mask = np.load(E0 / "masks_random_cell.npz")[MASK_ID].astype(bool)
    if mask.shape != (len(test_rows), len(variants)) or int(mask.sum()) != 545:
        raise AssertionError("Frozen E0 pilot mask changed")

    e1 = json.loads((E1 / "predictions_fold0.json").read_text(encoding="utf-8"))
    e1_metrics = json.loads((E1 / "donor_diagnostics_fold0.json").read_text(encoding="utf-8"))
    loaded = {m: load_e1_arrays(e1, m) for m in ("hron_2a", "aitchison_complete")}
    for method, (rows, truth, _, _, hidden) in loaded.items():
        if [int(r["row_id"]) for r in rows] != test_rows.tolist():
            raise AssertionError(f"E1 row order mismatch for {method}")
        if not np.array_equal(hidden, mask):
            raise AssertionError(f"E1 mask mismatch for {method}")

    adjustment = json.loads((E2 / "positivity_adjustments_fold0.json").read_text(encoding="utf-8"))
    delta = float(adjustment["metadata"]["positivity_delta"])
    if delta != 0.5:
        raise AssertionError("E2 train-only positivity delta changed")
    transform_objects, reference_sets = make_reference_transforms(df, variants, fold0, delta)
    train_dates = pd.to_datetime(train_frame["date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)[complete_train]
    test_dates = pd.to_datetime(test_frame["date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)
    # The completion builder is a deterministic train-only stand-in for an
    # initializer while fitting CSDI; test initialization comes from E1.
    col_medians = np.nanmedian(train_raw_all, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)

    def init_builder(masked, visible):
        return np.where(visible, masked, col_medians[None, :])

    config = csdi.CSDIConfig(epochs=20, diffusion_steps=20, mc_samples=2,
                             channels=16, masks_per_epoch=4, learning_rate=0.025)
    metrics = {
        "metadata": {
            "data_sha256": DATA_SHA,
            "fold_id": 0,
            "test_location": "Canada",
            "mask_id": MASK_ID,
            "n_test_rows": int(len(test_rows)),
            "n_hidden_cells": int(mask.sum()),
            "feature_order": variants,
            "positivity_delta": delta,
            "train_complete_rows": int(len(train_raw)),
            "train_complete_row_ids": train_ids.tolist(),
            "outer_protocol": "U",
            "single_outer_pass": True,
            "diffusion_is_internal_reverse_chain": True,
            "frozen_inputs": {
                "e0_manifest": sha256(E0 / "manifest.json"),
                "e0_cv_splits": sha256(E0 / "cv_splits.json"),
                "e0_random_masks": sha256(E0 / "masks_random_cell.npz"),
                "e1_predictions": sha256(E1 / "predictions_fold0.json"),
                "e2_transform_code": sha256(E2 / "transforms.py"),
            },
        },
        "reference_sets_all_folds": reference_sets,
        "methods": {},
    }
    predictions = {"metadata": metrics["metadata"], "methods": {}}

    # Init-only ceiling controls. They are deliberately reported separately
    # from learned CSDI rows and are not allowed to tune any model statistic.
    for initializer in ("hron_2a", "aitchison_complete"):
        rows, truth, query, init, hidden = loaded[initializer]
        visible = np.isfinite(query)
        pred = init.copy()
        pred[visible] = truth[visible]
        restored = {"scales": np.ones(len(pred)), "n_no_positive_visible_anchor": 0}
        key = f"init_only__{initializer}"
        metrics["methods"][key] = evaluate(
            truth, pred, hidden, visible, delta, restored, "init_only",
            initializer, None, {"source": "E1 prediction_fold0", "selected_k": 4},
        )
        predictions["methods"][key] = [{
            "row_id": int(row["row_id"]), "prediction_raw": pred[i].tolist(),
            "hidden_eval_mask": hidden[i].tolist(), "observed_restoration_exact": True,
        } for i, row in enumerate(rows)]

    method_counter = 0
    for transform_name, transform in transform_objects.items():
        target_z = np.asarray(transform.forward(train_raw), dtype=np.float64)
        latent_dim = target_z.shape[1]
        # The same train-only target is reused for the three conditioning
        # ablations; only the permitted information in condition changes.
        for initializer in ("hron_2a", "aitchison_complete"):
            rows, truth, query, init, hidden = loaded[initializer]
            visible = np.isfinite(query)
            init_z_test = np.asarray(transform.forward(init), dtype=np.float64)
            seed = 3000 + method_counter
            model = csdi.CSDICore(latent_dim, len(variants), "init", config, seed=seed)
            model.fit(target_z, train_raw, train_dates, transform,
                      init_builder=init_builder)
            latent = model.sample_latent(query, visible, test_dates,
                                         init_z=init_z_test, n_samples=2,
                                         seed=seed + 50000)
            comp = np.asarray(transform.inverse(latent), dtype=np.float64)
            pred, restore_info = transforms.restore_observed_counts(comp, query)
            if not np.array_equal(pred[visible], truth[visible]):
                raise AssertionError(f"Observed restoration failed: {initializer}/{transform_name}")
            key = f"{initializer}__{transform_name}"
            mm = evaluate(truth, pred, hidden, visible, delta, restore_info,
                          "init_csdi", initializer, transform_name,
                          model.metadata())
            mm["donor_diagnostics_from_e1"] = e1_metrics["methods"][initializer]
            mm["latent_dim"] = latent_dim
            metrics["methods"][key] = mm
            predictions["methods"][key] = [{
                "row_id": int(row["row_id"]), "prediction_raw": pred[i].tolist(),
                "hidden_eval_mask": hidden[i].tolist(),
                "observed_restoration_exact": True,
            } for i, row in enumerate(rows)]
            method_counter += 1

        # No-init CSDI: raw visible counts + mask + time, without E1 values.
        rows, truth, query, _, hidden = loaded["hron_2a"]
        visible = np.isfinite(query)
        seed = 5000 + method_counter
        model = csdi.CSDICore(latent_dim, len(variants), "no_init", config, seed=seed)
        model.fit(target_z, train_raw, train_dates, transform, init_builder=None)
        latent = model.sample_latent(query, visible, test_dates, init_z=None,
                                     n_samples=2, seed=seed + 50000)
        comp = np.asarray(transform.inverse(latent), dtype=np.float64)
        pred, restore_info = transforms.restore_observed_counts(comp, query)
        if not np.array_equal(pred[visible], truth[visible]):
            raise AssertionError(f"Observed restoration failed: no_init/{transform_name}")
        key = f"no_init__{transform_name}"
        mm = evaluate(truth, pred, hidden, visible, delta, restore_info,
                      "no_init_csdi", None, transform_name, model.metadata())
        mm["latent_dim"] = latent_dim
        metrics["methods"][key] = mm
        predictions["methods"][key] = [{
            "row_id": int(row["row_id"]), "prediction_raw": pred[i].tolist(),
            "hidden_eval_mask": hidden[i].tolist(),
            "observed_restoration_exact": True,
        } for i, row in enumerate(rows)]
        method_counter += 1

    # One mask-only control is sufficient because the target transform is an
    # algebraic output adapter; its condition has no value slots at all.
    transform_name = "clr16"
    transform = transform_objects[transform_name]
    target_z = np.asarray(transform.forward(train_raw), dtype=np.float64)
    rows, truth, query, _, hidden = loaded["hron_2a"]
    visible = np.isfinite(query)
    seed = 7000 + method_counter
    model = csdi.CSDICore(target_z.shape[1], len(variants), "mask_only", config, seed=seed)
    model.fit(target_z, train_raw, train_dates, transform, init_builder=None)
    latent = model.sample_latent(query, visible, test_dates, init_z=None,
                                 n_samples=2, seed=seed + 50000)
    comp = np.asarray(transform.inverse(latent), dtype=np.float64)
    pred, restore_info = transforms.restore_observed_counts(comp, query)
    if not np.array_equal(pred[visible], truth[visible]):
        raise AssertionError("Observed restoration failed: mask_only")
    key = "mask_only__clr16"
    mm = evaluate(truth, pred, hidden, visible, delta, restore_info,
                  "mask_only_csdi", None, transform_name, model.metadata())
    mm["latent_dim"] = target_z.shape[1]
    metrics["methods"][key] = mm
    predictions["methods"][key] = [{
        "row_id": int(row["row_id"]), "prediction_raw": pred[i].tolist(),
        "hidden_eval_mask": hidden[i].tolist(),
        "observed_restoration_exact": True,
    } for i, row in enumerate(rows)]

    if len(metrics["methods"]) != 12:
        raise AssertionError(f"E3 requires 12 methods, got {len(metrics['methods'])}")
    no_init_m2 = [v["m2_mean_squared_aitchison_affected_rows"]
                  for k, v in metrics["methods"].items() if k.startswith("no_init__")]
    mask_m2 = metrics["methods"]["mask_only__clr16"]["m2_mean_squared_aitchison_affected_rows"]
    matching_no_init = metrics["methods"]["no_init__clr16"]["m2_mean_squared_aitchison_affected_rows"]
    gate = {
        "criterion": "mask-only CLR-16 M2 must be strictly worse than no-init CLR-16 M2",
        "mask_only_m2": float(mask_m2),
        "no_init_m2_by_transform": {k: float(v["m2_mean_squared_aitchison_affected_rows"])
                                     for k, v in metrics["methods"].items() if k.startswith("no_init__")},
        "no_init_mean_m2": float(np.mean(no_init_m2)),
        "matching_no_init_transform": "clr16",
        "matching_no_init_m2": float(matching_no_init),
        "passed": bool(mask_m2 > matching_no_init),
        "passed_against_no_init_mean": bool(mask_m2 > float(np.mean(no_init_m2))),
        "passed_against_each_no_init_transform": bool(all(mask_m2 > x for x in no_init_m2)),
    }
    metrics["mask_only_gate"] = gate
    metrics["method_count"] = len(metrics["methods"])

    (OUT / "e3_metrics_fold0.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    (OUT / "predictions_fold0.json").write_text(json.dumps(jsonable(predictions), indent=2), encoding="utf-8")

    lines = [
        "# E3 pilot: CSDI core and conditioning controls",
        "",
        "**Status:** completed on frozen fold 0 (Canada) with one outer pass.",
        "",
        "## Frozen protocol",
        "",
        f"- Data SHA-256: `{DATA_SHA}`; test rows: **{len(test_rows)}**; hidden cells: **{int(mask.sum())}**.",
        f"- Mask: `{MASK_ID}`; outer-train complete rows used for target fitting: **{len(train_raw)}**.",
        f"- Train-only positivity delta: **{delta:.17g}**; feature order has {len(variants)} variants.",
        "- All runs use float64, natural-log LR transforms, observed restoration after inverse, and no outer refinement.",
        "",
        "## Core",
        "",
        "The denoiser is a NumPy CSDI-style epsilon model with a shared width-16 encoder. Its condition layout is raw visible log1p counts, a binary visibility mask, optional initializer LR values, and a two-coordinate time embedding. The target branch is the transform-specific LR latent. Each fit uses 20 training epochs, 20 internal DDPM reverse steps, and two latent samples whose mean is inverted.",
        "",
        "## Metrics",
        "",
        "| Method | Type | M2 | nonzero CLR MAE | observed exact |"]
    lines.append("|---|---|---:|---:|:---:|")
    for key, value in metrics["methods"].items():
        lines.append(f"| `{key}` | {value['method_type']} | {value['m2_mean_squared_aitchison_affected_rows']:.6g} | {value['clr_mae_nonzero_hidden']:.6g} | {value['observed_restoration_exact']} |")
    lines.extend([
        "",
        "## Mandatory mask-only gate",
        "",
        f"- Mask-only M2: **{gate['mask_only_m2']:.6g}**.",
        f"- No-init M2 by transform: " + ", ".join(f"`{k.replace('no_init__','')}`={v:.6g}" for k, v in gate["no_init_m2_by_transform"].items()) + ".",
        f"- Matching CLR-16 gate: **{'PASS' if gate['passed'] else 'FAIL'}** ({gate['mask_only_m2']:.6g} > {gate['matching_no_init_m2']:.6g}). The all-transform mean comparison is **{'PASS' if gate['passed_against_no_init_mean'] else 'FAIL'}** and is reported only as a sensitivity check.",
        "",
        "The gate is a conditioning diagnostic. A pass means the mask-only model is less accurate on this pilot, consistent with value conditioning carrying information; it is not a claim of generalization beyond the frozen fold.",
        "",
        "## Integrity checks",
        "",
        "- All 12 requested rows are present in `e3_metrics_fold0.json`.",
        "- Every learned output passed exact observed-cell restoration; nonfinite and negative output counts are recorded per method.",
        "- E1 donor diagnostics are embedded in each Init+CSDI method, and reference sets are recorded for all folds.",
        "- The full per-row predictions and method provenance are in `predictions_fold0.json`.",
        "",
        "This is a pilot-scale CSDI core under Protocol U. The 17-part total remains unidentified, so hidden raw counts use the same visible-positive scale anchor as E2 after inverse composition. No natural-missing cell is included in the hidden-cell accuracy mask.",
    ])
    (OUT / "e3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashes = {}
    for path in (OUT / "csdi_core.py", OUT / "e3_metrics_fold0.json",
                 OUT / "predictions_fold0.json", OUT / "e3_report.md",
                 Path(__file__)):
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    (OUT / "file_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "methods": len(metrics["methods"]),
                      "mask_only_gate": gate, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
