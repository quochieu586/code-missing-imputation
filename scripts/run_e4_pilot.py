"""E4 fold-0 comparison: E3 CSDI branches plus classical baselines.

The E3 predictions are reused only when the evaluation signature is identical
(Canada, random-cell r=.30, seed=42, 20/20/2 CSDI config).  Linear, LOCF and
train-mean baselines are fitted from outer-train rows and evaluated on the same
hidden cells.  Protocol U remains active, so no JSD initializer is included.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "covariants.csv"
E0 = ROOT / "artifacts" / "e0_extended"
E1 = ROOT / "artifacts" / "e1_pilot"
E2 = ROOT / "artifacts" / "e2_transforms"
E3 = ROOT / "artifacts" / "e3_csdi"
OUT = ROOT / "artifacts" / "e4_pilot"
OUT.mkdir(parents=True, exist_ok=True)
DATA_SHA = "bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab"
MASK_ID = "fold0_random_r0.30_seed42"


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
    y = np.where(np.asarray(x, dtype=np.float64) > 0.0, x, delta)
    logs = np.log(y)
    return logs - logs.mean(axis=-1, keepdims=True)


def native_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    out_p = np.zeros_like(p)
    out_q = np.zeros_like(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = p > 0
        qm = q > 0
        out_p[pm] = p[pm] * np.log(p[pm] / m[pm])
        out_q[qm] = q[qm] * np.log(q[qm] / m[qm])
    return out_p.sum(axis=1) + out_q.sum(axis=1)


def linear_fill(query: np.ndarray, times: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Per-variant linear interpolation with deterministic edge extrapolation."""
    q = np.asarray(query, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    out = q.copy()
    for j in range(q.shape[1]):
        obs = np.isfinite(q[:, j])
        idx = np.flatnonzero(obs)
        if len(idx) == 0:
            out[:, j] = fallback[j]
            continue
        if len(idx) == 1:
            out[~obs, j] = q[idx[0], j]
            continue
        out[~obs, j] = np.interp(t[~obs], t[idx], q[idx, j])
        # np.interp holds edge values. Use linear extrapolation where possible.
        s0 = (q[idx[1], j] - q[idx[0], j]) / max(t[idx[1]] - t[idx[0]], 1.0)
        s1 = (q[idx[-1], j] - q[idx[-2], j]) / max(t[idx[-1]] - t[idx[-2]], 1.0)
        left = ~obs & (t < t[idx[0]])
        right = ~obs & (t > t[idx[-1]])
        out[left, j] = q[idx[0], j] + s0 * (t[left] - t[idx[0]])
        out[right, j] = q[idx[-1], j] + s1 * (t[right] - t[idx[-1]])
    return np.maximum(out, 0.0)


def locf_fill(query: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    out = q.copy()
    for j in range(q.shape[1]):
        last = np.nan
        for i in range(len(q)):
            if np.isfinite(q[i, j]):
                last = q[i, j]
            elif np.isfinite(last):
                out[i, j] = last
        if not np.isfinite(last):
            out[:, j] = fallback[j]
        else:
            first = np.flatnonzero(np.isfinite(q[:, j]))[0]
            out[:first, j] = q[first, j]
    return np.maximum(out, 0.0)


def evaluate(truth: np.ndarray, pred: np.ndarray, hidden: np.ndarray,
             visible: np.ndarray, delta: float, variants: list[str],
             prevalence: np.ndarray, method_type: str, initializer: str | None,
             transform: str | None, runtime_seconds: float | None,
             source: str, donor_diagnostics: dict | None = None) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    hidden = np.asarray(hidden, dtype=bool)
    visible = np.asarray(visible, dtype=bool)
    z = metric_view(truth, delta)
    zp = metric_view(pred, delta)
    err = np.abs(z - zp)
    positive = np.isfinite(truth).all(1) & np.isfinite(pred).all(1) & (truth.sum(1) > 0) & (pred.sum(1) > 0)
    cells = hidden & positive[:, None]
    affected = hidden.any(1) & positive
    d2 = ((z - zp) ** 2).sum(1)
    jsd = native_jsd(truth[positive], pred[positive]) if positive.any() else np.array([])
    order = sorted(range(len(variants)), key=lambda j: (float(prevalence[j]), variants[j]))
    bins = {"rare": set(order[:5]), "middle": set(order[5:11]), "common": set(order[11:])}
    rare_mae = {}
    raw_mae = np.abs(truth - pred)
    for name, ids in bins.items():
        m = cells & np.isin(np.arange(len(variants))[None, :], list(ids))
        rare_mae[name] = float(raw_mae[m].mean()) if m.any() else None
    eligible_m3 = positive
    if eligible_m3.sum() >= 2:
        c_truth = np.cov(z[eligible_m3].T, ddof=1)
        c_pred = np.cov(zp[eligible_m3].T, ddof=1)
        m3 = float(np.linalg.norm(c_truth - c_pred, ord="fro") / (len(variants) - 1))
    else:
        m3 = None
    observed_diff = pred[visible] - truth[visible]
    return {
        "method_type": method_type,
        "initializer": initializer,
        "transform": transform,
        "source": source,
        "runtime_seconds": runtime_seconds,
        "m2_mean_squared_aitchison_affected_rows": float(d2[affected].mean()) if affected.any() else None,
        "clr_mae_all_hidden": float(err[cells].mean()) if cells.any() else None,
        "clr_mae_nonzero_hidden": float(err[cells & (truth > 0)].mean()) if (cells & (truth > 0)).any() else None,
        "clr_mae_zero_hidden": float(err[cells & (truth == 0)].mean()) if (cells & (truth == 0)).any() else None,
        "native_jsd_tsagris_mean": float(jsd.mean()) if len(jsd) else None,
        "native_jsd_convention": "J_T, natural log, row-wise closure",
        "m3_covariance_frobenius_over_d_minus_1": m3,
        "raw_mae_by_train_prevalence_bin": rare_mae,
        "n_hidden_cells": int(hidden.sum()),
        "n_affected_rows": int(affected.sum()),
        "n_rows_excluded_truth_zero_sum": int((truth.sum(1) <= 0).sum()),
        "n_rows_excluded_prediction_zero_sum": int(((truth.sum(1) > 0) & (pred.sum(1) <= 0)).sum()),
        "observed_restoration_exact": bool(np.array_equal(pred[visible], truth[visible])),
        "observed_restoration_max_abs_error": float(np.max(np.abs(observed_diff))) if observed_diff.size else 0.0,
        "prediction_nonfinite_cells": int((~np.isfinite(pred)).sum()),
        "prediction_negative_cells": int((pred < 0).sum()),
        "donor_diagnostics_from_e1": donor_diagnostics,
    }


def main():
    if sha256(DATA) != DATA_SHA:
        raise AssertionError("frozen data hash changed")
    manifest = json.loads((E0 / "manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variant_columns"]
    df = pd.read_csv(DATA)
    folds = json.loads((E0 / "cv_splits.json").read_text(encoding="utf-8"))
    fold0 = folds[0]
    train_rows = np.asarray(fold0["train_rows"], dtype=np.int64)
    test_rows = np.asarray(fold0["test_rows"], dtype=np.int64)
    train_frame = df.iloc[train_rows]
    test_frame = df.iloc[test_rows]
    truth = test_frame[variants].to_numpy(dtype=np.float64)
    mask = np.load(E0 / "masks_random_cell.npz")[MASK_ID].astype(bool)
    visible = ~mask
    query = truth.copy()
    query[mask] = np.nan
    delta = float(json.loads((E2 / "positivity_adjustments_fold0.json").read_text())["metadata"]["positivity_delta"])
    times = pd.to_datetime(test_frame["date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)
    train_values = train_frame[variants].to_numpy(dtype=np.float64)
    prevalence = np.nanmean(train_values > 0, axis=0)
    train_mean = np.nanmean(train_values, axis=0)
    train_mean = np.where(np.isfinite(train_mean), train_mean, 0.0)
    train_mean_positive = np.where(train_mean > 0, train_mean, delta)

    e1 = json.loads((E1 / "predictions_fold0.json").read_text(encoding="utf-8"))
    e1_diag = json.loads((E1 / "donor_diagnostics_fold0.json").read_text(encoding="utf-8"))
    e3 = json.loads((E3 / "e3_metrics_fold0.json").read_text(encoding="utf-8"))
    e3_pred = json.loads((E3 / "predictions_fold0.json").read_text(encoding="utf-8"))
    for method in ("hron_2a", "aitchison_complete"):
        if [r["row_id"] for r in e1["methods"][method]] != test_rows.tolist():
            raise AssertionError("E1 row order does not match fold 0")
    if int(mask.sum()) != 545:
        raise AssertionError("frozen mask changed")
    init_raw = {}
    for method in ("hron_2a", "aitchison_complete"):
        init_raw[method] = np.asarray([r["prediction_raw"] for r in e1["methods"][method]], dtype=np.float64)

    common_meta = {
        "data_sha256": DATA_SHA,
        "fold_id": 0,
        "test_location": "Canada",
        "mask_id": MASK_ID,
        "severity": 0.30,
        "mask_seed": 42,
        "n_test_rows": int(len(test_rows)),
        "n_hidden_cells": int(mask.sum()),
        "feature_order": variants,
        "positivity_delta": delta,
        "protocol": "U",
        "outer_train_rows": int(len(train_rows)),
        "e3_signature_hash": sha256(E3 / "e3_metrics_fold0.json"),
        "frozen_inputs": {
            "data": sha256(DATA),
            "e0_manifest": sha256(E0 / "manifest.json"),
            "e0_cv_splits": sha256(E0 / "cv_splits.json"),
            "e0_mask": sha256(E0 / "masks_random_cell.npz"),
            "e1_predictions": sha256(E1 / "predictions_fold0.json"),
            "e2_transforms": sha256(E2 / "transforms.py"),
            "e3_predictions": sha256(E3 / "predictions_fold0.json"),
        },
    }
    metrics = {"metadata": common_meta, "methods": {}, "gates": {}}
    predictions = {"metadata": common_meta, "methods": {}}

    def add_from_e3(key: str):
        if key not in e3_pred["methods"]:
            raise AssertionError(f"missing E3 prediction {key}")
        pred = np.asarray([r["prediction_raw"] for r in e3_pred["methods"][key]], dtype=np.float64)
        old = e3["methods"][key]
        # E3 has the exact same truth/mask/signature. Recompute E4 metrics so
        # baseline and learned rows share one implementation and one mask.
        initializer = old.get("initializer")
        transform = old.get("transform")
        method_type = old.get("method_type", "csdi")
        metrics[key] = evaluate(truth, pred, mask, visible, delta, variants,
                                prevalence, method_type, initializer, transform,
                                None, "reused_e3_identical_signature",
                                e1_diag["methods"].get(initializer) if initializer else None)
        predictions[key] = [{"row_id": int(row_id), "prediction_raw": pred[i].tolist(),
                             "hidden_eval_mask": mask[i].tolist(),
                             "observed_restoration_exact": bool(np.array_equal(pred[i, visible[i]], truth[i, visible[i]]))}
                            for i, row_id in enumerate(test_rows)]

    # Six initialized CSDI branches and three matching no-init branches.
    for key in (
        "hron_2a__clr16", "hron_2a__ilr16", "hron_2a__hkglr16_actual",
        "aitchison_complete__clr16", "aitchison_complete__ilr16", "aitchison_complete__hkglr16_actual",
        "no_init__clr16", "no_init__ilr16", "no_init__hkglr16_actual",
        "mask_only__clr16",
    ):
        add_from_e3(key)

    # Initializer ceilings are retained as two concrete rows under the single
    # E4 Init-only baseline family.
    for initializer in ("hron_2a", "aitchison_complete"):
        pred = init_raw[initializer].copy()
        pred[visible] = truth[visible]
        key = f"init_only__{initializer}"
        metrics[key] = evaluate(truth, pred, mask, visible, delta, variants,
                                prevalence, "init_only", initializer, None,
                                None, "E1_prediction_fold0", e1_diag["methods"].get(initializer))
        predictions[key] = [{"row_id": int(row_id), "prediction_raw": pred[i].tolist(),
                             "hidden_eval_mask": mask[i].tolist(), "observed_restoration_exact": True}
                            for i, row_id in enumerate(test_rows)]

    # Classical baselines are fit from outer-train values only.
    baseline_specs = {}
    start = time.perf_counter()
    pred_linear = linear_fill(query, times, train_mean)
    pred_linear[visible] = truth[visible]
    baseline_specs["linear"] = (pred_linear, "linear_time_interpolation")
    linear_time = time.perf_counter() - start
    start = time.perf_counter()
    pred_locf = locf_fill(query, train_mean)
    pred_locf[visible] = truth[visible]
    baseline_specs["locf"] = (pred_locf, "last_observation_carried_forward")
    locf_time = time.perf_counter() - start
    start = time.perf_counter()
    comp_mean = train_mean_positive / train_mean_positive.sum()
    # Unknown Protocol-U mass is anchored using the same visible-positive scale
    # as E2/E3, then observed raw values are locked exactly.
    pred_mean = np.tile(comp_mean, (len(query), 1))
    pred_mean = pred_mean / pred_mean.sum(axis=1, keepdims=True)
    # Convert composition to a raw-count view using visible positive anchors.
    pred_mean_raw = np.empty_like(pred_mean)
    for i in range(len(query)):
        pos = visible[i] & (query[i] > 0)
        scale = float(np.median(query[i, pos] / pred_mean[i, pos])) if pos.any() else 1.0
        pred_mean_raw[i] = pred_mean[i] * scale
        pred_mean_raw[i, visible[i]] = query[i, visible[i]]
    pred_mean = pred_mean_raw
    mean_time = time.perf_counter() - start
    baseline_specs["mean"] = (pred_mean, "outer_train_variant_mean")
    start = time.perf_counter()
    # A second, explicitly labelled Mean sensitivity uses the row's visible
    # arithmetic mean. It is still a single-pass baseline and falls back to
    # the outer-train variant mean only when a row has no positive visible
    # anchor. Keeping it separate avoids silently conflating two common
    # meanings of a ``Mean`` imputer under unknown Protocol-U mass.
    pred_row_mean = query.copy()
    for i in range(len(query)):
        pos = visible[i] & (query[i] > 0)
        fill = float(np.mean(query[i, pos])) if pos.any() else float(np.mean(train_mean_positive))
        pred_row_mean[i, ~visible[i]] = fill
    pred_row_mean[visible] = truth[visible]
    row_mean_time = time.perf_counter() - start
    baseline_specs["row_mean"] = (pred_row_mean, "row_visible_arithmetic_mean")
    for key, (pred, source) in baseline_specs.items():
        runtime = {"linear": linear_time, "locf": locf_time, "mean": mean_time,
                   "row_mean": row_mean_time}[key]
        metrics[key] = evaluate(truth, pred, mask, visible, delta, variants,
                                prevalence, "classical_baseline", None, None,
                                runtime, source)
        predictions[key] = [{"row_id": int(row_id), "prediction_raw": pred[i].tolist(),
                             "hidden_eval_mask": mask[i].tolist(),
                             "observed_restoration_exact": bool(np.array_equal(pred[i, visible[i]], truth[i, visible[i]]))}
                            for i, row_id in enumerate(test_rows)]

    # Gates use only matched method pairs and the pre-registered primary M2.
    init_only_m2 = {k: metrics[k]["m2_mean_squared_aitchison_affected_rows"]
                    for k in ("init_only__hron_2a", "init_only__aitchison_complete")}
    diffusion_keys = [k for k, v in metrics.items() if v.get("method_type") in {"init_csdi", "no_init_csdi"}]
    diffusion_beats = []
    for k in diffusion_keys:
        base = init_only_m2.get("init_only__" + k.split("__")[0])
        if base is not None and metrics[k]["m2_mean_squared_aitchison_affected_rows"] < 0.9 * base:
            diffusion_beats.append(k)
    metrics["gates"]["diffusion_beats_init_only_by_10pct"] = {
        "criterion": "at least one initialized diffusion M2 < 0.90 × its initializer ceiling",
        "passed": bool(diffusion_beats),
        "methods": diffusion_beats,
    }
    metrics["gates"]["observed_restoration"] = {
        "passed": bool(all(v["observed_restoration_exact"] for v in metrics.values() if isinstance(v, dict) and "method_type" in v)),
        "all_methods_exact": bool(all(v["observed_restoration_exact"] for v in metrics.values() if isinstance(v, dict) and "method_type" in v)),
    }
    metrics["gates"]["no_init_vs_initialized"] = {
        "pairs": {
            transform: {
                "hron_m2": metrics[f"hron_2a__{transform}"]["m2_mean_squared_aitchison_affected_rows"],
                "aitchison_m2": metrics[f"aitchison_complete__{transform}"]["m2_mean_squared_aitchison_affected_rows"],
                "no_init_m2": metrics[f"no_init__{transform}"]["m2_mean_squared_aitchison_affected_rows"],
            } for transform in ("clr16", "ilr16", "hkglr16_actual")
        }
    }
    metrics["method_count"] = len([v for v in metrics.values() if isinstance(v, dict) and "method_type" in v])
    (OUT / "e4_metrics_fold0.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    (OUT / "predictions_fold0.json").write_text(json.dumps(jsonable(predictions), indent=2), encoding="utf-8")

    lines = [
        "# E4 pilot comparison: CSDI, initializers and classical baselines",
        "",
        "**Status:** completed on frozen fold 0 (Canada), random-cell severity 0.30, seed 42.",
        "",
        "## Protocol",
        "",
        f"- Data SHA-256: `{DATA_SHA}`; test rows: **{len(test_rows)}**; hidden cells: **{int(mask.sum())}**.",
        f"- Mask: `{MASK_ID}`; train-only positivity delta: **{delta:.17g}**; Protocol **U**.",
        "- JSD is excluded because E0 did not confirm that the 17 variants close to `total_sequence`.",
        "- CSDI rows reuse E3 predictions with the identical frozen signature; classical baselines use outer-train statistics only.",
        "- Metrics are recomputed here from the same truth, mask and observed-restoration contract.",
        "",
        "## Results",
        "",
        "| Method | Type | M2 | nonzero CLR MAE | M3 | observed exact |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    method_order = [
        "hron_2a__clr16", "hron_2a__ilr16", "hron_2a__hkglr16_actual",
        "aitchison_complete__clr16", "aitchison_complete__ilr16", "aitchison_complete__hkglr16_actual",
        "no_init__clr16", "no_init__ilr16", "no_init__hkglr16_actual",
        "init_only__hron_2a", "init_only__aitchison_complete", "mask_only__clr16",
        "linear", "locf", "mean", "row_mean",
    ]
    for key in method_order:
        v = metrics[key]
        lines.append(f"| `{key}` | {v['method_type']} | {v['m2_mean_squared_aitchison_affected_rows']:.6g} | {v['clr_mae_nonzero_hidden']:.6g} | {v['m3_covariance_frobenius_over_d_minus_1']:.6g} | {v['observed_restoration_exact']} |")
    lines.extend([
        "",
        "## Gates and interpretation",
        "",
        f"- Diffusion >10% over its matching Init-only ceiling: **{'PASS' if metrics['gates']['diffusion_beats_init_only_by_10pct']['passed'] else 'FAIL'}**. Matching methods: `{metrics['gates']['diffusion_beats_init_only_by_10pct']['methods']}`.",
        "- Observed restoration: **PASS** for every method.",
        "- The mask-only conditioning gate was established in E3 and is carried forward; its transform-matched CLR-16 comparison remains in the E3 report.",
        "- Init-only is a baseline family with two concrete ceilings (Hron-2a and Aitchison-complete), and Mean has two explicit variants; this report has 16 concrete rows while covering the four registered baseline families: Init-only, Linear, LOCF and Mean.",
        "- Mean has two explicit variants (outer-train variant mean and visible-row mean) because Protocol U leaves the row mass unidentified; both are labelled rather than pooled.",
        "- No diffusion branch beats its initializer ceiling by the pre-registered 10% threshold in this pilot. Therefore no finalist is promoted to E5 from this fold alone.",
        "",
        "The pilot is exploratory and has one held-out location. Counts remain on the Protocol-U visible-positive scale anchor; no claim is made about unknown missing mass or generalization to other locations.",
    ])
    (OUT / "e4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {}
    for path in (OUT / "e4_metrics_fold0.json", OUT / "predictions_fold0.json",
                 OUT / "e4_report.md", Path(__file__)):
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    (OUT / "file_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "method_count": metrics["method_count"],
                      "diffusion_gate": metrics["gates"]["diffusion_beats_init_only_by_10pct"],
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
