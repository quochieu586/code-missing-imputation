"""Run the registered three-family baseline benchmark on an easy/medium/hard bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from longitudinal_impute.baselines.adapters import CountBaselineAdapter
from longitudinal_impute.baselines.panels import subset
from longitudinal_impute.baselines.registry import make_core
from longitudinal_impute.contracts import FitScope
from longitudinal_impute.synthetic import load_conditioning, load_evaluation
from longitudinal_impute.synthetic.scoring import score


METHODS = {
    "hron_knn": [
        {"k": 3}, {"k": 5}, {"k": 10},
    ],
    "micoda_v2": [
        {"W": 1e-6, "max_iter": 50},
        {"W": 1e-4, "max_iter": 50},
        {"W": 1e-3, "max_iter": 50},
    ],
    "tsagris_alpha": [
        {"k": k, "alpha": alpha}
        for k, alpha in itertools.product((3, 5, 10), (0.25, 0.5, 1.0))
    ],
}
MASKS = ("01_mcar_declared_patterns", "02_mar_declared_patterns")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=json_default) + "\n",
                    encoding="utf-8")




def new_adapter(method, parameters):
    return CountBaselineAdapter(
        make_core(method, degenerate_policies=True, **parameters),
        fallback="train_mean",
        degenerate_policies=True,
    )


def fit_scope(panel, allowed_ids, split_id):
    return FitScope("inductive", tuple(allowed_ids), split_id, panel.mask_id)


def run_mask(bundle: Path, mask_name: str, output: Path):
    panel = load_conditioning(bundle, mask_name)
    train_panel = load_conditioning(bundle, mask_name, split="train")
    validation_panel = load_conditioning(bundle, mask_name, split="validation")
    test_panel = load_conditioning(bundle, mask_name, split="test")
    train_ids = train_panel.row_ids
    train_validation_ids = train_ids + validation_panel.row_ids
    validation_labels, validation_latent = load_evaluation(bundle, mask_name, split="validation")
    test_labels, test_latent = load_evaluation(bundle, mask_name, split="test")

    mask_dir = output / mask_name
    mask_dir.mkdir(parents=True, exist_ok=False)
    selection = {}
    final_results = []
    for method, candidates in METHODS.items():
        candidate_results = []
        for index, parameters in enumerate(candidates):
            start = time.perf_counter()
            record = {"candidate_index": index, "method": method, "parameters": parameters}
            try:
                model = new_adapter(method, parameters)
                scope = fit_scope(panel, train_ids, f"{mask_name}:train")
                model.fit(panel, scope)
                result = model.predict_result(validation_panel)
                scored = score(validation_labels, validation_latent, validation_panel, result.counts)
                record.update({
                    "status": "ok",
                    "validation": scored["overall"],
                    "validation_checks": scored["checks"],
                    "fallback_rows": len(result.diagnostics.get("fallback_records", [])),
                    "wall_seconds": time.perf_counter() - start,
                })
            except Exception as exc:  # Keep candidate failures explicit in the tuning record.
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                               "wall_seconds": time.perf_counter() - start})
            candidate_results.append(record)
            print(json.dumps({"event": "candidate_complete", "mask": mask_name,
                              "method": method, "candidate": index,
                              "status": record["status"]}), flush=True)

        successful = [item for item in candidate_results if item["status"] == "ok"
                      and item["validation"]["sample_ratio_mae"] is not None]
        if not successful:
            raise RuntimeError(f"All validation candidates failed for {mask_name}/{method}")
        chosen = min(successful, key=lambda item: (item["validation"]["sample_ratio_mae"],
                                                    item["candidate_index"]))
        selection[method] = {"selected_candidate": chosen, "candidates": candidate_results,
                             "selection_metric": "validation hidden-cell sample_ratio_mae against X/N; tie by candidate index; no test retuning"}

        start = time.perf_counter()
        final = new_adapter(method, chosen["parameters"])
        scope = fit_scope(panel, train_validation_ids, f"{mask_name}:train+validation")
        final.fit(panel, scope)
        fit_seconds = time.perf_counter() - start
        predict_start = time.perf_counter()
        result = final.predict_result(test_panel)
        predict_seconds = time.perf_counter() - predict_start
        scored = score(test_labels, test_latent, test_panel, result.counts)
        final_dir = mask_dir / method
        final_dir.mkdir(exist_ok=False)
        np.savez_compressed(final_dir / "test_predictions.npz",
                            row_ids=np.asarray(test_panel.row_ids, dtype=str),
                            counts=result.counts)
        write_json(final_dir / "diagnostics.json", result.diagnostics)
        final_record = {
            "method": method,
            "parameters": chosen["parameters"],
            "selected_validation_sample_ratio_mae": chosen["validation"]["sample_ratio_mae"],
            "test": scored,
            "fit_rows": len(train_validation_ids),
            "validation_rows": len(validation_panel.row_ids),
            "test_rows": len(test_panel.row_ids),
            "fallback_rows": len(result.diagnostics.get("fallback_records", [])),
            "fit_wall_seconds": fit_seconds,
            "predict_wall_seconds": predict_seconds,
        }
        write_json(final_dir / "result.json", final_record)
        final_results.append(final_record)
        print(json.dumps({"event": "method_complete", "mask": mask_name,
                          "method": method, "test_sample_ratio_mae": scored["overall"]["sample_ratio_mae"]}),
              flush=True)

    return {"mask": mask_name, "selection": selection, "results": final_results,
            "split_rows": {"train": len(train_ids), "validation": len(validation_panel.row_ids),
                           "test": len(test_panel.row_ids)},
            "hidden_test_cells": int(test_labels.target_mask.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path("datasets/synthetic/wave_panel_v1/easy"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/synthetic_easy_baselines_v2_2026-10-23"))
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    dataset_manifest = bundle / "manifest.json"
    manifest_data = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    profile = manifest_data.get("config", {}).get("profile")
    if profile not in ("easy", "medium", "hard"):
        raise ValueError("The selected bundle profile must be easy, medium, or hard")
    if {item.get("name") for item in manifest_data.get("masks", [])} != set(MASKS):
        raise ValueError("Unexpected declared MCAR/MAR mask set")

    run_manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle),
        "bundle_manifest_sha256": sha256(dataset_manifest),
        "profile": manifest_data["config"]["profile"],
        "scale": manifest_data["config"]["scale"],
        "dataset_seed": manifest_data["config"]["seed"],
        "selection_objective": "validation hidden-cell sample_ratio_mae against X/N; tie by candidate index; no test retuning",
        "metric_version": "dual-truth-v2",
        "protocol": "entity-disjoint train/validation/test; validation selects candidates; final fit uses train+validation; test used once for reporting",
        "methods_and_validation_candidates": METHODS,
        "masks": list(MASKS),
        "fallback": "train_mean",
        "degenerate_policies": True,
        "metadata_used": [],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
            for path in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "nextgen/src/longitudinal_impute/baselines/adapters.py",
                Path(__file__).resolve().parents[1] / "nextgen/src/longitudinal_impute/baselines/hron.py",
                Path(__file__).resolve().parents[1] / "nextgen/src/longitudinal_impute/baselines/micoda.py",
                Path(__file__).resolve().parents[1] / "nextgen/src/longitudinal_impute/baselines/tsagris.py",
                Path(__file__).resolve().parents[1] / "nextgen/src/longitudinal_impute/synthetic/datasets.py",
                Path(__file__).resolve().parents[1] / "nextgen/src/longitudinal_impute/synthetic/scoring.py",
            ]
        },
    }
    write_json(output / "run_manifest.json", run_manifest)

    results = [run_mask(bundle, mask_name, output) for mask_name in MASKS]
    run_manifest["status"] = "complete"
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run_manifest.json", run_manifest)
    summary = {"experiment": output.name, "profile": profile, "bundle": str(bundle),
               "results": results}
    write_json(output / "summary.json", summary)

    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["mask", "method", "parameters", "validation_sample_ratio_mae", "test_hidden_cells",
                  "test_latent_p_mae", "test_latent_p_rmse", "test_count_mae", "test_count_rmse", "test_sample_ratio_mae", "test_sample_ratio_rmse",
                  "true_zero_sample_ratio_mae", "sampling_zero_sample_ratio_mae", "positive_count_sample_ratio_mae",
                  "fallback_rows", "finite", "nonnegative", "observed_unchanged", "budget_violations"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mask_result in results:
            chosen = mask_result["selection"]
            for row in mask_result["results"]:
                overall = row["test"]["overall"]
                strata = row["test"]["by_stratum"]
                checks = row["test"]["checks"]
                writer.writerow({
                    "mask": mask_result["mask"], "method": row["method"],
                    "parameters": json.dumps(row["parameters"], sort_keys=True),
                    "validation_sample_ratio_mae": chosen[row["method"]]["selected_candidate"]["validation"]["sample_ratio_mae"],
                    "test_latent_p_mae": overall["latent_p_mae"], "test_latent_p_rmse": overall["latent_p_rmse"],
                    "test_hidden_cells": overall["n"], "test_count_mae": overall["count_mae"],
                    "test_count_rmse": overall["count_rmse"], "test_sample_ratio_mae": overall["sample_ratio_mae"],
                    "test_sample_ratio_rmse": overall["sample_ratio_rmse"],
                    "true_zero_sample_ratio_mae": strata["true_zero"]["sample_ratio_mae"],
                    "sampling_zero_sample_ratio_mae": strata["sampling_zero"]["sample_ratio_mae"],
                    "positive_count_sample_ratio_mae": strata["positive_count"]["sample_ratio_mae"],
                    "fallback_rows": row["fallback_rows"], "finite": checks["finite"],
                    "nonnegative": checks["nonnegative"],
                    "observed_unchanged": checks["observed_unchanged"],
                    "budget_violations": checks["budget_violations"],
                })

    lines = [f"# Synthetic {profile} — baseline results", "",
             f"Dataset: `{bundle}` (scale {manifest_data['config']['scale']}, seed {manifest_data['config']['seed']}).",
             "The benchmark used the two declared MCAR/MAR masks.", "",
             "Protocol: entity-disjoint train/validation/test. Candidate parameters were selected by hidden-cell sample ratio MAE (X/N) on validation; each selected model was refit on train+validation and evaluated on test. Test results did not select parameters. Fallback was `train_mean`, degenerate policies were enabled, and no metadata covariates were passed to the baselines.", "",
             "## Test metrics", "",
             "| Mask | Baseline | Selected parameters | Hidden cells | Count MAE | Count RMSE | Sample ratio MAE | Sample ratio RMSE | Latent p MAE | Latent p RMSE | Fallback rows |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mask_result in results:
        for row in mask_result["results"]:
            metric = row["test"]["overall"]
            params = json.dumps(row["parameters"], sort_keys=True)
            lines.append(f"| {mask_result['mask']} | {row['method']} | `{params}` | {metric['n']} | {metric['count_mae']:.6g} | {metric['count_rmse']:.6g} | {metric['sample_ratio_mae']:.6g} | {metric['sample_ratio_rmse']:.6g} | {metric['latent_p_mae']:.6g} | {metric['latent_p_rmse']:.6g} | {row['fallback_rows']} |")
    lines.extend(["", "## Integrity checks", "",
                  "Per-run finite, nonnegative, observed-value preservation, and count-budget checks are in `summary.json`. Count, sample ratio (X/N), and latent p MAE/RMSE are distinct. Hidden test error by latent stratum (true zero, sampling zero, positive count) is also recorded there.", "",
                  "Candidate validation scores and failures, prediction arrays, and model diagnostics are retained under each mask directory. See `metrics.csv` for a compact spreadsheet-ready summary and `run_manifest.json` for provenance.", ""])
    (output / "report_vi.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output),
                      "summary": str(output / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
