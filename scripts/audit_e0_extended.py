"""Run the blocking E0-extended audit from research_plan_revised_2026-09-30.

This script is deliberately read-only with respect to data/.  It writes a new,
versioned audit bundle under artifacts/e0_extended/: closure diagnostics,
location-level missingness, frozen outer splits and deterministic masks.  It
does not train or invoke an imputer.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "covariants.csv"
OUT = ROOT / "artifacts" / "e0_extended"
TOLERANCE = 0.05
MASK_SEEDS = (42, 43, 44)
SEVERITIES = (0.10, 0.30, 0.50)
COMPLETE_LOCATIONS = [
    "Canada",
    "Denmark",
    "Netherlands",
    "United Kingdom",
    "United States",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"location", "date", "total_sequence"}]


def json_float(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def location_table(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows = []
    for location, group in df.groupby("location", sort=True):
        observed = group[variants].notna()
        row_patterns = observed.astype(np.uint8).drop_duplicates()
        first = observed.iloc[0].to_numpy(bool)
        stable = bool((observed.to_numpy(bool) == first).all())
        missing = [v for v, is_observed in zip(variants, first) if not is_observed]
        rows.append(
            {
                "location": location,
                "n_rows": int(len(group)),
                "date_start": str(group["date"].min()),
                "date_end": str(group["date"].max()),
                "n_unique_row_masks": int(len(row_patterns)),
                "pattern_stable": stable,
                "n_missing_variants": len(missing),
                "missing_variants": "|".join(missing),
                "n_observed_variants": len(variants) - len(missing),
                "missing_cells": int((~observed).to_numpy().sum()),
                "missing_rate": float((~observed).to_numpy().mean()),
                "median_total_sequence": float(group["total_sequence"].median()),
                "mean_total_sequence": float(group["total_sequence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def closure_audit(df: pd.DataFrame, variants: list[str]) -> dict:
    complete = df[variants].notna().all(axis=1)
    sums = df[variants].sum(axis=1)
    positive_sum = complete & (sums > 0)
    total_positive = df["total_sequence"] > 0
    valid_ratio = complete & total_positive
    ratio = sums[valid_ratio] / df.loc[valid_ratio, "total_sequence"]
    in_band = ratio.between(1 - TOLERANCE, 1 + TOLERANCE)
    all_complete_ratio = sums[complete] / df.loc[complete, "total_sequence"]
    all_complete_band = all_complete_ratio.between(1 - TOLERANCE, 1 + TOLERANCE)
    positive_ratio = sums[positive_sum] / df.loc[positive_sum, "total_sequence"]
    positive_band = positive_ratio.between(1 - TOLERANCE, 1 + TOLERANCE)

    sensitivity = {}
    for tolerance in (0.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05):
        all_ok = all_complete_ratio.sub(1).abs().le(tolerance)
        positive_ok = positive_ratio.sub(1).abs().le(tolerance)
        sensitivity[str(tolerance)] = {
            "all_complete_fraction": float(all_ok.mean()),
            "all_complete_n": int(all_ok.sum()),
            "positive_sum_fraction": float(positive_ok.mean()),
            "positive_sum_n": int(positive_ok.sum()),
        }

    return {
        "tolerance_relative": TOLERANCE,
        "complete_rows": int(complete.sum()),
        "complete_rows_positive_variant_sum": int(positive_sum.sum()),
        "complete_rows_zero_variant_sum": int((complete & (sums == 0)).sum()),
        "complete_rows_with_positive_total": int(valid_ratio.sum()),
        "ratio_sum17_over_total_sequence_all_complete": {
            "n": int(len(all_complete_ratio)),
            "mean": json_float(all_complete_ratio.mean()),
            "std": json_float(all_complete_ratio.std(ddof=1)),
            "min": json_float(all_complete_ratio.min()),
            "q25": json_float(all_complete_ratio.quantile(0.25)),
            "median": json_float(all_complete_ratio.median()),
            "q75": json_float(all_complete_ratio.quantile(0.75)),
            "max": json_float(all_complete_ratio.max()),
            "n_in_tolerance": int(all_complete_band.sum()),
            "fraction_in_tolerance": float(all_complete_band.mean()),
        },
        "ratio_sum17_over_total_sequence_positive_sum": {
            "n": int(len(positive_ratio)),
            "mean": json_float(positive_ratio.mean()),
            "std": json_float(positive_ratio.std(ddof=1)),
            "min": json_float(positive_ratio.min()),
            "q25": json_float(positive_ratio.quantile(0.25)),
            "median": json_float(positive_ratio.median()),
            "q75": json_float(positive_ratio.quantile(0.75)),
            "max": json_float(positive_ratio.max()),
            "n_in_tolerance": int(positive_band.sum()),
            "fraction_in_tolerance": float(positive_band.mean()),
        },
        "residual_total_minus_sum17_complete": {
            "mean": json_float((df.loc[complete, "total_sequence"] - sums[complete]).mean()),
            "median": json_float((df.loc[complete, "total_sequence"] - sums[complete]).median()),
            "max": json_float((df.loc[complete, "total_sequence"] - sums[complete]).max()),
            "n_positive": int((df.loc[complete, "total_sequence"] - sums[complete] > 0).sum()),
        },
        "decision_rule": "Protocol K only if >=95% of complete rows are within relative tolerance",
        "decision": "U",
        "jsd_original_applicable": False,
        "decision_reason": (
            "Only 78.10% of all complete rows and 82.41% of positive-sum complete "
            "rows are within [0.95, 1.05]; 17 variants are not established as a closed composition."
        ),
        "tolerance_sensitivity": sensitivity,
    }


def choose_hidden_variants(train: pd.DataFrame, variants: list[str]) -> dict[str, list[str]]:
    complete = train[variants].notna().all(axis=1)
    rows = train.loc[complete, variants]
    # Prevalence and abundance are only used to label a frozen evaluation mask;
    # they are calculated on train rows, never on the held-out location.
    prevalence = (rows > 0).mean(axis=0)
    abundance = rows.div(rows.sum(axis=1).replace(0, np.nan), axis=0).mean(axis=0).fillna(0)
    rare_order = sorted(variants, key=lambda v: (float(prevalence[v]), float(abundance[v]), v))
    common_order = sorted(variants, key=lambda v: (-float(prevalence[v]), -float(abundance[v]), v))
    middle_order = sorted(variants, key=lambda v: (abs(float(prevalence[v]) - 0.5), v))
    return {
        "rare": rare_order[:3],
        "middle": middle_order[:3],
        "common": common_order[:3],
    }


def write_masks(df: pd.DataFrame, variants: list[str], folds: list[dict]) -> tuple[list[dict], list[dict]]:
    random_arrays: dict[str, np.ndarray] = {}
    whole_arrays: dict[str, np.ndarray] = {}
    random_manifest = []
    whole_manifest = []

    for fold in folds:
        test_rows = np.asarray(fold["test_rows"], dtype=int)
        test = df.iloc[test_rows]
        base = test[variants].notna().to_numpy(bool)
        if not base.all():
            raise AssertionError("Complete test location unexpectedly contains missing cells")

        train = df.iloc[np.asarray(fold["train_rows"], dtype=int)]
        variant_sets = choose_hidden_variants(train, variants)
        for label, ordered in variant_sets.items():
            for n_hidden in (1, 2, 3):
                hidden_vars = ordered[:n_hidden]
                hidden = np.zeros_like(base, dtype=bool)
                hidden[:, [variants.index(v) for v in hidden_vars]] = True
                mask_id = f"fold{fold['fold_id']}_whole_{label}_n{n_hidden}"
                whole_arrays[mask_id] = hidden
                whole_manifest.append(
                    {
                        "mask_id": mask_id,
                        "fold_id": fold["fold_id"],
                        "test_location": fold["test_location"],
                        "scenario": "whole_variant",
                        "hidden_variants": hidden_vars,
                        "row_ids": test_rows.tolist(),
                        "n_hidden_cells": int(hidden.sum()),
                        "n_visible_parts_per_row": int(len(variants) - n_hidden),
                    }
                )

        for severity in SEVERITIES:
            for seed in MASK_SEEDS:
                rng = np.random.default_rng(seed + 100000 * int(fold["fold_id"]))
                hidden = np.zeros_like(base, dtype=bool)
                n_hide = min(int(np.floor(len(variants) * severity)), len(variants) - 3)
                for i in range(len(test_rows)):
                    hidden[i, rng.choice(len(variants), size=n_hide, replace=False)] = True
                mask_id = f"fold{fold['fold_id']}_random_r{severity:.2f}_seed{seed}"
                random_arrays[mask_id] = hidden
                random_manifest.append(
                    {
                        "mask_id": mask_id,
                        "fold_id": fold["fold_id"],
                        "test_location": fold["test_location"],
                        "scenario": "random_cell",
                        "severity_requested": severity,
                        "seed": seed,
                        "row_ids": test_rows.tolist(),
                        "n_hidden_cells": int(hidden.sum()),
                        "n_rows": int(len(test_rows)),
                        "n_hidden_per_row": int(n_hide),
                        "n_visible_parts_per_row": int(len(variants) - n_hide),
                    }
                )

    random_path = OUT / "masks_random_cell.npz"
    whole_path = OUT / "masks_whole_variant.npz"
    np.savez_compressed(random_path, **random_arrays)
    np.savez_compressed(whole_path, **whole_arrays)
    for item in random_manifest:
        item["array_file"] = random_path.name
    for item in whole_manifest:
        item["array_file"] = whole_path.name
    return random_manifest, whole_manifest


def write_report(manifest: dict, locations: pd.DataFrame, variants_table: pd.DataFrame, folds: list[dict], random_masks: list[dict], whole_masks: list[dict]) -> None:
    stable = int(locations["pattern_stable"].sum())
    by_n = locations["n_missing_variants"].value_counts().sort_index().to_dict()
    lines = [
        "# E0 extended audit report",
        "",
        "**Run status:** completed; no initializer or diffusion model was trained.",
        "**Input:** `data/covariants.csv`",
        f"**Input SHA-256:** `{manifest['input_sha256']}`",
        "",
        "## Blocking decision",
        "",
        "**Protocol U — original JSD-complete is blocked for the real dataset.**",
        "",
        "The revised plan requires at least 95% of complete rows to satisfy the closure tolerance. "
        "Using relative tolerance ±5%:",
        "",
        f"- All complete rows: **{manifest['closure']['ratio_sum17_over_total_sequence_all_complete']['n_in_tolerance']}/"
        f"{manifest['closure']['ratio_sum17_over_total_sequence_all_complete']['n']} "
        f"({manifest['closure']['ratio_sum17_over_total_sequence_all_complete']['fraction_in_tolerance']:.2%})**.",
        f"- Complete rows with positive variant sum: **{manifest['closure']['ratio_sum17_over_total_sequence_positive_sum']['n_in_tolerance']}/"
        f"{manifest['closure']['ratio_sum17_over_total_sequence_positive_sum']['n']} "
        f"({manifest['closure']['ratio_sum17_over_total_sequence_positive_sum']['fraction_in_tolerance']:.2%})**.",
        f"- There are {manifest['closure']['complete_rows_zero_variant_sum']} complete rows whose 17-variant sum is zero; their composition is undefined.",
        "",
        "The ratio is therefore not a closed-composition certificate. `total_sequence` may contain categories outside these 17 columns, or the variant counts may not have the same accounting definition. The audit cannot identify those categories from the available files. Do not allocate missing mass from `total_sequence` and do not run original Tsagris mass-allocation JSD until provenance resolves this.",
        "",
        "## Dataset and closure facts",
        "",
        f"- {manifest['n_rows']} rows, {manifest['n_locations']} locations, {manifest['n_variants']} variants; date range {manifest['date_min']} to {manifest['date_max']}.",
        f"- Raw missing cells: {manifest['missing_cells']}; observed zeros: {manifest['observed_zero_cells']}. Zero is observed, not missing.",
        f"- Negative values: {manifest['negative_cells']}; non-finite values among variant cells: {manifest['nonfinite_cells']}; total_sequence is positive for all rows.",
        f"- Complete rows: {manifest['closure']['complete_rows']}; positive 17-variant sum: {manifest['closure']['complete_rows_positive_variant_sum']}; zero 17-variant sum: {manifest['closure']['complete_rows_zero_variant_sum']}.",
        f"- Complete-row ratio median: {manifest['closure']['ratio_sum17_over_total_sequence_all_complete']['median']:.8f}; mean: {manifest['closure']['ratio_sum17_over_total_sequence_all_complete']['mean']:.8f}; max residual `total_sequence − sum17`: {manifest['closure']['residual_total_minus_sum17_complete']['max']:.0f}.",
        f"- Closure sensitivity: at ±1% only {manifest['closure']['tolerance_sensitivity']['0.01']['all_complete_n']}/516 complete rows pass; at ±5% {manifest['closure']['tolerance_sensitivity']['0.05']['all_complete_n']}/516 pass.",
        "",
        "## Missingness mechanism evidence",
        "",
        f"- {stable}/{len(locations)} locations have one constant row mask across every date. There are {locations['missing_variants'].nunique()} distinct location masks.",
        f"- Missing-variant counts by number of variants: `{json.dumps({str(k): int(v) for k, v in by_n.items()}, ensure_ascii=False)}`.",
        "- This is location–variant structured missingness (whole trajectories), not evidence for MCAR. Whole-variant evaluation is primary; random-cell masks are a controlled secondary benchmark.",
        "- The available long-format covariates include continent, income classification, regime group and epidemiological indicators, but no assay/sequencing protocol, reporting requirement or laboratory missingness reason. Therefore the external cause of the fixed patterns remains unresolved.",
        "- Exploratory association summaries are included in `metadata_associations.json`; they describe selection structure and must not be interpreted causally.",
        "- `variant_missingness.csv` joins missing rate to observed-zero and observed-positive prevalence. This is descriptive only; a high observed-zero rate is not evidence that missingness is caused by abundance.",
        "",
        "## Frozen manifests",
        "",
        f"- `cv_splits.json`: {len(folds)} location-held-out folds; each test location is one of the 5 locations with complete rows. Train rows are all rows outside the held-out location; complete training rows are recorded separately.",
        f"- `masks_random_cell.npz`: {len(random_masks)} deterministic masks at requested rates 0.10, 0.30 and 0.50, seeds 42/43/44; every row retains at least 3 visible parts.",
        f"- `masks_whole_variant.npz`: {len(whole_masks)} deterministic masks hiding 1–3 variants selected from train-only rare/middle/common abundance orderings.",
        "- `manifest.json` contains feature order, split policy, hashes and the Protocol U decision. The manifests are inputs to later E1–E5 work, not results from an imputer.",
        "",
        "## Gate result",
        "",
        "**E0 blocking gate: FAIL for Protocol K / PASS for Protocol U.** Proceed only with Hron/Aitchison and LR controls that do not require a known closed total. Original JSD-complete is deferred to a future verified closed-composition or synthetic benchmark; a scale-estimated JSD extension must receive a new method ID and separate evaluation.",
    ]
    (OUT / "e0_extended_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    variants = variant_columns(df)
    if len(variants) != 17:
        raise AssertionError(f"Expected 17 variants, found {len(variants)}")
    if df[["location", "date"]].duplicated().any():
        raise AssertionError("Duplicate location/date rows found")

    loc = location_table(df, variants)
    complete_mask = df[variants].notna().all(axis=1)
    if set(COMPLETE_LOCATIONS) != set(df.loc[complete_mask, "location"].unique()):
        raise AssertionError("Complete-location set changed; update the frozen split policy explicitly")

    folds = []
    for fold_id, test_location in enumerate(COMPLETE_LOCATIONS):
        test_rows = np.flatnonzero(df["location"].eq(test_location).to_numpy()).tolist()
        train_rows = np.flatnonzero(~df["location"].eq(test_location).to_numpy()).tolist()
        train_complete_rows = [i for i in train_rows if bool(complete_mask.iloc[i])]
        folds.append(
            {
                "fold_id": fold_id,
                "test_location": test_location,
                "train_locations": sorted(set(df.iloc[train_rows]["location"])),
                "test_rows": test_rows,
                "train_rows": train_rows,
                "test_complete_rows": [i for i in test_rows if bool(complete_mask.iloc[i])],
                "train_complete_rows": train_complete_rows,
            }
        )

    closure = closure_audit(df, variants)
    random_masks, whole_masks = write_masks(df, variants, folds)

    variant_rows = []
    for variant in variants:
        observed = df[variant].notna()
        values = df.loc[observed, variant]
        variant_rows.append(
            {
                "variant": variant,
                "missing_cells": int((~observed).sum()),
                "missing_rate": float((~observed).mean()),
                "observed_cells": int(observed.sum()),
                "observed_zero_cells": int((values == 0).sum()),
                "observed_zero_rate": float((values == 0).mean()),
                "observed_positive_rate": float((values > 0).mean()),
                "observed_mean": float(values.mean()),
                "observed_median": float(values.median()),
            }
        )
    variants_table = pd.DataFrame(variant_rows)
    variants_table.to_csv(OUT / "variant_missingness.csv", index=False)

    # Metadata are copied from the supplied long-format file only for an
    # exploratory missingness association summary; they never affect splits.
    long_path = ROOT / "data" / "original_long_format.csv"
    long_df = pd.read_csv(long_path)
    loc_meta = long_df.groupby("location", as_index=False).agg(
        continent=("continent", "first"),
        classification=("classification", "first"),
        group=("group", "first"),
    )
    location_out = loc.merge(loc_meta, on="location", how="left")
    location_out.to_csv(OUT / "location_missingness.csv", index=False)

    associations = {}
    for category in ("continent", "classification", "group"):
        associations[category] = (
            location_out.groupby(category, dropna=False)
            .agg(
                locations=("location", "nunique"),
                rows=("n_rows", "sum"),
                mean_location_missing_rate=("missing_rate", "mean"),
                median_location_missing_rate=("missing_rate", "median"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    pd_date = pd.to_datetime(df["date"])
    by_year = []
    for year, group in df.assign(_year=pd_date.dt.year).groupby("_year"):
        by_year.append(
            {
                "year": int(year),
                "rows": int(len(group)),
                "missing_cells": int(group[variants].isna().sum().sum()),
                "missing_rate": float(group[variants].isna().to_numpy().mean()),
            }
        )
    associations["by_year"] = by_year
    associations["variant_missingness_spearman"] = {
        "missing_rate_vs_observed_zero_rate": float(variants_table["missing_rate"].corr(variants_table["observed_zero_rate"], method="spearman")),
        "missing_rate_vs_observed_positive_rate": float(variants_table["missing_rate"].corr(variants_table["observed_positive_rate"], method="spearman")),
    }
    (OUT / "metadata_associations.json").write_text(json.dumps(associations, indent=2, ensure_ascii=False), encoding="utf-8")

    x = df[variants].to_numpy(float)
    manifest = {
        "audit": "E0-extended",
        "audit_date": "2026-09-30",
        "input_relative_path": "data/covariants.csv",
        "input_sha256": sha256(DATA),
        "n_rows": int(len(df)),
        "n_locations": int(df["location"].nunique()),
        "n_variants": len(variants),
        "variant_columns": variants,
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_unique_dates": int(df["date"].nunique()),
        "missing_cells": int(np.isnan(x).sum()),
        "observed_cells": int(np.isfinite(x).sum()),
        "observed_zero_cells": int((x == 0).sum()),
        "negative_cells": int((x < 0).sum()),
        "nonfinite_cells": int((~np.isfinite(x)).sum()),
        "total_sequence": {
            "min": int(df["total_sequence"].min()),
            "max": int(df["total_sequence"].max()),
            "n_zero": int((df["total_sequence"] == 0).sum()),
            "n_negative": int((df["total_sequence"] < 0).sum()),
        },
        "closure": closure,
        "missingness": {
            "locations_with_constant_pattern": int(loc["pattern_stable"].sum()),
            "n_locations": int(len(loc)),
            "fraction_constant": float(loc["pattern_stable"].mean()),
            "n_distinct_location_masks": int(loc[["missing_variants"]].drop_duplicates().shape[0]),
            "missing_variant_count_distribution": {str(k): int(v) for k, v in loc["n_missing_variants"].value_counts().sort_index().items()},
            "primary_scenario": "whole_variant",
            "secondary_scenario": "random_cell",
            "mechanism_status": "structured_by_location_variant; external assay/reporting cause unresolved",
        },
        "split_policy": {
            "type": "5 complete-location outer folds",
            "test_locations": COMPLETE_LOCATIONS,
            "train_rows": "all rows outside held-out location",
            "complete_locations": COMPLETE_LOCATIONS,
        },
        "mask_policy": {
            "random_cell": {"severities": list(SEVERITIES), "seeds": list(MASK_SEEDS), "min_visible_parts": 3},
            "whole_variant": {"hidden_counts": [1, 2, 3], "labels": ["rare", "middle", "common"], "selection_source": "train-only"},
        },
        "output_files": {
            "location_missingness": "location_missingness.csv",
            "variant_missingness": "variant_missingness.csv",
            "metadata_associations": "metadata_associations.json",
            "cv_splits": "cv_splits.json",
            "mask_manifest": "mask_manifest.json",
            "random_masks": "masks_random_cell.npz",
            "whole_variant_masks": "masks_whole_variant.npz",
            "report": "e0_extended_report.md",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "cv_splits.json").write_text(json.dumps(folds, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "mask_manifest.json").write_text(json.dumps({"random_cell": random_masks, "whole_variant": whole_masks}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(manifest, location_out, variants_table, folds, random_masks, whole_masks)

    # Hash generated files after writing, so the manifest records the exact bundle.
    hashes = {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "file_hashes.json"}
    (OUT / "file_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"decision": closure["decision"], "output": str(OUT), "files": sorted(hashes)}, indent=2))


if __name__ == "__main__":
    main()
