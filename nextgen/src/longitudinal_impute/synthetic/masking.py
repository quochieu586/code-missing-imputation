"""Count adapters for approved Schouten MCAR/MAR and fixed-wave patterns.

MAR sees placeholders at ALL targets and only always-observed N/trait values.
This prevents truth-dependent source preprocessing even with zero target weights.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

from .backend import run_author, validate_seed, write_request, workspace_temp
from .capabilities import require_capability


@dataclass(frozen=True)
class MaskResult:
    hidden: np.ndarray
    row_ids: tuple[str, ...]
    diagnostics: dict

    def condition(self, raw, *, approved_metadata=("entity_trait_01",)):
        from ..baselines.panels import hide_targets
        if tuple(raw.row_ids) != self.row_ids:
            raise ValueError("Mask row identities differ from the supplied panel")
        # Private latent proportions/residuals cannot be whitelisted by this adapter.
        if set(approved_metadata) - {"entity_trait_01"}:
            raise ValueError("Only entity_trait_01 is approved; latent truth is evaluator-only")
        approved = tuple(k for k in approved_metadata if k in raw.metadata)
        return hide_targets(raw, self.hidden, mask_id=self.diagnostics["mask_id"],
                            approved_metadata=approved)


def _validate_panel(raw, mechanism):
    if mechanism not in ("MCAR", "MAR"):
        require_capability("latent_p_mnar" if mechanism == "MNAR" else mechanism)
    x = np.asarray(raw.counts, dtype=float)
    observed = np.asarray(raw.original_observed)
    n = np.asarray(raw.totals, dtype=float)
    if x.ndim != 2 or not all(x.shape) or observed.dtype != bool or observed.shape != x.shape:
        raise ValueError("A nonempty complete N x D count panel with boolean O is required")
    if not observed.all() or not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError("This source adapter requires complete nonnegative integer count truth")
    if n.shape != (len(x),) or not np.isfinite(n).all() or (n <= 0).any() or (n != np.floor(n)).any():
        raise ValueError("N must be a positive integer for every sample")
    if (x.sum(axis=1) > n).any():
        raise ValueError("Target counts exceed N")
    if len(raw.row_ids) != len(x) or len(set(raw.row_ids)) != len(x):
        raise ValueError("Unique row IDs must align with counts")
    if len(raw.entity_ids) != len(x) or len(raw.timestamps) != len(x):
        raise ValueError("Entity/time IDs must align with counts")
    if len(set(zip(map(str, raw.entity_ids), map(str, raw.timestamps)))) != len(x):
        raise ValueError("Duplicate entity/time sample")
    trait = np.zeros(len(x))
    if mechanism == "MAR":
        if "entity_trait_01" not in raw.metadata:
            raise ValueError("MAR requires always-observed entity_trait_01")
        trait = np.asarray(raw.metadata["entity_trait_01"], dtype=float)
        if trait.shape != n.shape or not np.isfinite(trait).all():
            raise ValueError("MAR trait must be finite and present at every sample")
        for entity in set(map(str, raw.entity_ids)):
            ids = np.flatnonzero(np.asarray(raw.entity_ids).astype(str) == entity)
            if not np.all(trait[ids] == trait[ids[0]]):
                raise ValueError("entity_trait_01 must be constant within each entity")
    return x, n, trait


def _standardize_observed(values):
    scale = np.std(values, ddof=1) if len(values) > 1 else 0.
    return (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)


def _execute(target_shape, patterns, frequencies, totals, trait, *, mechanism,
             case_probability, seed, n_weight, trait_weight, rscript):
    validate_seed(seed)
    if not np.isfinite(case_probability) or not 0 < case_probability < 1:
        raise ValueError("case_probability must be strictly between zero and one")
    pat = np.asarray(patterns)
    if pat.ndim != 2 or pat.shape[1] != target_shape[1] or not len(pat):
        raise ValueError("patterns must have K x D target columns; 0=hide, 1=retain")
    if not np.isin(pat, [0, 1]).all() or not (pat == 0).any(axis=1).all():
        raise ValueError("Every declared pattern must hide at least one target; use only 0/1")
    freq = np.asarray(frequencies, dtype=float)
    if freq.shape != (len(pat),) or not np.isfinite(freq).all() or (freq <= 0).any() or not np.isclose(freq.sum(), 1):
        raise ValueError("Positive frequencies must match patterns and sum to one")
    if not np.isfinite([n_weight, trait_weight]).all() or n_weight < 0 or trait_weight < 0:
        raise ValueError("Observed-covariate relative weights must be finite and nonnegative")
    if mechanism == "MAR" and n_weight == trait_weight == 0:
        raise ValueError("MAR needs a nonzero observed-covariate weight; select MCAR explicitly")
    # Zero placeholders are not simulated truth. Source std=FALSE is explicit:
    # only observed covariates are scaled here; source sumscores still scales scores.
    cov = np.column_stack([_standardize_observed(totals), _standardize_observed(trait)])
    data = np.column_stack([np.zeros(target_shape), cov])
    source_patterns = np.column_stack([pat, np.ones((len(pat), 2), dtype=int)])
    weights = np.zeros_like(source_patterns, dtype=float)
    if mechanism == "MAR":
        weights[:, -2] = -n_weight
        weights[:, -1] = trait_weight
    with workspace_temp("schouten-") as directory:
        directory = Path(directory); output = directory / "output"
        for name, value in [("data", data), ("patterns", source_patterns), ("weights", weights),
                            ("frequencies", freq[None, :])]:
            np.savetxt(directory / f"{name}.csv", value, delimiter=",", fmt="%.17g")
        write_request(directory / "request.dcf", Seed=seed, Probability=case_probability, Mechanism=mechanism)
        provenance = run_author("mask", directory, output, rscript=rscript)
        full_hidden = np.loadtxt(output / "hidden.csv", delimiter=",", ndmin=2).astype(bool)
        candidates = np.loadtxt(output / "candidates.csv", delimiter=",", ndmin=1).astype(int)
        scores = np.genfromtxt(output / "scores.csv", delimiter=",", ndmin=1,
                               missing_values="NA", filling_values=np.nan)
    if full_hidden.shape != data.shape or full_hidden[:, -2:].any():
        raise RuntimeError("Author output shape/observed-covariate invariant failed")
    hidden = full_hidden[:, :-2].copy()
    per_pattern = []
    for k in range(len(pat)):
        ids = candidates == k + 1
        # A case is selected exactly when its registered target pattern is applied.
        selected = hidden[ids].any(axis=1)
        if len(selected) and not np.array_equal(hidden[ids], selected[:, None] & (pat[k] == 0)):
            raise RuntimeError("Mask differs from a declared source pattern")
        per_pattern.append({"pattern": k + 1, "candidates": int(ids.sum()),
                            "hidden_cases": int(selected.sum()),
                            "score_sd": float(np.std(scores[ids], ddof=1))
                            if ids.sum() > 1 and np.isfinite(scores[ids]).all() else None})
    return hidden, {"source": provenance, "mechanism": mechanism, "seed": seed,
        "rate_definition": "candidate_case_probability_not_target_cell_rate",
        "target_case_probability": case_probability,
        "nominal_target_cell_rate": float(case_probability * np.dot(freq, (pat == 0).mean(axis=1))),
        "achieved_case_rate": float(hidden.any(axis=1).mean()),
        "patterns": pat.astype(int).tolist(), "frequencies": freq.tolist(),
        "relative_observed_weights": {"N": -n_weight, "entity_trait_01": trait_weight},
        "source_std": False, "covariate_scaling": "observed_panel_sample_sd; independent_of_target_truth",
        "source_score_scaling": True, "pattern_diagnostics": per_pattern,
        "calibration": "original_standard_normal_reference; nominal_rate_not_empirical_guarantee"}


def _result(raw, hidden, diagnostics, scenario):
    hidden = hidden.copy(); hidden.setflags(write=False)
    identity = json.dumps(list(raw.row_ids), ensure_ascii=False).encode("utf-8")
    mask_id = hashlib.sha256(identity + hidden.tobytes()).hexdigest()
    diagnostics = {**diagnostics, "scenario": scenario, "mask_id": mask_id,
                   "achieved_target_cell_rate": float(hidden.mean()),
                   "hidden_target_cells": int(hidden.sum()), "target_cells": int(hidden.size)}
    return MaskResult(hidden, tuple(raw.row_ids), diagnostics)


def mask_counts(raw, *, patterns, frequencies, mechanism, case_probability, seed,
                n_weight=1., trait_weight=1., rscript=None):
    """One source candidate pattern per sample. No latent p parameter is accepted."""
    x, totals, trait = _validate_panel(raw, mechanism)
    hidden, diagnostics = _execute(x.shape, patterns, frequencies, totals, trait,
        mechanism=mechanism, case_probability=case_probability, seed=seed,
        n_weight=n_weight, trait_weight=trait_weight, rscript=rscript)
    return _result(raw, hidden, diagnostics, "declared_patterns")


def mask_fixed_wave(raw, *, wave_time, mechanism, case_probability, seed,
                    n_weight=1., trait_weight=1., rscript=None):
    """One fixed wave on a balanced panel; each case is a whole entity.

    L=1 measurement wave, ALL components. Not a random temporal-block generator.
    """
    x, totals, trait = _validate_panel(raw, mechanism)
    entities = np.asarray(raw.entity_ids).astype(str)
    times = np.asarray(raw.timestamps)
    ordered = []
    for entity in sorted(set(entities)):
        ids = np.flatnonzero(entities == entity)
        ids = ids[np.argsort(times[ids], kind="stable")]
        ordered.append(ids)
    reference = times[ordered[0]]
    if any(not np.array_equal(times[ids], reference) for ids in ordered):
        raise ValueError("fixed_wave requires the same ordered time grid for every entity")
    matched = np.flatnonzero(reference == wave_time)
    if len(matched) != 1:
        raise ValueError("wave_time must identify exactly one measurement on the shared grid")
    wave = int(matched[0]); t = len(reference); d = x.shape[1]
    pat = np.ones((1, t * d), dtype=int); pat[:, wave*d:(wave+1)*d] = 0
    wave_ids = np.array([ids[wave] for ids in ordered])
    wide_hidden, diagnostics = _execute((len(ordered), t*d), pat, [1.], totals[wave_ids], trait[wave_ids],
        mechanism=mechanism, case_probability=case_probability, seed=seed,
        n_weight=n_weight, trait_weight=trait_weight, rscript=rscript)
    hidden = np.zeros_like(x, dtype=bool)
    for i, ids in enumerate(ordered): hidden[ids] = wide_hidden[i].reshape(t, d)
    diagnostics["fixed_wave_time"] = str(wave_time)
    diagnostics["case_unit"] = "entity"
    return _result(raw, hidden, diagnostics, "fixed_wave")
