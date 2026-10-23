"""Reproduce Table 1 experimental DESIGNS, not the paper's inference results."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .backend import run_author, validate_seed, write_request, workspace_temp


def reproduce_table1(output, *, seed=2018, replicates=3, rhos=None, rscript=None):
    validate_seed(seed, maximum=2**31-1-1000000)
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 1 <= replicates <= 1000:
        raise ValueError("replicates must be an integer from 1 to 1000")
    rhos = np.asarray(list(rhos) if rhos is not None else np.arange(1, 10) / 10, dtype=float)
    if rhos.ndim != 1 or not len(rhos) or not np.isfinite(rhos).all() or not np.isin(np.round(rhos, 10), np.arange(1, 10)/10).all():
        raise ValueError("Only the source rho grid .1, .2, ..., .9 is supported")
    if len(np.unique(rhos)) != len(rhos): raise ValueError("rho values must be unique")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with workspace_temp("schouten-table1-") as directory:
        directory = Path(directory)
        write_request(directory / "request.dcf", Seed=seed, Replicates=replicates)
        np.savetxt(directory / "rhos.csv", rhos[None, :], delimiter=",")
        provenance = run_author("table1", directory, output, rscript=rscript)
    with (output / "table1_design_runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(rhos) * replicates * 5
    if len(rows) != expected or any(int(row["x1_hidden"]) != 0 for row in rows):
        raise RuntimeError("Table 1 design validation failed")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.iterdir()) if p.is_file()}
    manifest = {"status": "COMPLETED_TABLE1_DESIGN_ONLY", "source": provenance,
        "n_per_dataset": 1000, "mean": [5, 5, 10], "rho_grid": rhos.tolist(),
        "replicates": replicates, "master_seed": seed, "design_runs": expected,
        "source_monte_carlo_replicates": 1000,
        "is_full_1000_replicate_design_run": replicates == 1000,
        "post_imputation_inference_tables": "NOT-READY",
        "source_locator": "Schouten2018 PDF 9-10 / printed 8-9, Section 4.1, Table 1",
        "design_files_sha256": hashes,
        "scope": "Five missingness designs and source masks; no reproduced PMM/bias/coverage claims."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
