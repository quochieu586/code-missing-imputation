"""Author-R parity, masking information boundary, scope guards and Table 1 designs."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

from longitudinal_impute.contracts import DatasetSchema, RawPanel
from longitudinal_impute.synthetic import (NotReadyError, capabilities, mask_counts,
    mask_fixed_wave, reproduce_table1, require_capability)
from longitudinal_impute.synthetic.backend import VENDOR, resolve_rscript, source_provenance
from longitudinal_impute.synthetic.cli import main


def panel(e=24, t=5, d=4):
    rng = np.random.default_rng(6)
    n = e*t
    totals = np.tile(np.arange(t)*40 + 100, e)
    counts = np.array([rng.multinomial(int(total), np.ones(d+1)/(d+1))[:d] for total in totals])
    counts[::4, 0] = 0
    entities = np.repeat([f"entity_{i:03}" for i in range(e)], t)
    times = np.tile(np.arange(t), e)
    return RawPanel(tuple(f"{entity}_{time}" for entity, time in zip(entities, times)), counts,
        np.ones((n, d), dtype=bool), totals, entities, times,
        {"entity_trait_01": np.repeat(np.linspace(-2, 2, e), t),
         "private_p": np.ones(n)*.3, "residual_truth": totals-counts.sum(axis=1)},
        DatasetSchema("entity", "time", "N", tuple(f"c{i}" for i in range(d)), ("entity_trait_01",)))


@pytest.fixture
def rscript():
    try: return resolve_rscript()
    except RuntimeError: pytest.skip("Set RSCRIPT to run original-author R integration checks")


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_mask_matches_direct_original_ampute(mechanism, tmp_path, rscript):
    raw = panel()
    patterns = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
    result = mask_counts(raw, patterns=patterns, frequencies=[.5, .5], mechanism=mechanism,
                         case_probability=.3, seed=53, rscript=rscript)
    n = np.asarray(raw.totals, dtype=float); a = np.asarray(raw.metadata["entity_trait_01"])
    matrix = np.column_stack([np.zeros_like(raw.counts), (n-n.mean())/n.std(ddof=1),
                              (a-a.mean())/a.std(ddof=1)])
    np.savetxt(tmp_path/"data.csv", matrix, delimiter=",")
    script = tmp_path/"oracle.R"
    script.write_text('''a <- commandArgs(TRUE)
for (f in c("check.R","mads.R","ampute.default.R","ampute.mcar.R","ampute.continuous.R","ampute.discrete.R","ampute.R")) source(file.path(a[1],f))
x <- read.csv(a[2],header=FALSE)
p <- rbind(c(0,0,1,1,1,1),c(1,1,0,0,1,1))
w <- matrix(0,2,6)
if (a[4]=="MAR") { w[,5] <- -1; w[,6] <- 1 }
set.seed(53)
z <- ampute(x,prop=.3,patterns=p,freq=c(.5,.5),weights=w,mech=a[4],std=FALSE,type="RIGHT",cont=TRUE,bycases=TRUE)
write.table(is.na(z$amp[,1:4])*1L,a[3],sep=",",row.names=FALSE,col.names=FALSE)
''', encoding="utf-8")
    run = subprocess.run([rscript, "--vanilla", str(script), str(VENDOR), str(tmp_path/"data.csv"),
                          str(tmp_path/"expected.csv"), mechanism], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    expected = np.loadtxt(tmp_path/"expected.csv", delimiter=",").astype(bool)
    np.testing.assert_array_equal(result.hidden, expected)
    assert result.diagnostics["nominal_target_cell_rate"] == pytest.approx(.15)
    assert not result.hidden.flags.writeable


def test_mar_independent_of_counts_and_private_truth_including_constant_columns(rscript):
    raw = panel(); altered = copy.deepcopy(raw)
    altered.counts[:] = 0  # Does not change zero-weight preprocessing/scaling branches.
    altered.metadata["private_p"][:] = 999
    altered.metadata["residual_truth"][:] = -1e8
    options = dict(patterns=[[0, 0, 1, 1]], frequencies=[1.], mechanism="MAR",
                   case_probability=.3, seed=19, rscript=rscript)
    first = mask_counts(raw, **options); second = mask_counts(altered, **options)
    np.testing.assert_array_equal(first.hidden, second.hidden)
    conditioning, labels = first.condition(raw)
    assert set(conditioning.metadata) == {"entity_trait_01"}
    assert np.isnan(conditioning.counts[first.hidden]).all()
    np.testing.assert_array_equal(conditioning.counts[~first.hidden], raw.counts[~first.hidden])
    assert np.isnan(labels.values[~first.hidden]).all()
    with pytest.raises(ValueError, match="Only entity_trait"):
        first.condition(raw, approved_metadata=("private_p",))


def test_fixed_wave_preserves_all_other_times_and_input_order(rscript):
    raw = panel(e=50)
    order = np.random.default_rng(7).permutation(len(raw.counts))
    raw = RawPanel(tuple(raw.row_ids[i] for i in order), raw.counts[order], raw.original_observed[order],
        raw.totals[order], raw.entity_ids[order], raw.timestamps[order],
        {k: v[order] for k,v in raw.metadata.items()}, raw.schema)
    result = mask_fixed_wave(raw, wave_time=2, mechanism="MAR", case_probability=.4,
                             seed=31, rscript=rscript)
    assert not result.hidden[raw.timestamps != 2].any()
    wave = result.hidden[raw.timestamps == 2]
    assert np.all(wave == wave[:, :1])
    assert 0 < wave.sum() < wave.size
    assert result.diagnostics["nominal_target_cell_rate"] == pytest.approx(.4/5)
    assert result.row_ids == raw.row_ids


def test_table1_all_five_conditions_and_source_grid(tmp_path, rscript):
    out = tmp_path/"table1"
    manifest = reproduce_table1(out, replicates=1, rscript=rscript)
    assert manifest["design_runs"] == 45
    assert manifest["post_imputation_inference_tables"] == "NOT-READY"
    assert not manifest["is_full_1000_replicate_design_run"]
    with (out/"table1_design_runs.csv").open() as f: rows = list(csv.DictReader(f))
    assert {r["condition"] for r in rows} == {"mcar", "mar", "weak_mar", "mnar", "weak_mnar"}
    for row in rows:
        key = row["key"]
        hidden = np.loadtxt(out/f"{key}_hidden.csv", delimiter=",").astype(bool)
        np.testing.assert_array_equal(hidden[:, 0], hidden[:, 1])
        assert not hidden[:, 2].any()
        assert float(row["achieved_all_column_cell_rate"]) == pytest.approx(hidden.mean())
        weights = np.loadtxt(out/f"{key}_weights.csv", delimiter=",", ndmin=2)
        expected = {"mcar": [[0,0,0]], "mar": [[0,0,1]], "weak_mar": [[0,0,0],[0,0,1]],
                    "mnar": [[1,0,0]], "weak_mnar": [[1,0,5]]}[row["condition"]]
        np.testing.assert_array_equal(weights, expected)
    with pytest.raises(FileExistsError): reproduce_table1(out, rscript=rscript)


@pytest.mark.parametrize("feature", ["latent_p_mnar", "source_mnar", "random_temporal_blocks",
    "whole_component_within_entity", "beta_strength_grid", "full_cell_rate_grid", "exact_realized_rate"])
def test_unsupported_capabilities_fail_closed(feature):
    with pytest.raises(NotReadyError, match="NOT-READY"): require_capability(feature)


def test_public_mnar_never_runs_r():
    with pytest.raises(NotReadyError, match="latent_p_mnar"):
        mask_counts(panel(), patterns=[[0, 1, 1, 1]], frequencies=[1.], mechanism="MNAR",
                    case_probability=.3, seed=42, rscript="DO_NOT_EXECUTE")


def test_invalid_and_ambiguous_requests_rejected_before_r():
    raw = panel(); raw.original_observed[0, 0] = False
    with pytest.raises(ValueError, match="complete"):
        mask_counts(raw, patterns=[[0,1,1,1]], frequencies=[1.], mechanism="MCAR", case_probability=.3, seed=1)
    assert capabilities()["random_temporal_blocks"]["status"] == "NOT-READY"
    source_provenance()  # All original source hashes must match their pinned manifest.


def test_cli_not_ready_and_private_csv_boundary(tmp_path, rscript, capsys):
    spec = tmp_path/"spec.json"
    spec.write_text(json.dumps({"scenario":"declared_patterns", "mechanism":"MNAR",
        "case_probability":.3, "seed":1,"patterns":[[0,1]],"frequencies":[1]}))
    schema = tmp_path/"schema.toml"
    schema.write_text('[dataset]\nentity="entity"\ntime="date"\ntotal="N"\ntargets=["A","B"]\nmetadata_candidates=["entity_trait_01"]\n')
    data = tmp_path/"input.csv"
    data.write_text('entity,date,N,A,B,entity_trait_01,private_p,residual_truth\n'+''.join(
        f'e{i},2020-01-01,{100+i},00010,0,{i},SECRET,SECRET\n' for i in range(40)))
    common = ["--rscript", rscript, "mask", "--data", str(data), "--schema-config", str(schema),
              "--spec", str(spec), "--output", str(tmp_path/"result")]
    assert main(common) == 2
    assert not (tmp_path/"result").exists()
    config = json.loads(spec.read_text()); config["mechanism"] = "MAR"; spec.write_text(json.dumps(config))
    assert main(common) == 0
    text = (tmp_path/"result"/"conditioning.csv").read_text()
    assert "SECRET" not in text and "private_p" not in text and "residual_truth" not in text
    assert "00010" in text  # Observed counts retain original formatting.
    assert (tmp_path/"result"/"manifest.json").exists()
