"""Integration test for full ZINB zero-state pipeline on covariants.csv."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# This test requires the full data and is marked as integration
# Run with: pytest tests/integration/test_zinb_pipeline.py -v

@pytest.mark.integration
def test_full_zinb_pipeline(tmp_path):
    """Test full ZINB pipeline on covariants.csv (if available)."""
    data_path = Path("data/covariants.csv")
    if not data_path.exists():
        pytest.skip("covariants.csv not found")

    config_path = Path("configs/zero_state/zinb.yaml")
    if not config_path.exists():
        pytest.skip("ZINB config not found")

    observed_mask_path = Path("artifacts/M_observed.npz")
    if not observed_mask_path.exists():
        pytest.skip("M_observed.npz not found - run baselines first")

    from src.missing_imputation.pipeline.run_zinb import run_zinb_pipeline

    output_dir = tmp_path / "zero_state_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_zinb_pipeline(
        data_path=data_path,
        config_path=config_path,
        observed_mask_path=observed_mask_path,
        output_dir=output_dir,
        seed=42,
        gate_mode="calibrated_threshold",
    )

    # Verify outputs exist
    assert Path(result.dataset_01_zinb_path).exists()
    assert Path(result.posterior_path).exists()
    assert Path(result.masks_path).exists()
    assert Path(result.sampled_states_path).exists()

    # Verify invariants
    assert result.invariants_ok, "Invariants violated"

    # Verify dataset_01_zinb has correct structure
    df_zinb = pd.read_csv(result.dataset_01_zinb_path)
    assert len(df_zinb) > 0
    assert "total_sequence" in df_zinb.columns
    assert "other" in df_zinb.columns

    # Verify posterior loads
    from src.missing_imputation.zero_state import ZeroStatePosterior
    posterior = ZeroStatePosterior.load(result.posterior_path)
    assert len(posterior.variant_posteriors) > 0

    # Verify masks load
    from src.missing_imputation.zero_state import ZeroStateMasks
    masks = ZeroStateMasks.load(result.masks_path)
    ok, violations = masks.verify_partition()
    assert ok, f"Mask partition violations: {violations}"

    # Verify calibration metrics
    assert len(result.calibration_metrics) > 0
    for variant, metrics in result.calibration_metrics.items():
        assert "test_metrics" in metrics
        assert "best_threshold" in metrics


@pytest.mark.integration
def test_fusion_pipeline(tmp_path):
    """Test fusion pipeline (T13)."""
    required_files = [
        "data/covariants.csv",
        "artifacts/dataset_00_tsagris.csv",
        "artifacts/zero_state/dataset_01_zinb.csv",
        "artifacts/zero_state/zero_state_posterior.npz",
    ]

    if not all(Path(f).exists() for f in required_files):
        pytest.skip("Required files for fusion not found")

    from src.missing_imputation.pipeline.fuse_initializations import fuse_initializations

    output_dir = tmp_path / "fusion_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = fuse_initializations(
        raw_path="data/covariants.csv",
        tsagris_path="artifacts/dataset_00_tsagris.csv",
        zinb_path="artifacts/zero_state/dataset_01_zinb.csv",
        posterior_path="artifacts/zero_state/zero_state_posterior.npz",
        output_dir=output_dir,
    )

    # Verify outputs
    assert Path(result.fused_path).exists()
    assert Path(result.m_fixed_path).exists()
    assert Path(result.m_gan_path).exists()
    assert Path(result.provenance_path).exists()
    assert result.invariants_ok

    # Verify fused dataset
    df_fused = pd.read_csv(result.fused_path)
    assert len(df_fused) > 0

    # Verify provenance
    provenance = pd.read_parquet(result.provenance_path)
    assert len(provenance) > 0
    assert "source" in provenance.columns
    assert set(provenance["source"].unique()).issubset(
        {"observed", "zero_state", "tsagris", "fused_closure", "unknown"}
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])