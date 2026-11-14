# Legacy code snapshots

`legacy_knn_csdi_2026-11-02/` preserves the previously Git-tracked root runtime,
scripts, tests, README and requirements before promoting `nextgen` to the root.
`nextgen_history_2026-11-02/` preserves the already tracked baseline review history.

The maintained package is now `src/longitudinal_impute/`; install and run it from
the repository root. Legacy paths and assumptions are preserved as history, not
as a second supported runtime. Current pytest collection excludes `archive/`.

This code PR does not publish the complete local research archive: no new raw
datasets, virtual environments, checkpoints, local agent logs, research PDFs or
presentation assets are added. Existing non-runtime historical material remains
at its prior tracked paths. The older tracked virtualenv is removed from Git;
recreate environments from `uv.lock`. The source workspace's local environment
is not deleted or modified by this publication.
