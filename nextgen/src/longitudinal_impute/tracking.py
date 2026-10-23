"""Run manifest requirements. Serialization is intentionally not implemented."""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    source_revision: str
    dataset_sha256: str
    config_sha256: str
    split_sha256: str
    mask_sha256: str
    environment: Mapping[str, str]
    seeds: Mapping[str, int]
    protocol: str
