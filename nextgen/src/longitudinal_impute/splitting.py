"""Group splits are explicit artifacts, not hidden runner-side selections."""
from dataclasses import dataclass
from typing import Protocol
from .contracts import RawPanel


@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    train_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    test_row_ids: tuple[str, ...]
    protocol: str
    seed: int


class Splitter(Protocol):
    def split(self, panel: RawPanel, *, seed: int) -> SplitManifest:
        """Inductive default: disjoint locations; persist IDs before tuning."""
        ...
