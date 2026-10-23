"""Dataset adapters: explicit schemas; no feature inference by column position."""
from pathlib import Path
from typing import Protocol
from .contracts import DatasetSchema, RawPanel


class DatasetAdapter(Protocol):
    def load(self, path: Path, schema: DatasetSchema) -> RawPanel: ...

    def validate(self, panel: RawPanel) -> None:
        """Check unique entity/time keys, counts, totals and observed mass <= N."""
        ...
