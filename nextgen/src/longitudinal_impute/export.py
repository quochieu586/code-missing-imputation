"""Final counts export and run records; no file operations in this scaffold."""
from pathlib import Path
from typing import Protocol
from .contracts import ImputationResult, RawPanel
from .tracking import RunManifest


class Exporter(Protocol):
    def write(self, result: ImputationResult, original: RawPanel,
              manifest: RunManifest, destination: Path) -> None:
        """Verify key alignment, observed identity and budget before writing.

        Refuse existing destination by default. Original raw access belongs at
        this trusted final-native-export boundary, never the prediction path.
        """
        ...
