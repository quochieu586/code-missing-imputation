"""Future orchestration; deliberately fails rather than simulating completion."""
from typing import Any, Mapping


def run(config: Mapping[str, Any]) -> None:
    raise NotImplementedError(
        'Scaffold only: implement and verify gates G1-G5 before running imputation.'
    )
