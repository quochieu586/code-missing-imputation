"""Publish complete directories by same-filesystem rename; clean failed staging."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile


@contextmanager
def staged_directory(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output directory already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        yield staging
        if output.exists():
            raise FileExistsError(f'Output directory already exists: {output}')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
