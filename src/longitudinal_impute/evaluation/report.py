"""Shared helpers of the evaluation module: paths, hashes, number formats, CSV and report blocks.

ROOT is the repository root (data/, configs/); paths in configs/evaluation/eval_v1.toml are relative to it.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def find_run(roots, name):
    """Run directory `name` in the first of `roots` (one directory or several) that has it."""
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots or ())
    for root in roots:
        if (Path(root) / name).is_dir():
            return Path(root) / name
    raise FileNotFoundError(f'run directory {name} not found in {[str(r) for r in roots]}')


def shown(path):
    """A path as written to tables and manifests: relative to ROOT when inside it."""
    path = Path(path)
    return path.relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else str(path)



def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def na(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def mean_sd(values):
    values = np.array([v for v in values if v is not None], float)
    if not len(values):
        return None, None
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else None)


def update_report(path, blocks):
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    for name, lines in blocks.items():
        start, end = f'<!-- auto:{name}:start -->', f'<!-- auto:{name}:end -->'
        if text.count(start) != 1 or text.count(end) != 1:
            continue
        head, rest = text.split(start)
        _, tail = rest.split(end)
        text = head + start + '\n' + '\n'.join(lines).rstrip() + '\n' + end + tail
    path.write_text(text, encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval='')
        writer.writeheader()
        writer.writerows([{k: 'NA' if v is None else v for k, v in r.items()} for r in rows])



def signed(value, digits=3):
    return 'NA' if value is None else f'{value:+.{digits}f}'



def ms(values, digits=4, fmt=None):
    """mean ± sd over the cases with a value; (n = k) when some case has none."""
    mean, sd = mean_sd(values)
    fmt = fmt or (lambda v, d: na(v, d))
    k = sum(v is not None for v in values)
    return 'NA' if mean is None else f'{fmt(mean, digits)} ± {na(sd, digits)}' + (
        f' (n = {k})' if k < len(values) else '')



def digest(value):
    """sha256 of bytes, of a numpy array (dtype, shape, data) or of a JSON-able object."""
    if isinstance(value, np.ndarray):
        value = f'{value.dtype}{value.shape}'.encode() + np.ascontiguousarray(value).tobytes()
    elif not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()
