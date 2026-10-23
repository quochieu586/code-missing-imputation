"""R transport to pinned, unmodified author sources; never downloads at runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "_vendor"
SOURCE_COMMIT = "61083e667fa41cc4bc655492591b100bf8a7e443"


def workspace_temp(prefix):
    # Windows sandbox system TEMP may permit mkdir but deny later file writes.
    workspace = Path.cwd().resolve()
    base = workspace / "tmp" / "schouten_work"
    base.mkdir(parents=True, exist_ok=True)
    if not base.resolve().is_relative_to(workspace):
        raise ValueError("Temporary workspace must remain inside the working directory")
    return tempfile.TemporaryDirectory(prefix=prefix, dir=base)


def source_provenance():
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != SOURCE_COMMIT:
        raise RuntimeError("Unexpected author-source commit")
    for entry in manifest["files"]:
        path = VENDOR / entry["file"]
        if path.parent != VENDOR or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Author-source hash mismatch: {entry['file']}")
    return manifest


def resolve_rscript(explicit=None):
    candidate = explicit or os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError("Rscript is required; supply rscript=... or set RSCRIPT. No Python fallback.")
    return str(Path(candidate).resolve())


def run_author(mode, input_dir, output_dir, *, rscript=None):
    if mode not in ("mask", "table1"):
        raise ValueError("Unknown author mode")
    source = source_provenance()
    process = subprocess.run([resolve_rscript(rscript), "--vanilla", str(ROOT / "author_bridge.R"),
                              mode, str(VENDOR), str(input_dir), str(output_dir)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=600)
    if process.returncode:
        raise RuntimeError(f"Original ampute failed (no fallback): {process.stderr}\n{process.stdout}")
    output_dir = Path(output_dir)
    return {"paper_doi": "10.1080/00949655.2018.1491577", "author_source": source,
            "implementation": "unmodified_R_ampute", "bridge_sha256": hashlib.sha256(
                (ROOT / "author_bridge.R").read_bytes()).hexdigest(),
            "r_session": (output_dir / "r_session.txt").read_text(encoding="utf-8"),
            "source_warnings": (output_dir / "source_warnings.txt").read_text(encoding="utf-8").splitlines(),
            "runtime_stderr": process.stderr}


def write_request(path, **values):
    Path(path).write_text("".join(f"{key}: {value}\n" for key, value in values.items()), encoding="utf-8")


def validate_seed(seed, *, maximum=2**31-1):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= maximum:
        raise ValueError(f"Seed must be an integer between 0 and {maximum}")
