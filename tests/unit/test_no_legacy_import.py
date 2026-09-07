import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "missing_imputation"

NEW_MODULES = ["occurrence", "gan"]
LEGACY = "zero_state"


def test_new_modules_do_not_import_legacy():
    offenders = []
    for mod in NEW_MODULES:
        mod_dir = SRC / mod
        if not mod_dir.exists():
            continue
        for py in mod_dir.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if LEGACY in node.module:
                        offenders.append(str(py))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if LEGACY in alias.name:
                            offenders.append(str(py))
    assert not offenders, f"Legacy zero_state imported from new modules: {offenders}"


def test_pipeline_v2_does_not_import_legacy():
    offenders = []
    for name in ["run_occurrence.py", "fuse_initializations.py", "run_refinement.py"]:
        py = SRC / "pipeline" / name
        if not py.exists():
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and LEGACY in node.module:
                offenders.append(name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if LEGACY in alias.name:
                        offenders.append(name)
    assert not offenders, f"Legacy zero_state imported from pipeline v2: {offenders}"