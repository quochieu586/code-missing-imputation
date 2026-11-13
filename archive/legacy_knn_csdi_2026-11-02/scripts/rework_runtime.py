"""Use the working bundled CPython with the existing, compatible research wheels."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / '.venv-e3-revised/Lib/site-packages'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SITE))
sys.dont_write_bytecode = True
