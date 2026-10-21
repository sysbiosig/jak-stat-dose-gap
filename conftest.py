"""Make `src/` importable for the whole test suite.

CONTRACT
    pytest adds the rootdir to sys.path, but the package lives in `src/`, so
    `from jakstat_dosegap import ...` fails unless something inserts that
    directory first. Doing it here means it happens exactly once, before any
    test module is imported, and no individual test file has to remember.

    This is why the suite needs no PYTHONPATH, no editable install, and no
    packaging metadata: `python -m pytest` works in a bare clone.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
