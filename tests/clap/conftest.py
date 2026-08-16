"""Make ``src`` and this directory importable without an install step."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"

for path in (_SRC, _HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
