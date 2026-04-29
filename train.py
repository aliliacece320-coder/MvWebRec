"""Entry point: run from repo root with ``python train.py``."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from hetubv_gcl.train import main

if __name__ == "__main__":
    main()
