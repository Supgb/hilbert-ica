#!/usr/bin/env python3
"""Repository-root entry point for running ``hilbert-ica`` without installing first.

Prepends ``./src`` to ``sys.path`` and forwards execution to ``hilbert_ica.main:main``
so ``python main.py --data-dir ... --out ...`` works directly from a fresh checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hilbert_ica.main import main

if __name__ == "__main__":
    main()
