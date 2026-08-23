"""UI package for ChessBench Arena."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is on sys.path for robust module loading
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
