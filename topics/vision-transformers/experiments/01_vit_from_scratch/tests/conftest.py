"""Pytest configuration — adds src/ to import path."""

import sys
from pathlib import Path

# Add the experiment's src/ directory to sys.path
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
