"""Utilities for making repo-local imports deterministic in script mode."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Resolve the project root from the current file location."""
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_sys_path() -> Path:
    """Ensure project root is importable when scripts are run directly."""
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

