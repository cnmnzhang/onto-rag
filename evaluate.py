#!/usr/bin/env python3
"""Convenience entrypoint for official evaluation.

Run from repo root:
- `python3 evaluate.py`

Writes results to `results/`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from eval_official import main


if __name__ == "__main__":
    main()
