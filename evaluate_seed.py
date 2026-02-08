#!/usr/bin/env python3
"""Convenience entrypoint for seeded evaluation.

Run from repo root:
- `python3 evaluate_seed.py`

This writes results to `results/`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from seed_eval import main


if __name__ == "__main__":
    main()
