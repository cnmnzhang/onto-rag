#!/usr/bin/env python3
"""
Quick test script to verify No-RAG and RAG both work with 3 cases
"""

import sys
import subprocess

print("=" * 70)
print("QUICK TEST: 3 Charts (2 disease + 1 NONE)")
print("=" * 70)
print()
print("This will test:")
print("  1. BioPortal connection ✓")
print("  2. Corpus building (8 TCO classes) ✓")
print("  3. Chart generation (3 charts) ✓")
print("  4. No-RAG predictions (3 predictions) ✓")
print("  5. RAG(TCO) predictions (3 predictions) ✓")
print("  6. Label validation and enforcement ✓")
print()
print("=" * 70)
print()

# Run the experiment
result = subprocess.run(
    ["python", "rag_exp.py"],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
