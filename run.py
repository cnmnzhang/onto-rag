#!/usr/bin/env python3
"""
Simple runner script for TCO RAG Comparison.
Run from project root: python run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run
from rag_exp import main

if __name__ == "__main__":
    main()
