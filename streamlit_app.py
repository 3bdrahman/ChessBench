"""Streamlit app entry point for Chess LLM Benchmark."""

import os
import sys

# Ensure repository root is on sys.path for Streamlit Cloud deployment
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ruff: noqa: E402
from chessbench.ui.streamlit_app import main

if __name__ == "__main__":
    main()
