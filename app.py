"""Streamlit entry point.

Run from this folder (the repo root):
    streamlit run app.py
"""

import sys
from pathlib import Path

# Make `humaneval_t` importable even without `pip install -e extended/`.
_EXTENDED = Path(__file__).resolve().parent / "extended"
if str(_EXTENDED) not in sys.path:
    sys.path.insert(0, str(_EXTENDED))

from humaneval_t.streamlit_app import main

main()
