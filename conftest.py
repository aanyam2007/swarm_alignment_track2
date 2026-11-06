"""Ensure the repo root is importable so `python -m pytest` finds packages."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
