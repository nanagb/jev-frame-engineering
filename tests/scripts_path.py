"""Puts the skill's scripts/ directory on sys.path so the tests can import jevlib and the scripts.
Every test module imports this once; the path lives here and nowhere else."""
import os, sys

SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "skills", "jev-frame-engineering", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
