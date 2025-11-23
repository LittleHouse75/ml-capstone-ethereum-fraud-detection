"""
Global random seed utilities for the project.

This module sets a single canonical seed and applies it consistently to:
  - Python's built-in `random` module
  - NumPy's random generator
  - PYTHONHASHSEED (for more reproducible hashing)

Importing this module once near the start of a run helps make experiments
more repeatable.
"""

import os
import random

import numpy as np

# Canonical project-wide seed
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """
    Set global random seeds for Python, NumPy, and hashing.

    Parameters
    ----------
    seed : int, optional
        Seed value to use everywhere. Defaults to the module-level SEED.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# Apply the default seed on import
set_global_seed()