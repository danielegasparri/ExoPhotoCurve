"""Small numeric helper functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .constants import NONE_COL


def parse_float(value: object, default: Optional[float] = None) -> Optional[float]:
    """Parse a float from a GUI string, accepting empty values."""
    if value is None:
        return default

    text = str(value).strip()
    if text == "":
        return default

    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: object, default: int) -> int:
    """Parse an integer from a GUI string."""
    try:
        return int(str(value).strip())
    except Exception:
        return default


def to_numeric_array(df: pd.DataFrame, column: str) -> Optional[np.ndarray]:
    """Convert a selected dataframe column to a NumPy array."""
    if column == NONE_COL or column not in df.columns:
        return None

    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def finite_mask(*arrays: Optional[np.ndarray]) -> np.ndarray:
    """Return a common finite-data mask for all provided arrays."""
    valid_arrays = [arr for arr in arrays if arr is not None]

    if not valid_arrays:
        return np.array([], dtype=bool)

    mask = np.ones_like(valid_arrays[0], dtype=bool)

    for arr in valid_arrays:
        mask &= np.isfinite(arr)

    return mask


def format_number(value: float) -> str:
    """Format a number compactly for axis labels and annotations."""
    if not np.isfinite(value):
        return "nan"

    abs_value = abs(value)
    if abs_value >= 1e5:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if abs_value >= 100:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:.5f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"
