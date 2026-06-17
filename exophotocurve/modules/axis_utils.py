"""X-axis transformation helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .numeric_utils import format_number


def resolve_x_offset(x: np.ndarray, x_offset_text: object) -> float:
    """Resolve the X-axis offset.

    If the GUI field is empty or set to 'auto', use the integer part of the
    first finite X value. This is convenient for JD/BJD/HJD axes.
    """
    text = str(x_offset_text).strip().lower()
    finite_x = x[np.isfinite(x)]

    if finite_x.size == 0:
        return 0.0

    if text in ("", "auto"):
        return float(np.floor(np.nanmin(finite_x)))

    try:
        return float(text)
    except ValueError:
        return float(np.floor(np.nanmin(finite_x)))


def transform_x_axis(
    x: np.ndarray,
    values: Dict[str, object],
    x_column_name: str,
    xlabel_text: str,
) -> Tuple[np.ndarray, str, float]:
    """Apply the selected X-axis transformation and return the axis label."""
    x_mode = str(values.get("-XMODE-", "Raw X"))
    x_offset = resolve_x_offset(x, values.get("-XOFFSET-", "auto"))

    if x_mode == "X - offset":
        x_plot = x - x_offset
        automatic_label = f"{x_column_name} - {format_number(x_offset)}"
    elif x_mode == "Hours from offset":
        x_plot = (x - x_offset) * 24.0
        automatic_label = f"Time from {x_column_name} = {format_number(x_offset)} [hours]"
    else:
        x_plot = x.copy()
        automatic_label = x_column_name

    xlabel = xlabel_text if xlabel_text else automatic_label
    return x_plot, xlabel, x_offset
