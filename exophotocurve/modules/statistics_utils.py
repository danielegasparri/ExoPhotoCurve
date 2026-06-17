"""General photometric statistics for light curves and residuals."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import numpy as np

from .numeric_utils import finite_mask


@dataclass
class SeriesStatistics:
    """Container for basic time-series statistics."""

    name: str
    y_type: str
    n_points: int
    x_start: float
    x_end: float
    time_span: float
    cadence_median: float
    cadence_mean: float
    y_mean: float
    y_median: float
    y_std: float
    y_rms_zero: float
    y_rms_median: float
    y_min: float
    y_max: float
    y_amplitude: float
    y_sem: float
    err_median: float
    err_mean: float
    rms_over_median_err: float

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable dictionary."""
        return asdict(self)


def _safe_float(value: float) -> float:
    """Return a normal float, preserving NaN for invalid values."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def compute_series_statistics(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    name: str = "Light curve",
    y_type: str = "Relative flux",
) -> SeriesStatistics:
    """Compute general statistics for a photometric time series.

    The RMS about zero is useful for residuals. The RMS about the median is a
    robust scatter estimator for generic light curves or variable stars.
    """
    mask = finite_mask(x, y)
    if yerr is not None:
        # Do not require valid yerr for a point to enter the light-curve stats;
        # error statistics are computed from the valid errors separately.
        err_values = np.asarray(yerr, dtype=float)
    else:
        err_values = None

    if not np.any(mask):
        return SeriesStatistics(
            name=name,
            y_type=y_type,
            n_points=0,
            x_start=np.nan,
            x_end=np.nan,
            time_span=np.nan,
            cadence_median=np.nan,
            cadence_mean=np.nan,
            y_mean=np.nan,
            y_median=np.nan,
            y_std=np.nan,
            y_rms_zero=np.nan,
            y_rms_median=np.nan,
            y_min=np.nan,
            y_max=np.nan,
            y_amplitude=np.nan,
            y_sem=np.nan,
            err_median=np.nan,
            err_mean=np.nan,
            rms_over_median_err=np.nan,
        )

    x_valid = np.asarray(x[mask], dtype=float)
    y_valid = np.asarray(y[mask], dtype=float)
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]

    n_points = int(len(y_valid))
    x_start = _safe_float(np.nanmin(x_valid))
    x_end = _safe_float(np.nanmax(x_valid))
    time_span = _safe_float(x_end - x_start)

    if n_points > 1:
        dx = np.diff(x_valid)
        dx = dx[np.isfinite(dx) & (dx >= 0)]
        cadence_median = _safe_float(np.nanmedian(dx)) if dx.size else np.nan
        cadence_mean = _safe_float(np.nanmean(dx)) if dx.size else np.nan
        y_std = _safe_float(np.nanstd(y_valid, ddof=1))
        y_sem = _safe_float(y_std / np.sqrt(n_points))
    else:
        cadence_median = np.nan
        cadence_mean = np.nan
        y_std = np.nan
        y_sem = np.nan

    y_mean = _safe_float(np.nanmean(y_valid))
    y_median = _safe_float(np.nanmedian(y_valid))
    y_rms_zero = _safe_float(np.sqrt(np.nanmean(y_valid**2)))
    y_rms_median = _safe_float(np.sqrt(np.nanmean((y_valid - y_median) ** 2)))
    y_min = _safe_float(np.nanmin(y_valid))
    y_max = _safe_float(np.nanmax(y_valid))
    y_amplitude = _safe_float(y_max - y_min)

    err_median = np.nan
    err_mean = np.nan
    rms_over_median_err = np.nan
    if err_values is not None:
        err_mask = np.isfinite(err_values) & (err_values > 0)
        if np.any(err_mask):
            err_median = _safe_float(np.nanmedian(err_values[err_mask]))
            err_mean = _safe_float(np.nanmean(err_values[err_mask]))
            if np.isfinite(err_median) and err_median > 0:
                rms_over_median_err = _safe_float(y_rms_median / err_median)

    return SeriesStatistics(
        name=name,
        y_type=y_type,
        n_points=n_points,
        x_start=x_start,
        x_end=x_end,
        time_span=time_span,
        cadence_median=cadence_median,
        cadence_mean=cadence_mean,
        y_mean=y_mean,
        y_median=y_median,
        y_std=y_std,
        y_rms_zero=y_rms_zero,
        y_rms_median=y_rms_median,
        y_min=y_min,
        y_max=y_max,
        y_amplitude=y_amplitude,
        y_sem=y_sem,
        err_median=err_median,
        err_mean=err_mean,
        rms_over_median_err=rms_over_median_err,
    )


def compute_residual_statistics(residuals: np.ndarray) -> tuple[float, int]:
    """Return residual RMS and number of valid residual points."""
    mask = finite_mask(residuals)

    if not np.any(mask):
        return np.nan, 0

    rms = float(np.sqrt(np.nanmean(residuals[mask] ** 2)))
    n_points = int(np.count_nonzero(mask))
    return rms, n_points


def _fmt(value: float, precision: int = 6) -> str:
    """Format a float for a text report."""
    if value is None or not np.isfinite(value):
        return "NaN"
    if abs(value) >= 1e5:
        return f"{value:.6f}"
    if abs(value) >= 100:
        return f"{value:.4f}"
    if abs(value) >= 1:
        return f"{value:.6f}"
    return f"{value:.{precision}g}"


def _unit_suffix(y_type: str) -> str:
    if y_type == "Magnitude":
        return " mag"
    if y_type == "Relative flux":
        return " rel. flux"
    return ""


def format_single_statistics(stats: SeriesStatistics) -> str:
    """Format one statistics block as human-readable text."""
    unit = _unit_suffix(stats.y_type)

    lines = [
        f"[{stats.name}]",
        f"Data type: {stats.y_type}",
        f"N points: {stats.n_points:d}",
        f"X start: {_fmt(stats.x_start)}",
        f"X end: {_fmt(stats.x_end)}",
        f"Time span: {_fmt(stats.time_span)}",
        f"Median cadence: {_fmt(stats.cadence_median)}",
        f"Mean cadence: {_fmt(stats.cadence_mean)}",
        "",
        f"Mean: {_fmt(stats.y_mean)}{unit}",
        f"Median: {_fmt(stats.y_median)}{unit}",
        f"Std dev: {_fmt(stats.y_std)}{unit}",
        f"RMS about zero: {_fmt(stats.y_rms_zero)}{unit}",
        f"RMS about median: {_fmt(stats.y_rms_median)}{unit}",
        f"Minimum: {_fmt(stats.y_min)}{unit}",
        f"Maximum: {_fmt(stats.y_max)}{unit}",
        f"Peak-to-peak amplitude: {_fmt(stats.y_amplitude)}{unit}",
        f"Standard error of the mean: {_fmt(stats.y_sem)}{unit}",
    ]

    if stats.y_type == "Relative flux":
        lines.extend(
            [
                f"RMS about median: {_fmt(stats.y_rms_median * 1000.0, 4)} ppt",
                f"Peak-to-peak amplitude: {_fmt(stats.y_amplitude * 1000.0, 4)} ppt",
            ]
        )

    lines.extend(
        [
            "",
            f"Median error: {_fmt(stats.err_median)}{unit}",
            f"Mean error: {_fmt(stats.err_mean)}{unit}",
            f"RMS/median error: {_fmt(stats.rms_over_median_err)}",
        ]
    )

    return "\n".join(lines)


def format_statistics_report(blocks: Iterable[SeriesStatistics], title: str = "PhotoCurve Lab statistics") -> str:
    """Format several statistics blocks into one report."""
    sep = "=" * 64
    out: List[str] = [title, sep, ""]

    for stats in blocks:
        out.append(format_single_statistics(stats))
        out.append("")
        out.append("-" * 64)
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def statistics_to_dataframe_rows(blocks: Iterable[SeriesStatistics]) -> List[Dict[str, object]]:
    """Convert statistics blocks to row dictionaries for CSV export."""
    return [stats.to_dict() for stats in blocks]
