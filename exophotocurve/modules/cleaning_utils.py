"""Interactive cleaning and sigma-clipping utilities for PhotoCurve Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .numeric_utils import finite_mask, parse_float, parse_int


@dataclass
class CleaningResult:
    """Result of a cleaning-mask computation."""

    keep_mask: np.ndarray
    rejected_mask: np.ndarray
    enabled: bool
    n_total: int
    n_valid_for_clipping: int
    n_rejected: int
    target: str
    sigma: float
    iterations: int
    centre: str
    scale: str


def _robust_sigma(values: np.ndarray, centre_value: float, scale_method: str) -> float:
    """Return either a standard-deviation or MAD-based sigma estimate."""
    if values.size == 0:
        return np.nan

    if scale_method == "Std dev":
        sigma = float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan
    else:
        mad = float(np.nanmedian(np.abs(values - centre_value)))
        sigma = 1.4826 * mad

        # Fallback for very small or very flat samples.
        if (not np.isfinite(sigma)) or sigma <= 0:
            sigma = float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan

    return sigma


def sigma_clip_keep_mask(
    values: np.ndarray,
    sigma_threshold: float = 4.0,
    max_iterations: int = 3,
    centre_method: str = "Median",
    scale_method: str = "MAD",
) -> tuple[np.ndarray, int]:
    """Return a Boolean mask selecting points that survive sigma clipping.

    The returned mask has the same length as ``values``. Non-finite values are
    rejected. The clipping is iterative and is performed around either the mean
    or the median. The scale can be the classical standard deviation or the
    MAD-based robust sigma, 1.4826 * MAD.
    """
    arr = np.asarray(values, dtype=float)
    keep = np.isfinite(arr)

    if arr.size == 0 or not np.any(keep):
        return keep, 0

    sigma_threshold = max(0.1, float(sigma_threshold))
    max_iterations = max(1, int(max_iterations))
    n_iter_done = 0

    for _ in range(max_iterations):
        current = arr[keep]
        if current.size < 3:
            break

        if centre_method == "Mean":
            centre = float(np.nanmean(current))
        else:
            centre = float(np.nanmedian(current))

        sigma = _robust_sigma(current, centre, scale_method)
        if (not np.isfinite(sigma)) or sigma <= 0:
            break

        new_keep = keep & (np.abs(arr - centre) <= sigma_threshold * sigma)
        n_iter_done += 1

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep

    return keep, n_iter_done


def _choose_clipping_series(
    y: Optional[np.ndarray],
    model: Optional[np.ndarray],
    residuals: Optional[np.ndarray],
    target: str,
) -> Dict[str, np.ndarray]:
    """Return one or more arrays to use for sigma clipping."""
    series: Dict[str, np.ndarray] = {}

    if target == "Residuals":
        if residuals is not None:
            series["Residuals"] = residuals
    elif target == "Light curve - model":
        if y is not None and model is not None:
            series["Light curve - model"] = y - model
        elif y is not None:
            series["Light curve"] = y
    elif target == "Light curve":
        if y is not None:
            series["Light curve"] = y
    elif target == "Both":
        if residuals is not None:
            series["Residuals"] = residuals
        if y is not None and model is not None:
            series["Light curve - model"] = y - model
        elif y is not None:
            series["Light curve"] = y

    return series


def _parse_index_set(value: object, n_total: int) -> set[int]:
    """Parse a comma-separated list of integer point indices."""
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    indices: set[int] = set()
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(token)
        except Exception:
            continue
        if 0 <= index < n_total:
            indices.add(index)
    return indices


def compute_cleaning_mask(
    x: np.ndarray,
    y: Optional[np.ndarray],
    model: Optional[np.ndarray],
    residuals: Optional[np.ndarray],
    values: Dict[str, object],
) -> CleaningResult:
    """Compute the global keep/reject mask requested by the GUI.

    The mask always has the same length as ``x``. If cleaning is disabled, all
    finite X positions are kept. If cleaning is enabled, the selected clipping
    series are sigma-clipped and combined with logical AND, so a point rejected
    by any selected diagnostic is excluded from the cleaned data set.
    """
    n_total = int(len(x))
    base_keep = np.isfinite(np.asarray(x, dtype=float))

    sigma_enabled = bool(values.get("-CLEAN_ACTIVE-", False))
    manual_enabled = bool(values.get("-MANUAL_CLEAN_ACTIVE-", True))
    manual_reject_indices = _parse_index_set(values.get("-MANUAL_REJECT_INDICES-", ""), n_total)
    manual_keep_indices = _parse_index_set(values.get("-MANUAL_KEEP_INDICES-", ""), n_total)
    has_manual_mask = manual_enabled and (bool(manual_reject_indices) or bool(manual_keep_indices))
    enabled = sigma_enabled or has_manual_mask
    target = str(values.get("-CLEAN_TARGET-", "Residuals"))
    sigma = parse_float(values.get("-CLEAN_SIGMA-", 4.0), 4.0) or 4.0
    max_iter = parse_int(values.get("-CLEAN_MAXITER-", 3), 3)
    centre = str(values.get("-CLEAN_CENTRE-", "Median"))
    scale = str(values.get("-CLEAN_SCALE-", "MAD"))

    if not enabled:
        rejected = ~base_keep
        return CleaningResult(
            keep_mask=base_keep,
            rejected_mask=rejected,
            enabled=False,
            n_total=n_total,
            n_valid_for_clipping=int(np.count_nonzero(base_keep)),
            n_rejected=int(np.count_nonzero(rejected)),
            target=target,
            sigma=float(sigma),
            iterations=0,
            centre=centre,
            scale=scale,
        )

    # Start from all finite X positions. Sigma clipping and manual masks are
    # then combined. Manual keep points override sigma clipping, while manual
    # reject points are always removed. Non-finite X positions are never kept.
    keep = base_keep.copy()
    valid_for_clipping = base_keep.copy()
    iterations_done = 0

    series_dict = _choose_clipping_series(y, model, residuals, target) if sigma_enabled else {}

    # If sigma clipping was requested but the selected diagnostic does not
    # exist, do not clip the wrong quantity. Manual masks, if present, are still
    # applied below.
    target_label = target
    if sigma_enabled and not series_dict:
        target_label = f"{target} (not available)"

    for _, series in series_dict.items():
        series = np.asarray(series, dtype=float)
        if series.shape != keep.shape:
            continue

        series_keep, n_iter = sigma_clip_keep_mask(
            series,
            sigma_threshold=float(sigma),
            max_iterations=max_iter,
            centre_method=centre,
            scale_method=scale,
        )
        keep &= series_keep
        valid_for_clipping &= np.isfinite(series)
        iterations_done = max(iterations_done, n_iter)

    if has_manual_mask:
        for index in manual_keep_indices:
            if 0 <= index < n_total and base_keep[index]:
                keep[index] = True
        for index in manual_reject_indices:
            if 0 <= index < n_total:
                keep[index] = False

    rejected = base_keep & ~keep

    if has_manual_mask:
        if sigma_enabled:
            target_label = f"{target_label} + manual"
        else:
            target_label = "Manual points"

    return CleaningResult(
        keep_mask=keep,
        rejected_mask=rejected,
        enabled=True,
        n_total=n_total,
        n_valid_for_clipping=int(np.count_nonzero(valid_for_clipping)),
        n_rejected=int(np.count_nonzero(rejected)),
        target=target_label,
        sigma=float(sigma),
        iterations=int(iterations_done),
        centre=centre,
        scale=scale,
    )
