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
    n_auto_rejected: int = 0
    n_manual_rejected: int = 0
    n_manual_restored: int = 0


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
    initial_keep: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int]:
    """Return a Boolean mask selecting points that survive sigma clipping.

    The returned mask has the same length as ``values``. Non-finite values are
    rejected. The clipping is iterative and is performed around either the mean
    or the median. The scale can be the classical standard deviation or the
    MAD-based robust sigma, 1.4826 * MAD.
    """
    arr = np.asarray(values, dtype=float)
    keep = np.isfinite(arr)
    if initial_keep is not None:
        initial = np.asarray(initial_keep, dtype=bool)
        if initial.shape == keep.shape:
            keep &= initial

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


def compute_auto_sigma_clip_reject_indices(
    x: np.ndarray,
    y: Optional[np.ndarray],
    model: Optional[np.ndarray],
    residuals: Optional[np.ndarray],
    values: Dict[str, object],
) -> tuple[set[int], str, int]:
    """Return new point indices rejected by one explicit sigma-clipping pass.

    This function is used by the GUI ``Apply sigma clipping`` button.  It does
    not modify an existing auto-clipping mask; it only returns additional
    indices to append to that persistent mask. Existing auto-rejected points and
    manual rejects are excluded from the statistics used by the new pass, while
    manual kept/restored points are protected from automatic rejection.
    """
    n_total = int(len(x))
    base_keep = np.isfinite(np.asarray(x, dtype=float))

    target = str(values.get("-CLEAN_TARGET-", "Residuals"))
    sigma = parse_float(values.get("-CLEAN_SIGMA-", 4.0), 4.0) or 4.0
    max_iter = parse_int(values.get("-CLEAN_MAXITER-", 3), 3)
    centre = str(values.get("-CLEAN_CENTRE-", "Median"))
    scale = str(values.get("-CLEAN_SCALE-", "MAD"))

    auto_reject_indices = _parse_index_set(values.get("-AUTO_REJECT_INDICES-", ""), n_total)
    manual_reject_indices = _parse_index_set(values.get("-MANUAL_REJECT_INDICES-", ""), n_total)
    manual_keep_indices = _parse_index_set(values.get("-MANUAL_KEEP_INDICES-", ""), n_total)

    initial_keep = base_keep.copy()
    for index in auto_reject_indices | manual_reject_indices:
        if 0 <= index < n_total:
            initial_keep[index] = False

    series_dict = _choose_clipping_series(y, model, residuals, target)
    if not series_dict:
        return set(), f"{target} (not available)", 0

    combined_keep = initial_keep.copy()
    valid_for_any_series = False
    iterations_done = 0

    for _, series in series_dict.items():
        series = np.asarray(series, dtype=float)
        if series.shape != combined_keep.shape:
            continue
        valid_for_any_series = True
        series_keep, n_iter = sigma_clip_keep_mask(
            series,
            sigma_threshold=float(sigma),
            max_iterations=max_iter,
            centre_method=centre,
            scale_method=scale,
            initial_keep=initial_keep,
        )
        combined_keep &= series_keep
        iterations_done = max(iterations_done, n_iter)

    if not valid_for_any_series:
        return set(), f"{target} (not compatible)", 0

    # Only append genuinely new automatic rejections. Manual keep/restored
    # points are never auto-rejected by this button.
    new_reject_mask = initial_keep & ~combined_keep
    for index in manual_keep_indices:
        if 0 <= index < n_total:
            new_reject_mask[index] = False

    new_indices = {int(i) for i in np.flatnonzero(new_reject_mask)}
    return new_indices, target, int(iterations_done)


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

    # ``-CLEAN_ACTIVE-`` is kept for backward compatibility with settings files
    # from older versions. The current GUI uses a persistent auto-clipping mask
    # filled only by the explicit ``Apply sigma clipping`` button.
    sigma_enabled = bool(values.get("-CLEAN_ACTIVE-", False))
    auto_reject_indices = _parse_index_set(values.get("-AUTO_REJECT_INDICES-", ""), n_total)
    has_auto_mask = bool(auto_reject_indices)
    manual_enabled = bool(values.get("-MANUAL_CLEAN_ACTIVE-", True))
    manual_reject_indices = _parse_index_set(values.get("-MANUAL_REJECT_INDICES-", ""), n_total)
    manual_keep_indices = _parse_index_set(values.get("-MANUAL_KEEP_INDICES-", ""), n_total)
    has_manual_mask = manual_enabled and (bool(manual_reject_indices) or bool(manual_keep_indices))
    enabled = sigma_enabled or has_auto_mask or has_manual_mask
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
            n_auto_rejected=0,
            n_manual_rejected=0,
            n_manual_restored=0,
        )

    # Start from all finite X positions. The persistent auto-clipping mask,
    # optional legacy dynamic sigma clipping and manual masks are then combined.
    # Manual keep points override sigma/auto clipping, while manual rejects are
    # always removed. Non-finite X positions are never kept.
    keep = base_keep.copy()
    valid_for_clipping = base_keep.copy()
    iterations_done = 0

    for index in auto_reject_indices:
        if 0 <= index < n_total:
            keep[index] = False

    series_dict = _choose_clipping_series(y, model, residuals, target) if sigma_enabled else {}

    # If sigma clipping was requested but the selected diagnostic does not
    # exist, do not clip the wrong quantity. Manual/auto masks, if present, are
    # still applied below.
    target_label = target
    if has_auto_mask and not sigma_enabled:
        target_label = "Locked auto sigma clipping"
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

    if has_auto_mask and sigma_enabled:
        target_label = f"{target_label} + locked auto"

    if has_manual_mask:
        if sigma_enabled or has_auto_mask:
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
        n_auto_rejected=len(auto_reject_indices),
        n_manual_rejected=len(manual_reject_indices) if manual_enabled else 0,
        n_manual_restored=len(manual_keep_indices) if manual_enabled else 0,
    )
