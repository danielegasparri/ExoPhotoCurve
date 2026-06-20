"""Photometric decorrelation/detrending utilities.

This module provides the first PhotoCurve Lab detrending engine.  It is meant
for differential light curves exported by AstroImageJ or reconstructed by the
comparison-star optimiser.  The detrending model is a multiplicative baseline:

    flux_observed(t) = baseline(regressors) * astrophysical_signal(t)

The output light curve is therefore flux / baseline.  This is the appropriate
operation for relative fluxes; subtractive detrending would be more natural for
magnitudes and is intentionally not used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DetrendRegressorDetection:
    """Description of usable regressors found in a loaded table."""

    compatible: bool
    columns: List[str] = field(default_factory=list)
    suggested: List[str] = field(default_factory=list)
    warning: str = ""


@dataclass
class PhotometricDetrendResult:
    """Result returned by the photometric detrending routine."""

    x: np.ndarray
    input_flux: np.ndarray
    input_flux_err: Optional[np.ndarray]
    detrended_flux: np.ndarray
    detrended_flux_err: np.ndarray
    baseline: np.ndarray
    residuals: np.ndarray
    selected_regressors: List[str]
    fit_mask: np.ndarray
    robust_keep_mask: np.ndarray
    rms_before_ppt: float
    rms_after_ppt: float
    improvement_percent: float
    coefficients: Dict[str, float]
    report: str
    meridian_flip_enabled: bool = False
    meridian_flip_time: Optional[float] = None
    meridian_flip_mode: str = "Off"
    transit_model_used: bool = False


def _finite_numeric_fraction(df: pd.DataFrame, column: str) -> float:
    """Return the fraction of finite numeric entries in *column*."""
    try:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    except Exception:
        return 0.0
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(values)) / values.size)


def _is_raw_flux_or_error_column(column: str) -> bool:
    """Return True for AIJ source/error columns that should not be regressors."""
    text = str(column).strip().lower()
    if re.match(r"^source-sky_[tc]\d+$", text):
        return True
    if re.match(r"^source_error_[tc]\d+$", text):
        return True
    if re.match(r"^rel_flux(_err)?_[tc]\d+", text):
        return True
    if text.startswith("photocurve_fit_") or text.startswith("photocurve_expected_"):
        return True
    if text.startswith("photocurve_det_"):
        return True
    return False


def _regressor_priority(column: str, x_column: str = "") -> Tuple[int, str]:
    """Sort likely detrending regressors into a useful GUI order."""
    text = str(column).strip().lower()
    x_text = str(x_column).strip().lower()
    if text == x_text:
        return (0, text)
    if "airmass" in text:
        return (1, text)
    if "jd_utc" in text or "bjd" in text or "hjd" in text or "time" in text or "j.d." in text:
        return (2, text)
    if "fwhm" in text or "width" in text or "seeing" in text:
        return (3, text)
    if "sky" in text and not _is_raw_flux_or_error_column(text):
        return (4, text)
    if "x(fits" in text or "x_" in text or text.startswith("x"):
        return (5, text)
    if "y(fits" in text or "y_" in text or text.startswith("y"):
        return (6, text)
    if "peak" in text or "max" in text or "tot" in text:
        return (7, text)
    return (20, text)


def detect_detrending_regressors(
    df: pd.DataFrame,
    x_column: str = "",
    y_column: str = "",
    yerr_column: str = "",
) -> DetrendRegressorDetection:
    """Detect candidate detrending regressors in a photometry table.

    The detector is intentionally conservative.  It favours columns that are
    commonly exported by AstroImageJ and that describe observing conditions or
    image/centroid diagnostics: time, airmass, FWHM/width, sky background,
    centroid positions and peak/total counts.  Raw target/comparison fluxes are
    excluded to avoid using the astrophysical signal itself as a detrending
    regressor.
    """
    if df is None or df.empty:
        return DetrendRegressorDetection(False, warning="Load a photometry table before using photometric detrending.")

    excluded = {str(y_column), str(yerr_column)}
    candidates: List[str] = []
    suggested: List[str] = []

    keywords = (
        "airmass",
        "jd_utc",
        "bjd",
        "hjd",
        "j.d.",
        "time",
        "fwhm",
        "width",
        "seeing",
        "sky",
        "x(fits",
        "y(fits",
        "x(ij",
        "y(ij",
        "xcent",
        "ycent",
        "centroid",
        "peak",
        "tot",
    )

    for column in df.columns:
        col = str(column)
        lower = col.strip().lower()
        if col in excluded or _is_raw_flux_or_error_column(col):
            continue
        if _finite_numeric_fraction(df, col) < 0.70:
            continue
        if col == x_column:
            candidates.append(col)
            suggested.append(col)
            continue
        if any(key in lower for key in keywords):
            candidates.append(col)
            if "airmass" in lower or col == x_column:
                suggested.append(col)

    # The selected X column is often the most important trend proxy when the
    # user wants to mimic AIJ's JD_UTC detrending.  Add it even when its name is
    # non-standard, as long as it is numeric and not already present.
    if x_column and x_column in df.columns and x_column not in candidates:
        if _finite_numeric_fraction(df, x_column) >= 0.70:
            candidates.append(x_column)
            suggested.append(x_column)

    unique: List[str] = []
    for column in sorted(candidates, key=lambda name: _regressor_priority(name, x_column)):
        if column not in unique:
            unique.append(column)

    suggested_unique = [column for column in unique if column in set(suggested)]

    if not unique:
        return DetrendRegressorDetection(
            False,
            warning=(
                "Photometric detrending inactive: no usable columns such as JD_UTC, "
                "AIRMASS, FWHM/Width, sky background or centroid positions were found."
            ),
        )

    return DetrendRegressorDetection(True, columns=unique, suggested=suggested_unique)


def _to_float_array(df: pd.DataFrame, column: str) -> np.ndarray:
    """Convert a DataFrame column to a float array."""
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _normalise_regressor(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return a robustly centred/scaled regressor and its centre/scale."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=float), float("nan"), float("nan")
    centre = float(np.nanmedian(finite))
    q16, q84 = np.nanpercentile(finite, [16, 84])
    scale = float(0.5 * (q84 - q16))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (values - centre) / scale, centre, scale


def _build_design_matrix(
    df: pd.DataFrame,
    selected_regressors: Sequence[str],
    polynomial_order: int = 1,
    x_values: Optional[np.ndarray] = None,
    meridian_flip_time: Optional[float] = None,
    meridian_flip_mode: str = "Off",
    allow_constant_only: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Build a linear design matrix from selected and synthetic regressors."""
    n_rows = len(df)
    columns = [np.ones(n_rows, dtype=float)]
    names = ["constant"]
    order = int(max(1, min(2, polynomial_order)))

    for regressor in selected_regressors:
        if regressor not in df.columns:
            continue
        raw = _to_float_array(df, regressor)
        norm, _, _ = _normalise_regressor(raw)
        columns.append(norm)
        names.append(str(regressor))
        if order >= 2:
            columns.append(norm ** 2)
            names.append(f"{regressor}^2")

    # A meridian flip is not a continuous observing-condition trend; it is best
    # represented as a step function.  Optionally we also allow a separate
    # after-flip slope, which can model a changed trend after the pier flip.
    flip_time = float("nan") if meridian_flip_time is None else float(meridian_flip_time)
    if x_values is not None and np.isfinite(flip_time):
        x_arr = np.asarray(x_values, dtype=float)
        if x_arr.size == n_rows:
            step = np.where(np.isfinite(x_arr) & (x_arr >= flip_time), 1.0, 0.0)
            columns.append(step)
            names.append("meridian_flip_step")

            if str(meridian_flip_mode).strip().lower().startswith("step +"):
                after = step > 0.5
                delta = np.asarray(x_arr - flip_time, dtype=float)
                finite_after = delta[after & np.isfinite(delta)]
                if finite_after.size >= 2:
                    scale = float(np.nanpercentile(finite_after, 84) - np.nanpercentile(finite_after, 16))
                    if not np.isfinite(scale) or scale <= 0:
                        scale = float(np.nanstd(finite_after))
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0
                    after_slope = np.where(after & np.isfinite(delta), delta / scale, 0.0)
                    columns.append(after_slope)
                    names.append("meridian_flip_after_slope")

    if len(columns) == 1 and not allow_constant_only:
        raise ValueError("Select at least one valid detrending regressor or enable meridian-flip detrending.")

    return np.vstack(columns).T, names


def _weighted_lstsq(design: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Solve a weighted least-squares problem on rows selected by *mask*."""
    fit_mask = np.asarray(mask, dtype=bool)
    fit_mask &= np.all(np.isfinite(design), axis=1)
    fit_mask &= np.isfinite(y)
    if weights is not None:
        fit_mask &= np.isfinite(weights) & (weights > 0)

    if np.count_nonzero(fit_mask) < design.shape[1] + 2:
        raise ValueError("Not enough valid points to fit the selected detrending model.")

    x_fit = design[fit_mask]
    y_fit = y[fit_mask]
    if weights is not None:
        w = np.sqrt(weights[fit_mask])
        x_fit = x_fit * w[:, None]
        y_fit = y_fit * w

    coeff, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    return coeff


def _robust_sigma_clip(residual: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Return an updated robust keep mask from residuals."""
    keep = np.asarray(mask, dtype=bool).copy()
    use = keep & np.isfinite(residual)
    if np.count_nonzero(use) < 8:
        return keep
    subset = residual[use]
    centre = float(np.nanmedian(subset))
    scale = float(1.4826 * np.nanmedian(np.abs(subset - centre)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(subset))
    if not np.isfinite(scale) or scale <= 0:
        return keep
    keep &= np.isfinite(residual) & (np.abs(residual - centre) <= sigma * scale)
    return keep


def _rms_ppt(values: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return RMS in ppt after subtracting the median."""
    values = np.asarray(values, dtype=float)
    use = np.isfinite(values)
    if mask is not None and len(mask) == len(values):
        use &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(use) < 3:
        return float("nan")
    subset = values[use]
    residual = subset - np.nanmedian(subset)
    return float(1000.0 * np.sqrt(np.nanmean(residual ** 2)))


def _expected_transit_outside_mask(
    x: np.ndarray,
    planet: Optional[object],
    duration_factor: float = 1.35,
) -> Optional[np.ndarray]:
    """Return True outside the expected transit, when a planet is available."""
    if planet is None:
        return None
    try:
        period = float(getattr(planet, "period_days"))
        t0 = float(getattr(planet, "t0_bjd_tdb"))
        duration_hours = float(getattr(planet, "duration_hours"))
    except Exception:
        return None
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0) and np.isfinite(duration_hours) and duration_hours > 0):
        return None

    time = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(time)
    if np.count_nonzero(finite) < 3:
        return None
    median_time = float(np.nanmedian(time[finite]))
    if 50000.0 < median_time < 100000.0:
        time = time + 2400000.0
    epoch = int(np.round((float(np.nanmedian(time[finite])) - t0) / period))
    predicted_tmid = t0 + epoch * period
    half_duration_days = 0.5 * duration_factor * duration_hours / 24.0
    outside = np.abs(time - predicted_tmid) > half_duration_days
    outside &= np.isfinite(time)
    if np.count_nonzero(outside) < 6:
        return None
    return outside



def _is_robust_meridian_flip_mode(mode: object) -> bool:
    """Return True when the user selected robust level matching."""
    text = str(mode or "").strip().lower()
    return text.startswith("robust") or "level" in text


def _constant_baseline(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a constant baseline from the median of finite masked values."""
    values = np.asarray(values, dtype=float)
    use = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if np.count_nonzero(use) >= 3:
        centre = float(np.nanmedian(values[use]))
    else:
        finite = values[np.isfinite(values)]
        centre = float(np.nanmedian(finite)) if finite.size else 1.0
    if not np.isfinite(centre) or abs(centre) <= 1.0e-12:
        centre = 1.0
    return np.full_like(values, centre, dtype=float)


def _preliminary_slow_baseline(
    df: pd.DataFrame,
    selected_regressors: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    mask: np.ndarray,
    polynomial_order: int,
    robust_sigma: float,
    robust_iterations: int,
) -> np.ndarray:
    """Estimate a slow baseline without any meridian-flip term.

    This is used only to measure the flip jump on a flattened light curve.  If
    the requested slow model is not sufficiently constrained, the routine falls
    back to a constant median baseline rather than applying an unstable model.
    """
    selected = [str(col) for col in selected_regressors if str(col) in df.columns]
    if not selected:
        return _constant_baseline(y, mask)
    try:
        design, _names = _build_design_matrix(
            df,
            selected,
            polynomial_order=polynomial_order,
            x_values=x,
            meridian_flip_time=None,
            meridian_flip_mode="Off",
        )
        keep = np.asarray(mask, dtype=bool).copy()
        coeff = _weighted_lstsq(design, y, weights, keep)
        for _ in range(max(0, int(robust_iterations))):
            baseline_trial = design @ coeff
            residual = y - baseline_trial
            new_keep = _robust_sigma_clip(residual, keep, float(robust_sigma))
            if np.array_equal(new_keep, keep):
                break
            if np.count_nonzero(new_keep) < design.shape[1] + 2:
                break
            keep = new_keep
            coeff = _weighted_lstsq(design, y, weights, keep)
        baseline = design @ coeff
        valid = np.isfinite(baseline) & (np.abs(baseline) > 1.0e-8)
        if np.count_nonzero(valid) < max(6, len(y) // 4):
            return _constant_baseline(y, mask)
        return baseline
    except Exception:
        return _constant_baseline(y, mask)


def _sigma_clipped_median(values: np.ndarray, sigma: float = 4.0, iterations: int = 3) -> Tuple[float, int]:
    """Return a robust median and the number of points retained."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), 0
    keep = np.ones(arr.size, dtype=bool)
    for _ in range(max(0, int(iterations))):
        subset = arr[keep]
        if subset.size < 4:
            break
        centre = float(np.nanmedian(subset))
        scale = float(1.4826 * np.nanmedian(np.abs(subset - centre)))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(subset))
        if not np.isfinite(scale) or scale <= 0:
            break
        new_keep = np.abs(arr - centre) <= float(sigma) * scale
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    subset = arr[keep]
    if subset.size == 0:
        return float("nan"), 0
    return float(np.nanmedian(subset)), int(subset.size)



def _robust_segment_level_at_time(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    reference_time: float,
    robust_sigma: float = 4.0,
    robust_iterations: int = 3,
) -> Tuple[float, int, str]:
    """Estimate a segment level at a reference time.

    A meridian flip often occurs during the transit, so the safest quantity is
    not the global median before/after the flip.  Instead, when enough points are
    available, this function fits a robust straight line to one side of the
    light curve and evaluates that line at the flip time.  This estimates the
    level immediately before or after the flip while ignoring the in-transit
    points already removed from the mask.  If the line is not constrained, the
    routine falls back to a robust median.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    use = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(use) == 0:
        return float("nan"), 0, "none"

    xx = x[use]
    yy = y[use]
    median_level, median_n = _sigma_clipped_median(yy, robust_sigma, robust_iterations)
    if xx.size < 8:
        return median_level, median_n, "median"

    span = float(np.nanmax(xx) - np.nanmin(xx))
    if not np.isfinite(span) or span <= 0:
        return median_level, median_n, "median"

    x_norm = (xx - float(reference_time)) / span
    design = np.vstack([np.ones_like(x_norm), x_norm]).T
    keep = np.isfinite(x_norm) & np.isfinite(yy)
    if np.count_nonzero(keep) < 8:
        return median_level, median_n, "median"

    coeff, *_ = np.linalg.lstsq(design[keep], yy[keep], rcond=None)
    for _ in range(max(0, int(robust_iterations))):
        pred = design @ coeff
        resid = yy - pred
        subset = resid[keep & np.isfinite(resid)]
        if subset.size < 8:
            break
        centre = float(np.nanmedian(subset))
        scale = float(1.4826 * np.nanmedian(np.abs(subset - centre)))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(subset))
        if not np.isfinite(scale) or scale <= 0:
            break
        new_keep = np.isfinite(resid) & (np.abs(resid - centre) <= float(robust_sigma) * scale)
        if np.array_equal(new_keep, keep):
            break
        if np.count_nonzero(new_keep) < 8:
            break
        keep = new_keep
        coeff, *_ = np.linalg.lstsq(design[keep], yy[keep], rcond=None)

    level_at_reference = float(coeff[0])
    # Guard against a badly extrapolated line.  The level at the flip should be
    # close to the observed segment scale for normal relative-flux light curves.
    if not np.isfinite(level_at_reference):
        return median_level, median_n, "median"
    if np.isfinite(median_level) and abs(median_level) > 1.0e-12:
        ratio = abs(level_at_reference / median_level)
        if not (0.5 <= ratio <= 1.5):
            return median_level, median_n, "median"
    return level_at_reference, int(np.count_nonzero(keep)), "linear-at-flip"


def _estimate_robust_meridian_flip_factor(
    x: np.ndarray,
    y: np.ndarray,
    preliminary_baseline: np.ndarray,
    fit_mask: np.ndarray,
    flip_time: float,
    robust_sigma: float,
    robust_iterations: int,
    min_points_per_side: int = 4,
) -> Dict[str, object]:
    """Estimate a multiplicative post-flip correction from robust levels.

    The correction is intentionally direct and sign-safe: it estimates the
    relative level immediately before and after the flip, then multiplies the
    post-flip data by ``before / after``.  If the flip occurs during the transit,
    the in-transit points are normally absent from ``fit_mask``; in that case a
    robust straight line is fitted separately to the out-of-transit points on
    each side and evaluated at the flip time.  This is much safer than fitting a
    single step coefficient together with all other regressors.

    ``preliminary_baseline`` is accepted for backwards compatibility with the
    first implementation, but the current estimator deliberately works on the
    original relative flux.  A global preliminary baseline can absorb part of a
    real step and make the measured correction too small.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.asarray(fit_mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    before = valid & (x < flip_time)
    after = valid & (x >= flip_time)

    level_before, n_before, method_before = _robust_segment_level_at_time(
        x,
        y,
        before,
        flip_time,
        robust_sigma=robust_sigma,
        robust_iterations=robust_iterations,
    )
    level_after, n_after, method_after = _robust_segment_level_at_time(
        x,
        y,
        after,
        flip_time,
        robust_sigma=robust_sigma,
        robust_iterations=robust_iterations,
    )

    applied = False
    factor = 1.0
    warning = ""
    if n_before < int(min_points_per_side) or n_after < int(min_points_per_side):
        warning = (
            "Meridian flip robust correction not applied: not enough good out-of-transit "
            f"points on both sides of the flip ({n_before} before, {n_after} after)."
        )
    elif not (np.isfinite(level_before) and np.isfinite(level_after) and abs(level_after) > 1.0e-12):
        warning = "Meridian flip robust correction not applied: the pre/post levels could not be measured reliably."
    else:
        factor = float(level_before / level_after)
        if np.isfinite(factor) and 0.2 <= factor <= 5.0:
            applied = True
        else:
            warning = f"Meridian flip robust correction not applied: derived factor is unrealistic ({factor:.6g})."
            factor = 1.0

    factor_array = np.ones_like(y, dtype=float)
    if applied:
        factor_array[np.isfinite(x) & (x >= flip_time)] = factor

    corrected = y * factor_array
    corrected_before, _, _ = _robust_segment_level_at_time(
        x,
        corrected,
        before,
        flip_time,
        robust_sigma=robust_sigma,
        robust_iterations=robust_iterations,
    )
    corrected_after, _, _ = _robust_segment_level_at_time(
        x,
        corrected,
        after,
        flip_time,
        robust_sigma=robust_sigma,
        robust_iterations=robust_iterations,
    )
    residual_jump_ppt = float("nan")
    if np.isfinite(corrected_before) and np.isfinite(corrected_after):
        ref = corrected_before if abs(corrected_before) > 1.0e-12 else 1.0
        residual_jump_ppt = float(1000.0 * (corrected_after - corrected_before) / ref)

    return {
        "applied": applied,
        "factor": float(factor),
        "factor_array": factor_array,
        "level_before": float(level_before),
        "level_after": float(level_after),
        "n_before": int(n_before),
        "n_after": int(n_after),
        "method_before": method_before,
        "method_after": method_after,
        "residual_jump_ppt": residual_jump_ppt,
        "warning": warning,
    }


def apply_photometric_detrending(
    df: pd.DataFrame,
    x: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray],
    selected_regressors: Sequence[str],
    planet: Optional[object] = None,
    mask_expected_transit: bool = True,
    external_keep_mask: Optional[np.ndarray] = None,
    polynomial_order: int = 1,
    robust_sigma: float = 4.0,
    robust_iterations: int = 3,
    meridian_flip_time: Optional[float] = None,
    meridian_flip_mode: str = "Off",
    transit_model: Optional[np.ndarray] = None,
) -> PhotometricDetrendResult:
    """Fit and remove a multiplicative photometric baseline.

    The baseline is fitted to selected regressors using weighted least squares.
    When requested and possible, expected in-transit points are excluded from
    the baseline fit so that the astrophysical transit is preserved.

    Meridian flips can be handled in two ways.  The recommended mode is robust
    level matching: the routine measures the pre/post-flip level offset on a
    preliminary flattened curve and applies a direct multiplicative correction
    to the post-flip data before the final slow-baseline fit.  This is much less
    sensitive to sign mistakes and to flips occurring during the transit than a
    simple step coefficient inside the linear detrending model.
    """
    selected = [str(col) for col in selected_regressors if str(col) in df.columns]

    x = np.asarray(x, dtype=float)
    y_original = np.asarray(flux, dtype=float)
    if flux_err is not None:
        err_original = np.asarray(flux_err, dtype=float)
    else:
        err_original = None

    # Optional model-aware detrending.  When a previous transit fit is
    # available, use its pure transit model to estimate the instrumental
    # baseline on flux / transit_model.  The final output still preserves the
    # transit signal: only the baseline and optional flip factor are removed.
    transit_model_used = False
    transit_model_arr = np.ones_like(y_original, dtype=float)
    if transit_model is not None:
        candidate = np.asarray(transit_model, dtype=float)
        if candidate.shape == y_original.shape:
            good_model = np.isfinite(candidate) & (np.abs(candidate) > 1.0e-6) & (candidate > 0.2) & (candidate < 2.0)
            if np.count_nonzero(good_model) >= max(5, min(10, candidate.size // 3)):
                transit_model_arr[good_model] = candidate[good_model]
                transit_model_used = True

    valid_transit_model = np.isfinite(transit_model_arr) & (np.abs(transit_model_arr) > 1.0e-12)
    baseline_target = np.divide(
        y_original,
        transit_model_arr,
        out=np.full_like(y_original, np.nan, dtype=float),
        where=valid_transit_model,
    )
    if err_original is not None:
        err_for_fit_original = np.divide(
            err_original,
            np.abs(transit_model_arr),
            out=np.full_like(err_original, np.nan, dtype=float),
            where=valid_transit_model,
        )
    else:
        err_for_fit_original = None

    flip_time = float("nan") if meridian_flip_time is None else float(meridian_flip_time)
    flip_enabled = bool(np.isfinite(flip_time))
    robust_flip_mode = bool(flip_enabled and _is_robust_meridian_flip_mode(meridian_flip_mode))
    if not selected and not flip_enabled:
        raise ValueError("Select at least one detrending regressor or enable meridian-flip detrending.")

    # In robust level-matching mode, the flip is applied as a direct
    # multiplicative correction, not as a column in the final design matrix.
    design, design_names = _build_design_matrix(
        df,
        selected,
        polynomial_order=polynomial_order,
        x_values=x,
        meridian_flip_time=None if robust_flip_mode else (flip_time if flip_enabled else None),
        meridian_flip_mode=meridian_flip_mode,
        allow_constant_only=robust_flip_mode,
    )

    base_mask = np.isfinite(x) & np.isfinite(y_original) & np.isfinite(baseline_target) & np.all(np.isfinite(design), axis=1)
    if external_keep_mask is not None and len(external_keep_mask) == len(base_mask):
        base_mask &= np.asarray(external_keep_mask, dtype=bool)

    transit_mask_used = False
    if mask_expected_transit and not transit_model_used:
        outside = _expected_transit_outside_mask(x, planet)
        if outside is not None and len(outside) == len(base_mask):
            base_mask &= outside
            transit_mask_used = True

    # Initial weights, before any optional robust flip correction.
    preliminary_weights = None
    if err_for_fit_original is not None:
        good_weight = np.isfinite(err_for_fit_original) & (err_for_fit_original > 0)
        min_needed = design.shape[1] + 2 if not robust_flip_mode else max(3, min(design.shape[1] + 2, 5))
        if np.count_nonzero(base_mask & good_weight) >= min_needed:
            preliminary_weights = np.zeros_like(baseline_target, dtype=float)
            preliminary_weights[good_weight] = 1.0 / np.maximum(err_for_fit_original[good_weight], 1.0e-12) ** 2

    flip_diagnostics: Optional[Dict[str, object]] = None
    flip_factor_array = np.ones_like(y_original, dtype=float)
    if robust_flip_mode:
        preliminary_baseline = _preliminary_slow_baseline(
            df,
            selected,
            x,
            baseline_target,
            preliminary_weights,
            base_mask,
            polynomial_order=int(max(1, min(2, polynomial_order))),
            robust_sigma=float(robust_sigma),
            robust_iterations=int(robust_iterations),
        )
        flip_diagnostics = _estimate_robust_meridian_flip_factor(
            x,
            baseline_target,
            preliminary_baseline,
            base_mask,
            flip_time,
            robust_sigma=float(robust_sigma),
            robust_iterations=int(robust_iterations),
        )
        flip_factor_array = np.asarray(flip_diagnostics.get("factor_array", flip_factor_array), dtype=float)

    y = baseline_target * flip_factor_array
    if err_for_fit_original is not None:
        err = err_for_fit_original * np.abs(flip_factor_array)
    else:
        err = None

    weights = None
    if err is not None:
        good_weight = np.isfinite(err) & (err > 0)
        if np.count_nonzero(base_mask & good_weight) >= design.shape[1] + 2:
            weights = np.zeros_like(y, dtype=float)
            weights[good_weight] = 1.0 / np.maximum(err[good_weight], 1.0e-12) ** 2

    robust_keep = base_mask.copy()
    coeff = _weighted_lstsq(design, y, weights, robust_keep)
    for _ in range(max(0, int(robust_iterations))):
        baseline_corrected = design @ coeff
        residual = y - baseline_corrected
        new_keep = _robust_sigma_clip(residual, robust_keep, float(robust_sigma))
        if np.array_equal(new_keep, robust_keep):
            break
        if np.count_nonzero(new_keep) < design.shape[1] + 2:
            break
        robust_keep = new_keep
        coeff = _weighted_lstsq(design, y, weights, robust_keep)

    baseline_corrected = design @ coeff
    valid_baseline_corrected = np.isfinite(baseline_corrected) & (np.abs(baseline_corrected) > 1.0e-8)
    # Remove only the fitted instrumental baseline and flip factor from the
    # original flux.  If a transit model was used to estimate the baseline, the
    # astrophysical transit remains in the output light curve.
    detrended = np.divide(
        y_original * flip_factor_array,
        baseline_corrected,
        out=np.full_like(y_original, np.nan, dtype=float),
        where=valid_baseline_corrected,
    )

    # Put the detrended light curve on a clean relative-flux scale.  Use the
    # same mask used for the baseline fit so the out-of-transit baseline is 1.
    scale_mask = robust_keep & np.isfinite(detrended)
    if np.count_nonzero(scale_mask) >= 3:
        scale = float(np.nanmedian(detrended[scale_mask]))
    else:
        finite_det = detrended[np.isfinite(detrended)]
        scale = float(np.nanmedian(finite_det)) if finite_det.size else 1.0
    if np.isfinite(scale) and scale != 0:
        detrended = detrended / scale
        baseline_corrected = baseline_corrected * scale

    # ``baseline`` is the total multiplicative baseline relative to the
    # original input flux.  For robust flip correction, the final detrended data
    # are y_original * factor / slow_baseline, therefore the equivalent baseline
    # to divide the original flux by is slow_baseline / factor.
    valid_factor = np.isfinite(flip_factor_array) & (np.abs(flip_factor_array) > 1.0e-12)
    baseline = np.divide(
        baseline_corrected,
        flip_factor_array,
        out=np.full_like(baseline_corrected, np.nan, dtype=float),
        where=valid_factor,
    )
    valid_baseline = np.isfinite(baseline) & (np.abs(baseline) > 1.0e-8)

    if err_original is not None:
        detrended_err = np.divide(
            err_original,
            np.abs(baseline),
            out=np.full_like(err_original, np.nan, dtype=float),
            where=valid_baseline,
        )
    else:
        detrended_err = np.full_like(y_original, np.nan, dtype=float)

    residuals = detrended - 1.0
    finite_input = baseline_target[np.isfinite(baseline_target)]
    input_median = float(np.nanmedian(finite_input)) if finite_input.size else float("nan")
    if np.isfinite(input_median) and abs(input_median) > 1.0e-12:
        before_for_rms = baseline_target / input_median
    else:
        before_for_rms = baseline_target

    if transit_model_used:
        after_for_rms = np.divide(
            detrended,
            transit_model_arr,
            out=np.full_like(detrended, np.nan, dtype=float),
            where=valid_transit_model,
        )
    else:
        after_for_rms = detrended

    rms_before = _rms_ppt(before_for_rms, base_mask)
    rms_after = _rms_ppt(after_for_rms, base_mask)
    if np.isfinite(rms_before) and rms_before > 0 and np.isfinite(rms_after):
        improvement = 100.0 * (rms_before - rms_after) / rms_before
    else:
        improvement = float("nan")

    coefficients = {name: float(value) for name, value in zip(design_names, coeff)}
    if robust_flip_mode and flip_diagnostics is not None:
        coefficients["meridian_flip_factor_after"] = float(flip_diagnostics.get("factor", 1.0))

    report_lines = [
        "Photometric detrending",
        f"Selected regressors: {', '.join(selected) if selected else 'none'}",
        f"Meridian flip detrending: {'yes' if flip_enabled else 'no'}",
        f"Meridian flip time: {flip_time:.8f}" if flip_enabled else "Meridian flip time: not used",
        f"Meridian flip mode: {str(meridian_flip_mode) if flip_enabled else 'not used'}",
        f"Transit fit model used for detrending: {'yes' if transit_model_used else 'no'}",
    ]
    if robust_flip_mode and flip_diagnostics is not None:
        applied = bool(flip_diagnostics.get("applied", False))
        level_before = float(flip_diagnostics.get("level_before", float("nan")))
        level_after = float(flip_diagnostics.get("level_after", float("nan")))
        factor = float(flip_diagnostics.get("factor", 1.0))
        residual_jump = float(flip_diagnostics.get("residual_jump_ppt", float("nan")))
        report_lines.extend(
            [
                "Meridian flip robust level matching",
                f"Robust correction applied: {'yes' if applied else 'no'}",
                f"Good points before/after flip: {int(flip_diagnostics.get('n_before', 0))} / {int(flip_diagnostics.get('n_after', 0))}",
                f"Level estimator before/after: {flip_diagnostics.get('method_before', 'n/a')} / {flip_diagnostics.get('method_after', 'n/a')}",
                f"Estimated level before flip: {level_before:.8f}",
                f"Estimated level after flip:  {level_after:.8f}",
                f"Applied multiplier to post-flip data: x{factor:.8f}",
                f"Equivalent post-flip jump: {(factor - 1.0) * 1000.0:+.3f} ppt",
                f"Residual pre/post level difference after correction: {residual_jump:+.3f} ppt",
            ]
        )
        warning = str(flip_diagnostics.get("warning", "") or "")
        if warning:
            report_lines.append(f"Warning: {warning}")

    report_lines.extend(
        [
            f"Polynomial order per regressor: {int(max(1, min(2, polynomial_order)))}",
            f"Expected transit masked: {'yes' if transit_mask_used else ('no (fit model used)' if transit_model_used else 'no')}",
            f"External cleaning mask used: {'yes' if external_keep_mask is not None else 'no'}",
            f"Fit points used: {int(np.count_nonzero(robust_keep))} / {len(y_original)}",
            f"Robust sigma: {float(robust_sigma):.2f}",
            "",
            f"RMS before detrending = {rms_before:.3f} ppt",
            f"RMS after detrending  = {rms_after:.3f} ppt",
            f"RMS improvement       = {improvement:+.1f} %",
            "",
            "Baseline coefficients",
        ]
    )
    for name, value in coefficients.items():
        report_lines.append(f"{name}: {value:+.8e}")
    report = "\n".join(report_lines)

    return PhotometricDetrendResult(
        x=x,
        input_flux=y_original,
        input_flux_err=err_original,
        detrended_flux=detrended,
        detrended_flux_err=detrended_err,
        baseline=baseline,
        residuals=residuals,
        selected_regressors=selected,
        fit_mask=base_mask,
        robust_keep_mask=robust_keep,
        rms_before_ppt=rms_before,
        rms_after_ppt=rms_after,
        improvement_percent=improvement,
        coefficients=coefficients,
        report=report,
        meridian_flip_enabled=flip_enabled,
        meridian_flip_time=flip_time if flip_enabled else None,
        meridian_flip_mode=str(meridian_flip_mode) if flip_enabled else "Off",
        transit_model_used=transit_model_used,
    )
