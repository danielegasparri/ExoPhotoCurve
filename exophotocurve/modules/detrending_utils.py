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

    if len(columns) == 1:
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
) -> PhotometricDetrendResult:
    """Fit and remove a multiplicative photometric baseline.

    The baseline is fitted to selected regressors using weighted least squares.
    When requested and possible, expected in-transit points are excluded from
    the baseline fit so that the astrophysical transit is preserved.
    """
    selected = [str(col) for col in selected_regressors if str(col) in df.columns]

    x = np.asarray(x, dtype=float)
    y = np.asarray(flux, dtype=float)
    if flux_err is not None:
        err = np.asarray(flux_err, dtype=float)
    else:
        err = None

    flip_time = float("nan") if meridian_flip_time is None else float(meridian_flip_time)
    flip_enabled = bool(np.isfinite(flip_time))
    if not selected and not flip_enabled:
        raise ValueError("Select at least one detrending regressor or enable meridian-flip detrending.")

    design, design_names = _build_design_matrix(
        df,
        selected,
        polynomial_order=polynomial_order,
        x_values=x,
        meridian_flip_time=flip_time if flip_enabled else None,
        meridian_flip_mode=meridian_flip_mode,
    )

    base_mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if err is not None:
        good_err = np.isfinite(err) & (err > 0)
        # Bad error bars should not necessarily remove a point from the fit;
        # they just disable weighting for that point if there are too few good
        # uncertainties.  Keep them in base_mask for the unweighted fallback.
    if external_keep_mask is not None and len(external_keep_mask) == len(base_mask):
        base_mask &= np.asarray(external_keep_mask, dtype=bool)

    transit_mask_used = False
    if mask_expected_transit:
        outside = _expected_transit_outside_mask(x, planet)
        if outside is not None and len(outside) == len(base_mask):
            base_mask &= outside
            transit_mask_used = True

    weights = None
    if err is not None:
        good_weight = np.isfinite(err) & (err > 0)
        if np.count_nonzero(base_mask & good_weight) >= design.shape[1] + 2:
            weights = np.zeros_like(y, dtype=float)
            weights[good_weight] = 1.0 / np.maximum(err[good_weight], 1.0e-12) ** 2

    robust_keep = base_mask.copy()
    coeff = _weighted_lstsq(design, y, weights, robust_keep)
    for _ in range(max(0, int(robust_iterations))):
        baseline_trial = design @ coeff
        residual = y - baseline_trial
        new_keep = _robust_sigma_clip(residual, robust_keep, float(robust_sigma))
        if np.array_equal(new_keep, robust_keep):
            break
        if np.count_nonzero(new_keep) < design.shape[1] + 2:
            break
        robust_keep = new_keep
        coeff = _weighted_lstsq(design, y, weights, robust_keep)

    baseline = design @ coeff
    valid_baseline = np.isfinite(baseline) & (np.abs(baseline) > 1.0e-8)
    detrended = np.divide(y, baseline, out=np.full_like(y, np.nan, dtype=float), where=valid_baseline)

    # Put the detrended light curve on a clean relative-flux scale.  Use the
    # same mask used for the baseline fit so the out-of-transit baseline is 1.
    scale_mask = robust_keep & np.isfinite(detrended)
    if np.count_nonzero(scale_mask) >= 3:
        scale = float(np.nanmedian(detrended[scale_mask]))
    else:
        scale = float(np.nanmedian(detrended[np.isfinite(detrended)]))
    if np.isfinite(scale) and scale != 0:
        detrended = detrended / scale
        baseline = baseline * scale

    if err is not None:
        detrended_err = np.divide(err, np.abs(baseline), out=np.full_like(err, np.nan, dtype=float), where=valid_baseline)
    else:
        detrended_err = np.full_like(y, np.nan, dtype=float)

    residuals = detrended - 1.0
    rms_before = _rms_ppt(y / np.nanmedian(y[np.isfinite(y)]) if np.isfinite(np.nanmedian(y[np.isfinite(y)])) else y, base_mask)
    rms_after = _rms_ppt(detrended, base_mask)
    if np.isfinite(rms_before) and rms_before > 0 and np.isfinite(rms_after):
        improvement = 100.0 * (rms_before - rms_after) / rms_before
    else:
        improvement = float("nan")

    coefficients = {name: float(value) for name, value in zip(design_names, coeff)}

    report_lines = [
        "Photometric detrending",
        f"Selected regressors: {', '.join(selected) if selected else 'none'}",
        f"Meridian flip detrending: {'yes' if flip_enabled else 'no'}",
        f"Meridian flip time: {flip_time:.8f}" if flip_enabled else "Meridian flip time: not used",
        f"Meridian flip mode: {str(meridian_flip_mode) if flip_enabled else 'not used'}",
        f"Polynomial order per regressor: {int(max(1, min(2, polynomial_order)))}",
        f"Expected transit masked: {'yes' if transit_mask_used else 'no'}",
        f"External cleaning mask used: {'yes' if external_keep_mask is not None else 'no'}",
        f"Fit points used: {int(np.count_nonzero(robust_keep))} / {len(y)}",
        f"Robust sigma: {float(robust_sigma):.2f}",
        "",
        f"RMS before detrending = {rms_before:.3f} ppt",
        f"RMS after detrending  = {rms_after:.3f} ppt",
        f"RMS improvement       = {improvement:+.1f} %",
        "",
        "Baseline coefficients",
    ]
    for name, value in coefficients.items():
        report_lines.append(f"{name}: {value:+.8e}")
    report = "\n".join(report_lines)

    return PhotometricDetrendResult(
        x=x,
        input_flux=y,
        input_flux_err=err,
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
    )
