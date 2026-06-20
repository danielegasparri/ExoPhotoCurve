"""Comparison-star optimisation utilities for AstroImageJ tables.

The routines in this module work only when an input table contains the raw
AstroImageJ aperture flux columns, usually named ``Source-Sky_T1`` and
``Source-Sky_C2`` etc.  The optimiser builds differential light curves from
raw target/check-star counts and subsets of comparison-star counts, then uses a
robust scatter metric to select a stable comparison ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_SOURCE_RE = re.compile(r"^Source-Sky_([TC]\d+)$", re.IGNORECASE)
_ERROR_TEMPLATE = "Source_Error_{star_id}"


@dataclass
class AijFluxDetection:
    """Description of raw AstroImageJ flux columns found in a table."""

    compatible: bool
    target_ids: List[str] = field(default_factory=list)
    comparison_ids: List[str] = field(default_factory=list)
    source_columns: Dict[str, str] = field(default_factory=dict)
    error_columns: Dict[str, str] = field(default_factory=dict)
    time_column: str = ""
    warning: str = ""


@dataclass
class CandidateMetric:
    """Quality metrics for one differential light curve."""

    rms_ppt: float
    mad_ppt: float
    autocorr_lag1: float
    beta_factor: float
    objective: float
    n_points: int




@dataclass
class ComparisonDiagnosticCurve:
    """Leave-one-out diagnostic curve for one comparison star.

    The tested comparison star is treated as a temporary target and is divided
    by the ensemble of the other active comparison stars.  This helps users
    detect comparison stars that are variable, noisy or affected by local
    systematics before they are used in the science-target light curve.
    """

    star_id: str
    reference_stars: List[str]
    flux: np.ndarray
    flux_err: np.ndarray
    ensemble: np.ndarray
    metric: CandidateMetric
    warning: str = ""

@dataclass
class ComparisonOptimisationResult:
    """Result returned by the comparison-star optimiser."""

    target_id: str
    mode: str
    check_id: str
    selected_comparisons: List[str]
    rejected_comparisons: List[str]
    initial_comparisons: List[str]
    removed_sequence: List[str]
    current_curve_metric: Optional[CandidateMetric]
    all_comparisons_metric: CandidateMetric
    optimised_metric: CandidateMetric
    improvement_vs_current_percent: Optional[float]
    improvement_vs_all_percent: float
    x: np.ndarray
    optimised_flux: np.ndarray
    optimised_flux_err: np.ndarray
    comparison_ensemble: np.ndarray
    report: str


def _normalise_star_id(star_id: str) -> str:
    """Return a canonical AIJ star identifier such as ``T1`` or ``C12``."""
    text = str(star_id).strip().upper().replace(" ", "")
    match = re.search(r"([TC])\s*-?\s*(\d+)", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return text


def _star_sort_key(star_id: str) -> Tuple[str, int]:
    """Sort target/check-star identifiers in their natural AIJ order."""
    star_id = _normalise_star_id(star_id)
    match = re.match(r"([TC])(\d+)", star_id)
    if not match:
        return (star_id, 0)
    return (match.group(1), int(match.group(2)))


def detect_aij_flux_columns(df: pd.DataFrame) -> AijFluxDetection:
    """Detect raw AstroImageJ source flux columns in *df*.

    A compatible table must contain at least one target-like ``Source-Sky_T*``
    column and at least two comparison-like ``Source-Sky_C*`` columns.
    """
    source_columns: Dict[str, str] = {}
    error_columns: Dict[str, str] = {}

    for column in df.columns:
        column_text = str(column).strip()
        match = _SOURCE_RE.match(column_text)
        if match:
            star_id = _normalise_star_id(match.group(1))
            source_columns[star_id] = column_text

    for star_id in source_columns:
        err_name = _ERROR_TEMPLATE.format(star_id=star_id)
        if err_name in df.columns:
            error_columns[star_id] = err_name

    target_ids = sorted([sid for sid in source_columns if sid.startswith("T")], key=_star_sort_key)
    comparison_ids = sorted([sid for sid in source_columns if sid.startswith("C")], key=_star_sort_key)

    time_candidates = [
        # Prefer usable AIJ UTC columns for raw exports.  Several AIJ tables
        # contain a BJD_TDB column that is present but entirely NaN.
        "JD_UTC_B",
        "JD_UTC",
        "BJD_TDB",
        "BJD_UTC",
        "HJD_UTC",
        "J.D.-2400000",
        "J.D.-2400000_1",
        "slice",
    ]
    time_column = ""
    for name in time_candidates:
        if name not in df.columns:
            continue
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        if np.count_nonzero(np.isfinite(values)) > 0:
            time_column = name
            break

    if not time_column:
        # Last-resort fallback for non-standard but still usable time columns.
        for column in df.columns:
            lower = str(column).lower()
            if not any(key in lower for key in ("jd", "time", "slice")):
                continue
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            if np.count_nonzero(np.isfinite(values)) > 0:
                time_column = str(column)
                break

    if not target_ids and not comparison_ids:
        return AijFluxDetection(
            compatible=False,
            warning=(
                "Comparison-star optimiser inactive: no AstroImageJ raw flux "
                "columns such as Source-Sky_T1 and Source-Sky_C2 were found."
            ),
        )

    if not target_ids:
        return AijFluxDetection(
            compatible=False,
            target_ids=target_ids,
            comparison_ids=comparison_ids,
            source_columns=source_columns,
            error_columns=error_columns,
            time_column=time_column,
            warning="Comparison-star optimiser inactive: no Source-Sky_T* target column was found.",
        )

    if len(comparison_ids) < 2:
        return AijFluxDetection(
            compatible=False,
            target_ids=target_ids,
            comparison_ids=comparison_ids,
            source_columns=source_columns,
            error_columns=error_columns,
            time_column=time_column,
            warning="Comparison-star optimiser inactive: at least two Source-Sky_C* comparison stars are required.",
        )

    return AijFluxDetection(
        compatible=True,
        target_ids=target_ids,
        comparison_ids=comparison_ids,
        source_columns=source_columns,
        error_columns=error_columns,
        time_column=time_column,
        warning="",
    )


def _to_float_array(df: pd.DataFrame, column: str) -> np.ndarray:
    """Convert a table column to a float NumPy array."""
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _safe_median(values: np.ndarray) -> float:
    """Return a finite median, or NaN if the array has no finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmedian(finite))


def _normalised_flux(values: np.ndarray) -> np.ndarray:
    """Normalise a raw flux series by its robust median."""
    median = _safe_median(values)
    if not np.isfinite(median) or median <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return values / median


def _relative_scatter(values: np.ndarray) -> float:
    """Return robust relative scatter for a normalised stellar flux."""
    finite = values[np.isfinite(values)]
    if finite.size < 5:
        return float("inf")
    med = np.nanmedian(finite)
    mad = 1.4826 * np.nanmedian(np.abs(finite - med))
    if not np.isfinite(mad) or mad <= 0:
        std = np.nanstd(finite)
        return float(std) if np.isfinite(std) else float("inf")
    return float(mad)


def _valid_fraction(values: np.ndarray) -> float:
    """Fraction of finite positive values in a raw flux series."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(values) & (values > 0)) / values.size)


def preselect_comparison_stars(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    comparison_ids: Optional[Sequence[str]] = None,
    min_valid_fraction: float = 0.80,
) -> Tuple[List[str], List[str]]:
    """Reject comparison stars with unusable raw flux series."""
    if comparison_ids is None:
        comparison_ids = detection.comparison_ids

    accepted: List[str] = []
    rejected: List[str] = []

    scatter_values: List[Tuple[str, float]] = []
    for star_id in comparison_ids:
        column = detection.source_columns.get(star_id)
        if not column or column not in df.columns:
            rejected.append(star_id)
            continue
        raw = _to_float_array(df, column)
        if _valid_fraction(raw) < min_valid_fraction:
            rejected.append(star_id)
            continue
        norm = _normalised_flux(raw)
        scatter = _relative_scatter(norm)
        if not np.isfinite(scatter) or scatter <= 0:
            rejected.append(star_id)
            continue
        scatter_values.append((star_id, scatter))

    if not scatter_values:
        return [], rejected

    scatters = np.array([item[1] for item in scatter_values], dtype=float)
    median_scatter = float(np.nanmedian(scatters))
    loose_limit = max(0.20, 5.0 * median_scatter)

    for star_id, scatter in scatter_values:
        if scatter <= loose_limit:
            accepted.append(star_id)
        else:
            rejected.append(star_id)

    return sorted(accepted, key=_star_sort_key), sorted(rejected, key=_star_sort_key)


def make_differential_light_curve(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    target_id: str,
    comparison_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a differential light curve from raw AIJ source counts.

    Each star is normalised by its median.  The comparison ensemble is the
    unweighted mean of the normalised comparison stars.  This deliberately
    matches the curve written by the integrated aperture-photometry builder, so
    selecting all comparison stars in the main program reproduces the original
    ``rel_flux_T1`` curve.
    """
    target_id = _normalise_star_id(target_id)
    comparison_ids = [_normalise_star_id(sid) for sid in comparison_ids]

    if target_id not in detection.source_columns:
        raise ValueError(f"Target/check star {target_id} was not found in the AIJ Source-Sky columns.")
    if not comparison_ids:
        raise ValueError("At least one comparison star is required.")

    target_raw = _to_float_array(df, detection.source_columns[target_id])
    target_norm = _normalised_flux(target_raw)

    comp_norms: List[np.ndarray] = []
    comp_rel_errs: List[np.ndarray] = []

    for comp_id in comparison_ids:
        column = detection.source_columns.get(comp_id)
        if not column:
            continue
        raw = _to_float_array(df, column)
        norm = _normalised_flux(raw)
        scatter = _relative_scatter(norm)
        if not np.isfinite(scatter) or scatter <= 0:
            continue
        comp_norms.append(norm)

        err_column = detection.error_columns.get(comp_id)
        if err_column and err_column in df.columns:
            err_raw = _to_float_array(df, err_column)
            rel_err = np.divide(err_raw, raw, out=np.full_like(raw, np.nan, dtype=float), where=(raw > 0))
        else:
            rel_err = np.full_like(raw, np.nan, dtype=float)
        comp_rel_errs.append(rel_err)

    if not comp_norms:
        raise ValueError("No valid comparison-star fluxes were available for this subset.")

    comp_matrix = np.vstack(comp_norms)

    # Keep the manual comparison-star curves strictly reproducible.  The
    # aperture-photometry builder writes rel_flux_T1 using an AIJ-like ensemble:
    # every comparison star is normalised by its own median and the ensemble is
    # the unweighted mean of those normalised comparison curves.  The main
    # program must use the same rule when the user toggles comparison stars;
    # otherwise selecting all stars again would not recover the original curve.
    ensemble = np.nanmean(comp_matrix, axis=0)

    rel_flux = np.divide(target_norm, ensemble, out=np.full_like(target_norm, np.nan), where=np.isfinite(ensemble) & (ensemble != 0))
    rel_flux = rel_flux / _safe_median(rel_flux)

    # Approximate propagated relative error.  This is intended for plotting and
    # quick diagnostics, not as a substitute for a full photometric noise model.
    target_err_col = detection.error_columns.get(target_id)
    if target_err_col and target_err_col in df.columns:
        target_err_raw = _to_float_array(df, target_err_col)
        target_rel_err = np.divide(target_err_raw, target_raw, out=np.full_like(target_raw, np.nan, dtype=float), where=(target_raw > 0))
    else:
        target_rel_err = np.full_like(target_raw, np.nan, dtype=float)

    if comp_rel_errs:
        comp_rel_err_matrix = np.vstack(comp_rel_errs)
        comp_rel_err = np.nanmean(comp_rel_err_matrix, axis=0) / np.sqrt(max(1, len(comp_rel_errs)))
    else:
        comp_rel_err = np.full_like(target_raw, np.nan, dtype=float)

    rel_err = rel_flux * np.sqrt(target_rel_err ** 2 + comp_rel_err ** 2)
    if np.count_nonzero(np.isfinite(rel_err)) < 3:
        # Fallback to the robust point-to-point scatter if AIJ error columns are
        # missing or unusable.
        fallback = _relative_scatter(rel_flux)
        rel_err = np.full_like(rel_flux, fallback if np.isfinite(fallback) else np.nan)

    return rel_flux, rel_err, ensemble


def _sigma_clip_mask(values: np.ndarray, sigma: float = 5.0, max_iter: int = 4) -> np.ndarray:
    """Return a robust sigma-clipping mask."""
    mask = np.isfinite(values)
    for _ in range(max_iter):
        if np.count_nonzero(mask) < 5:
            break
        subset = values[mask]
        centre = np.nanmedian(subset)
        scale = 1.4826 * np.nanmedian(np.abs(subset - centre))
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nanstd(subset)
        if not np.isfinite(scale) or scale <= 0:
            break
        new_mask = np.isfinite(values) & (np.abs(values - centre) <= sigma * scale)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask


def _expected_transit_mask(
    x: np.ndarray,
    x_column: str,
    planet: Optional[object],
    duration_factor: float = 1.35,
) -> Optional[np.ndarray]:
    """Return a mask that is True outside the expected transit, when possible."""
    if planet is None or x is None:
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

    # AIJ's J.D.-2400000 column is common in exported tables.  It is close
    # enough for defining a transit mask, even before full BJD_TDB conversion.
    median_time = float(np.nanmedian(time[finite]))
    if median_time < 100000.0 and "2400000" in str(x_column):
        time = time + 2400000.0
    elif median_time < 100000.0 and median_time > 50000.0:
        # Many JD-minus-offset columns do not keep the exact original name.
        time = time + 2400000.0

    median_time = float(np.nanmedian(time[finite]))
    epoch = int(np.round((median_time - t0) / period))
    predicted_tmid = t0 + epoch * period
    half_duration_days = 0.5 * duration_factor * duration_hours / 24.0

    outside = np.abs(time - predicted_tmid) > half_duration_days
    outside &= np.isfinite(time)
    if np.count_nonzero(outside) < 5:
        return None
    return outside


def evaluate_light_curve_metric(
    x: np.ndarray,
    flux: np.ndarray,
    mask: Optional[np.ndarray] = None,
    polynomial_order: int = 1,
) -> CandidateMetric:
    """Evaluate robust scatter, red-noise proxy and objective for a light curve."""
    x = np.asarray(x, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(x) & np.isfinite(flux)
    if mask is not None and len(mask) == len(flux):
        finite &= np.asarray(mask, dtype=bool)

    if np.count_nonzero(finite) < max(6, polynomial_order + 3):
        return CandidateMetric(np.inf, np.inf, np.nan, np.nan, np.inf, int(np.count_nonzero(finite)))

    x_use = x[finite]
    y_use = flux[finite]
    med = np.nanmedian(y_use)
    if np.isfinite(med) and med != 0:
        y_use = y_use / med

    clip_mask = _sigma_clip_mask(y_use, sigma=5.0, max_iter=4)
    if np.count_nonzero(clip_mask) >= max(6, polynomial_order + 3):
        x_use = x_use[clip_mask]
        y_use = y_use[clip_mask]

    x_c = x_use - np.nanmedian(x_use)
    order = int(max(0, min(2, polynomial_order)))
    if order > 0 and np.count_nonzero(np.isfinite(x_c) & np.isfinite(y_use)) > order + 2:
        try:
            coeff = np.polyfit(x_c, y_use, order)
            trend = np.polyval(coeff, x_c)
        except Exception:
            trend = np.full_like(y_use, np.nanmedian(y_use))
    else:
        trend = np.full_like(y_use, np.nanmedian(y_use))

    residual = y_use - trend
    residual = residual[np.isfinite(residual)]
    if residual.size < 5:
        return CandidateMetric(np.inf, np.inf, np.nan, np.nan, np.inf, int(residual.size))

    rms = float(np.sqrt(np.nanmean(residual ** 2)))
    mad = float(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual))))

    if residual.size > 2 and np.nanstd(residual[:-1]) > 0 and np.nanstd(residual[1:]) > 0:
        autocorr = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    else:
        autocorr = float("nan")

    beta = _beta_factor(residual)
    penalty = 1.0
    if np.isfinite(autocorr):
        penalty += 0.35 * abs(autocorr)
    if np.isfinite(beta) and beta > 1.0:
        penalty += 0.25 * (beta - 1.0)
    objective = rms * penalty

    return CandidateMetric(
        rms_ppt=1000.0 * rms,
        mad_ppt=1000.0 * mad,
        autocorr_lag1=autocorr,
        beta_factor=beta,
        objective=1000.0 * objective,
        n_points=int(residual.size),
    )


def _beta_factor(residual: np.ndarray, bin_size: int = 5) -> float:
    """Return a simple time-correlated-noise beta factor."""
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    n = residual.size
    if n < bin_size * 3:
        return float("nan")
    sigma1 = np.nanstd(residual, ddof=1)
    if not np.isfinite(sigma1) or sigma1 <= 0:
        return float("nan")
    n_bins = n // bin_size
    trimmed = residual[: n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    sigma_bin = np.nanstd(binned, ddof=1)
    expected = sigma1 / np.sqrt(bin_size) * np.sqrt(n_bins / max(n_bins - 1, 1))
    if not np.isfinite(expected) or expected <= 0:
        return float("nan")
    return float(sigma_bin / expected)


def _metric_for_subset(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    x: np.ndarray,
    target_id: str,
    comparison_ids: Sequence[str],
    metric_mask: Optional[np.ndarray],
    polynomial_order: int,
) -> Tuple[CandidateMetric, np.ndarray, np.ndarray, np.ndarray]:
    """Build and score a subset differential light curve."""
    flux, flux_err, ensemble = make_differential_light_curve(df, detection, target_id, comparison_ids)
    metric = evaluate_light_curve_metric(x, flux, mask=metric_mask, polynomial_order=polynomial_order)
    return metric, flux, flux_err, ensemble


def optimise_comparison_stars(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    x: np.ndarray,
    x_column: str,
    current_flux: Optional[np.ndarray],
    target_id: str,
    mode: str = "Target light curve",
    check_id: str = "",
    planet: Optional[object] = None,
    mask_expected_transit: bool = True,
    min_stars: int = 2,
    max_stars: Optional[int] = None,
    polynomial_order: int = 1,
    improvement_threshold_percent: float = 0.5,
    allowed_comparisons: Optional[Sequence[str]] = None,
) -> ComparisonOptimisationResult:
    """Select a comparison-star subset using greedy backward elimination."""
    if not detection.compatible:
        raise ValueError(detection.warning or "The loaded table is not compatible with the comparison-star optimiser.")

    target_id = _normalise_star_id(target_id or detection.target_ids[0])
    check_id = _normalise_star_id(check_id) if check_id else ""

    if allowed_comparisons is None:
        candidate_pool = list(detection.comparison_ids)
    else:
        allowed_set = {_normalise_star_id(sid) for sid in allowed_comparisons}
        candidate_pool = [sid for sid in detection.comparison_ids if sid in allowed_set]

    if mode == "Check star stability":
        if not check_id:
            raise ValueError("Select a check star before using Check star stability mode.")
        working_target = check_id
        raw_comparisons = [sid for sid in candidate_pool if sid != check_id]
    else:
        working_target = target_id
        raw_comparisons = list(candidate_pool)

    accepted, rejected = preselect_comparison_stars(df, detection, raw_comparisons)
    if len(accepted) < min_stars:
        raise ValueError(
            f"Only {len(accepted)} usable comparison stars were found after pre-screening; "
            f"at least {min_stars} are required."
        )

    if max_stars is not None and max_stars > 0 and len(accepted) > max_stars:
        # Keep the most stable individual comparison stars as the starting set.
        ranked: List[Tuple[str, float]] = []
        for sid in accepted:
            metric, _, _, _ = _metric_for_subset(
                df, detection, x, working_target, [sid], None, polynomial_order
            )
            ranked.append((sid, metric.objective))
        ranked.sort(key=lambda item: item[1])
        accepted = sorted([sid for sid, _ in ranked[:max_stars]], key=_star_sort_key)
        rejected = sorted(set(rejected + [sid for sid, _ in ranked[max_stars:]]), key=_star_sort_key)

    metric_mask = None
    if mode == "Target light curve" and mask_expected_transit:
        metric_mask = _expected_transit_mask(x, x_column, planet)

    current_metric: Optional[CandidateMetric] = None
    if current_flux is not None and mode == "Target light curve":
        current_metric = evaluate_light_curve_metric(x, current_flux, mask=metric_mask, polynomial_order=polynomial_order)
        if not np.isfinite(current_metric.objective):
            current_metric = None

    all_metric, all_flux, all_err, all_ensemble = _metric_for_subset(
        df, detection, x, working_target, accepted, metric_mask, polynomial_order
    )

    selected = list(accepted)
    best_metric = all_metric
    best_flux = all_flux
    best_err = all_err
    best_ensemble = all_ensemble
    removed_sequence: List[str] = []

    while len(selected) > min_stars:
        trial_results: List[Tuple[str, CandidateMetric, np.ndarray, np.ndarray, np.ndarray]] = []
        for sid in selected:
            trial_subset = [candidate for candidate in selected if candidate != sid]
            metric, flux, flux_err, ensemble = _metric_for_subset(
                df, detection, x, working_target, trial_subset, metric_mask, polynomial_order
            )
            trial_results.append((sid, metric, flux, flux_err, ensemble))

        trial_results.sort(key=lambda item: item[1].objective)
        remove_id, trial_metric, trial_flux, trial_err, trial_ensemble = trial_results[0]
        improvement = (best_metric.objective - trial_metric.objective) / best_metric.objective * 100.0

        if np.isfinite(improvement) and improvement >= improvement_threshold_percent:
            selected.remove(remove_id)
            removed_sequence.append(remove_id)
            best_metric = trial_metric
            best_flux = trial_flux
            best_err = trial_err
            best_ensemble = trial_ensemble
        else:
            break

    improvement_vs_all = (all_metric.objective - best_metric.objective) / all_metric.objective * 100.0
    if current_metric is not None:
        improvement_vs_current = (current_metric.objective - best_metric.objective) / current_metric.objective * 100.0
    else:
        improvement_vs_current = None

    # The subset may be selected using a check star, but the generated output
    # curve should normally be the science target.  This lets users optimise a
    # stable check star and then send the target light curve directly to the
    # Transit tab.
    output_flux = best_flux
    output_err = best_err
    output_ensemble = best_ensemble
    if working_target != target_id:
        output_flux, output_err, output_ensemble = make_differential_light_curve(
            df, detection, target_id, selected
        )

    report = format_comparison_optimisation_report(
        method="Automatic backward-elimination optimiser",
        target_id=target_id,
        working_target=working_target,
        mode=mode,
        check_id=check_id,
        initial=accepted,
        rejected=rejected,
        selected=selected,
        removed_sequence=removed_sequence,
        current_metric=current_metric,
        all_metric=all_metric,
        best_metric=best_metric,
        improvement_vs_current=improvement_vs_current,
        improvement_vs_all=improvement_vs_all,
        metric_mask_used=metric_mask is not None,
        polynomial_order=polynomial_order,
    )

    return ComparisonOptimisationResult(
        target_id=target_id,
        mode=mode,
        check_id=check_id,
        selected_comparisons=selected,
        rejected_comparisons=rejected,
        initial_comparisons=accepted,
        removed_sequence=removed_sequence,
        current_curve_metric=current_metric,
        all_comparisons_metric=all_metric,
        optimised_metric=best_metric,
        improvement_vs_current_percent=improvement_vs_current,
        improvement_vs_all_percent=improvement_vs_all,
        x=np.asarray(x, dtype=float),
        optimised_flux=output_flux,
        optimised_flux_err=output_err,
        comparison_ensemble=output_ensemble,
        report=report,
    )


def build_manual_comparison_result(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    x: np.ndarray,
    x_column: str,
    current_flux: Optional[np.ndarray],
    target_id: str,
    selected_comparisons: Sequence[str],
    mode: str = "Target light curve",
    check_id: str = "",
    planet: Optional[object] = None,
    mask_expected_transit: bool = True,
    polynomial_order: int = 1,
) -> ComparisonOptimisationResult:
    """Build a light curve from a user-selected comparison-star subset.

    This is the manual counterpart of :func:`optimise_comparison_stars`.  It
    evaluates the same scatter metrics, writes the same output arrays and can
    therefore be sent to the Data/Transit tabs in exactly the same way as an
    automatically optimised subset.
    """
    if not detection.compatible:
        raise ValueError(detection.warning or "The loaded table is not compatible with comparison-star selection.")

    target_id = _normalise_star_id(target_id or detection.target_ids[0])
    check_id = _normalise_star_id(check_id) if check_id else ""
    selected = []
    seen = set()
    for sid in selected_comparisons:
        sid = _normalise_star_id(sid)
        if sid in detection.comparison_ids and sid not in seen:
            selected.append(sid)
            seen.add(sid)

    if mode == "Check star stability":
        if not check_id:
            raise ValueError("Select a check star before using Check star stability mode.")
        working_target = check_id
        selected_for_metric = [sid for sid in selected if sid != check_id]
    else:
        working_target = target_id
        selected_for_metric = list(selected)

    if not selected_for_metric:
        raise ValueError("Select at least one comparison star.")

    metric_mask = None
    if mode == "Target light curve" and mask_expected_transit:
        metric_mask = _expected_transit_mask(x, x_column, planet)

    current_metric: Optional[CandidateMetric] = None
    if current_flux is not None and mode == "Target light curve":
        current_metric = evaluate_light_curve_metric(x, current_flux, mask=metric_mask, polynomial_order=polynomial_order)
        if not np.isfinite(current_metric.objective):
            current_metric = None

    metric, metric_flux, metric_err, metric_ensemble = _metric_for_subset(
        df,
        detection,
        x,
        working_target,
        selected_for_metric,
        metric_mask,
        polynomial_order,
    )

    output_flux = metric_flux
    output_err = metric_err
    output_ensemble = metric_ensemble
    if working_target != target_id:
        output_flux, output_err, output_ensemble = make_differential_light_curve(
            df,
            detection,
            target_id,
            selected_for_metric,
        )

    if current_metric is not None:
        improvement_vs_current = (current_metric.objective - metric.objective) / current_metric.objective * 100.0
    else:
        improvement_vs_current = None

    report = format_comparison_optimisation_report(
        method="Manual comparison-star selection",
        target_id=target_id,
        working_target=working_target,
        mode=mode,
        check_id=check_id,
        initial=selected_for_metric,
        rejected=[],
        selected=selected_for_metric,
        removed_sequence=[],
        current_metric=current_metric,
        all_metric=metric,
        best_metric=metric,
        improvement_vs_current=improvement_vs_current,
        improvement_vs_all=0.0,
        metric_mask_used=metric_mask is not None,
        polynomial_order=polynomial_order,
    )

    return ComparisonOptimisationResult(
        target_id=target_id,
        mode=mode,
        check_id=check_id,
        selected_comparisons=selected_for_metric,
        rejected_comparisons=[],
        initial_comparisons=selected_for_metric,
        removed_sequence=[],
        current_curve_metric=current_metric,
        all_comparisons_metric=metric,
        optimised_metric=metric,
        improvement_vs_current_percent=improvement_vs_current,
        improvement_vs_all_percent=0.0,
        x=np.asarray(x, dtype=float),
        optimised_flux=output_flux,
        optimised_flux_err=output_err,
        comparison_ensemble=output_ensemble,
        report=report,
    )



def build_comparison_diagnostics(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    x: np.ndarray,
    selected_comparisons: Sequence[str],
    polynomial_order: int = 1,
) -> Dict[str, ComparisonDiagnosticCurve]:
    """Build leave-one-out relative light curves for active comparison stars.

    For each active comparison star ``Ci`` the diagnostic curve is computed as
    ``Ci / ensemble(other active comparisons)``.  The tested star is never used
    in its own denominator.  With fewer than two active comparison stars the
    diagnostic cannot identify which star is responsible for a mismatch, so no
    curve is generated.
    """
    if not detection.compatible:
        return {}

    selected: List[str] = []
    seen = set()
    for sid in selected_comparisons:
        sid = _normalise_star_id(sid)
        if sid in detection.comparison_ids and sid not in seen:
            selected.append(sid)
            seen.add(sid)
    selected = sorted(selected, key=_star_sort_key)

    diagnostics: Dict[str, ComparisonDiagnosticCurve] = {}
    if len(selected) < 2:
        return diagnostics

    for star_id in selected:
        reference_stars = [sid for sid in selected if sid != star_id]
        if not reference_stars:
            continue
        try:
            flux, flux_err, ensemble = make_differential_light_curve(
                df,
                detection,
                target_id=star_id,
                comparison_ids=reference_stars,
            )
            metric = evaluate_light_curve_metric(
                x,
                flux,
                mask=None,
                polynomial_order=polynomial_order,
            )
            warning = ""
            if not np.isfinite(metric.objective):
                warning = "metric unavailable"
            elif len(reference_stars) < 2:
                warning = "only one reference star; diagnostic is ambiguous"
            diagnostics[star_id] = ComparisonDiagnosticCurve(
                star_id=star_id,
                reference_stars=reference_stars,
                flux=flux,
                flux_err=flux_err,
                ensemble=ensemble,
                metric=metric,
                warning=warning,
            )
        except Exception as exc:
            diagnostics[star_id] = ComparisonDiagnosticCurve(
                star_id=star_id,
                reference_stars=reference_stars,
                flux=np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float),
                flux_err=np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float),
                ensemble=np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float),
                metric=CandidateMetric(np.inf, np.inf, np.nan, np.nan, np.inf, 0),
                warning=str(exc),
            )

    return diagnostics


def _diagnostic_flag(metric: CandidateMetric, reference_count: int) -> str:
    """Return a compact human-readable flag for one comparison diagnostic."""
    if metric is None or not np.isfinite(metric.objective):
        return "unusable"
    if reference_count < 2:
        return "ambiguous"
    if metric.rms_ppt <= 4.0 and metric.mad_ppt <= 3.0:
        return "OK"
    if metric.rms_ppt <= 8.0 and metric.mad_ppt <= 6.0:
        return "check"
    return "suspect"


def format_comparison_diagnostics_report(
    diagnostics: Dict[str, ComparisonDiagnosticCurve],
) -> str:
    """Return a readable report for comparison-star leave-one-out curves."""
    lines = [
        "Comparison-star diagnostics",
        "",
        "Each comparison star is divided by the ensemble of the other active comparison stars.",
        "This leave-one-out test helps identify noisy or possibly variable comparison stars.",
    ]
    if not diagnostics:
        lines.extend(
            [
                "",
                "Not enough active comparison stars for leave-one-out diagnostics.",
                "Use at least two active comparison stars; three or more are recommended.",
            ]
        )
        return "\n".join(lines)

    if len(diagnostics) < 3:
        lines.extend(
            [
                "",
                "Note: diagnostics are most reliable with three or more comparison stars.",
                "With only two active comparisons, a mismatch can be detected but the responsible star is ambiguous.",
            ]
        )

    lines.extend(["", "Star   Ref. stars          RMS[ppt]  MAD[ppt]  beta   lag-1    flag"])
    for star_id in sorted(diagnostics, key=_star_sort_key):
        diag = diagnostics[star_id]
        metric = diag.metric
        flag = _diagnostic_flag(metric, len(diag.reference_stars))
        beta = "nan" if not np.isfinite(metric.beta_factor) else f"{metric.beta_factor:5.2f}"
        lag = "nan" if not np.isfinite(metric.autocorr_lag1) else f"{metric.autocorr_lag1:+6.3f}"
        rms = "nan" if not np.isfinite(metric.rms_ppt) else f"{metric.rms_ppt:8.3f}"
        mad = "nan" if not np.isfinite(metric.mad_ppt) else f"{metric.mad_ppt:8.3f}"
        refs = ",".join(diag.reference_stars)
        if len(refs) > 18:
            refs = refs[:15] + "..."
        line = f"{star_id:<5s}  {refs:<18s} {rms} {mad} {beta} {lag}  {flag}"
        if diag.warning:
            line += f"  ({diag.warning})"
        lines.append(line)

    lines.extend(
        [
            "",
            "Generated diagnostic columns",
            "PhotoCurve_compdiag_time",
            "PhotoCurve_compdiag_<star>_flux",
            "PhotoCurve_compdiag_<star>_err",
            "",
            "Tip: select a diagnostic star and enable 'Plot selected comp' to inspect its relative curve.",
        ]
    )
    return "\n".join(lines)

def _format_metric(metric: Optional[CandidateMetric]) -> str:
    """Format a metric block for the report."""
    if metric is None:
        return "not available"
    beta = "nan" if not np.isfinite(metric.beta_factor) else f"{metric.beta_factor:.2f}"
    ac = "nan" if not np.isfinite(metric.autocorr_lag1) else f"{metric.autocorr_lag1:+.3f}"
    return (
        f"RMS={metric.rms_ppt:.3f} ppt, MAD={metric.mad_ppt:.3f} ppt, "
        f"objective={metric.objective:.3f}, beta={beta}, lag-1={ac}, N={metric.n_points}"
    )


def format_comparison_optimisation_report(
    target_id: str,
    working_target: str,
    mode: str,
    check_id: str,
    initial: Sequence[str],
    rejected: Sequence[str],
    selected: Sequence[str],
    removed_sequence: Sequence[str],
    current_metric: Optional[CandidateMetric],
    all_metric: CandidateMetric,
    best_metric: CandidateMetric,
    improvement_vs_current: Optional[float],
    improvement_vs_all: float,
    metric_mask_used: bool,
    polynomial_order: int,
    method: str = "Automatic backward-elimination optimiser",
) -> str:
    """Build a readable comparison-star optimisation report."""
    lines = [
        "Comparison-star optimiser",
        "",
        f"Method: {method}",
        f"Mode: {mode}",
        f"Target star: {target_id}",
        f"Optimised star: {working_target}",
    ]
    if check_id:
        lines.append(f"Check star: {check_id}")
    lines.extend(
        [
            f"Expected-transit mask used: {'yes' if metric_mask_used else 'no'}",
            f"Polynomial order used for scatter metric: {polynomial_order}",
            "",
            f"Initial usable comparison stars ({len(initial)}): {', '.join(initial) if initial else 'none'}",
            f"Automatically rejected stars ({len(rejected)}): {', '.join(rejected) if rejected else 'none'}",
            f"Removed by optimiser ({len(removed_sequence)}): {', '.join(removed_sequence) if removed_sequence else 'none'}",
            f"Selected comparison stars ({len(selected)}): {', '.join(selected) if selected else 'none'}",
            "",
            "Quality metrics",
            f"Current selected light curve: {_format_metric(current_metric)}",
            f"All usable comparisons:       {_format_metric(all_metric)}",
            f"Optimised subset:            {_format_metric(best_metric)}",
            "",
            f"Improvement versus all usable comparisons: {improvement_vs_all:+.1f}%",
        ]
    )
    if improvement_vs_current is not None:
        lines.append(f"Improvement versus current selected light curve: {improvement_vs_current:+.1f}%")
    else:
        lines.append("Improvement versus current selected light curve: not available")

    lines.extend(
        [
            "",
            "Generated columns",
            "PhotoCurve_compopt_time",
            "PhotoCurve_compopt_flux",
            "PhotoCurve_compopt_err",
            "PhotoCurve_compopt_ensemble",
            "",
            "Notes",
            "The optimiser uses raw AIJ Source-Sky fluxes.  It does not change the original table.",
            "For transit work, the metric is measured outside the predicted transit when a catalogue ephemeris is available.",
        ]
    )
    return "\n".join(lines)
