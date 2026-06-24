"""Empirical exoplanet transit diagnostic fitting.

These routines are intentionally conservative: they are designed to assess the
quality of an observed light curve and to compare it with a catalogue prediction,
not to provide a full physical characterisation of the exoplanetary system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .exoplanet_catalog import PlanetTransitParameters
from .numeric_utils import finite_mask
from .time_conversion_utils import convert_input_times_to_bjd_tdb, get_observatory_parameters

try:
    from scipy.optimize import least_squares
    from scipy.stats import shapiro
except Exception:  # pragma: no cover - optional runtime dependency fallback
    least_squares = None
    shapiro = None

try:
    import batman  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency fallback
    batman = None


BASELINE_ORDERS = ["Constant", "Linear", "Quadratic"]
TIME_SYSTEMS = ["BJD_TDB", "JD_UTC", "HJD_UTC", "Other"]
FILTERS = [
    "Clear", "L", "B", "V", "R", "I", "G", "g'", "r'", "i'", "z'",
    "Sloan g", "Sloan r", "Sloan i", "Sloan z", "TESS", "Other",
]

TIMESTAMP_REFERENCES = ["Exposure start", "Mid-exposure", "Exposure end"]
MODEL_ENGINES = ["Auto", "Batman physical", "Empirical"]


def timestamp_correction_seconds(timestamp_reference: str, exposure_time_s: float) -> float:
    """Return the correction needed to convert the input time to mid-exposure.

    The transit timing should be measured at the middle of each exposure. If the
    input timestamp refers to the exposure start, add EXPTIME/2. If it refers to
    the exposure end, subtract EXPTIME/2.
    """
    try:
        exptime = float(exposure_time_s)
    except Exception:
        return 0.0

    if not np.isfinite(exptime) or exptime <= 0:
        return 0.0

    ref = str(timestamp_reference).strip().lower()
    if ref == "exposure start":
        return +0.5 * exptime
    if ref == "exposure end":
        return -0.5 * exptime
    return 0.0


def apply_timestamp_correction(
    time_days: np.ndarray,
    timestamp_reference: str,
    exposure_time_s: float,
) -> Tuple[np.ndarray, float]:
    """Return mid-exposure times and the applied correction in seconds."""
    correction_s = timestamp_correction_seconds(timestamp_reference, exposure_time_s)
    return np.asarray(time_days, dtype=float) + correction_s / 86400.0, correction_s


@dataclass
class TransitDiagnosticResult:
    """Results of the empirical transit diagnostic fit."""

    planet_name: str
    filter_name: str
    exposure_time_s: float
    timestamp_reference: str
    time_correction_s: float
    model_integration_subsamples: int
    time_system: str
    bjd_conversion_applied: bool
    bjd_correction_min: float
    time_conversion_note: str
    target_ra_deg: float
    target_dec_deg: float
    observatory_lat_deg: float
    observatory_lon_deg: float
    observatory_alt_m: float
    baseline_order: str
    display_mode: str
    model_engine: str
    model_engine_note: str
    limb_darkening_law: str
    limb_darkening_coefficients: Tuple[float, float]
    limb_darkening_source: str
    catalogue_source: str
    catalogue_notes: str
    catalogue_type: str
    catalogue_depth_source: str
    catalogue_rprs_source: str
    catalogue_geometry_source: str
    catalogue_stellar_source: str
    catalogue_t0_bjd_tdb: float
    catalogue_period_days: float
    predicted_epoch: int
    tmid_catalogue_predicted: float
    tmid_reference_source: str
    tmid_reference_override: float
    tmid_predicted: float
    tmid_observed: float
    oc_minutes: float
    oc_unc_minutes: float
    catalogue_depth_ppt: float
    geometric_depth_ppt: float
    expected_depth_ppt: float
    observed_depth_ppt: float
    expected_rp_rs: float
    observed_rp_rs: float
    depth_scale: float
    depth_scale_unc: float
    expected_duration_hours: float
    observed_duration_hours: float
    duration_scale: float
    duration_scale_unc: float
    residual_rms_ppt: float
    median_error_ppt: float
    rms_over_median_error: float
    transit_snr: float
    lag1_autocorr: float
    shapiro_p: float
    n_points: int
    n_in_transit: int
    cadence_median_min: float
    reduced_chi2: float
    detection_label: str
    depth_label: str
    duration_label: str
    timing_label: str
    scatter_label: str
    autocorr_label: str
    warning: str
    corrected_time: np.ndarray
    baseline: np.ndarray
    detrended_flux: np.ndarray
    detrended_flux_err: np.ndarray
    fit_transit_model: np.ndarray
    expected_transit_model: np.ndarray
    fit_full_model: np.ndarray
    expected_full_model: np.ndarray
    full_residuals: np.ndarray
    detrended_residuals: np.ndarray
    model: np.ndarray
    expected_model: np.ndarray
    residuals: np.ndarray
    keep_mask: np.ndarray


# -----------------------------------------------------------------------------
# Empirical transit template
# -----------------------------------------------------------------------------

def smooth_transit_profile(
    time_days: np.ndarray,
    tmid: float,
    duration_hours: float,
    ingress_fraction: float = 0.18,
    bottom_curvature: float = 0.08,
) -> np.ndarray:
    """Return a smooth empirical transit profile between 0 and 1.

    The profile is not a physical Mandel & Agol model. It is a rounded trapezoid
    designed for automatic observational diagnostics:

    * 0 outside the transit;
    * a smooth cosine ingress and egress;
    * a mildly curved bottom to avoid a too-boxy appearance.

    This gives a more ExoClock-like diagnostic model while requiring only the
    catalogue duration and depth.
    """
    t = np.asarray(time_days, dtype=float)
    duration_days = max(float(duration_hours) / 24.0, 1e-8)
    half_duration = 0.5 * duration_days

    ingress_fraction = float(np.clip(ingress_fraction, 0.05, 0.45))
    ingress_days = max(duration_days * ingress_fraction, 1e-8)
    ingress_days = min(ingress_days, 0.95 * half_duration)
    half_flat = max(half_duration - ingress_days, 0.0)

    x = np.abs(t - float(tmid))
    profile = np.zeros_like(x, dtype=float)

    bottom_curvature = float(np.clip(bottom_curvature, 0.0, 0.30))
    edge_level = 1.0 - bottom_curvature

    if half_flat > 1e-10:
        inner = x <= half_flat
        if np.any(inner):
            r = x[inner] / half_flat
            profile[inner] = 1.0 - bottom_curvature * r**2

        ingress = (x > half_flat) & (x <= half_duration)
        if np.any(ingress):
            s = (x[ingress] - half_flat) / ingress_days
            profile[ingress] = edge_level * 0.5 * (1.0 + np.cos(np.pi * s))
    else:
        # Grazing/V-shaped fallback when ingress dominates the whole transit.
        inside = x <= half_duration
        if np.any(inside):
            r = x[inside] / half_duration
            profile[inside] = np.clip(1.0 - r**2, 0.0, 1.0)

    return np.clip(profile, 0.0, 1.0)


def exposure_integrated_transit_profile(
    time_days: np.ndarray,
    tmid: float,
    duration_hours: float,
    exposure_time_s: float,
    n_subsamples: int = 7,
) -> np.ndarray:
    """Return the empirical transit profile averaged over each exposure.

    The input times are assumed to be mid-exposure times. The function samples
    the model across the finite exposure and averages it, which softens ingress
    and egress for long integrations.
    """
    t = np.asarray(time_days, dtype=float)
    try:
        exptime = float(exposure_time_s)
    except Exception:
        exptime = 0.0

    n_subsamples = int(max(1, n_subsamples))
    if (not np.isfinite(exptime)) or exptime <= 0 or n_subsamples <= 1:
        return smooth_transit_profile(t, tmid, duration_hours)

    offsets = np.linspace(-0.5, 0.5, n_subsamples) * exptime / 86400.0
    profiles = [smooth_transit_profile(t + offset, tmid, duration_hours) for offset in offsets]
    return np.nanmean(np.vstack(profiles), axis=0)



# -----------------------------------------------------------------------------
# Optional physical transit model with batman
# -----------------------------------------------------------------------------

def _limb_darkening_info(filter_name: str) -> Tuple[str, Tuple[float, float], str]:
    """Return approximate limb-darkening information for a given filter.

    These coefficients are deliberately simple defaults for observational
    diagnostics. They are not a replacement for stellar-atmosphere based
    coefficients. The goal is to obtain a more realistic transit shape than the
    empirical rounded box while keeping the GUI simple and automatic.
    """
    filt = str(filter_name).strip().lower()
    law = "quadratic"

    table = {
        "b": (0.55, 0.18),
        "v": (0.48, 0.20),
        "g": (0.48, 0.20),
        "g'": (0.48, 0.20),
        "sloan g": (0.48, 0.20),
        "r": (0.38, 0.22),
        "r'": (0.38, 0.22),
        "sloan r": (0.38, 0.22),
        "i": (0.28, 0.22),
        "i'": (0.28, 0.22),
        "sloan i": (0.28, 0.22),
        "z": (0.22, 0.20),
        "z'": (0.22, 0.20),
        "sloan z": (0.22, 0.20),
        "tess": (0.30, 0.20),
    }

    if filt in table:
        coeffs = table[filt]
        source = "internal approximate table for the selected filter"
    else:
        # Clear/L/other broad filters: middle-of-the-road optical coefficients.
        coeffs = (0.40, 0.20)
        source = "internal approximate broad-optical default"

    return law, coeffs, source


def _limb_darkening_coefficients(filter_name: str) -> list[float]:
    """Return approximate quadratic limb-darkening coefficients."""
    _, coeffs, _ = _limb_darkening_info(filter_name)
    return [float(coeffs[0]), float(coeffs[1])]


def physical_model_is_available(planet: PlanetTransitParameters) -> bool:
    """Return True if batman can be used for this planet."""
    if batman is None:
        return False
    required = [planet.rp_rs, planet.a_rs, planet.inclination_deg]
    if not all(np.isfinite(value) for value in required):
        return False
    if planet.rp_rs <= 0 or planet.a_rs <= 1.0 or planet.inclination_deg <= 0:
        return False
    return True


def choose_model_engine(planet: PlanetTransitParameters, values: Dict[str, object]) -> Tuple[str, str]:
    """Choose the internal transit model engine."""
    requested = str(values.get("-TR_MODEL_ENGINE-", "Auto"))
    can_use_batman = physical_model_is_available(planet)

    if requested == "Empirical":
        return "Empirical", "Using the empirical rounded-transit template selected by the user."

    if requested == "Batman physical":
        if can_use_batman:
            return "Batman physical", "Using batman with catalogue geometry and approximate limb darkening."
        reason = "batman is not installed" if batman is None else "the catalogue lacks a/Rs or inclination"
        return "Empirical", f"Requested batman, but {reason}; falling back to the empirical template."

    # Auto mode.
    if can_use_batman:
        return "Batman physical", "Auto mode selected batman because catalogue geometry is available."
    reason = "batman is not installed" if batman is None else "the catalogue lacks a/Rs or inclination"
    return "Empirical", f"Auto mode used the empirical template because {reason}."


def batman_instant_flux(
    time_days: np.ndarray,
    planet: PlanetTransitParameters,
    tmid: float,
    rp_rs: float,
    filter_name: str,
) -> np.ndarray:
    """Return a batman transit model without finite-exposure averaging."""
    if batman is None:
        raise RuntimeError("batman is not installed.")

    t = np.asarray(time_days, dtype=float)
    params = batman.TransitParams()
    params.t0 = float(tmid)
    params.per = float(planet.period_days)
    params.rp = float(max(rp_rs, 1e-6))
    params.a = float(planet.a_rs)
    params.inc = float(planet.inclination_deg)

    ecc = float(planet.ecc) if np.isfinite(planet.ecc) else 0.0
    params.ecc = float(np.clip(ecc, 0.0, 0.95))
    params.w = float(planet.omega_deg) if np.isfinite(planet.omega_deg) else 90.0
    params.u = _limb_darkening_coefficients(filter_name)
    params.limb_dark = "quadratic"

    model = batman.TransitModel(params, t)
    return np.asarray(model.light_curve(params), dtype=float)


def exposure_integrated_batman_flux(
    time_days: np.ndarray,
    planet: PlanetTransitParameters,
    tmid: float,
    rp_rs: float,
    filter_name: str,
    exposure_time_s: float,
    n_subsamples: int = 7,
) -> np.ndarray:
    """Return a batman model averaged over each finite exposure."""
    t = np.asarray(time_days, dtype=float)
    try:
        exptime = float(exposure_time_s)
    except Exception:
        exptime = 0.0

    n_subsamples = int(max(1, n_subsamples))
    if (not np.isfinite(exptime)) or exptime <= 0 or n_subsamples <= 1:
        return batman_instant_flux(t, planet, tmid, rp_rs, filter_name)

    offsets = np.linspace(-0.5, 0.5, n_subsamples) * exptime / 86400.0
    fluxes = [batman_instant_flux(t + offset, planet, tmid, rp_rs, filter_name) for offset in offsets]
    return np.nanmean(np.vstack(fluxes), axis=0)


def transit_flux_model(
    time_days: np.ndarray,
    planet: PlanetTransitParameters,
    tmid: float,
    depth_scale: float,
    duration_scale: float,
    filter_name: str,
    exposure_time_s: float,
    n_subsamples: int,
    model_engine: str,
) -> np.ndarray:
    """Return the internal transit model in relative flux.

    For the physical engine, depth_scale modifies Rp/Rs as sqrt(depth_scale),
    while duration_scale stretches the time axis around Tmid.  This keeps the
    fit diagnostic and automatic: catalogue geometry is used, but the observed
    duration can still differ from the prediction, similarly to ExoClock-style
    diagnostics.
    """
    t = np.asarray(time_days, dtype=float)
    depth_scale = float(max(depth_scale, 0.0))
    duration_scale = float(np.clip(duration_scale, 0.1, 10.0))

    if model_engine == "Batman physical" and physical_model_is_available(planet):
        rp = float(planet.rp_rs) * np.sqrt(depth_scale)
        # Stretching the time coordinate changes the apparent duration without
        # fitting a/Rs or inclination freely, which would be unstable for many
        # short ground-based light curves.
        t_model = float(tmid) + (t - float(tmid)) / duration_scale
        exptime_model_s = float(exposure_time_s) / duration_scale if duration_scale > 0 else float(exposure_time_s)
        return exposure_integrated_batman_flux(
            t_model,
            planet,
            tmid=float(tmid),
            rp_rs=rp,
            filter_name=filter_name,
            exposure_time_s=exptime_model_s,
            n_subsamples=n_subsamples,
        )

    duration = planet.duration_hours * duration_scale
    profile = exposure_integrated_transit_profile(
        t,
        tmid,
        duration,
        exposure_time_s=exposure_time_s,
        n_subsamples=n_subsamples,
    )
    return 1.0 - planet.depth_relative * depth_scale * profile


def _model_depth_ppt_from_transit_model(
    planet: PlanetTransitParameters,
    tmid: float,
    depth_scale: float,
    duration_scale: float,
    filter_name: str,
    exposure_time_s: float,
    n_subsamples: int,
    model_engine: str,
) -> float:
    """Measure the apparent depth directly from the transit model.

    This is deliberately based on the actual model flux, not on the approximation
    catalogue_depth * depth_scale.  For physical batman models, especially
    high-impact or grazing transits, Rp/Rs, a/Rs, inclination and limb darkening
    can make the apparent depth substantially different from simple scalings.
    """
    try:
        duration_hours = float(planet.duration_hours) * float(duration_scale)
    except Exception:
        duration_hours = np.nan

    if not np.isfinite(duration_hours) or duration_hours <= 0:
        duration_hours = float(planet.duration_hours) if np.isfinite(planet.duration_hours) else 2.0

    duration_days = max(duration_hours / 24.0, 1.0 / 24.0)
    half_window_days = max(1.25 * duration_days, 0.08)

    # A dense local grid is more reliable than using the observed cadence, because
    # sparse observations can miss the exact minimum of a grazing transit.
    time_grid = float(tmid) + np.linspace(-half_window_days, half_window_days, 2501)

    try:
        model_flux = transit_flux_model(
            time_grid,
            planet=planet,
            tmid=float(tmid),
            depth_scale=float(depth_scale),
            duration_scale=float(duration_scale),
            filter_name=filter_name,
            exposure_time_s=float(exposure_time_s),
            n_subsamples=int(max(1, n_subsamples)),
            model_engine=model_engine,
        )
    except Exception:
        return np.nan

    model_flux = np.asarray(model_flux, dtype=float)
    finite = model_flux[np.isfinite(model_flux)]
    if finite.size == 0:
        return np.nan

    # The transit models are normalised close to 1 outside transit.  Use a high
    # percentile for the continuum so that numerical edge effects or finite
    # exposure averaging do not bias the depth estimate.
    continuum = float(np.nanpercentile(finite, 95.0))
    minimum = float(np.nanmin(finite))
    depth_ppt = (continuum - minimum) * 1000.0

    if not np.isfinite(depth_ppt) or depth_ppt < 0:
        return np.nan
    return float(depth_ppt)


# Backwards-compatible name used by older parts of the code/package.
def smooth_box_transit_profile(
    time_days: np.ndarray,
    tmid: float,
    duration_hours: float,
    ingress_fraction: float = 0.18,
) -> np.ndarray:
    """Alias for the improved empirical transit profile."""
    return smooth_transit_profile(time_days, tmid, duration_hours, ingress_fraction)


def expected_transit_model(
    time_days: np.ndarray,
    planet: PlanetTransitParameters,
    tmid: float,
    depth_scale: float = 1.0,
    duration_scale: float = 1.0,
    exposure_time_s: float = 0.0,
    n_subsamples: int = 7,
) -> np.ndarray:
    """Return an empirical expected transit model in relative flux."""
    duration = planet.duration_hours * float(duration_scale)
    profile = exposure_integrated_transit_profile(
        time_days,
        tmid,
        duration,
        exposure_time_s=exposure_time_s,
        n_subsamples=n_subsamples,
    )
    return 1.0 - planet.depth_relative * float(depth_scale) * profile


def nearest_predicted_tmid(time_days: np.ndarray, planet: PlanetTransitParameters) -> Tuple[float, int]:
    """Return the predicted mid-transit nearest to the median observed time."""
    finite_time = np.asarray(time_days, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    if finite_time.size == 0:
        raise ValueError("No finite time values are available for the transit diagnostics.")

    epoch = int(np.round((np.nanmedian(finite_time) - planet.t0_bjd_tdb) / planet.period_days))
    tmid_pred = planet.t0_bjd_tdb + epoch * planet.period_days
    return float(tmid_pred), epoch


def parse_tmid_override(values: Dict[str, object]) -> Tuple[float, str]:
    """Parse an optional user-supplied predicted Tmid in BJD_TDB.

    The override is intended for cases where the user wants to compare the
    fitted observed mid-transit against an external prediction, for example the
    T prediction mid value provided by ExoClock. The value must be an absolute
    BJD_TDB timestamp, not an offset or phase.
    """
    text = str(values.get("-TR_TMID_OVERRIDE-", "")).strip()
    if not text:
        return np.nan, ""

    try:
        value = float(text)
    except ValueError:
        return np.nan, f"The optional predicted Tmid override could not be parsed: {text!r}. It was ignored."

    if not np.isfinite(value):
        return np.nan, f"The optional predicted Tmid override is not finite: {text!r}. It was ignored."

    return value, ""


def baseline_terms(time_days: np.ndarray, tmid_ref: float, order: str) -> np.ndarray:
    """Build baseline design terms centred on the predicted mid-transit."""
    dt = np.asarray(time_days, dtype=float) - float(tmid_ref)
    if order == "Quadratic":
        return np.vstack([np.ones_like(dt), dt, dt**2]).T
    if order == "Linear":
        return np.vstack([np.ones_like(dt), dt]).T
    return np.vstack([np.ones_like(dt)]).T


def _initial_baseline(y: np.ndarray, n_terms: int) -> np.ndarray:
    coeffs = np.zeros(n_terms, dtype=float)
    finite_y = y[np.isfinite(y)]
    if finite_y.size:
        # Use the upper half of the flux distribution as a conservative estimate
        # of the out-of-transit level.
        coeffs[0] = float(np.nanpercentile(finite_y, 75))
    else:
        coeffs[0] = 1.0
    return coeffs


def estimate_transit_centre_from_flux(time_days: np.ndarray, flux: np.ndarray) -> float:
    """Estimate a first-guess transit centre directly from the observed flux."""
    mask = finite_mask(time_days, flux)
    if not np.any(mask):
        return np.nan

    t = np.asarray(time_days[mask], dtype=float)
    y = np.asarray(flux[mask], dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    n = len(y)
    if n < 5:
        return float(t[int(np.nanargmin(y))])

    window = max(3, min(9, n // 6))
    if window % 2 == 0:
        window += 1
    half = window // 2

    smoothed = np.empty_like(y)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = np.nanmedian(y[lo:hi])

    return float(t[int(np.nanargmin(smoothed))])


def estimate_duration_scale_from_flux(
    time_days: np.ndarray,
    flux: np.ndarray,
    expected_duration_hours: float,
) -> float:
    """Estimate a first-guess duration scale from the observed flux width."""
    mask = finite_mask(time_days, flux)
    if not np.any(mask):
        return 1.0

    t = np.asarray(time_days[mask], dtype=float)
    y = np.asarray(flux[mask], dtype=float)
    if t.size < 8:
        return 1.0

    high = np.nanpercentile(y, 80)
    low = np.nanpercentile(y, 10)
    depth = high - low
    if not np.isfinite(depth) or depth <= 0:
        return 1.0

    threshold = high - 0.45 * depth
    inside = y < threshold
    if np.count_nonzero(inside) < 3:
        return 1.0

    observed_duration_days = float(np.nanmax(t[inside]) - np.nanmin(t[inside]))
    expected_duration_days = max(float(expected_duration_hours) / 24.0, 1e-8)
    return float(np.clip(observed_duration_days / expected_duration_days, 0.60, 1.60))


def transit_template_overlaps_data(
    time_days: np.ndarray,
    tmid: float,
    duration_hours: float,
) -> bool:
    """Return True when the expected transit template overlaps the data span."""
    profile = smooth_transit_profile(time_days, tmid, duration_hours)
    finite_profile = profile[np.isfinite(profile)]
    return bool(finite_profile.size and np.nanmax(finite_profile) > 0.02)


def _initial_depth_scale(flux: np.ndarray, expected_depth_relative: float) -> float:
    """Estimate a conservative initial depth-scale value from the data."""
    finite_flux = flux[np.isfinite(flux)]
    if finite_flux.size == 0 or expected_depth_relative <= 0:
        return 1.0
    observed_depth = np.nanpercentile(finite_flux, 80) - np.nanpercentile(finite_flux, 5)
    return float(np.clip(observed_depth / expected_depth_relative, 0.15, 3.0))


def _safe_uncertainties(jac: np.ndarray, residual_vector: np.ndarray, dof: int) -> Optional[np.ndarray]:
    """Estimate parameter uncertainties from the least-squares Jacobian."""
    try:
        # Use a pseudo-inverse because the baseline and depth can be partially
        # degenerate for partial transits.
        jtj_inv = np.linalg.pinv(jac.T @ jac)
        if dof > 0:
            chi2_red = float(np.sum(residual_vector**2) / dof)
            jtj_inv *= chi2_red
        return np.sqrt(np.diag(jtj_inv))
    except Exception:
        return None


def _lag1_autocorrelation(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return np.nan
    a = finite[:-1] - np.nanmean(finite[:-1])
    b = finite[1:] - np.nanmean(finite[1:])
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom <= 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def _quality_labels(
    transit_snr: float,
    depth_scale: float,
    duration_scale: float,
    oc_min: float,
    oc_unc_min: float,
    rms_over_err: float,
    autocorr: float,
) -> Tuple[str, str, str, str, str, str]:
    """Assign simple observational-quality labels."""
    if not np.isfinite(transit_snr):
        detection = "not assessed"
    elif transit_snr >= 10.0:
        detection = "strong detection"
    elif transit_snr >= 5.0:
        detection = "moderate detection"
    else:
        detection = "weak detection"

    if not np.isfinite(depth_scale):
        depth = "not assessed"
    elif 0.75 <= depth_scale <= 1.25:
        depth = "consistent with expected depth"
    elif 0.50 <= depth_scale <= 1.50:
        depth = "possible depth difference"
    else:
        depth = "large depth difference"

    if not np.isfinite(duration_scale):
        duration = "not assessed"
    elif 0.75 <= duration_scale <= 1.25:
        duration = "consistent with expected duration"
    elif 0.55 <= duration_scale <= 1.60:
        duration = "possible duration difference"
    else:
        duration = "large duration difference"

    if np.isfinite(oc_unc_min) and oc_unc_min > 0:
        sigma = abs(oc_min) / oc_unc_min
        if sigma < 2.0:
            timing = "consistent timing"
        elif sigma < 3.0:
            timing = "possible timing offset"
        else:
            timing = "significant timing offset"
    else:
        if np.isfinite(oc_min) and abs(oc_min) < 5.0:
            timing = "small timing offset"
        elif np.isfinite(oc_min) and abs(oc_min) < 15.0:
            timing = "possible timing offset"
        else:
            timing = "large timing offset"

    if not np.isfinite(rms_over_err):
        scatter = "not assessed"
    elif rms_over_err < 1.5:
        scatter = "good scatter"
    elif rms_over_err < 2.5:
        scatter = "acceptable scatter"
    else:
        scatter = "excess scatter or underestimated errors"

    if not np.isfinite(autocorr):
        autocorr_label = "not assessed"
    elif abs(autocorr) < 0.30:
        autocorr_label = "low correlated noise"
    elif abs(autocorr) < 0.50:
        autocorr_label = "possible correlated noise"
    else:
        autocorr_label = "significant correlated noise"

    return detection, depth, duration, timing, scatter, autocorr_label


def run_transit_diagnostics(
    time_days: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray],
    planet: PlanetTransitParameters,
    values: Dict[str, object],
    keep_mask: Optional[np.ndarray] = None,
) -> TransitDiagnosticResult:
    """Run an automatic empirical transit diagnostic fit."""
    if least_squares is None:
        raise RuntimeError("Transit diagnostics require scipy. Please install scipy and try again.")

    time_input_days = np.asarray(time_days, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err_arr = None if flux_err is None else np.asarray(flux_err, dtype=float)

    order = str(values.get("-TR_BASELINE-", "Linear"))
    display_mode = str(values.get("-TR_DISPLAY_MODE-", "Detrended flux"))
    fit_tmid = bool(values.get("-TR_FIT_TMID-", True))
    fit_depth = bool(values.get("-TR_FIT_DEPTH-", True))
    # Kept automatic and hidden from the GUI for simplicity.
    fit_duration = bool(values.get("-TR_FIT_DURATION-", True))
    filter_name = str(values.get("-TR_FILTER-", "Other"))
    model_engine, model_engine_note = choose_model_engine(planet, values)
    limb_darkening_law, limb_darkening_coeffs, limb_darkening_source = _limb_darkening_info(filter_name)
    if model_engine != "Batman physical":
        limb_darkening_law = "not used"
        limb_darkening_source = "not used by the empirical transit template"
    time_system = str(values.get("-TR_TIME_SYSTEM-", "BJD_TDB"))
    timestamp_reference = str(values.get("-TR_TIMESTAMP_REF-", "Mid-exposure"))
    try:
        exposure_time_s = float(str(values.get("-TR_EXPTIME-", "0") or "0"))
    except ValueError:
        exposure_time_s = np.nan

    time_mid_days, time_correction_s = apply_timestamp_correction(
        time_input_days,
        timestamp_reference,
        exposure_time_s,
    )

    # Transit ephemerides in the offline catalogue are assumed to be in BJD_TDB.
    # Therefore all timing diagnostics must run in BJD_TDB.  If the user
    # supplies JD_UTC, convert it using the target RA/Dec and observatory
    # coordinates.
    obs_lat = obs_lon = obs_alt = np.nan
    if str(time_system).strip().upper() == "JD_UTC":
        obs_lat, obs_lon, obs_alt = get_observatory_parameters(values)
    else:
        # Keep these in the report even when they were not used.
        try:
            obs_lat, obs_lon, obs_alt = get_observatory_parameters(values)
        except Exception:
            pass

    time_conversion = convert_input_times_to_bjd_tdb(
        time_mid_days,
        input_time_system=time_system,
        ra_deg=planet.ra_deg,
        dec_deg=planet.dec_deg,
        observatory_lat_deg=obs_lat,
        observatory_lon_deg=obs_lon,
        observatory_alt_m=obs_alt,
    )
    time_days = time_conversion.time_bjd_tdb
    integration_subsamples = 7

    base_mask = finite_mask(time_days, flux)
    if flux_err_arr is not None:
        base_mask &= np.isfinite(flux_err_arr) & (flux_err_arr > 0)
    if keep_mask is not None and keep_mask.shape == base_mask.shape:
        base_mask &= keep_mask

    if np.count_nonzero(base_mask) < 8:
        raise ValueError("Too few valid points for transit diagnostics.")

    t = time_days[base_mask]
    y = flux[base_mask]
    e = None if flux_err_arr is None else flux_err_arr[base_mask]

    tmid_catalogue_pred, _epoch = nearest_predicted_tmid(t, planet)
    tmid_pred = float(tmid_catalogue_pred)
    tmid_reference_source = "Catalogue ephemeris"
    tmid_reference_override, override_warning = parse_tmid_override(values)
    if np.isfinite(tmid_reference_override):
        tmid_pred = float(tmid_reference_override)
        tmid_reference_source = "User override"

    duration_days = max(planet.duration_hours / 24.0, 1e-6)
    n_baseline = baseline_terms(t, tmid_pred, order).shape[1]

    predicted_overlaps = transit_template_overlaps_data(t, tmid_pred, planet.duration_hours)
    data_tmid_guess = estimate_transit_centre_from_flux(t, y)

    if fit_tmid:
        if predicted_overlaps:
            # Ground-based timings can be noticeably shifted if the catalogue
            # ephemeris is old. Allow a few durations but still keep the fit local.
            tmid_half_window = max(1.25 * duration_days, 0.02)
            dt_guess = 0.0
            lower_tmid = -tmid_half_window
            upper_tmid = +tmid_half_window
            if np.isfinite(data_tmid_guess):
                dt_guess = float(np.clip(data_tmid_guess - tmid_pred, lower_tmid, upper_tmid))
        else:
            lower_tmid = float(np.nanmin(t) - tmid_pred - duration_days)
            upper_tmid = float(np.nanmax(t) - tmid_pred + duration_days)
            dt_guess = float(data_tmid_guess - tmid_pred) if np.isfinite(data_tmid_guess) else 0.0
            dt_guess = float(np.clip(dt_guess, lower_tmid, upper_tmid))
    else:
        lower_tmid = upper_tmid = dt_guess = 0.0

    p0 = []
    lower = []
    upper = []

    if fit_tmid:
        p0.append(dt_guess)
        lower.append(lower_tmid)
        upper.append(upper_tmid)
    if fit_depth:
        p0.append(_initial_depth_scale(y, planet.depth_relative))
        lower.append(0.05)
        upper.append(3.5)
    if fit_duration:
        p0.append(estimate_duration_scale_from_flux(t, y, planet.duration_hours))
        lower.append(0.55)
        upper.append(1.80)

    baseline0 = _initial_baseline(y, n_baseline)
    p0.extend(baseline0.tolist())
    lower.extend([-np.inf] * n_baseline)
    upper.extend([np.inf] * n_baseline)

    p0 = np.asarray(p0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    weights = np.ones_like(y)
    if e is not None:
        weights = 1.0 / e

    def unpack(params: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        index = 0
        dt = 0.0
        depth_scale = 1.0
        duration_scale = 1.0
        if fit_tmid:
            dt = float(params[index])
            index += 1
        if fit_depth:
            depth_scale = float(params[index])
            index += 1
        if fit_duration:
            duration_scale = float(params[index])
            index += 1
        baseline_coeffs = params[index:index + n_baseline]
        return dt, depth_scale, duration_scale, baseline_coeffs

    def transit_with_baseline(
        x: np.ndarray,
        tmid: float,
        depth_scale: float,
        duration_scale: float,
        baseline_coeffs: np.ndarray,
    ) -> np.ndarray:
        design = baseline_terms(x, tmid_pred, order)
        baseline = design @ baseline_coeffs
        transit = transit_flux_model(
            x,
            planet=planet,
            tmid=tmid,
            depth_scale=depth_scale,
            duration_scale=duration_scale,
            filter_name=filter_name,
            exposure_time_s=exposure_time_s,
            n_subsamples=integration_subsamples,
            model_engine=model_engine,
        )
        return baseline * transit

    def model_from_params(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        dt, depth_scale, duration_scale, baseline_coeffs = unpack(params)
        return transit_with_baseline(
            x,
            tmid_pred + dt,
            depth_scale,
            duration_scale,
            baseline_coeffs,
        )

    def residual_vector(params: np.ndarray) -> np.ndarray:
        return (y - model_from_params(params, t)) * weights

    # Use a robust loss so that a few cloud/systematics points do not dominate
    # the automatic diagnostic fit.
    result = least_squares(
        residual_vector,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=30000,
    )
    best = result.x
    dt_best, depth_scale_best, duration_scale_best, coeffs_best = unpack(best)

    # Build all output variants explicitly.  The fit itself is performed on the
    # observed flux with baseline × transit model.  For the default scientific
    # display we then divide the photometry by the fitted baseline and plot the
    # pure transit models.  This avoids the visually confusing effect where a
    # linear/quadratic baseline appears to distort the transit shape.
    full_baseline = np.full_like(time_days, np.nan, dtype=float)
    baseline_valid = finite_mask(time_days)
    if np.any(baseline_valid):
        full_baseline[baseline_valid] = baseline_terms(time_days[baseline_valid], tmid_pred, order) @ coeffs_best

    fit_transit_model = np.full_like(time_days, np.nan, dtype=float)
    expected_transit_model = np.full_like(time_days, np.nan, dtype=float)
    if np.any(baseline_valid):
        fit_transit_model[baseline_valid] = transit_flux_model(
            time_days[baseline_valid],
            planet=planet,
            tmid=tmid_pred + dt_best,
            depth_scale=depth_scale_best,
            duration_scale=duration_scale_best,
            filter_name=filter_name,
            exposure_time_s=exposure_time_s,
            n_subsamples=integration_subsamples,
            model_engine=model_engine,
        )
        expected_transit_model[baseline_valid] = transit_flux_model(
            time_days[baseline_valid],
            planet=planet,
            tmid=tmid_pred,
            depth_scale=1.0,
            duration_scale=1.0,
            filter_name=filter_name,
            exposure_time_s=exposure_time_s,
            n_subsamples=integration_subsamples,
            model_engine=model_engine,
        )

    full_model = full_baseline * fit_transit_model
    expected_model = full_baseline * expected_transit_model

    valid_baseline = np.isfinite(full_baseline) & (np.abs(full_baseline) > 1e-12)
    detrended_flux = np.full_like(time_days, np.nan, dtype=float)
    detrended_flux[valid_baseline] = flux[valid_baseline] / full_baseline[valid_baseline]

    detrended_flux_err = np.full_like(time_days, np.nan, dtype=float)
    if flux_err_arr is not None:
        detrended_flux_err[valid_baseline] = flux_err_arr[valid_baseline] / np.abs(full_baseline[valid_baseline])

    full_residuals = np.full_like(time_days, np.nan, dtype=float)
    full_residuals[base_mask] = flux[base_mask] - full_model[base_mask]

    detrended_residuals = np.full_like(time_days, np.nan, dtype=float)
    detrended_residuals[base_mask] = detrended_flux[base_mask] - fit_transit_model[base_mask]

    residual_fit = detrended_residuals[base_mask]
    n_points = int(np.count_nonzero(base_mask))
    dof = max(1, n_points - len(best))
    residual_weighted = residual_vector(best)
    reduced_chi2 = float(np.sum(residual_weighted**2) / dof) if e is not None else np.nan

    uncertainties = _safe_uncertainties(result.jac, residual_weighted, dof)
    dt_unc = np.nan
    depth_scale_unc = np.nan
    duration_scale_unc = np.nan
    p_index = 0
    if uncertainties is not None:
        if fit_tmid:
            dt_unc = float(uncertainties[p_index])
            p_index += 1
        if fit_depth:
            depth_scale_unc = float(uncertainties[p_index])
            p_index += 1
        if fit_duration:
            duration_scale_unc = float(uncertainties[p_index])

    tmid_observed = tmid_pred + dt_best
    oc_minutes = dt_best * 24.0 * 60.0
    oc_unc_minutes = dt_unc * 24.0 * 60.0 if np.isfinite(dt_unc) else np.nan

    catalogue_depth_ppt = float(planet.expected_model_depth_ppt)
    geometric_depth_ppt = (float(planet.rp_rs) ** 2) * 1000.0 if np.isfinite(planet.rp_rs) and planet.rp_rs > 0 else np.nan
    expected_model_depth_ppt = _model_depth_ppt_from_transit_model(
        planet=planet,
        tmid=tmid_pred,
        depth_scale=1.0,
        duration_scale=1.0,
        filter_name=filter_name,
        exposure_time_s=exposure_time_s,
        n_subsamples=integration_subsamples,
        model_engine=model_engine,
    )
    observed_model_depth_ppt = _model_depth_ppt_from_transit_model(
        planet=planet,
        tmid=tmid_pred + dt_best,
        depth_scale=depth_scale_best,
        duration_scale=duration_scale_best,
        filter_name=filter_name,
        exposure_time_s=exposure_time_s,
        n_subsamples=integration_subsamples,
        model_engine=model_engine,
    )

    # Fallbacks for unusual/failed model-depth measurements.
    if not np.isfinite(expected_model_depth_ppt):
        expected_model_depth_ppt = catalogue_depth_ppt
    if not np.isfinite(observed_model_depth_ppt):
        observed_model_depth_ppt = catalogue_depth_ppt * depth_scale_best

    expected_depth_ppt = float(expected_model_depth_ppt)
    observed_depth_ppt = float(observed_model_depth_ppt)
    expected_rp_rs = planet.rp_rs
    observed_rp_rs = expected_rp_rs * np.sqrt(depth_scale_best) if depth_scale_best >= 0 else np.nan
    observed_duration_hours = planet.duration_hours * duration_scale_best

    residual_rms_ppt = float(np.sqrt(np.nanmean(residual_fit**2)) * 1000.0)
    median_error_ppt = float(np.nanmedian(e) * 1000.0) if e is not None else np.nan
    rms_over_median_error = residual_rms_ppt / median_error_ppt if median_error_ppt > 0 else np.nan

    profile_best = exposure_integrated_transit_profile(
        t,
        tmid_observed,
        observed_duration_hours,
        exposure_time_s=exposure_time_s,
        n_subsamples=integration_subsamples,
    )
    n_in_transit = int(np.count_nonzero(profile_best > 0.20))
    transit_snr = (
        observed_depth_ppt / residual_rms_ppt * np.sqrt(n_in_transit)
        if residual_rms_ppt > 0 and n_in_transit > 0
        else np.nan
    )

    order_time = np.argsort(t)
    lag1 = _lag1_autocorrelation(residual_fit[order_time])

    if shapiro is not None and 3 <= residual_fit.size <= 5000:
        try:
            shapiro_p = float(shapiro(residual_fit).pvalue)
        except Exception:
            shapiro_p = np.nan
    else:
        shapiro_p = np.nan

    cadence_median_min = np.nan
    if n_points > 1:
        dt_sorted = np.diff(np.sort(t))
        dt_sorted = dt_sorted[np.isfinite(dt_sorted) & (dt_sorted > 0)]
        if dt_sorted.size:
            cadence_median_min = float(np.nanmedian(dt_sorted) * 24.0 * 60.0)

    depth_ratio_for_quality = (
        observed_depth_ppt / expected_depth_ppt
        if np.isfinite(observed_depth_ppt) and np.isfinite(expected_depth_ppt) and expected_depth_ppt > 0
        else depth_scale_best
    )
    detection, depth_label, duration_label, timing, scatter, autocorr_label = _quality_labels(
        transit_snr,
        depth_ratio_for_quality,
        duration_scale_best,
        oc_minutes,
        oc_unc_minutes,
        rms_over_median_error,
        lag1,
    )

    warnings = []
    if override_warning:
        warnings.append(override_warning)
    if np.isfinite(tmid_reference_override):
        delta_override_min = (tmid_reference_override - tmid_catalogue_pred) * 24.0 * 60.0
        warnings.append(
            "The predicted Tmid used for O-C and the expected model was overridden by the user. "
            f"Override - catalogue prediction = {delta_override_min:+.2f} min."
        )
    if model_engine != "Batman physical" and str(values.get("-TR_MODEL_ENGINE-", "Auto")) == "Batman physical":
        warnings.append(model_engine_note)
    if abs(time_correction_s) > 0:
        warnings.append(
            f"Input timestamps were treated as '{timestamp_reference}' and corrected to mid-exposure "
            f"by {time_correction_s:+.1f} s before the fit and O-C calculation."
        )
    elif str(timestamp_reference) != "Mid-exposure" and (not np.isfinite(exposure_time_s) or exposure_time_s <= 0):
        warnings.append(
            "A start/end timestamp reference was selected, but the exposure time is not valid; "
            "no timing correction was applied."
        )
    if str(time_system).strip().upper() != "BJD_TDB":
        if time_conversion.applied:
            warnings.append(time_conversion.note)
        else:
            warnings.append(
                "The catalogue ephemeris is assumed to be BJD_TDB, but the input time system "
                f"{time_system} was not converted. O-C values should be interpreted with caution."
            )
    if fit_tmid and not predicted_overlaps:
        warnings.append(
            "The expected catalogue transit does not overlap the observed time range. "
            "The fitted transit centre was initialised from the observed flux minimum. "
            "Check that the selected planet, ephemeris and time system are correct before interpreting O-C."
        )
    if (
        model_engine == "Batman physical"
        and np.isfinite(catalogue_depth_ppt)
        and np.isfinite(expected_depth_ppt)
        and catalogue_depth_ppt > 0
        and abs(expected_depth_ppt - catalogue_depth_ppt) > max(2.0, 0.25 * catalogue_depth_ppt)
    ):
        warnings.append(
            "The catalogue transit depth and the physical batman model depth are noticeably different. "
            "For high-impact or grazing transits this can happen because the apparent depth depends "
            "non-linearly on Rp/Rs, inclination, a/Rs and limb darkening. The reported fit depth is "
            "therefore measured from the plotted model, not from catalogue_depth × depth_scale."
        )
    if abs(duration_scale_best - 1.0) > 0.35:
        warnings.append(
            "The fitted duration differs substantially from the catalogue value. This can indicate "
            "a partial transit, imperfect detrending, an outdated ephemeris or a wrong target selection."
        )
    warning = "\n".join(warnings)

    return TransitDiagnosticResult(
        planet_name=planet.planet_name,
        filter_name=filter_name,
        exposure_time_s=exposure_time_s,
        timestamp_reference=timestamp_reference,
        time_correction_s=float(time_correction_s),
        model_integration_subsamples=int(integration_subsamples),
        time_system=time_system,
        bjd_conversion_applied=bool(time_conversion.applied),
        bjd_correction_min=float(time_conversion.correction_minutes),
        time_conversion_note=time_conversion.note,
        target_ra_deg=float(planet.ra_deg),
        target_dec_deg=float(planet.dec_deg),
        observatory_lat_deg=float(obs_lat),
        observatory_lon_deg=float(obs_lon),
        observatory_alt_m=float(obs_alt),
        baseline_order=order,
        display_mode=display_mode,
        model_engine=model_engine,
        model_engine_note=model_engine_note,
        limb_darkening_law=limb_darkening_law,
        limb_darkening_coefficients=(float(limb_darkening_coeffs[0]), float(limb_darkening_coeffs[1])),
        limb_darkening_source=limb_darkening_source,
        catalogue_source=str(getattr(planet, "source", "")),
        catalogue_notes=str(getattr(planet, "notes", "")),
        catalogue_type=str(getattr(planet, "catalogue_type", "")),
        catalogue_depth_source=str(getattr(planet, "depth_source", "")),
        catalogue_rprs_source=str(getattr(planet, "rprs_source", "")),
        catalogue_geometry_source=str(getattr(planet, "geometry_source", "")),
        catalogue_stellar_source=str(getattr(planet, "stellar_source", "")),
        catalogue_t0_bjd_tdb=float(planet.t0_bjd_tdb),
        catalogue_period_days=float(planet.period_days),
        predicted_epoch=int(_epoch),
        tmid_catalogue_predicted=float(tmid_catalogue_pred),
        tmid_reference_source=tmid_reference_source,
        tmid_reference_override=float(tmid_reference_override),
        tmid_predicted=float(tmid_pred),
        tmid_observed=float(tmid_observed),
        oc_minutes=float(oc_minutes),
        oc_unc_minutes=float(oc_unc_minutes),
        catalogue_depth_ppt=float(catalogue_depth_ppt),
        geometric_depth_ppt=float(geometric_depth_ppt),
        expected_depth_ppt=float(expected_depth_ppt),
        observed_depth_ppt=float(observed_depth_ppt),
        expected_rp_rs=float(expected_rp_rs),
        observed_rp_rs=float(observed_rp_rs),
        depth_scale=float(depth_scale_best),
        depth_scale_unc=float(depth_scale_unc),
        expected_duration_hours=float(planet.duration_hours),
        observed_duration_hours=float(observed_duration_hours),
        duration_scale=float(duration_scale_best),
        duration_scale_unc=float(duration_scale_unc),
        residual_rms_ppt=float(residual_rms_ppt),
        median_error_ppt=float(median_error_ppt),
        rms_over_median_error=float(rms_over_median_error),
        transit_snr=float(transit_snr),
        lag1_autocorr=float(lag1),
        shapiro_p=float(shapiro_p),
        n_points=n_points,
        n_in_transit=n_in_transit,
        cadence_median_min=float(cadence_median_min),
        reduced_chi2=float(reduced_chi2),
        detection_label=detection,
        depth_label=depth_label,
        duration_label=duration_label,
        timing_label=timing,
        scatter_label=scatter,
        autocorr_label=autocorr_label,
        warning=warning,
        corrected_time=time_days,
        baseline=full_baseline,
        detrended_flux=detrended_flux,
        detrended_flux_err=detrended_flux_err,
        fit_transit_model=fit_transit_model,
        expected_transit_model=expected_transit_model,
        fit_full_model=full_model,
        expected_full_model=expected_model,
        full_residuals=full_residuals,
        detrended_residuals=detrended_residuals,
        model=fit_transit_model if display_mode == "Detrended flux" else full_model,
        expected_model=expected_transit_model if display_mode == "Detrended flux" else expected_model,
        residuals=detrended_residuals if display_mode == "Detrended flux" else full_residuals,
        keep_mask=base_mask,
    )


def _fmt(value: float, fmt: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    return format(value, fmt)


def format_transit_report(result: TransitDiagnosticResult) -> str:
    """Return a readable text report for the transit diagnostics."""
    lines = []
    lines.append("PhotoCurve Lab - Transit diagnostics")
    lines.append("=" * 43)
    lines.append("")
    lines.append("Observation")
    lines.append(f"Planet: {result.planet_name}")
    lines.append(f"Filter: {result.filter_name}")
    lines.append(f"Exposure time: {_fmt(result.exposure_time_s, '.1f')} s")
    lines.append(f"Input time system: {result.time_system}")
    lines.append(f"Working time system: BJD_TDB")
    lines.append(f"Time stamp reference: {result.timestamp_reference}")
    lines.append(f"Applied mid-exposure correction: {_fmt(result.time_correction_s, '+.1f')} s")
    lines.append(f"BJD_TDB conversion correction: {_fmt(result.bjd_correction_min, '+.2f')} min")
    lines.append(f"Time conversion: {result.time_conversion_note}")
    if np.isfinite(result.target_ra_deg) and np.isfinite(result.target_dec_deg):
        lines.append(f"Target coordinates: RA={_fmt(result.target_ra_deg, '.6f')} deg, Dec={_fmt(result.target_dec_deg, '.6f')} deg")
    if np.isfinite(result.observatory_lat_deg) and np.isfinite(result.observatory_lon_deg):
        lines.append(
            f"Observatory: lat={_fmt(result.observatory_lat_deg, '.6f')} deg, "
            f"lon={_fmt(result.observatory_lon_deg, '.6f')} deg, "
            f"alt={_fmt(result.observatory_alt_m, '.1f')} m"
        )
    lines.append(f"Model exposure integration: {result.model_integration_subsamples:d} subsamples")
    lines.append(f"Baseline model: {result.baseline_order}")
    lines.append(f"Display mode: {result.display_mode}")
    lines.append(f"Transit model engine: {result.model_engine}")
    lines.append(f"Model note: {result.model_engine_note}")
    lines.append(f"Limb darkening law: {result.limb_darkening_law}")
    if result.model_engine == "Batman physical":
        u1, u2 = result.limb_darkening_coefficients
        lines.append(f"Limb darkening coefficients: u1={_fmt(u1, '.3f')}, u2={_fmt(u2, '.3f')}")
    lines.append(f"Limb darkening source: {result.limb_darkening_source}")
    lines.append("")
    lines.append("Ephemeris and timing reference")
    if str(result.catalogue_source).strip():
        lines.append(f"Catalogue source = {result.catalogue_source}")
    if str(result.catalogue_type).strip():
        lines.append(f"Catalogue type = {result.catalogue_type}")
    if str(result.catalogue_depth_source).strip():
        lines.append(f"Depth source = {result.catalogue_depth_source}")
    if str(result.catalogue_rprs_source).strip():
        lines.append(f"Rp/Rs source = {result.catalogue_rprs_source}")
    if str(result.catalogue_geometry_source).strip():
        lines.append(f"Geometry source = {result.catalogue_geometry_source}")
    if str(result.catalogue_stellar_source).strip():
        lines.append(f"Stellar source = {result.catalogue_stellar_source}")
    lines.append(f"Catalogue T0 = {_fmt(result.catalogue_t0_bjd_tdb, '.8f')} BJD_TDB")
    lines.append(f"Catalogue period = {_fmt(result.catalogue_period_days, '.10f')} d")
    lines.append(f"Epoch number = {result.predicted_epoch:d}")
    lines.append(f"Catalogue predicted Tmid = {_fmt(result.tmid_catalogue_predicted, '.8f')} BJD_TDB")
    lines.append(f"Timing reference source = {result.tmid_reference_source}")
    if np.isfinite(result.tmid_reference_override):
        delta_override_min = (result.tmid_reference_override - result.tmid_catalogue_predicted) * 24.0 * 60.0
        lines.append(f"User predicted Tmid override = {_fmt(result.tmid_reference_override, '.8f')} BJD_TDB")
        lines.append(f"Override - catalogue prediction = {_fmt(delta_override_min, '+.2f')} min")
    lines.append(f"Reference Tmid used for O-C = {_fmt(result.tmid_predicted, '.8f')} BJD_TDB")
    lines.append("")
    lines.append("Results")
    lines.append(f"Predicted/reference Tmid = {_fmt(result.tmid_predicted, '.8f')} BJD_TDB")
    lines.append(f"Observed fitted Tmid    = {_fmt(result.tmid_observed, '.8f')} BJD_TDB")
    lines.append(
        f"O-C = {_fmt(result.oc_minutes, '+.2f')} ± {_fmt(result.oc_unc_minutes, '.2f')} min"
    )
    lines.append(
        f"Rp/Rs = {_fmt(result.observed_rp_rs, '.4f')} "
        f"(expected {_fmt(result.expected_rp_rs, '.4f')})"
    )
    lines.append(
        f"Depth = {_fmt(result.observed_depth_ppt, '.2f')} ppt "
        f"(expected model {_fmt(result.expected_depth_ppt, '.2f')} ppt)"
    )
    if np.isfinite(result.catalogue_depth_ppt):
        lines.append(f"Catalogue flux depth = {_fmt(result.catalogue_depth_ppt, '.2f')} ppt")
    if (
        np.isfinite(result.geometric_depth_ppt)
        and np.isfinite(result.catalogue_depth_ppt)
        and abs(result.geometric_depth_ppt - result.catalogue_depth_ppt) > 0.25
    ):
        lines.append(f"Geometric depth Rp/Rs^2 = {_fmt(result.geometric_depth_ppt, '.2f')} ppt")
    lines.append(
        f"Duration = {_fmt(result.observed_duration_hours, '.2f')} h "
        f"(expected {_fmt(result.expected_duration_hours, '.2f')} h)"
    )
    lines.append("")
    lines.append("Quality check")
    lines.append(f"Residual RMS = {_fmt(result.residual_rms_ppt, '.2f')} ppt")
    lines.append(f"Median photometric error = {_fmt(result.median_error_ppt, '.2f')} ppt")
    lines.append(f"RMS / median error = {_fmt(result.rms_over_median_error, '.2f')}")
    lines.append(f"Transit SNR = {_fmt(result.transit_snr, '.2f')}")
    lines.append(f"N points = {result.n_points}")
    lines.append(f"N in transit = {result.n_in_transit}")
    lines.append(f"Median cadence = {_fmt(result.cadence_median_min, '.2f')} min")
    lines.append(f"Lag-1 autocorrelation = {_fmt(result.lag1_autocorr, '.3f')}")
    lines.append(f"Shapiro p-value = {_fmt(result.shapiro_p, '.3f')}")
    lines.append(f"Reduced chi2 = {_fmt(result.reduced_chi2, '.2f')}")
    lines.append("")
    lines.append("Diagnostics")
    lines.append(f"Transit detection: {result.detection_label}")
    lines.append(f"Timing: {result.timing_label}")
    lines.append(f"Depth: {result.depth_label}")
    lines.append(f"Duration: {result.duration_label}")
    lines.append(f"Scatter: {result.scatter_label}")
    lines.append(f"Autocorrelation: {result.autocorr_label}")
    if result.warning:
        lines.append("")
        lines.append("Warning")
        lines.append(result.warning)
    lines.append("")
    lines.append(
        "Note: this is a preliminary observational diagnostic. When the batman "
        "engine is used, the transit shape is physical but the fit is still "
        "deliberately limited to timing, depth, duration scaling and baseline."
    )
    return "\n".join(lines)


def result_to_dict(result: TransitDiagnosticResult) -> Dict[str, object]:
    """Convert a transit diagnostic result to a serialisable dictionary."""
    out = result.__dict__.copy()
    for key in [
        "corrected_time", "baseline", "detrended_flux", "detrended_flux_err",
        "fit_transit_model", "expected_transit_model",
        "fit_full_model", "expected_full_model",
        "full_residuals", "detrended_residuals",
        "model", "expected_model", "residuals", "keep_mask",
    ]:
        out.pop(key, None)
    return out
