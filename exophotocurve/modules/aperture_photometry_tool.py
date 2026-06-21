"""Interactive aperture-photometry builder for PhotoCurve Lab.

This module intentionally behaves like a small, separate sub-program.  It
measures simple aperture photometry on an already calibrated and aligned FITS
sequence, then writes an AstroImageJ-like text table.  The main PhotoCurve Lab
pipeline can load that table exactly as it would load an AIJ export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import glob
import json
import math
import os
import time
from . import misc

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .numeric_utils import parse_float, parse_int
from .sg_loader import sg

try:  # pragma: no cover - environment-dependent optional import
    from astropy.io import fits
    from astropy.time import Time
except Exception:  # pragma: no cover
    fits = None
    Time = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(BASE_DIR, "ExoPhotoCurve.ico")

@dataclass
class ApertureStar:
    """One manually selected aperture."""

    star_id: str
    x: float
    y: float
    role: str = "comparison"
    peak_quality: str = "unknown"
    peak_median: float = math.nan
    peak_max: float = math.nan
    peak_source: str = "not checked"


@dataclass
class PhotometryMeasurement:
    """Photometry measured for one star in one image."""

    flux: float = math.nan
    error: float = math.nan
    sky_median: float = math.nan
    sky_std: float = math.nan
    peak: float = math.nan
    mean_aperture: float = math.nan
    x: float = math.nan
    y: float = math.nan
    n_aperture: int = 0
    n_sky: int = 0
    saturated: bool = False


@dataclass
class PhotometrySession:
    """Mutable state for the photometry builder window."""

    files: List[str] = field(default_factory=list)
    current_index: int = 0
    image: Optional[np.ndarray] = None
    header: object = None
    stars: List[ApertureStar] = field(default_factory=list)
    figure_agg: object = None
    figure: Optional[Figure] = None
    ax: object = None
    output_path: Optional[str] = None
    view_xlim: Optional[Tuple[float, float]] = None
    view_ylim: Optional[Tuple[float, float]] = None
    hover_artists: List[object] = field(default_factory=list)
    overlay_artists: List[object] = field(default_factory=list)
    pan_active: bool = False
    # Panning is tracked in canvas/display pixels, not image pixels.  This is
    # much more stable in embedded Tk/Matplotlib canvases because xdata/ydata
    # change continuously as the axes limits are updated during the drag.
    pan_start_xy: Optional[Tuple[float, float]] = None
    pan_start_xlim: Optional[Tuple[float, float]] = None
    pan_start_ylim: Optional[Tuple[float, float]] = None
    pending_click_active: bool = False
    pending_click_pixel: Optional[Tuple[float, float]] = None
    pending_click_data: Optional[Tuple[float, float]] = None
    pending_click_xlim: Optional[Tuple[float, float]] = None
    pending_click_ylim: Optional[Tuple[float, float]] = None
    last_motion_emit_time: float = 0.0
    last_motion_pixel: Optional[Tuple[float, float]] = None


def _split_keywords(text: object) -> List[str]:
    """Return normalised FITS keyword candidates from a comma list."""
    if text is None:
        return []
    out: List[str] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _safe_float(value: object, default: float = math.nan) -> float:
    """Parse a float while accepting FITS strings and empty values."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
            # FITS cards occasionally contain a unit/comment-like suffix.  Keep
            # the leading numeric token if present.
            value = value.split()[0].replace(",", ".")
        return float(value)
    except Exception:
        return default


def _header_get_float(header, keys: Sequence[str], default: float = math.nan) -> float:
    """Return the first finite numeric value from a list of FITS keywords."""
    if header is None:
        return default
    for key in keys:
        if key in header:
            value = _safe_float(header.get(key), default=math.nan)
            if np.isfinite(value):
                return float(value)
    return default


def _header_get_text(header, keys: Sequence[str], default: str = "") -> str:
    """Return the first non-empty text value from a list of FITS keywords."""
    if header is None:
        return default
    for key in keys:
        if key in header:
            value = str(header.get(key)).strip()
            if value:
                return value
    return default


def _jd_from_header_with_key(header, time_keywords: Sequence[str]) -> Tuple[float, str]:
    """Return a JD_UTC-like time and the FITS keyword used, if available.

    The returned value is the timestamp exactly represented by the FITS
    keyword.  It is not automatically shifted to mid-exposure because FITS
    headers differ between acquisition programs: DATE-OBS is commonly exposure
    start, while some JD-like keywords may already be mid-exposure.  The
    photometry builder therefore records both the raw timestamp and derived
    start/mid/end columns.
    """
    if header is None:
        return math.nan, ""

    for key in time_keywords:
        if key not in header:
            continue
        raw = header.get(key)
        key_upper = str(key).upper()

        if "MJD" in key_upper:
            value = _safe_float(raw, default=math.nan)
            if np.isfinite(value):
                return float(value + 2400000.5), str(key)
            continue

        if "DATE" in key_upper:
            if Time is None:
                continue
            try:
                return float(Time(str(raw).strip(), scale="utc").jd), str(key)
            except Exception:
                try:
                    return float(Time(str(raw).strip(), format="isot", scale="utc").jd), str(key)
                except Exception:
                    continue

        value = _safe_float(raw, default=math.nan)
        if np.isfinite(value):
            if value < 100000.0 and "JD" not in key_upper:
                continue
            return float(value), str(key)

    return math.nan, ""


def _jd_from_header(header, time_keywords: Sequence[str]) -> float:
    """Return a JD_UTC-like time from a FITS header, if available."""
    value, _key = _jd_from_header_with_key(header, time_keywords)
    return value


def _time_reference_offsets_days(header_time_jd: float, exptime_seconds: float, time_reference: str) -> Tuple[float, float, float]:
    """Return start, mid and end JD values implied by a header timestamp.

    ``time_reference`` describes what the header timestamp means.  Most DATE-OBS
    values are exposure start, but users can change this if their acquisition
    software writes mid-exposure or end-exposure times.
    """
    if not np.isfinite(header_time_jd):
        return math.nan, math.nan, math.nan
    exp_days = float(exptime_seconds) / 86400.0 if np.isfinite(exptime_seconds) else 0.0
    ref = str(time_reference or "Exposure start").strip().lower()
    if ref.startswith("mid"):
        start = header_time_jd - 0.5 * exp_days
        mid = header_time_jd
        end = header_time_jd + 0.5 * exp_days
    elif ref.startswith("exposure end") or ref.startswith("end"):
        start = header_time_jd - exp_days
        mid = header_time_jd - 0.5 * exp_days
        end = header_time_jd
    else:
        start = header_time_jd
        mid = header_time_jd + 0.5 * exp_days
        end = header_time_jd + exp_days
    return float(start), float(mid), float(end)


def _read_fits_image(path: str) -> Tuple[np.ndarray, object]:
    """Read the first 2D image HDU from a FITS file."""
    if fits is None:
        raise RuntimeError("Astropy is required for FITS aperture photometry. Install it with: pip install astropy")

    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim == 2:
                return arr.astype(float), hdu.header
            if arr.ndim > 2:
                # Use the first plane of a cube-like image, but keep this simple.
                squeezed = np.squeeze(arr)
                if squeezed.ndim == 2:
                    return squeezed.astype(float), hdu.header
        raise ValueError(f"No 2D image HDU found in {path}")


def _load_current_image(session: PhotometrySession) -> None:
    """Load the current FITS image into the session."""
    if not session.files:
        session.image = None
        session.header = None
        return
    session.current_index = max(0, min(session.current_index, len(session.files) - 1))
    session.image, session.header = _read_fits_image(session.files[session.current_index])


def _image_stretch(image: np.ndarray, low_percent: float, high_percent: float) -> Tuple[float, float]:
    """Return robust image display limits."""
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.nanpercentile(finite, low_percent))
    high = float(np.nanpercentile(finite, high_percent))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        med = float(np.nanmedian(finite))
        std = float(np.nanstd(finite))
        low, high = med - 2.0 * std, med + 5.0 * std
    return low, high


def _aperture_stats_at(image: Optional[np.ndarray], x: float, y: float, radius: float) -> Tuple[float, float, float]:
    """Return pixel value, aperture peak and aperture mean near a cursor point."""
    if image is None:
        return math.nan, math.nan, math.nan
    ny, nx = image.shape
    ix = int(round(x))
    iy = int(round(y))
    pixel = math.nan
    if 0 <= ix < nx and 0 <= iy < ny:
        pixel = float(image[iy, ix])

    r = max(1.0, float(radius))
    x_min = max(0, int(math.floor(x - r)))
    x_max = min(nx - 1, int(math.ceil(x + r)))
    y_min = max(0, int(math.floor(y - r)))
    y_max = min(ny - 1, int(math.ceil(y + r)))
    if x_max < x_min or y_max < y_min:
        return pixel, math.nan, math.nan
    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    values = image[y_min : y_max + 1, x_min : x_max + 1][mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return pixel, math.nan, math.nan
    return pixel, float(np.nanmax(values)), float(np.nanmean(values))



def _peak_feedback_thresholds(values: Optional[Dict[str, object]] = None) -> Tuple[float, float, float, float, float]:
    """Return saturation, OK fractions and absolute peak thresholds.

    The limits are defined as fractions of a configurable non-linearity or
    saturation level.  With the default 60000 ADU and fractions 0.25--0.58,
    the green range is approximately 15000--35000 ADU.
    """
    values = values or {}
    saturation = parse_float(values.get("-PHOTO_SAT_LEVEL-", 60000.0), 60000.0)
    low_frac = parse_float(values.get("-PHOTO_PEAK_LOW_FRAC-", 0.25), 0.25)
    high_frac = parse_float(values.get("-PHOTO_PEAK_HIGH_FRAC-", 0.58), 0.58)
    if not np.isfinite(saturation) or saturation <= 0:
        saturation = 60000.0
    low_frac = min(max(float(low_frac), 0.01), 0.95)
    high_frac = min(max(float(high_frac), low_frac + 0.01), 0.99)
    low_peak = saturation * low_frac
    high_peak = saturation * high_frac
    return float(saturation), float(low_frac), float(high_frac), float(low_peak), float(high_peak)


def _peak_quality_from_value(peak: float, values: Optional[Dict[str, object]] = None) -> str:
    """Classify one peak value without reading any FITS files."""
    _sat, _lf, _hf, low_peak, high_peak = _peak_feedback_thresholds(values)
    if not np.isfinite(peak):
        return "unknown"
    if peak > high_peak:
        return "high"
    if peak < low_peak:
        return "low"
    return "ok"


def _peak_quality_from_stats(peak_median: float, peak_max: float, values: Optional[Dict[str, object]] = None) -> str:
    """Classify sequence/sample peak statistics.

    A single high frame is enough to flag the star as HIGH.  A LOW flag uses
    the median so that a brief cloudy frame does not make a good star look bad.
    """
    _sat, _lf, _hf, low_peak, high_peak = _peak_feedback_thresholds(values)
    if np.isfinite(peak_max) and peak_max > high_peak:
        return "high"
    if np.isfinite(peak_median) and peak_median < low_peak:
        return "low"
    if np.isfinite(peak_median) or np.isfinite(peak_max):
        return "ok"
    return "unknown"


def _peak_quality_style(quality: str) -> Tuple[str, str, str]:
    """Return text label, Matplotlib color and GUI background color."""
    quality = str(quality or "unknown").lower()
    if quality == "ok":
        return "OK", "limegreen", "#d8f5d0"
    if quality == "high":
        return "HIGH", "red", "#ffd6d6"
    if quality == "low":
        return "LOW", "gold", "#fff2b8"
    return "?", "tab:orange", "#eeeeee"


def _reclassify_star_peak_quality(star: ApertureStar, values: Dict[str, object]) -> None:
    """Reclassify an already measured star without re-reading images."""
    star.peak_quality = _peak_quality_from_stats(star.peak_median, star.peak_max, values)


def _reclassify_all_star_peak_quality(session: PhotometrySession, values: Dict[str, object]) -> None:
    """Reclassify all cached peak diagnostics without doing any I/O."""
    for star in session.stars:
        _reclassify_star_peak_quality(star, values)


def _set_star_peak_from_image(
    star: ApertureStar,
    image: Optional[np.ndarray],
    values: Dict[str, object],
    source: str = "current frame",
) -> None:
    """Update one star using only the currently loaded image.

    This is intentionally fast and is used for click feedback and aperture
    loading.  It never scans the full FITS sequence.
    """
    if image is None:
        star.peak_quality = "unknown"
        star.peak_median = math.nan
        star.peak_max = math.nan
        star.peak_source = "not checked"
        return
    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    _pixel, peak, _mean = _aperture_stats_at(image, star.x, star.y, aper)
    star.peak_median = float(peak) if np.isfinite(peak) else math.nan
    star.peak_max = float(peak) if np.isfinite(peak) else math.nan
    star.peak_quality = _peak_quality_from_value(peak, values)
    star.peak_source = source


def _update_all_star_peaks_from_current_image(session: PhotometrySession, values: Dict[str, object]) -> None:
    """Refresh all stars from the currently displayed frame only."""
    for star in session.stars:
        _set_star_peak_from_image(star, session.image, values, source="current frame")


def _sample_frame_indices(n_files: int, max_frames: int) -> List[int]:
    """Return representative frame indices for a quick peak check."""
    if n_files <= 0:
        return []
    if max_frames <= 0 or max_frames >= n_files:
        return list(range(n_files))
    raw = np.linspace(0, n_files - 1, max_frames)
    indices = sorted({int(round(v)) for v in raw})
    # Always include first, current-ish middle and last frames.
    indices = sorted(set(indices) | {0, n_files // 2, n_files - 1})
    return indices


def _check_star_peaks_over_sequence(
    window: sg.Window,
    session: PhotometrySession,
    values: Dict[str, object],
    max_frames: Optional[int] = None,
) -> None:
    """Check selected-star peak quality using one pass over sampled frames.

    The previous implementation was expensive because each star scanned the
    sequence independently.  This function reads each sampled FITS file once and
    measures all selected stars on that image.  The default is a representative
    sample, not the full sequence, so aperture loading and threshold changes
    remain instantaneous.  Set Peak frames to 0 to force a full sequence check.
    """
    if not session.files or not session.stars:
        return
    if max_frames is None:
        max_frames = parse_int(values.get("-PHOTO_PEAK_MAX_FRAMES-", 15), 15)
    max_frames = int(max_frames)
    indices = _sample_frame_indices(len(session.files), max_frames)
    if not indices:
        return

    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    recenter = bool(values.get("-PHOTO_RECENTER-", True))
    search_radius = parse_float(values.get("-PHOTO_SEARCH_R-", 8.0), 8.0)
    peaks_by_id: Dict[str, List[float]] = {star.star_id: [] for star in session.stars}

    n_indices = len(indices)
    for count, file_index in enumerate(indices, start=1):
        try:
            image, _header = _read_fits_image(session.files[file_index])
        except Exception:
            continue
        for star in session.stars:
            x = float(star.x)
            y = float(star.y)
            if recenter:
                x, y = _centroid_near(image, x, y, search_radius)
            _pixel, peak, _mean = _aperture_stats_at(image, x, y, aper)
            if np.isfinite(peak):
                peaks_by_id.setdefault(star.star_id, []).append(float(peak))
        try:
            window["-PHOTO_PROGRESS-"].update(int(count / max(1, n_indices) * 100.0))
            if count % 3 == 0:
                window.refresh()
        except Exception:
            pass

    full = n_indices >= len(session.files)
    source = "full sequence" if full else f"sample {n_indices}/{len(session.files)} frames"
    for star in session.stars:
        peaks = peaks_by_id.get(star.star_id, [])
        if peaks:
            star.peak_median = float(np.nanmedian(peaks))
            star.peak_max = float(np.nanmax(peaks))
            star.peak_quality = _peak_quality_from_stats(star.peak_median, star.peak_max, values)
            star.peak_source = source
        else:
            star.peak_median = math.nan
            star.peak_max = math.nan
            star.peak_quality = "unknown"
            star.peak_source = source
    try:
        window["-PHOTO_PROGRESS-"].update(0)
    except Exception:
        pass


def _update_peak_quality_from_table(session: PhotometrySession, table: pd.DataFrame, values: Dict[str, object]) -> None:
    """Update peak diagnostics from a just-computed photometry table."""
    for star in session.stars:
        col = f"Peak_{star.star_id}"
        if col not in table:
            continue
        peaks = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        peaks = peaks[np.isfinite(peaks)]
        if peaks.size == 0:
            continue
        star.peak_median = float(np.nanmedian(peaks))
        star.peak_max = float(np.nanmax(peaks))
        star.peak_quality = _peak_quality_from_stats(star.peak_median, star.peak_max, values)
        star.peak_source = "full sequence"



def _distance_to_positions(x: float, y: float, positions: Sequence[Tuple[float, float]]) -> float:
    """Return the minimum distance from a point to a list of positions."""
    if not positions:
        return math.inf
    return float(min(math.hypot(float(px) - x, float(py) - y) for px, py in positions))


def _auto_candidate_score(
    x: float,
    y: float,
    peak: float,
    image_shape: Tuple[int, int],
    values: Dict[str, object],
    nearest_distance: float,
) -> float:
    """Return a ranking score for an automatic comparison-star candidate.

    Lower scores are better.  The ranking favours stars whose peak is near the
    center of the recommended range, stars far from other selected/candidate
    apertures, and stars safely away from the image borders.
    """
    _sat, _lf, _hf, low_peak, high_peak = _peak_feedback_thresholds(values)
    peak_mid = 0.5 * (low_peak + high_peak)
    peak_half_width = max(0.5 * (high_peak - low_peak), 1.0)
    peak_term = abs(float(peak) - peak_mid) / peak_half_width

    ny, nx = image_shape
    edge_distance = min(float(x), float(y), float(nx - 1) - float(x), float(ny - 1) - float(y))
    edge_term = 1.0 / max(edge_distance, 1.0)
    isolation_term = 1.0 / max(float(nearest_distance), 1.0)
    return float(peak_term + 8.0 * isolation_term + 20.0 * edge_term)


def _auto_find_comparison_stars(window: sg.Window, session: PhotometrySession, values: Dict[str, object]) -> None:
    """Automatically add candidate comparison stars from the current FITS image.

    ``max`` is treated as the desired maximum total number of comparison-like
    stars in the aperture list, not as the number of raw candidates to try in a
    single click.  The function therefore keeps testing replacement candidates
    internally until the requested total is reached or no valid candidates are
    left.  This avoids the previous behaviour where the user had to press
    ``Auto find comps`` several times because some initially selected candidates
    were rejected only after the sampled sequence peak check.
    """
    if session.image is None:
        raise ValueError("Load a FITS sequence first.")
    if not any(star.role == "target" for star in session.stars):
        raise ValueError("Select the target first. Auto-comparison search uses the target aperture as an exclusion zone.")

    requested_total = parse_int(values.get("-PHOTO_AUTO_MAX_COMPS-", 15), 15)
    requested_total = max(1, int(requested_total))
    current_comp_count = sum(1 for star in session.stars if star.role in ("comparison", "check") or star.star_id.upper().startswith("C"))
    slots_to_fill = max(0, requested_total - current_comp_count)
    if slots_to_fill <= 0:
        window["-PHOTO_REPORT-"].update(
            (
                "Automatic comparison-star search skipped.\n"
                f"Requested maximum comparison stars: {requested_total}\n"
                f"Existing comparison/check stars: {current_comp_count}\n"
                "No additional comparison stars are needed.\n"
            ),
            append=True,
        )
        return

    aperture_radius = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    sky_outer = parse_float(values.get("-PHOTO_SKY_OUT-", 16.0), 16.0)
    search_radius = parse_float(values.get("-PHOTO_SEARCH_R-", 8.0), 8.0)
    min_distance = parse_float(values.get("-PHOTO_AUTO_MIN_DIST-", 0.0), 0.0)
    if not np.isfinite(min_distance) or min_distance <= 0:
        min_distance = max(3.0 * aperture_radius, 1.25 * sky_outer, 18.0)
    edge_margin = max(float(sky_outer) + 5.0, float(aperture_radius) + 8.0)

    _sat, _low_frac, _high_frac, low_peak, high_peak = _peak_feedback_thresholds(values)
    image = np.asarray(session.image, dtype=float)
    ny, nx = image.shape
    if nx <= 2 * edge_margin or ny <= 2 * edge_margin:
        raise ValueError("The image is too small for the current aperture/sky annulus and edge margin.")

    finite = np.isfinite(image)
    y_indices, x_indices = np.nonzero(finite & (image >= low_peak) & (image <= high_peak))
    if x_indices.size == 0:
        raise ValueError("No pixels in the current image fall inside the selected OK peak range.")

    peaks = image[y_indices, x_indices]
    # Keep enough bright pixels to build a generous candidate pool.  Validation
    # is done later, before any star is added, so a larger pool is needed to
    # replace candidates rejected by the sequence peak check in the same click.
    max_pixels_to_consider = min(int(peaks.size), max(6000, requested_total * 3500))
    if peaks.size > max_pixels_to_consider:
        top_idx = np.argpartition(peaks, -max_pixels_to_consider)[-max_pixels_to_consider:]
        order_local = top_idx[np.argsort(peaks[top_idx])[::-1]]
    else:
        order_local = np.argsort(peaks)[::-1]

    existing_positions = [(float(star.x), float(star.y)) for star in session.stars]
    provisional_positions: List[Tuple[float, float]] = []
    candidates: List[Tuple[float, float, float, float]] = []
    max_raw_candidates = max(200, min(600, requested_total * 40))

    for idx in order_local:
        x0 = float(x_indices[idx])
        y0 = float(y_indices[idx])
        if x0 < edge_margin or x0 > nx - 1 - edge_margin or y0 < edge_margin or y0 > ny - 1 - edge_margin:
            continue
        if _distance_to_positions(x0, y0, existing_positions) < min_distance:
            continue
        if _distance_to_positions(x0, y0, provisional_positions) < min_distance:
            continue

        cx, cy = _centroid_near(image, x0, y0, search_radius)
        if cx < edge_margin or cx > nx - 1 - edge_margin or cy < edge_margin or cy > ny - 1 - edge_margin:
            continue
        if _distance_to_positions(cx, cy, existing_positions) < min_distance:
            continue
        if _distance_to_positions(cx, cy, provisional_positions) < min_distance:
            continue

        _pixel, peak, _mean = _aperture_stats_at(image, cx, cy, aperture_radius)
        if _peak_quality_from_value(peak, values) != "ok":
            continue

        nearest = min(_distance_to_positions(cx, cy, existing_positions), _distance_to_positions(cx, cy, provisional_positions))
        if not np.isfinite(nearest):
            nearest = min_distance
        score = _auto_candidate_score(cx, cy, peak, image.shape, values, nearest)
        candidates.append((score, float(cx), float(cy), float(peak)))
        provisional_positions.append((float(cx), float(cy)))
        if len(candidates) >= max_raw_candidates:
            break

    if not candidates:
        raise ValueError("No suitable comparison-star candidates were found in the current image.")

    candidates.sort(key=lambda row: row[0])

    # Validate the whole candidate pool using the same sampled-frame idea used
    # by Check peaks, but without adding/removing stars repeatedly.  Each FITS
    # image is read once and all candidates are tested on that image.
    n_peak_frames = parse_int(values.get("-PHOTO_PEAK_MAX_FRAMES-", 15), 15)
    sample_indices = _sample_frame_indices(len(session.files), int(n_peak_frames)) if session.files else []
    if not sample_indices:
        sample_indices = [session.current_index]

    recenter = bool(values.get("-PHOTO_RECENTER-", True))
    peaks_by_candidate: List[List[float]] = [[] for _ in candidates]
    valid_candidate = [True for _ in candidates]

    n_indices = len(sample_indices)
    for count, file_index in enumerate(sample_indices, start=1):
        try:
            sample_image, _header = _read_fits_image(session.files[file_index]) if session.files else (image, None)
        except Exception:
            continue
        for cand_idx, (_score, cx0, cy0, _peak0) in enumerate(candidates):
            if not valid_candidate[cand_idx]:
                # A high candidate is already rejected; still skip it to keep the
                # validation pass responsive on large fields.
                continue
            x = float(cx0)
            y = float(cy0)
            if recenter:
                x, y = _centroid_near(sample_image, x, y, search_radius)
            _pixel, peak, _mean = _aperture_stats_at(sample_image, x, y, aperture_radius)
            if not np.isfinite(peak):
                valid_candidate[cand_idx] = False
                continue
            peaks_by_candidate[cand_idx].append(float(peak))
            # A single frame above the high limit is enough to reject a star as
            # potentially non-linear/saturated.
            if peak > high_peak:
                valid_candidate[cand_idx] = False
        try:
            window["-PHOTO_PROGRESS-"].update(int(count / max(1, n_indices) * 100.0))
            if count % 3 == 0:
                window.refresh()
        except Exception:
            pass

    accepted_rows: List[Tuple[float, float, float, float, float, str]] = []
    rejected_high = 0
    rejected_low = 0
    rejected_unknown = 0
    for cand_idx, (score, cx, cy, current_peak) in enumerate(candidates):
        peaks_for_candidate = peaks_by_candidate[cand_idx]
        if not peaks_for_candidate:
            rejected_unknown += 1
            continue
        peak_median = float(np.nanmedian(peaks_for_candidate))
        peak_max = float(np.nanmax(peaks_for_candidate))
        quality = _peak_quality_from_stats(peak_median, peak_max, values)
        if not valid_candidate[cand_idx] or quality == "high":
            rejected_high += 1
            continue
        if quality == "low":
            rejected_low += 1
            continue
        if quality != "ok":
            rejected_unknown += 1
            continue
        accepted_rows.append((score, cx, cy, peak_median, peak_max, "sampled sequence"))

    added_ids: List[str] = []
    occupied = list(existing_positions)
    skipped_distance_after_validation = 0
    for _score, cx, cy, peak_median, peak_max, source in accepted_rows:
        if len(added_ids) >= slots_to_fill:
            break
        if _distance_to_positions(cx, cy, occupied) < min_distance:
            skipped_distance_after_validation += 1
            continue
        star = _add_star(session, cx, cy, "comparison")
        star.peak_median = float(peak_median)
        star.peak_max = float(peak_max)
        star.peak_quality = _peak_quality_from_stats(star.peak_median, star.peak_max, values)
        star.peak_source = source
        added_ids.append(star.star_id)
        occupied.append((cx, cy))

    try:
        window["-PHOTO_PROGRESS-"].update(0)
    except Exception:
        pass

    if not added_ids:
        raise ValueError(
            "Candidate stars were detected, but none passed the sampled peak validation and distance filters. "
            "Try widening the OK fraction range, increasing the saturation/non-linearity level if appropriate, "
            "or reducing the minimum distance."
        )

    _update_star_list(window, session)
    _refresh_star_overlays(window, session, values)

    mode = "full sequence" if len(sample_indices) >= len(session.files) else f"sample {len(sample_indices)}/{len(session.files)} frames"
    final_total = current_comp_count + len(added_ids)
    report = [
        "Automatic comparison-star search complete.",
        f"Requested maximum comparison stars: {requested_total}",
        f"Existing comparison/check stars before search: {current_comp_count}",
        f"Additional slots requested: {slots_to_fill}",
        f"Raw candidates detected on current image: {len(candidates)}",
        f"Valid candidates after peak check: {len(accepted_rows)}",
        f"Candidates added in this click: {len(added_ids)}",
        f"Final comparison/check stars: {final_total}",
        f"Rejected as HIGH in sampled frames: {rejected_high}",
        f"Rejected as LOW in sampled frames: {rejected_low}",
        f"Rejected/unknown peak check: {rejected_unknown}",
        f"Skipped by final distance check: {skipped_distance_after_validation}",
        f"Peak validation: {mode}",
        f"Peak OK range: {low_peak:.0f} - {high_peak:.0f} ADU",
        f"Minimum separation used: {min_distance:.1f} px",
        "These are candidate comparison stars; inspect their light curves in the main Comp stars tab before final analysis.",
    ]
    window["-PHOTO_REPORT-"].update("\n".join(report) + "\n", append=True)

def _robust_sky_std(values: np.ndarray) -> float:
    """Return a robust sky scatter estimate."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return math.nan
    med = np.nanmedian(finite)
    mad = 1.4826 * np.nanmedian(np.abs(finite - med))
    if np.isfinite(mad) and mad > 0:
        return float(mad)
    return float(np.nanstd(finite))


def _centroid_near(image: np.ndarray, x: float, y: float, search_radius: float) -> Tuple[float, float]:
    """Estimate a local centroid near an input position."""
    ny, nx = image.shape
    r = max(1.0, float(search_radius))
    x_min = max(0, int(math.floor(x - r)))
    x_max = min(nx - 1, int(math.ceil(x + r)))
    y_min = max(0, int(math.floor(y - r)))
    y_max = min(ny - 1, int(math.ceil(y + r)))
    if x_max <= x_min or y_max <= y_min:
        return x, y

    cutout = image[y_min : y_max + 1, x_min : x_max + 1].astype(float)
    if not np.isfinite(cutout).any():
        return x, y
    background = np.nanmedian(cutout)
    weights = cutout - background
    weights[~np.isfinite(weights)] = 0.0
    weights[weights < 0.0] = 0.0
    total = float(np.nansum(weights))
    if total <= 0:
        return x, y
    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    cx = float(np.nansum(xx * weights) / total)
    cy = float(np.nansum(yy * weights) / total)
    if not np.isfinite(cx) or not np.isfinite(cy):
        return x, y
    return cx, cy


def _measure_one_star(
    image: np.ndarray,
    star: ApertureStar,
    aperture_radius: float,
    sky_inner_radius: float,
    sky_outer_radius: float,
    recenter: bool,
    search_radius: float,
    saturation_level: float,
) -> PhotometryMeasurement:
    """Measure aperture photometry for one star in one image."""
    x = float(star.x)
    y = float(star.y)
    if recenter:
        x, y = _centroid_near(image, x, y, search_radius)

    ny, nx = image.shape
    r_out = max(float(sky_outer_radius), float(aperture_radius) + 1.0)
    x_min = max(0, int(math.floor(x - r_out)))
    x_max = min(nx - 1, int(math.ceil(x + r_out)))
    y_min = max(0, int(math.floor(y - r_out)))
    y_max = min(ny - 1, int(math.ceil(y + r_out)))
    if x_max <= x_min or y_max <= y_min:
        return PhotometryMeasurement(x=x, y=y)

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    aperture_mask = rr2 <= float(aperture_radius) ** 2
    sky_mask = (rr2 >= float(sky_inner_radius) ** 2) & (rr2 <= float(sky_outer_radius) ** 2)

    cutout = image[y_min : y_max + 1, x_min : x_max + 1]
    aperture_values = cutout[aperture_mask]
    sky_values = cutout[sky_mask]
    aperture_values = aperture_values[np.isfinite(aperture_values)]
    sky_values = sky_values[np.isfinite(sky_values)]

    if aperture_values.size == 0:
        return PhotometryMeasurement(x=x, y=y)

    if sky_values.size > 0:
        sky_median = float(np.nanmedian(sky_values))
        sky_std = _robust_sky_std(sky_values)
    else:
        sky_median = 0.0
        sky_std = math.nan

    source_sum = float(np.nansum(aperture_values))
    n_ap = int(aperture_values.size)
    net_flux = source_sum - sky_median * n_ap
    peak = float(np.nanmax(aperture_values))
    mean_ap = float(np.nanmean(aperture_values))

    if np.isfinite(sky_std):
        # Approximate CCD noise in data units.  This intentionally avoids
        # assuming a gain/read-noise that may not be present in reduced images.
        error = math.sqrt(max(abs(source_sum), 0.0) + n_ap * sky_std * sky_std)
    else:
        error = math.sqrt(max(abs(source_sum), 0.0))

    saturated = bool(np.isfinite(saturation_level) and peak >= saturation_level)

    return PhotometryMeasurement(
        flux=net_flux,
        error=float(error),
        sky_median=sky_median,
        sky_std=sky_std,
        peak=peak,
        mean_aperture=mean_ap,
        x=x,
        y=y,
        n_aperture=n_ap,
        n_sky=int(sky_values.size),
        saturated=saturated,
    )


def _next_comparison_id(stars: Sequence[ApertureStar]) -> str:
    """Return the next AIJ-like comparison star ID, C2, C3, ..."""
    used_numbers = []
    for star in stars:
        if star.star_id.upper().startswith("C"):
            try:
                used_numbers.append(int(star.star_id[1:]))
            except Exception:
                pass
    number = 2
    while number in used_numbers:
        number += 1
    return f"C{number}"


def _add_star(session: PhotometrySession, x: float, y: float, role: str) -> ApertureStar:
    """Add or replace a star aperture and return the created star."""
    if role == "target":
        # Keep the first target simple and compatible with the downstream AIJ
        # column detector.
        session.stars = [s for s in session.stars if not s.star_id.upper().startswith("T")]
        star = ApertureStar("T1", x, y, role="target")
        session.stars.insert(0, star)
        return star
    if role == "check":
        star_id = _next_comparison_id(session.stars)
        star = ApertureStar(star_id, x, y, role="check")
        session.stars.append(star)
        return star
    star_id = _next_comparison_id(session.stars)
    star = ApertureStar(star_id, x, y, role="comparison")
    session.stars.append(star)
    return star


def _delete_nearest_star(session: PhotometrySession, x: float, y: float) -> None:
    """Delete the selected aperture closest to an image coordinate."""
    if not session.stars:
        return
    distances = [math.hypot(star.x - x, star.y - y) for star in session.stars]
    index = int(np.argmin(distances))
    if distances[index] <= 30.0:
        del session.stars[index]


def _star_sort_key(star: ApertureStar) -> Tuple[int, int]:
    """Sort T stars before C stars in natural order."""
    sid = star.star_id.upper()
    prefix_rank = 0 if sid.startswith("T") else 1
    try:
        number = int(sid[1:])
    except Exception:
        number = 999
    return prefix_rank, number


def _renumber_comparisons(session: PhotometrySession) -> None:
    """Renumber comparison/check stars consecutively after deletion/load."""
    target_seen = False
    comparisons: List[ApertureStar] = []
    new_stars: List[ApertureStar] = []
    for star in session.stars:
        if star.role == "target" and not target_seen:
            new_stars.append(ApertureStar(
                "T1", star.x, star.y, role="target",
                peak_quality=getattr(star, "peak_quality", "unknown"),
                peak_median=getattr(star, "peak_median", math.nan),
                peak_max=getattr(star, "peak_max", math.nan),
                peak_source=getattr(star, "peak_source", "not checked"),
            ))
            target_seen = True
        elif star.role in ("comparison", "check") or star.star_id.upper().startswith("C"):
            comparisons.append(star)
    for idx, star in enumerate(comparisons, start=2):
        new_stars.append(ApertureStar(
            f"C{idx}", star.x, star.y, role=star.role,
            peak_quality=getattr(star, "peak_quality", "unknown"),
            peak_median=getattr(star, "peak_median", math.nan),
            peak_max=getattr(star, "peak_max", math.nan),
            peak_source=getattr(star, "peak_source", "not checked"),
        ))
    session.stars = new_stars


def _star_list_labels(session: PhotometrySession) -> List[str]:
    """Return labels for the aperture listbox, including cached peak feedback."""
    labels = []
    for star in sorted(session.stars, key=_star_sort_key):
        role = star.role
        if role == "check":
            role = "check/C"
        quality_label, _color, _bg = _peak_quality_style(getattr(star, "peak_quality", "unknown"))
        peak_med = getattr(star, "peak_median", math.nan)
        peak_max = getattr(star, "peak_max", math.nan)
        source = str(getattr(star, "peak_source", "not checked") or "not checked")
        if np.isfinite(peak_med) or np.isfinite(peak_max):
            peak_text = f"{quality_label:<4s} med={peak_med:7.0f} max={peak_max:7.0f} [{source}]"
        else:
            peak_text = f"{quality_label:<4s} med=   n/a max=   n/a [{source}]"
        labels.append(f"{star.star_id:>3s}  x={star.x:.0f}  y={star.y:.0f}  {peak_text}")
    return labels


def _parse_star_id_from_label(label: str) -> str:
    """Extract the star ID from a listbox label."""
    return str(label).strip().split()[0]


def _update_star_list(window: sg.Window, session: PhotometrySession) -> None:
    """Update aperture list and counter labels."""
    try:
        window["-PHOTO_STAR_LIST-"].update(values=_star_list_labels(session))
        n_targets = sum(1 for s in session.stars if s.role == "target")
        n_comp = sum(1 for s in session.stars if s.role in ("comparison", "check"))
        # window["-PHOTO_STAR_COUNT-"].update(f"T={n_targets}, C={n_comp}")
    except Exception:
        pass



def _delete_photometry_figure(session: PhotometrySession) -> None:
    """Remove the current embedded photometry figure from the GUI."""
    if session.figure_agg is not None:
        try:
            session.figure_agg.get_tk_widget().destroy()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt

            plt.close(session.figure_agg.figure)
        except Exception:
            pass
    session.figure_agg = None
    session.figure = None
    session.ax = None
    session.hover_artists = []
    session.overlay_artists = []


def _remember_view(session: PhotometrySession) -> None:
    """Store the current image view limits before a redraw."""
    try:
        if session.ax is not None:
            xlim = tuple(float(v) for v in session.ax.get_xlim())
            ylim = tuple(float(v) for v in session.ax.get_ylim())
            if all(np.isfinite(xlim)) and all(np.isfinite(ylim)):
                session.view_xlim = xlim
                session.view_ylim = ylim
    except Exception:
        pass


def _full_image_view(session: PhotometrySession) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return default full-frame image limits."""
    if session.image is None:
        return (0.0, 1.0), (0.0, 1.0)
    ny, nx = session.image.shape
    return (-0.5, float(nx) - 0.5), (-0.5, float(ny) - 0.5)


def _reset_image_view(session: PhotometrySession) -> None:
    """Reset the stored zoom/pan state to the full FITS frame."""
    session.view_xlim = None
    session.view_ylim = None


def _draw_embedded_figure(canvas_elem: sg.Canvas, figure: Figure) -> FigureCanvasTkAgg:
    """Draw a Matplotlib figure in a Tk canvas, local to this sub-program."""
    canvas_agg = FigureCanvasTkAgg(figure, canvas_elem.TKCanvas)
    canvas_agg.draw()
    canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=True)
    return canvas_agg


def _star_color(star: ApertureStar) -> str:
    """Return a high-contrast display color for an aperture role."""
    if star.role == "target":
        return "tab:red"
    if star.role == "check":
        return "tab:cyan"
    return "tab:orange"


def _star_display_color(star: ApertureStar) -> str:
    """Return diagnostic color when peak feedback is available."""
    quality = str(getattr(star, "peak_quality", "unknown") or "unknown").lower()
    if quality != "unknown":
        _label, color, _bg = _peak_quality_style(quality)
        return color
    return _star_color(star)


def _update_hover_overlay(window: sg.Window, session: PhotometrySession, values: Dict[str, object], x: float, y: float, color: str = "lime") -> None:
    """Draw or update the live aperture preview under the mouse cursor.

    The first implementation removed and recreated three Matplotlib patches at
    every mouse-motion event.  On large FITS images this was CPU-heavy and made
    the preview lag behind the cursor.  Here the hover patches are created once
    and only their center/radius/visibility are updated.
    """
    if session.ax is None or session.figure_agg is None or session.image is None:
        return

    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    sky_in = parse_float(values.get("-PHOTO_SKY_IN-", 10.0), 10.0)
    sky_out = parse_float(values.get("-PHOTO_SKY_OUT-", 16.0), 16.0)

    if len(session.hover_artists) != 3:
        for artist in list(session.hover_artists):
            try:
                artist.remove()
            except Exception:
                pass
        session.hover_artists = [
            Circle((0.0, 0.0), aper, fill=False, lw=1.8, ec=color, alpha=0.95),
            Circle((0.0, 0.0), sky_in, fill=False, lw=1.0, ls="--", ec=color, alpha=0.70),
            Circle((0.0, 0.0), sky_out, fill=False, lw=1.0, ls=":", ec=color, alpha=0.70),
        ]
        for artist in session.hover_artists:
            session.ax.add_patch(artist)

    visible = bool(np.isfinite(x) and np.isfinite(y))
    for artist, radius in zip(session.hover_artists, (aper, sky_in, sky_out)):
        try:
            artist.center = (float(x), float(y))
            artist.radius = float(radius)
            artist.set_visible(visible)
            artist.set_edgecolor(color)
        except Exception:
            pass

    try:
        session.figure_agg.draw_idle()
    except Exception:
        pass


def _zoom_image(session: PhotometrySession, x: float, y: float, step: float) -> None:
    """Zoom the FITS preview around an image coordinate."""
    if session.ax is None or session.figure_agg is None or not np.isfinite(x) or not np.isfinite(y):
        return
    xlim = session.ax.get_xlim()
    ylim = session.ax.get_ylim()
    if step > 0:
        scale = 0.80
    else:
        scale = 1.25
    new_xlim = (x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
    new_ylim = (y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
    session.ax.set_xlim(new_xlim)
    session.ax.set_ylim(new_ylim)
    session.view_xlim = tuple(float(v) for v in new_xlim)
    session.view_ylim = tuple(float(v) for v in new_ylim)
    try:
        session.figure_agg.draw_idle()
    except Exception:
        pass


def _zoom_image_center(session: PhotometrySession, step: float) -> None:
    """Zoom around the center of the current displayed image."""
    if session.ax is None:
        return
    xlim = session.ax.get_xlim()
    ylim = session.ax.get_ylim()
    _zoom_image(session, 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1]), step)


def _start_pan(session: PhotometrySession, x_pixel: float, y_pixel: float) -> None:
    """Start panning in canvas/display pixel coordinates."""
    if session.ax is None or not np.isfinite(x_pixel) or not np.isfinite(y_pixel):
        return
    session.pan_active = True
    session.pan_start_xy = (float(x_pixel), float(y_pixel))
    session.pan_start_xlim = tuple(float(v) for v in session.ax.get_xlim())
    session.pan_start_ylim = tuple(float(v) for v in session.ax.get_ylim())


def _pan_to(session: PhotometrySession, x_pixel: float, y_pixel: float) -> None:
    """Pan the image using canvas/display pixel deltas.

    The first zoom/pan implementation used event.xdata/event.ydata.  That is
    fragile because those values are recomputed after every axes-limit update;
    in some Tk backends this produced flickering and almost no visible motion.
    Pixel-based panning is independent of the current data transform and behaves
    like a normal image viewer.
    """
    if not session.pan_active or session.ax is None or session.figure_agg is None:
        return
    if session.pan_start_xy is None or session.pan_start_xlim is None or session.pan_start_ylim is None:
        return
    if not np.isfinite(x_pixel) or not np.isfinite(y_pixel):
        return

    bbox = session.ax.bbox
    width = float(getattr(bbox, "width", 0.0) or 0.0)
    height = float(getattr(bbox, "height", 0.0) or 0.0)
    if width <= 0.0 or height <= 0.0:
        return

    start_xlim = session.pan_start_xlim
    start_ylim = session.pan_start_ylim
    x_span = start_xlim[1] - start_xlim[0]
    y_span = start_ylim[1] - start_ylim[0]

    dx_data = (float(x_pixel) - session.pan_start_xy[0]) * x_span / width
    dy_data = (float(y_pixel) - session.pan_start_xy[1]) * y_span / height

    # Dragging the image to the right/up should move the image content to the
    # right/up, hence the data window moves in the opposite direction.
    new_xlim = (start_xlim[0] - dx_data, start_xlim[1] - dx_data)
    new_ylim = (start_ylim[0] - dy_data, start_ylim[1] - dy_data)
    session.ax.set_xlim(new_xlim)
    session.ax.set_ylim(new_ylim)
    session.view_xlim = tuple(float(v) for v in new_xlim)
    session.view_ylim = tuple(float(v) for v in new_ylim)
    try:
        session.figure_agg.draw_idle()
    except Exception:
        pass


def _stop_pan(session: PhotometrySession) -> None:
    """End panning and clear any pending click state."""
    session.pan_active = False
    session.pan_start_xy = None
    session.pan_start_xlim = None
    session.pan_start_ylim = None
    session.pending_click_active = False
    session.pending_click_pixel = None
    session.pending_click_data = None
    session.pending_click_xlim = None
    session.pending_click_ylim = None


def _cancel_pending_click(session: PhotometrySession) -> None:
    """Clear a stored left-click without stopping a current pan."""
    session.pending_click_active = False
    session.pending_click_pixel = None
    session.pending_click_data = None
    session.pending_click_xlim = None
    session.pending_click_ylim = None

def _clear_star_overlays(session: PhotometrySession) -> None:
    """Remove fixed aperture overlays without touching image view limits."""
    for artist in list(session.overlay_artists):
        try:
            artist.remove()
        except Exception:
            pass
    session.overlay_artists = []


def _draw_star_overlays(
    session: PhotometrySession,
    aperture_radius: float,
    sky_inner_radius: float,
    sky_outer_radius: float,
    redraw: bool = True,
) -> None:
    """Draw fixed selected apertures on the current axes.

    This deliberately updates only patches/texts rather than recreating the
    whole Matplotlib canvas.  Recreating the embedded canvas after every click
    caused unpleasant jumps in zoomed views on some Tk/PyInstaller builds.
    """
    if session.ax is None:
        return
    _clear_star_overlays(session)
    artists: List[object] = []
    aper = float(aperture_radius)
    sky_in = float(sky_inner_radius)
    sky_out = float(sky_outer_radius)
    for star in session.stars:
        color = _star_display_color(star)
        if star.role == "target":
            line_width = 2.2
            label = star.star_id
        elif star.role == "check":
            line_width = 1.8
            label = f"{star.star_id}*"
        else:
            line_width = 1.6
            label = star.star_id
        patch_ap = Circle((star.x, star.y), aper, fill=False, lw=line_width, ec=color, alpha=0.98)
        patch_in = Circle((star.x, star.y), sky_in, fill=False, lw=1.0, ls="--", ec=color, alpha=0.80)
        patch_out = Circle((star.x, star.y), sky_out, fill=False, lw=1.0, ls=":", ec=color, alpha=0.80)
        txt = session.ax.text(
            star.x + aper + 2.0,
            star.y + aper + 2.0,
            label,
            fontsize=8,
            weight="bold",
            color=color,
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1.0},
        )
        for artist in (patch_ap, patch_in, patch_out):
            session.ax.add_patch(artist)
            artists.append(artist)
        artists.append(txt)
    session.overlay_artists = artists
    if redraw and session.figure_agg is not None:
        try:
            session.figure_agg.draw_idle()
        except Exception:
            pass


def _refresh_star_overlays(window: sg.Window, session: PhotometrySession, values: Dict[str, object]) -> None:
    """Refresh only the selected-aperture overlays in the existing image view."""
    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    sky_in = parse_float(values.get("-PHOTO_SKY_IN-", 10.0), 10.0)
    sky_out = parse_float(values.get("-PHOTO_SKY_OUT-", 16.0), 16.0)
    _draw_star_overlays(session, aper, sky_in, sky_out, redraw=True)
    try:
        window["-PHOTO_INDEX-"].update(f"{session.current_index + 1}/{len(session.files)}")
    except Exception:
        pass

def _draw_image(window: sg.Window, session: PhotometrySession, values: Dict[str, object]) -> None:
    """Redraw the FITS preview image and aperture overlays."""
    if session.image is None:
        return
    _remember_view(session)
    _delete_photometry_figure(session)

    fig = Figure(figsize=(8.2, 6.6), dpi=110)
    ax = fig.add_subplot(111)

    low_p = parse_float(values.get("-PHOTO_LOW_P-", 1.0), 1.0)
    high_p = parse_float(values.get("-PHOTO_HIGH_P-", 99.5), 99.5)
    vmin, vmax = _image_stretch(session.image, low_p, high_p)
    ax.imshow(session.image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    # Store the freshly created axes before drawing aperture overlays.  In the
    # previous version the overlays were requested while ``session.ax`` still
    # pointed to the old destroyed canvas (or to None after a full redraw), so
    # apertures loaded from JSON could appear in the text list but not on the
    # image.
    session.figure = fig
    session.ax = ax

    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    sky_in = parse_float(values.get("-PHOTO_SKY_IN-", 10.0), 10.0)
    sky_out = parse_float(values.get("-PHOTO_SKY_OUT-", 16.0), 16.0)

    # Draw fixed aperture overlays after the image has been displayed.
    session.overlay_artists = []
    _draw_star_overlays(session, aper, sky_in, sky_out, redraw=False)

    if session.view_xlim is not None and session.view_ylim is not None:
        try:
            ax.set_xlim(session.view_xlim)
            ax.set_ylim(session.view_ylim)
        except Exception:
            session.view_xlim, session.view_ylim = _full_image_view(session)
            ax.set_xlim(session.view_xlim)
            ax.set_ylim(session.view_ylim)
    else:
        session.view_xlim, session.view_ylim = _full_image_view(session)
        ax.set_xlim(session.view_xlim)
        ax.set_ylim(session.view_ylim)

    ax.set_title(Path(session.files[session.current_index]).name if session.files else "No image")
    ax.set_xlabel("X [pixel]")
    ax.set_ylabel("Y [pixel]")
    fig.tight_layout()

    session.figure_agg = _draw_embedded_figure(window["-PHOTO_CANVAS-"], fig)
    _connect_image_events(window, session)

    try:
        window["-PHOTO_INDEX-"].update(f"{session.current_index + 1}/{len(session.files)}")
    except Exception:
        pass


def _connect_image_events(window: sg.Window, session: PhotometrySession) -> None:
    """Connect matplotlib events for zoom, pan, frame navigation and aperture selection."""
    if session.figure_agg is None:
        return

    drag_threshold_px = 5.0

    def _click(event):
        # Right or middle button: immediate pan.  Left button is delayed until
        # release so that left-drag can be used for panning without adding an
        # accidental aperture.
        button = int(event.button or 1)
        if button in (2, 3):
            _cancel_pending_click(session)
            _start_pan(session, float(event.x), float(event.y))
            return
        if button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        session.pending_click_active = True
        session.pending_click_pixel = (float(event.x), float(event.y))
        session.pending_click_data = (float(event.xdata), float(event.ydata))
        if session.ax is not None:
            session.pending_click_xlim = tuple(float(v) for v in session.ax.get_xlim())
            session.pending_click_ylim = tuple(float(v) for v in session.ax.get_ylim())

    def _release(event):
        # End panning first.  If this was a left-drag promoted to pan, no
        # aperture is created.
        if session.pan_active:
            _stop_pan(session)
            return
        if not session.pending_click_active:
            return
        if session.pending_click_data is None:
            _cancel_pending_click(session)
            return
        x, y = session.pending_click_data
        _cancel_pending_click(session)
        window.write_event_value(
            "-PHOTO_IMAGE_CLICK-",
            {"x": float(x), "y": float(y), "button": 1},
        )

    def _motion(event):
        # Pan can continue even when the mouse is temporarily outside the axes,
        # therefore use event.x/event.y rather than xdata/ydata.
        if session.pan_active:
            _pan_to(session, float(event.x), float(event.y))
            return

        # Promote a held left click to panning once it has moved a few pixels.
        # This makes the viewer usable with a simple left-drag while keeping a
        # plain left click for aperture placement.
        if session.pending_click_active and session.pending_click_pixel is not None:
            dx = float(event.x) - session.pending_click_pixel[0]
            dy = float(event.y) - session.pending_click_pixel[1]
            if (dx * dx + dy * dy) ** 0.5 >= drag_threshold_px:
                if session.pending_click_xlim is not None and session.pending_click_ylim is not None:
                    session.pan_active = True
                    session.pan_start_xy = session.pending_click_pixel
                    session.pan_start_xlim = session.pending_click_xlim
                    session.pan_start_ylim = session.pending_click_ylim
                    _pan_to(session, float(event.x), float(event.y))
                session.pending_click_active = False
                return

        if event.xdata is None or event.ydata is None:
            return

        # Throttle expensive hover updates.  Matplotlib can emit hundreds of
        # motion events per second; forwarding all of them to the GUI event loop
        # makes the CPU jump to 100% on large images.
        now = time.monotonic()
        pixel = (float(event.x), float(event.y))
        if session.last_motion_pixel is not None:
            dx = pixel[0] - session.last_motion_pixel[0]
            dy = pixel[1] - session.last_motion_pixel[1]
            moved = (dx * dx + dy * dy) ** 0.5
        else:
            moved = 999.0
        if (now - session.last_motion_emit_time) < 0.070 and moved < 10.0:
            return
        session.last_motion_emit_time = now
        session.last_motion_pixel = pixel

        window.write_event_value(
            "-PHOTO_IMAGE_MOTION-",
            {"x": float(event.xdata), "y": float(event.ydata)},
        )

    def _scroll(event):
        # Default behaviour: wheel zooms the image around the cursor.  Hold Shift
        # while using the wheel to move through the FITS sequence.
        key = str(getattr(event, "key", "") or "").lower()
        step = float(getattr(event, "step", 0.0) or (1.0 if getattr(event, "button", "") == "up" else -1.0))
        if "shift" in key:
            direction = 1 if step > 0 else -1
            window.write_event_value("-PHOTO_IMAGE_SCROLL-", direction)
            return
        if event.xdata is None or event.ydata is None:
            _zoom_image_center(session, step)
        else:
            _zoom_image(session, float(event.xdata), float(event.ydata), step)

    def _key(event):
        key = str(getattr(event, "key", "") or "").lower()
        if key in ("right", "down", "pagedown"):
            window.write_event_value("-PHOTO_IMAGE_SCROLL-", -1)
        elif key in ("left", "up", "pageup"):
            window.write_event_value("-PHOTO_IMAGE_SCROLL-", 1)
        elif key in ("+", "=", "plus"):
            _zoom_image_center(session, 1.0)
        elif key in ("-", "minus", "_"):
            _zoom_image_center(session, -1.0)
        elif key in ("f", "home"):
            window.write_event_value("-PHOTO_FIT_VIEW-", None)

    session.figure_agg.mpl_connect("button_press_event", _click)
    session.figure_agg.mpl_connect("button_release_event", _release)
    session.figure_agg.mpl_connect("motion_notify_event", _motion)
    session.figure_agg.mpl_connect("scroll_event", _scroll)
    session.figure_agg.mpl_connect("key_press_event", _key)


def _build_layout() -> List[List[sg.Element]]:
    """Create the aperture-photometry builder layout."""
    input_frame = [
        [
            sg.Text("FITS folder", size=(12, 1)),
            sg.Input("", key="-PHOTO_FOLDER-", size=(36, 1)),
            sg.FolderBrowse("Browse", tooltip='Set the folder containing your image sequence.\nThe sequence MUST be already calibrated with dark, flat, bias and the frames aligned.\nfit(s) files are required'),
        ],
        [
            sg.Text("Pattern", size=(9, 1)),
            sg.Input("*.fit*", key="-PHOTO_PATTERN-", size=(12, 1)),
            sg.Button("Load sequence", tooltip='Load the browsed sequence here'),
            sg.Text("Image"),
            sg.Button("<", key="-PHOTO_PREV-"),
            sg.Text("0/0", key="-PHOTO_INDEX-", size=(8, 1)),
            sg.Button(">", key="-PHOTO_NEXT-"),
        ],
    ]

    aperture_frame = [
        [
            sg.Text("Aperture", size=(10, 1), tooltip='Set the aperture sizes here: inner radius, inner sky annulus, outer sky annulus.\nYou can change these values even after you set the apertures on the stars\nYou will see the changes in size in real time in the plot'),
            sg.Input("11", key="-PHOTO_APER_R-", size=(6, 1), enable_events=True),
            sg.Text("Sky in"),
            sg.Input("19", key="-PHOTO_SKY_IN-", size=(6, 1), enable_events=True),
            sg.Text("Sky out"),
            sg.Input("31", key="-PHOTO_SKY_OUT-", size=(6, 1), enable_events=True),
        ],
        [
            sg.Checkbox("Snap click", default=True, key="-PHOTO_SNAP_CLICK-", tooltip='Snap to the centroid of the selected star in the reference image'),
            sg.Checkbox("Recenter frames", default=True, key="-PHOTO_RECENTER-", tooltip='Recenter to the centroid any aperture, for any image of the sequence'),
            sg.Text("search"),
            sg.Input("8", key="-PHOTO_SEARCH_R-", size=(5, 1)),
            sg.Text("sat/nonlin"),
            sg.Input("60000", key="-PHOTO_SAT_LEVEL-", size=(7, 1), enable_events=True),
        ],
        [
            sg.Text("OK frac", tooltip='Min and max fraction of the saturation level to consider good the luminosity of the stars'),
            sg.Input("0.25", key="-PHOTO_PEAK_LOW_FRAC-", size=(5, 1), enable_events=True),
            sg.Input("0.66", key="-PHOTO_PEAK_HIGH_FRAC-", size=(5, 1), enable_events=True),
            sg.Text("Peak frames", tooltip='The number of frames of the sequence used to calculate the luminosity leveld of the selected stars\nand the automatic detection of the comparison stars\nSet zero to sample the whole sequence'),
            sg.Input("15", key="-PHOTO_PEAK_MAX_FRAMES-", size=(5, 1)),
            sg.Button("Check peaks", key="-PHOTO_CHECK_PEAKS-", tooltip='When you move the mouse over the stars on the right, the apertures will change colors\nYellow = star too faint, Green = star with good luminosity, Red = possibly saturated star\nWhen you select a star, the aperture will maintain it''s color. This color feedback is given considering only the image you are displaying\nThis button extends the luminosity analysis to a subsample (15 by default) of images.\nIt is important to check that the stars you have selected have the right luminosity\nAfter selecting the target and all the comparison stars you want, press this button to see if some of them are not suitable for photometry\nSet zero to extend the luminosity analysis to the whole sequence'),
        ],
        [
            sg.Button("Auto find comps", key="-PHOTO_AUTO_FIND_COMPS-", tooltip='Automatic algorithm to select the preliminary sample of suitable comparison stars.\nSuitable stars will have the peak luminosity within the OK fraction set above\nare not contaminated by other stars and will not contain other bright stars in the sky mearurement annulus\nThis preliminary sample will be fine-tuned in the main program, in the Comp stars tab'),
            sg.Text("max", tooltip='Maximum number of comparison stars found by the Auto find comps algorithm'),
            sg.Input("15", key="-PHOTO_AUTO_MAX_COMPS-", size=(5, 1)),
            sg.Text("min dist", tooltip='Minimum distance, in pixel, bewteen two stars to be considered eligible by the Auto find comp algorithm.\n Leave zero to automatic selection'),
            sg.Input("0", key="-PHOTO_AUTO_MIN_DIST-", size=(5, 1)),
            sg.Text("0 = auto", text_color="gray"),
        ],
        [
            sg.Text("Star type", size=(10, 1), tooltip='This is the most important section: you have to set which type of star you are considering\nand then click on that star in the image. One target only, as many comparison stars you want'),
            sg.Radio("Target", "PHOTO_MODE", key="-PHOTO_MODE_TARGET-", default=True, tooltip='Set the aperture to your target. Only one\nThis is required also if you want to use the automatic detection of comparison stars'),
            sg.Radio("Comparison", "PHOTO_MODE", key="-PHOTO_MODE_COMP-", tooltip='Set the apertures to comparison stars, if you prefer the manual mode.\nYou can select as many stars as you want'),
            sg.Radio("Check", "PHOTO_MODE", key="-PHOTO_MODE_CHECK-", tooltip='If you want to add also a check star. Optional, and often not useful'),
            sg.Radio("Delete", "PHOTO_MODE", key="-PHOTO_MODE_DELETE-", tooltip='To manually delete any aperture you want directly on the image\nActivate this option, then click on the aperture(s) you want to delete in the image'),
        ],
        [
            sg.Button("Delete selected", key="-PHOTO_DELETE_SELECTED-", tooltip='Select an aperture in the list and delete here'),
            sg.Button("Clear apertures", key="-PHOTO_CLEAR_STARS-", tooltip='Delete all the apertures'),
            sg.Button("Save apertures", key="-PHOTO_SAVE_APERTURES-", tooltip='Save the apertures. Very useful for reproducible results'),
            sg.Button("Load apertures", key="-PHOTO_LOAD_APERTURES-", tooltip='Load saved apertures. Of course, you should use the same sequence!'),
        ],
        # [sg.Text("Stars", size=(7, 1)), sg.Text("T=0, C=0", key="-PHOTO_STAR_COUNT-", size=(12, 1))],
        [sg.Listbox([], key="-PHOTO_STAR_LIST-", size=(59, 6), enable_events=False)],
    ]

    header_frame = [
        [sg.Text("Time keys", size=(12, 1), tooltip='Plausible fits file keywords to allow the program to recognise the time\n in your images. Pproblably you''ll never have to worry about them'), sg.Input("JD_UTC,JD,JD-OBS,MJD-OBS,MJD,DATE-OBS", key="-PHOTO_TIME_KEYS-", size=(44, 1))],
        [sg.Text("Exp key", tooltip='Plausible exposure time fits keywords to allow the program to recognise the exposure'), sg.Input("EXPTIME", key="-PHOTO_EXPTIME_KEYS-", size=(9, 1)), sg.Text("Airmass", tooltip='Plausible fits file keywords to allow the program to recognise the airmass values in your sequence'), sg.Input("AIRMASS", key="-PHOTO_AIRMASS_KEYS-", size=(10, 1)), sg.Text("Filter keys", tooltip='Plausible fits file keywords to allow the program to recognise the filters used. It''s just a plus, not very relevant'), sg.Input("FILTER,FILT,INSFLNAM", key="-PHOTO_FILTER_KEYS-", size=(12, 1))],
        [
            sg.Text("FITS time ref", tooltip='Time reference of fits files. Leave default'),
            sg.Combo(["Exposure start", "Mid-exposure", "Exposure end"], default_value="Exposure start", key="-PHOTO_TIME_REF-", size=(13, 1), readonly=True),
            sg.Text("Main JD_UTC", tooltip='Where take the Julian date info. If FITS time ref is set to Exposure start\nhere you should leave Mid-exposure corrected'),
            sg.Combo(["Header time", "Mid-exposure corrected"], default_value="Mid-exposure corrected", key="-PHOTO_MAIN_JDUTC-", size=(18, 1), readonly=True),
        ],
    ]

    display_frame = [
        [
            sg.Text("Display %", size=(10, 1)),
            sg.Input("1", key="-PHOTO_LOW_P-", size=(4, 1), enable_events=True),
            sg.Input("99.5", key="-PHOTO_HIGH_P-", size=(4, 1), enable_events=True),
            sg.Button("Refresh image", key="-PHOTO_REFRESH-"),
            sg.Button("Zoom +", key="-PHOTO_ZOOM_IN-"),
            sg.Button("Zoom -", key="-PHOTO_ZOOM_OUT-"),
            sg.Button("Fit view", key="-PHOTO_FIT_VIEW-"),
            # sg.Text("Wheel=zoom, drag=pan, click=aperture, Shift+wheel=images", size=(54, 1)),
        ],
        [sg.Text("Mouse/aperture:", key="-PHOTO_MOUSE_INFO-", size=(56, 1))],
    ]

    run_frame = [
        [
            sg.Text("Output", size=(8, 1)),
            sg.Input("", key="-PHOTO_OUTPUT-", size=(42, 1)),
            sg.FileSaveAs("Set path", file_types=(("Text table", "*.txt"), ("CSV", "*.csv"), ("All files", "*.*")), tooltip='Set the location and the name of the photometry file.\nIt will be saved automatically once you press Run photometry or Run + load in main'),
        ],
        [
            sg.Button("Run photometry", button_color=("white", "#2d6cdf"), tooltip='Run photometry and save the file, maintaining open this window'),
            sg.Button("Run + load in main", key="-PHOTO_RUN_AND_LOAD-", tooltip='Run photometry, save the file, close the window, and load the file to the main program'),
            sg.Button("Close"),
        ],
        [sg.ProgressBar(100, orientation="h", size=(44, 12), key="-PHOTO_PROGRESS-")],
        [sg.Multiline("", key="-PHOTO_REPORT-", size=(58, 6), disabled=True, autoscroll=True)],
    ]

    left_col = [
        [sg.Frame("1. Input sequence", input_frame, font=("Helvetica", 13, 'bold'))],
        [sg.Frame("2. Set Apertures", aperture_frame, font=("Helvetica", 13, 'bold'))],
        [sg.Frame("FITS header mapping", header_frame)],
        [sg.Frame("Display and cursor diagnostics", display_frame)],
        [sg.Frame("3. Run Photometry", run_frame, font=("Helvetica", 13, 'bold'))],
    ]

    right_col = [[sg.Canvas(key="-PHOTO_CANVAS-", size=(760, 620), expand_x=True, expand_y=True)]]

    return [[sg.Column(left_col, vertical_alignment="top"), sg.VSeparator(), sg.Column(right_col, expand_x=True, expand_y=True)]]




def _default_output_path(folder: str) -> str:
    """Return a sensible default photometry-table path."""
    if not folder:
        return ""
    return str(Path(folder) / "photocurve_aperture_photometry.txt")


def _load_sequence(window: sg.Window, session: PhotometrySession, values: Dict[str, object]) -> None:
    """Load the FITS sequence selected in the GUI."""
    folder = str(values.get("-PHOTO_FOLDER-", "")).strip()
    pattern = str(values.get("-PHOTO_PATTERN-", "*.fit*")).strip() or "*.fit*"
    if not folder or not Path(folder).is_dir():
        raise ValueError("Select a valid FITS folder.")
    files = sorted(glob.glob(str(Path(folder) / pattern)))
    files = [path for path in files if Path(path).is_file()]
    if not files:
        raise ValueError(f"No FITS files matching {pattern!r} were found in {folder}")
    session.files = files
    session.current_index = 0
    _reset_image_view(session)
    _load_current_image(session)
    if not str(values.get("-PHOTO_OUTPUT-", "")).strip():
        window["-PHOTO_OUTPUT-"].update(_default_output_path(folder))
    # Show immediately which time keyword is being used and what reference the
    # user says it represents.  This avoids the common 0.5*EXPTIME timing error
    # when DATE-OBS is exposure start but the transit fit is run as mid-exposure.
    time_keys = _split_keywords(values.get("-PHOTO_TIME_KEYS-", ""))
    exptime_keys = _split_keywords(values.get("-PHOTO_EXPTIME_KEYS-", "EXPTIME"))
    jd_header, time_key_used = _jd_from_header_with_key(session.header, time_keys)
    exptime = _header_get_float(session.header, exptime_keys, default=math.nan)
    time_ref = str(values.get("-PHOTO_TIME_REF-", "Exposure start"))
    start_jd, mid_jd, end_jd = _time_reference_offsets_days(jd_header, exptime, time_ref)
    report_lines = [f"Loaded {len(files)} FITS image(s)."]
    if time_key_used:
        report_lines.append(f"First image time key: {time_key_used} = {jd_header:.8f}")
        report_lines.append(f"Assumed FITS timestamp reference: {time_ref}")
        if np.isfinite(exptime):
            report_lines.append(f"Exposure time from header: {exptime:.3f} s")
        if np.isfinite(mid_jd):
            report_lines.append(f"Derived mid-exposure JD_UTC: {mid_jd:.8f}")
    else:
        report_lines.append("No valid time keyword found in the first FITS header.")
    window["-PHOTO_REPORT-"].update("\n".join(report_lines) + "\n")
    _draw_image(window, session, values)


def _selected_click_role(values: Dict[str, object]) -> str:
    """Return target/comparison/check/delete from the active radio button."""
    if bool(values.get("-PHOTO_MODE_DELETE-", False)):
        return "delete"
    if bool(values.get("-PHOTO_MODE_TARGET-", False)):
        return "target"
    if bool(values.get("-PHOTO_MODE_CHECK-", False)):
        return "check"
    return "comparison"


def _save_apertures(path: str, session: PhotometrySession, values: Dict[str, object]) -> None:
    """Save aperture definitions to JSON."""
    data = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "description": "PhotoCurve Lab aperture set. Coordinates are reference-frame FITS pixel coordinates.",
        "aperture_radius": str(values.get("-PHOTO_APER_R-", "")),
        "sky_inner_radius": str(values.get("-PHOTO_SKY_IN-", "")),
        "sky_outer_radius": str(values.get("-PHOTO_SKY_OUT-", "")),
        "snap_click": bool(values.get("-PHOTO_SNAP_CLICK-", True)),
        "recenter_frames": bool(values.get("-PHOTO_RECENTER-", True)),
        "search_radius": str(values.get("-PHOTO_SEARCH_R-", "")),
        "saturation_level": str(values.get("-PHOTO_SAT_LEVEL-", "")),
        "peak_low_fraction": str(values.get("-PHOTO_PEAK_LOW_FRAC-", "")),
        "peak_high_fraction": str(values.get("-PHOTO_PEAK_HIGH_FRAC-", "")),
        "peak_max_frames": str(values.get("-PHOTO_PEAK_MAX_FRAMES-", "")),
        "auto_max_comps": str(values.get("-PHOTO_AUTO_MAX_COMPS-", "")),
        "auto_min_distance": str(values.get("-PHOTO_AUTO_MIN_DIST-", "")),
        "current_reference_file": session.files[session.current_index] if session.files else "",
        "stars": [star.__dict__ for star in session.stars],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _load_apertures(path: str, session: PhotometrySession, window: sg.Window, values: Dict[str, object]) -> None:
    """Load aperture definitions from JSON and show them on the image."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stars = []
    for row in data.get("stars", []):
        try:
            stars.append(ApertureStar(
                str(row["star_id"]),
                float(row["x"]),
                float(row["y"]),
                str(row.get("role", "comparison")),
                str(row.get("peak_quality", "unknown")),
                _safe_float(row.get("peak_median", math.nan), default=math.nan),
                _safe_float(row.get("peak_max", math.nan), default=math.nan),
                str(row.get("peak_source", "loaded file")),
            ))
        except Exception:
            continue
    session.stars = stars
    _renumber_comparisons(session)

    # Restore aperture geometry and useful centring options when available.
    # Older aperture files only contain the first three fields, so all updates
    # are optional and backwards-compatible.
    field_map = [
        ("-PHOTO_APER_R-", "aperture_radius"),
        ("-PHOTO_SKY_IN-", "sky_inner_radius"),
        ("-PHOTO_SKY_OUT-", "sky_outer_radius"),
        ("-PHOTO_SEARCH_R-", "search_radius"),
        ("-PHOTO_SAT_LEVEL-", "saturation_level"),
        ("-PHOTO_PEAK_LOW_FRAC-", "peak_low_fraction"),
        ("-PHOTO_PEAK_HIGH_FRAC-", "peak_high_fraction"),
        ("-PHOTO_PEAK_MAX_FRAMES-", "peak_max_frames"),
        ("-PHOTO_AUTO_MAX_COMPS-", "auto_max_comps"),
        ("-PHOTO_AUTO_MIN_DIST-", "auto_min_distance"),
    ]
    updated_values = dict(values)
    for key, data_key in field_map:
        if data_key in data:
            value = str(data[data_key])
            try:
                window[key].update(value)
            except Exception:
                pass
            updated_values[key] = value

    checkbox_map = [
        ("-PHOTO_SNAP_CLICK-", "snap_click"),
        ("-PHOTO_RECENTER-", "recenter_frames"),
    ]
    for key, data_key in checkbox_map:
        if data_key in data:
            value = bool(data[data_key])
            try:
                window[key].update(value=value)
            except Exception:
                pass
            updated_values[key] = value

    # Loading an aperture file must stay immediate.  Do not scan the whole FITS
    # sequence here; use the current frame as a quick diagnostic fallback and
    # let the user run the sampled/full check with the Check peaks button.
    if session.image is not None:
        for star in session.stars:
            _set_star_peak_from_image(star, session.image, updated_values, source="current frame")
    else:
        _reclassify_all_star_peak_quality(session, updated_values)

    _update_star_list(window, session)

    if session.image is not None:
        # Do not rebuild the whole canvas unless needed.  Loaded apertures are
        # simply fixed overlays on the currently displayed FITS image, so a
        # lightweight overlay refresh preserves zoom/pan and makes the apertures
        # appear immediately.
        _refresh_star_overlays(window, session, updated_values)

    n_loaded = len(session.stars)
    recenter_text = "yes" if bool(updated_values.get("-PHOTO_RECENTER-", True)) else "no"
    search_radius = str(updated_values.get("-PHOTO_SEARCH_R-", ""))
    try:
        window["-PHOTO_REPORT-"].update(
            (
                f"Apertures loaded: {path}\n"
                f"Loaded apertures: {n_loaded}\n"
                "Loaded coordinates are reference-frame FITS pixel coordinates.\n"
                f"Recenter frames during photometry: {recenter_text}; search radius: {search_radius} px\n"
            ),
            append=True,
        )
    except Exception:
        pass

def _relative_curve_columns(table: pd.DataFrame, star_ids: Sequence[str]) -> pd.DataFrame:
    """Add a simple all-comparison differential light curve, if possible."""
    if "T1" not in star_ids:
        return table
    comp_ids = [sid for sid in star_ids if sid.startswith("C")]
    if not comp_ids:
        return table

    target = pd.to_numeric(table.get("Source-Sky_T1"), errors="coerce").to_numpy(dtype=float)
    target_err = pd.to_numeric(table.get("Source_Error_T1"), errors="coerce").to_numpy(dtype=float)

    comp_norms = []
    comp_rel_errs = []
    for comp_id in comp_ids:
        flux = pd.to_numeric(table.get(f"Source-Sky_{comp_id}"), errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(table.get(f"Source_Error_{comp_id}"), errors="coerce").to_numpy(dtype=float)
        finite = flux[np.isfinite(flux) & (flux > 0)]
        med = float(np.nanmedian(finite)) if finite.size else math.nan
        if not np.isfinite(med) or med <= 0:
            continue
        comp_norms.append(flux / med)
        comp_rel_errs.append(err / np.maximum(np.abs(flux), 1e-12))

    if not comp_norms:
        return table

    comp_ensemble = np.nanmean(np.vstack(comp_norms), axis=0)
    target_median = np.nanmedian(target[np.isfinite(target) & (target > 0)])
    if not np.isfinite(target_median) or target_median <= 0:
        return table
    target_norm = target / target_median
    rel_flux = target_norm / comp_ensemble
    rel_flux /= np.nanmedian(rel_flux[np.isfinite(rel_flux)])

    target_rel_err = target_err / np.maximum(np.abs(target), 1e-12)
    comp_rel_err = np.nanmean(np.vstack(comp_rel_errs), axis=0) / math.sqrt(max(1, len(comp_rel_errs))) if comp_rel_errs else np.full_like(target_rel_err, np.nan)
    rel_err = np.abs(rel_flux) * np.sqrt(target_rel_err ** 2 + comp_rel_err ** 2)

    table["rel_flux_T1"] = rel_flux
    table["rel_flux_err_T1"] = rel_err
    return table


def _run_photometry(window: sg.Window, session: PhotometrySession, values: Dict[str, object]) -> str:
    """Run aperture photometry over the full sequence and save a table."""
    if not session.files:
        raise ValueError("Load a FITS sequence first.")
    if not any(star.role == "target" for star in session.stars):
        raise ValueError("Select one target aperture before running photometry.")
    if not any(star.role in ("comparison", "check") for star in session.stars):
        raise ValueError("Select at least one comparison star before running photometry.")

    output_path = str(values.get("-PHOTO_OUTPUT-", "")).strip()
    if not output_path:
        folder = Path(session.files[0]).parent
        output_path = str(folder / "photocurve_aperture_photometry.txt")
        window["-PHOTO_OUTPUT-"].update(output_path)

    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
    sky_in = parse_float(values.get("-PHOTO_SKY_IN-", 10.0), 10.0)
    sky_out = parse_float(values.get("-PHOTO_SKY_OUT-", 16.0), 16.0)
    if sky_in <= aper:
        raise ValueError("The sky inner radius must be larger than the aperture radius.")
    if sky_out <= sky_in:
        raise ValueError("The sky outer radius must be larger than the sky inner radius.")

    recenter = bool(values.get("-PHOTO_RECENTER-", False))
    search_radius = parse_float(values.get("-PHOTO_SEARCH_R-", 8.0), 8.0)
    saturation_level = parse_float(values.get("-PHOTO_SAT_LEVEL-", 60000.0), 60000.0)

    time_keys = _split_keywords(values.get("-PHOTO_TIME_KEYS-", ""))
    exptime_keys = _split_keywords(values.get("-PHOTO_EXPTIME_KEYS-", "EXPTIME"))
    airmass_keys = _split_keywords(values.get("-PHOTO_AIRMASS_KEYS-", "AIRMASS"))
    filter_keys = _split_keywords(values.get("-PHOTO_FILTER_KEYS-", "FILTER"))
    time_reference = str(values.get("-PHOTO_TIME_REF-", "Exposure start"))
    main_jdutc_mode = str(values.get("-PHOTO_MAIN_JDUTC-", "Header time"))

    ordered_stars = sorted(session.stars, key=_star_sort_key)
    star_ids = [star.star_id for star in ordered_stars]

    rows: List[Dict[str, object]] = []
    n_total = len(session.files)
    for idx, path in enumerate(session.files, start=1):
        image, header = _read_fits_image(path)
        jd_header, time_key_used = _jd_from_header_with_key(header, time_keys)
        exptime = _header_get_float(header, exptime_keys, default=math.nan)
        airmass = _header_get_float(header, airmass_keys, default=math.nan)
        filt = _header_get_text(header, filter_keys, default="")
        jd_start, jd_mid, jd_end = _time_reference_offsets_days(jd_header, exptime, time_reference)
        jd_utc = jd_mid if main_jdutc_mode.lower().startswith("mid") else jd_header

        row: Dict[str, object] = {
            "Label": Path(path).name,
            "slice": idx,
            "JD_UTC": jd_utc,
            "J.D.-2400000": jd_utc - 2400000.0 if np.isfinite(jd_utc) else math.nan,
            "JD_UTC_header": jd_header,
            "JD_UTC_start": jd_start,
            "JD_UTC_mid": jd_mid,
            "JD_UTC_end": jd_end,
            "TIME_KEY": time_key_used,
            "TIME_REF": time_reference,
            "AIRMASS": airmass,
            "EXPTIME": exptime,
            "FILTER": filt,
        }

        any_saturated = False
        for star in ordered_stars:
            meas = _measure_one_star(
                image,
                star,
                aperture_radius=aper,
                sky_inner_radius=sky_in,
                sky_outer_radius=sky_out,
                recenter=recenter,
                search_radius=search_radius,
                saturation_level=saturation_level,
            )
            sid = star.star_id
            row[f"Source-Sky_{sid}"] = meas.flux
            row[f"Source_Error_{sid}"] = meas.error
            row[f"Sky/Pixel_{sid}"] = meas.sky_median
            row[f"Sky_Std_{sid}"] = meas.sky_std
            row[f"Peak_{sid}"] = meas.peak
            row[f"Mean_Aper_{sid}"] = meas.mean_aperture
            row[f"X(FITS)_{sid}"] = meas.x
            row[f"Y(FITS)_{sid}"] = meas.y
            row[f"N_Aper_{sid}"] = meas.n_aperture
            row[f"N_Sky_{sid}"] = meas.n_sky
            row[f"Saturated_{sid}"] = int(meas.saturated)
            any_saturated = any_saturated or meas.saturated

        row["Saturated"] = int(any_saturated)
        rows.append(row)
        try:
            window["-PHOTO_PROGRESS-"].update(int(idx / n_total * 100.0))
        except Exception:
            pass

    table = pd.DataFrame(rows)
    table = _relative_curve_columns(table, star_ids)
    _update_peak_quality_from_table(session, table, values)
    _update_star_list(window, session)

    suffix = Path(output_path).suffix.lower()
    if suffix == ".csv":
        table.to_csv(output_path, index=False, na_rep="nan")
    else:
        table.to_csv(output_path, index=False, sep="\t", na_rep="nan")

    # Save a compact companion aperture recipe.  This is not required by the
    # main pipeline, but makes the image photometry itself reproducible.
    recipe_path = str(Path(output_path).with_suffix(".apertures.json"))
    try:
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "input_files": session.files,
                    "output_table": output_path,
                    "aperture_radius": aper,
                    "sky_inner_radius": sky_in,
                    "sky_outer_radius": sky_out,
                    "recenter": recenter,
                    "search_radius": search_radius,
                    "peak_low_fraction": _peak_feedback_thresholds(values)[1],
                    "peak_high_fraction": _peak_feedback_thresholds(values)[2],
                    "stars": [star.__dict__ for star in ordered_stars],
                    "time_keywords": time_keys,
                    "fits_time_reference": time_reference,
                    "main_jdutc_export": main_jdutc_mode,
                    "exptime_keywords": exptime_keys,
                    "airmass_keywords": airmass_keys,
                    "filter_keywords": filter_keys,
                },
                f,
                indent=4,
            )
    except Exception:
        pass

    n_valid_time = int(np.count_nonzero(np.isfinite(pd.to_numeric(table["JD_UTC"], errors="coerce")))) if "JD_UTC" in table else 0
    report = [
        "Aperture photometry complete.",
        f"Images measured: {len(table)}",
        f"Stars measured: {', '.join(star_ids)}",
        f"Output table: {output_path}",
        f"Aperture recipe: {recipe_path}",
        f"Valid JD_UTC values: {n_valid_time}/{len(table)}",
        f"FITS timestamp reference assumed: {time_reference}",
        f"Main JD_UTC column: {main_jdutc_mode}",
        "Additional exported time columns: JD_UTC_header, JD_UTC_start, JD_UTC_mid, JD_UTC_end",
        "Use the matching Time stamp reference in the Transit tab, or select JD_UTC_mid and use Mid-exposure.",
        f"Aperture radius: {aper:.2f} px",
        f"Sky annulus: {sky_in:.2f} - {sky_out:.2f} px",
        f"Recentering: {'yes' if recenter else 'no'}",
    ]
    window["-PHOTO_REPORT-"].update("\n".join(report))
    window["-PHOTO_PROGRESS-"].update(100)
    session.output_path = output_path
    return output_path


def run_aperture_photometry_tool(parent_window: Optional[sg.Window] = None) -> Optional[str]:
    """Run the interactive aperture-photometry sub-program.

    Returns the generated table path when the user chooses ``Run + load in
    main``.  Returns ``None`` when the window is closed or the user only saves
    a table without requesting automatic loading.
    """
    if fits is None:
        sg.popup_error("Astropy is required for FITS aperture photometry.\nInstall it with: pip install astropy")
        return None

    session = PhotometrySession()
    window = sg.Window(
        "PhotoCurve Lab - Aperture Photometry Tool",
        _build_layout(),
        resizable=True,
        finalize=True,
        modal=True,
        icon=icon_path,
    )
    center_window(window)
    return_path: Optional[str] = None

    #Handling windows peculiarities: DPI awareness and mouse over buttons
    # current_os =   # 'posix' for Linux/Mac, 'nt' for Windows

    if os.name == "nt":
        misc.enable_hover_effect(window)
        # Keep DPI awareness on Windows for crisp rendering
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        

    try:
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, "Close"):
                break

            try:
                if event == "Load sequence":
                    _load_sequence(window, session, values)
                    if session.stars:
                        _update_all_star_peaks_from_current_image(session, values)
                    _update_star_list(window, session)

                elif event in ("-PHOTO_NEXT-", "-PHOTO_PREV-", "-PHOTO_IMAGE_SCROLL-"):
                    if not session.files:
                        continue
                    if event == "-PHOTO_NEXT-":
                        delta = 1
                    elif event == "-PHOTO_PREV-":
                        delta = -1
                    else:
                        # Mouse wheel up returns +1: previous image feels natural.
                        delta = -int(values.get("-PHOTO_IMAGE_SCROLL-", 0))
                    session.current_index = max(0, min(len(session.files) - 1, session.current_index + delta))
                    _load_current_image(session)
                    _draw_image(window, session, values)

                elif event in ("-PHOTO_APER_R-", "-PHOTO_SKY_IN-", "-PHOTO_SKY_OUT-"):
                    if session.image is not None:
                        # Geometry changes are refreshed only on the current
                        # image.  Full/sampled sequence peak checks are manual
                        # because scanning many FITS files during text edits is
                        # too slow for interactive use.
                        if session.stars:
                            _update_all_star_peaks_from_current_image(session, values)
                            _update_star_list(window, session)
                        _refresh_star_overlays(window, session, values)

                elif event in ("-PHOTO_SAT_LEVEL-", "-PHOTO_PEAK_LOW_FRAC-", "-PHOTO_PEAK_HIGH_FRAC-"):
                    _reclassify_all_star_peak_quality(session, values)
                    _update_star_list(window, session)
                    if session.image is not None:
                        _refresh_star_overlays(window, session, values)

                elif event == "-PHOTO_CHECK_PEAKS-":
                    if not session.files:
                        sg.popup_error("Load a FITS sequence first.")
                        continue
                    if not session.stars:
                        sg.popup_error("Select at least one aperture first.")
                        continue
                    _check_star_peaks_over_sequence(window, session, values)
                    _update_star_list(window, session)
                    if session.image is not None:
                        _refresh_star_overlays(window, session, values)
                    try:
                        n_frames = parse_int(values.get("-PHOTO_PEAK_MAX_FRAMES-", 15), 15)
                        indices = _sample_frame_indices(len(session.files), int(n_frames))
                        mode = "full sequence" if len(indices) >= len(session.files) else f"sample {len(indices)}/{len(session.files)} frames"
                        window["-PHOTO_REPORT-"].update(f"Peak check complete ({mode}).\n", append=True)
                    except Exception:
                        pass

                elif event == "-PHOTO_AUTO_FIND_COMPS-":
                    _auto_find_comparison_stars(window, session, values)

                elif event == "-PHOTO_REFRESH-" or event in ("-PHOTO_LOW_P-", "-PHOTO_HIGH_P-"):
                    if session.image is not None:
                        _draw_image(window, session, values)

                elif event == "-PHOTO_IMAGE_MOTION-":
                    payload = values.get("-PHOTO_IMAGE_MOTION-", {}) or {}
                    x = float(payload.get("x", math.nan))
                    y = float(payload.get("y", math.nan))
                    aper = parse_float(values.get("-PHOTO_APER_R-", 6.0), 6.0)
                    pixel, peak, mean_ap = _aperture_stats_at(session.image, x, y, aper)
                    quality = _peak_quality_from_value(peak, values)
                    quality_label, hover_color, hover_bg = _peak_quality_style(quality)
                    _update_hover_overlay(window, session, values, x, y, color=hover_color)
                    info_text = (
                        f"x={x:.0f}, y={y:.0f}, pixel={pixel:.0f}, "
                        f"peak={peak:.1f}, mean(ap)={mean_ap:.1f}  [{quality_label}]"
                    )
                    try:
                        window["-PHOTO_MOUSE_INFO-"].update(info_text, background_color=hover_bg, text_color="black")
                    except Exception:
                        window["-PHOTO_MOUSE_INFO-"].update(info_text)

                elif event in ("-PHOTO_ZOOM_IN-", "-PHOTO_ZOOM_OUT-"):
                    _zoom_image_center(session, 1.0 if event == "-PHOTO_ZOOM_IN-" else -1.0)

                elif event == "-PHOTO_FIT_VIEW-":
                    _reset_image_view(session)
                    if session.image is not None:
                        _draw_image(window, session, values)

                elif event == "-PHOTO_IMAGE_CLICK-":
                    if session.image is None:
                        continue
                    payload = values.get("-PHOTO_IMAGE_CLICK-", {}) or {}
                    x = float(payload.get("x", math.nan))
                    y = float(payload.get("y", math.nan))
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    role = _selected_click_role(values)
                    if role == "delete":
                        _delete_nearest_star(session, x, y)
                        _renumber_comparisons(session)
                    else:
                        if bool(values.get("-PHOTO_SNAP_CLICK-", True)) and session.image is not None:
                            search_radius = parse_float(values.get("-PHOTO_SEARCH_R-", 8.0), 8.0)
                            x, y = _centroid_near(session.image, x, y, search_radius)
                        new_star = _add_star(session, x, y, role)
                        _set_star_peak_from_image(new_star, session.image, values, source="current frame")
                    _update_star_list(window, session)
                    _refresh_star_overlays(window, session, values)

                elif event == "-PHOTO_CLEAR_STARS-":
                    session.stars.clear()
                    _update_star_list(window, session)
                    if session.image is not None:
                        _refresh_star_overlays(window, session, values)

                elif event == "-PHOTO_DELETE_SELECTED-":
                    selection = values.get("-PHOTO_STAR_LIST-", []) or []
                    if selection:
                        sid = _parse_star_id_from_label(selection[0])
                        session.stars = [star for star in session.stars if star.star_id != sid]
                        _renumber_comparisons(session)
                        _update_star_list(window, session)
                        if session.image is not None:
                            _refresh_star_overlays(window, session, values)

                elif event == "-PHOTO_SAVE_APERTURES-":
                    path = sg.popup_get_file(
                        "Save aperture set",
                        save_as=True,
                        no_window=True,
                        default_extension=".json",
                        file_types=(("JSON", "*.json"), ("All files", "*.*")),
                    )
                    if path:
                        _save_apertures(path, session, values)
                        window["-PHOTO_REPORT-"].update(f"Apertures saved: {path}\n", append=True)

                elif event == "-PHOTO_LOAD_APERTURES-":
                    path = sg.popup_get_file(
                        "Load aperture set",
                        no_window=True,
                        file_types=(("JSON", "*.json"), ("All files", "*.*")),
                    )
                    if path:
                        _load_apertures(path, session, window, values)

                elif event in ("Run photometry", "-PHOTO_RUN_AND_LOAD-"):
                    output_path = _run_photometry(window, session, values)
                    if session.image is not None:
                        _refresh_star_overlays(window, session, values)
                    if event == "-PHOTO_RUN_AND_LOAD-":
                        return_path = output_path
                        break

            except Exception as exc:
                sg.popup_error(str(exc))

    finally:
        try:
            _delete_photometry_figure(session)
        except Exception:
            pass
        window.close()

    return return_path




def center_window(window, margin=20):
    """
    Center a PySimpleGUI window on the current screen and make sure
    it does not open outside the visible area.

    Cross-platform: Windows, Linux, macOS.
    Does not change DPI awareness or scaling behavior.
    """
    try:
        root = window.TKroot

        # Force Tk to calculate the real window size
        root.update_idletasks()
        window.refresh()

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        win_w = root.winfo_width()
        win_h = root.winfo_height()

        # Fallback in case Tk has not updated the size yet
        if win_w <= 1:
            win_w = root.winfo_reqwidth()
        if win_h <= 1:
            win_h = root.winfo_reqheight()

        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)

        # Keep the window inside the screen
        x = max(margin, min(x, screen_w - win_w - margin))
        y = max(margin, min(y, screen_h - win_h - margin))

        # If the window is taller than the screen, at least keep the title bar visible
        if win_h > screen_h - 2 * margin:
            y = margin

        # If the window is wider than the screen, keep the left edge visible
        if win_w > screen_w - 2 * margin:
            x = margin

        root.geometry(f"+{x}+{y}")
        root.update_idletasks()

    except Exception as e:
        print(f"Warning: could not center window: {e}")
