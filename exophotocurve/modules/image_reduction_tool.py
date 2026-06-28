""" FITS image reduction tool for ExoPhotoCurve.

This module is intentionally independent from the light-curve analysis pipeline.
It performs only conservative scientific operations needed before aperture
photometry: calibration with master frames, optional Bayer-channel extraction,
quality diagnostics, and registration of the calibrated sequence.

No aesthetic processing is implemented on purpose: no background extraction,
no denoising, no sharpening, no gradient removal, no histogram stretching is
written to disk.  Display stretches are used only for preview/diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import glob
import json
import math
import os
import shutil
import traceback
import threading

import numpy as np
import pandas as pd

from .numeric_utils import parse_float, parse_int
from .sg_loader import sg
from . import misc
from .user_preferences import apply_preferences_to_window, save_window_preferences

try:  # pragma: no cover - optional import checked at runtime
    from astropy.io import fits
except Exception:  # pragma: no cover
    fits = None

try:  # pragma: no cover - optional import checked at runtime
    from scipy import ndimage
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    ndimage = None
    cKDTree = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(BASE_DIR, "ExoPhotoCurve.ico")

# Whitelisted preferences for the image-reduction tool.  These are stable
# user defaults, not a record of a particular reduction.  Reference-file and
# log/progress state are intentionally not persisted.
IMAGE_REDUCTION_PREFERENCE_KEYS = [
    "-RED_SCI_PATTERN-", "-RED_BIAS_PATTERN-", "-RED_DARK_PATTERN-",
    "-RED_FLAT_PATTERN-", "-RED_DARKFLAT_PATTERN-", "-RED_OUTPUT_FOLDER-",
    "-RED_COMBINE-", "-RED_SIGMA-", "-RED_DARK_SCALING-", "-RED_EXPTIME_KEYS-",
    "-RED_REUSE_MASTERS-", "-RED_MASTER_FOLDER-", "-RED_REUSE_BIAS-",
    "-RED_REUSE_DARK-", "-RED_REUSE_DARKFLAT-", "-RED_REUSE_FLAT-",
    "-RED_CAMERA-", "-RED_BAYER_PATTERN-", "-RED_BAYER_CHANNEL-",
    "-RED_ALIGN-", "-RED_REFERENCE_MODE-", "-RED_TRANSFORM-", "-RED_INTERP-",
    "-RED_DETECT_SIGMA-", "-RED_MAX_STARS-", "-RED_MATCH_TOL-",
    "-RED_CROP_COMMON-",
]


def _save_reduction_preferences(window: sg.Window) -> None:
    try:
        save_window_preferences(window, "image_reduction", IMAGE_REDUCTION_PREFERENCE_KEYS)
    except Exception:
        pass


@dataclass
class ReductionSettings:
    """User choices for one reduction run."""

    science_folder: str
    science_pattern: str
    output_folder: str
    bias_folder: str = ""
    bias_pattern: str = "*.fit*"
    dark_folder: str = ""
    dark_pattern: str = "*.fit*"
    flat_folder: str = ""
    flat_pattern: str = "*.fit*"
    darkflat_folder: str = ""
    darkflat_pattern: str = "*.fit*"
    reuse_masters: bool = False
    master_folder: str = ""
    reuse_master_bias: bool = True
    reuse_master_dark: bool = True
    reuse_master_darkflat: bool = True
    reuse_master_flat: bool = False
    combine_method: str = "Median + sigma clipping"
    sigma_clip: float = 4.0
    dark_scaling: str = "Auto"
    exptime_keywords: str = "EXPTIME,EXPOSURE,EXPOSURE_TIME"
    camera_mode: str = "Mono"
    bayer_pattern: str = "Auto"
    bayer_channel: str = "Mean G"
    run_alignment: bool = True
    reference_mode: str = "Auto best frame"
    reference_file: str = ""
    transform_mode: str = "Full affine / meridian flip safe"
    interpolation: str = "Linear"
    crop_common: bool = False
    max_stars: int = 60
    detect_sigma: float = 6.0
    match_tolerance: float = 5.0
    overwrite: bool = True


@dataclass
class ImageDiagnostics:
    """Quality-control quantities measured for one frame."""

    filename: str
    stage: str
    date_obs: str = ""
    exptime: float = math.nan
    filter_name: str = ""
    shape: str = ""
    sky_background: float = math.nan
    sky_noise: float = math.nan
    max_pixel: float = math.nan
    saturated_fraction: float = math.nan
    n_stars: int = 0
    fwhm: float = math.nan
    elongation: float = math.nan
    x_shift: float = math.nan
    y_shift: float = math.nan
    rotation_deg: float = math.nan
    scale: float = math.nan
    flip_detected: bool = False
    alignment_rms: float = math.nan
    alignment_inliers: int = 0
    status: str = "ok"
    output_file: str = ""


@dataclass
class StarDetection:
    """Simple star-detection result for registration and QC."""

    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    fwhm: np.ndarray
    elongation: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.x))

    def coords(self) -> np.ndarray:
        if self.n == 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack([self.x, self.y]).astype(float)

    def limited(self, max_stars: int) -> "StarDetection":
        if self.n <= max_stars:
            return self
        order = np.argsort(self.flux)[::-1][:max_stars]
        return StarDetection(
            x=self.x[order],
            y=self.y[order],
            flux=self.flux[order],
            peak=self.peak[order],
            fwhm=self.fwhm[order],
            elongation=self.elongation[order],
        )


@dataclass
class TransformResult:
    """Transformation mapping moving-image coordinates into reference coordinates."""

    matrix: np.ndarray
    offset: np.ndarray
    mode: str
    n_inliers: int = 0
    rms: float = math.nan
    median_residual: float = math.nan
    flip_detected: bool = False
    scale: float = math.nan
    rotation_deg: float = math.nan
    status: str = "ok"


@dataclass
class ReductionResult:
    """Summary returned to the main window after a reduction run."""

    output_root: str
    calibrated_folder: str
    aligned_folder: str
    report_path: str
    settings_path: str
    n_science: int
    n_calibrated: int
    n_aligned: int


ProgressCallback = Optional[Callable[[str, Optional[int]], None]]


class ReductionCancelled(RuntimeError):
    """Raised when the user stops an image-reduction run."""


class CancellationToken:
    """Small thread-safe cancellation flag for long reduction runs."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def _check_cancel(cancel_token: Optional[CancellationToken]) -> None:
    if cancel_token is not None and cancel_token.is_cancelled():
        raise ReductionCancelled("Image reduction stopped by user.")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _log(progress: ProgressCallback, message: str, percent: Optional[int] = None) -> None:
    if progress is not None:
        progress(message, percent)


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            txt = value.strip().replace(",", ".")
            if not txt:
                return default
            txt = txt.split()[0]
            return float(txt)
        return float(value)
    except Exception:
        return default


def _split_keywords(text: object) -> List[str]:
    out: List[str] = []
    for token in str(text or "").replace(";", ",").split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _header_get_float(header, keys: Sequence[str], default: float = math.nan) -> float:
    if header is None:
        return default
    for key in keys:
        if key in header:
            value = _safe_float(header.get(key), default=math.nan)
            if np.isfinite(value):
                return float(value)
    return default


def _header_get_text(header, keys: Sequence[str], default: str = "") -> str:
    if header is None:
        return default
    for key in keys:
        if key in header:
            value = str(header.get(key)).strip()
            if value:
                return value
    return default


def _list_fits(folder: str, pattern: str) -> List[str]:
    folder_path = Path(str(folder).strip())
    if not folder_path.is_dir():
        return []
    pattern = str(pattern or "*.fit*").strip() or "*.fit*"
    files = sorted(glob.glob(str(folder_path / pattern)))
    return [str(Path(f)) for f in files if Path(f).is_file()]


def _read_fits_image(path: str) -> Tuple[np.ndarray, object, int]:
    """Read the first 2D FITS image HDU as float64."""
    if fits is None:
        raise RuntimeError("Astropy is required for FITS reduction. Install it with: pip install astropy")
    with fits.open(path, memmap=False) as hdul:
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim == 2:
                return arr.astype(np.float64, copy=False), hdu.header.copy(), idx
            squeezed = np.squeeze(arr)
            if squeezed.ndim == 2:
                return squeezed.astype(np.float64, copy=False), hdu.header.copy(), idx
    raise ValueError(f"No 2D image HDU found in {path}")


def _copy_header(header) -> object:
    if header is None or fits is None:
        return None
    return header.copy()


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_fits(path: str | Path, data: np.ndarray, header=None, overwrite: bool = True) -> None:
    if fits is None:
        raise RuntimeError("Astropy is required for writing FITS files.")
    hdr = _copy_header(header) if header is not None else fits.Header()
    hdr["EPCRED"] = (True, "Reduced by ExoPhotoCurve reduction")
    hdr["BUNIT"] = ("adu", "Calibrated detector units")
    hdr.add_history("ExoPhotoCurve reduction: calibration/registration only.")
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=hdr, overwrite=overwrite)


def _json_safe(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isfinite(value):
            return value
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Robust statistics, combination and calibration
# ---------------------------------------------------------------------------


def _robust_sky(image: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    noise = 1.4826 * mad if mad > 0 else float(np.nanstd(finite))
    return med, noise


def _sigma_clip_stack(stack: np.ndarray, sigma: float) -> np.ndarray:
    """Return a copy with outliers replaced by NaN along the first axis."""
    if stack.shape[0] < 3 or sigma <= 0:
        return stack
    med = np.nanmedian(stack, axis=0)
    mad = np.nanmedian(np.abs(stack - med), axis=0)
    robust_std = 1.4826 * mad
    robust_std = np.where(robust_std <= 0, np.nan, robust_std)
    mask = np.abs(stack - med) > sigma * robust_std
    clipped = stack.copy()
    clipped[mask] = np.nan
    return clipped


def _combine_images(
    files: Sequence[str],
    method: str,
    sigma: float,
    progress: ProgressCallback = None,
    label: str = "frames",
    cancel_token: Optional[CancellationToken] = None,
) -> Tuple[np.ndarray, object, List[Dict[str, object]]]:
    if not files:
        raise ValueError(f"No {label} were selected.")
    arrays: List[np.ndarray] = []
    headers: List[object] = []
    rows: List[Dict[str, object]] = []
    shape0: Optional[Tuple[int, int]] = None

    for i, path in enumerate(files, start=1):
        arr, header, _hdu_idx = _read_fits_image(path)
        if shape0 is None:
            shape0 = arr.shape
        elif arr.shape != shape0:
            raise ValueError(f"Shape mismatch in {label}: {Path(path).name} has {arr.shape}, expected {shape0}")
        arrays.append(arr)
        headers.append(header)
        sky, noise = _robust_sky(arr)
        rows.append(
            {
                "filename": str(Path(path).name),
                "shape": str(arr.shape),
                "median": sky,
                "noise": noise,
                "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else math.nan,
            }
        )
        _log(progress, f"Read {label}: {i}/{len(files)} - {Path(path).name}")

    stack = np.stack(arrays, axis=0).astype(np.float64, copy=False)
    method_l = str(method or "").lower()
    if "sigma" in method_l:
        stack_use = _sigma_clip_stack(stack, sigma)
    else:
        stack_use = stack

    if "mean" in method_l:
        master = np.nanmean(stack_use, axis=0)
    else:
        master = np.nanmedian(stack_use, axis=0)

    return master.astype(np.float64, copy=False), headers[0], rows


def _same_shape_or_fail(arrays: Iterable[Tuple[str, np.ndarray]]) -> Tuple[int, int]:
    shape0: Optional[Tuple[int, int]] = None
    for name, arr in arrays:
        if arr is None:
            continue
        if shape0 is None:
            shape0 = arr.shape
        elif arr.shape != shape0:
            raise ValueError(f"Calibration-frame shape mismatch: {name} has {arr.shape}, expected {shape0}")
    if shape0 is None:
        raise ValueError("No image shape available.")
    return shape0


def _median_exptime(files: Sequence[str], keys: Sequence[str]) -> float:
    values: List[float] = []
    for path in files:
        try:
            _arr, hdr, _idx = _read_fits_image(path)
            exp = _header_get_float(hdr, keys, default=math.nan)
            if np.isfinite(exp):
                values.append(float(exp))
        except Exception:
            continue
    if not values:
        return math.nan
    return float(np.nanmedian(values))


def _normalise_flat(flat: np.ndarray) -> np.ndarray:
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("The master flat contains no finite pixels.")
    med = float(np.nanmedian(finite))
    if not np.isfinite(med) or med == 0:
        raise ValueError("The master flat median is zero or invalid; cannot normalise it.")
    norm = flat / med
    bad = ~np.isfinite(norm) | (np.abs(norm) < 1.0e-10)
    if np.any(bad):
        replacement = 1.0
        norm = norm.copy()
        norm[bad] = replacement
    return norm


# ---------------------------------------------------------------------------
# Existing master-frame reuse
# ---------------------------------------------------------------------------


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _candidate_master_dirs_from_folder(folder: str) -> List[Path]:
    """Return plausible masters directories from a user-selected folder.

    The user can select either the previous reduction root, its masters/
    subfolder, an ExoPhotoCurve_reduced container, or a generic calibration
    folder.  We keep the search shallow and deterministic to avoid accidentally
    picking unrelated files deep in the disk.
    """
    text = str(folder or "").strip()
    if not text:
        return []
    root = Path(text).expanduser()
    if not root.exists():
        return []

    candidates: List[Path] = []
    if root.is_dir():
        candidates.append(root)
        candidates.append(root / "masters")
        candidates.extend(sorted(root.glob("reduction_*/masters"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
        candidates.extend(sorted(root.glob("ExoPhotoCurve_reduced/reduction_*/masters"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    return [p for p in _unique_paths(candidates) if p.is_dir()]


def _auto_candidate_master_dirs(settings: ReductionSettings) -> List[Path]:
    bases: List[Path] = []
    for text in (settings.output_folder, settings.science_folder):
        text = str(text or "").strip()
        if text:
            bases.append(Path(text).expanduser())
    if settings.science_folder.strip():
        bases.append(Path(settings.science_folder.strip()).expanduser() / "ExoPhotoCurve_reduced")

    candidates: List[Path] = []
    for base in bases:
        if not base.exists():
            continue
        if base.is_dir():
            candidates.append(base / "masters")
            candidates.extend(sorted(base.glob("reduction_*/masters"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
            candidates.extend(sorted(base.glob("ExoPhotoCurve_reduced/reduction_*/masters"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    return [p for p in _unique_paths(candidates) if p.is_dir()]


def _find_first_matching_file(folder: Path, patterns: Sequence[str], reject_tokens: Sequence[str] = ()) -> Optional[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(folder.glob(pattern))
    matches = [m for m in _unique_paths(matches) if m.is_file()]
    if reject_tokens:
        rejects = tuple(t.lower() for t in reject_tokens)
        matches = [m for m in matches if not any(t in m.name.lower() for t in rejects)]
    if not matches:
        return None
    # Prefer canonical names over wildcard hits, then most recent files.
    matches.sort(key=lambda p: (0 if p.name.lower().startswith("master_") else 1, -p.stat().st_mtime))
    return matches[0]


def find_reusable_master_files(settings: ReductionSettings) -> Tuple[Dict[str, str], List[str]]:
    """Find reusable ExoPhotoCurve master files selected by the user.

    The returned keys are: bias, dark_raw, dark_signal, darkflat, flat_norm.
    Only normalised flats are considered reusable flat masters, because using a
    raw/uncorrected flat would risk a double or missing calibration step.
    """
    if settings.master_folder.strip():
        candidate_dirs = _candidate_master_dirs_from_folder(settings.master_folder)
    else:
        candidate_dirs = _auto_candidate_master_dirs(settings)

    found: Dict[str, str] = {}
    searched: List[str] = [str(p) for p in candidate_dirs]

    for folder in candidate_dirs:
        if "bias" not in found:
            path = _find_first_matching_file(folder, ["master_bias.fit*", "*master*bias*.fit*"], reject_tokens=("dark", "flat"))
            if path is not None:
                found["bias"] = str(path)

        if "dark_raw" not in found:
            path = _find_first_matching_file(
                folder,
                ["master_dark_raw.fit*", "master_dark.fit*", "*master*dark*raw*.fit*"],
                reject_tokens=("flat", "signal", "minus_bias"),
            )
            if path is not None:
                found["dark_raw"] = str(path)

        if "dark_signal" not in found:
            path = _find_first_matching_file(
                folder,
                ["master_dark_signal_minus_bias.fit*", "master_dark_signal.fit*", "*master*dark*signal*.fit*", "*master*dark*minus*bias*.fit*"],
                reject_tokens=("flat",),
            )
            if path is not None:
                found["dark_signal"] = str(path)

        if "darkflat" not in found:
            path = _find_first_matching_file(
                folder,
                ["master_darkflat.fit*", "master_dark_flat.fit*", "*master*darkflat*.fit*", "*master*dark-flat*.fit*", "*master*dark_flat*.fit*"],
            )
            if path is not None:
                found["darkflat"] = str(path)

        if "flat_norm" not in found:
            path = _find_first_matching_file(
                folder,
                ["master_flat_normalized.fit*", "master_flat_normalised.fit*", "*master*flat*norm*.fit*"],
                reject_tokens=("dark", "raw"),
            )
            if path is not None:
                found["flat_norm"] = str(path)

    return found, searched


def _load_master_file(path: str, label: str) -> Tuple[np.ndarray, object]:
    arr, hdr, _idx = _read_fits_image(path)
    if not np.isfinite(arr).any():
        raise ValueError(f"The selected {label} master contains no finite pixels: {path}")
    return arr.astype(np.float64, copy=False), hdr


def _copy_reused_master(src: str, masters_dir: Path, out_name: str, warnings: List[str]) -> None:
    try:
        dst = masters_dir / out_name
        if Path(src).resolve() != dst.resolve():
            shutil.copy2(src, dst)
    except Exception as exc:
        warnings.append(f"Could not copy reused master {Path(src).name} into the output masters folder: {exc}")


def _master_median(path_or_arr: object) -> float:
    try:
        if isinstance(path_or_arr, (str, Path)):
            arr, _hdr, _idx = _read_fits_image(str(path_or_arr))
        else:
            arr = np.asarray(path_or_arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(np.nanmedian(finite)) if finite.size else math.nan
    except Exception:
        return math.nan


def build_master_check_report(settings: ReductionSettings) -> str:
    found, searched = find_reusable_master_files(settings)
    lines: List[str] = []
    lines.append("Existing master-frame check")
    lines.append("")
    if settings.master_folder.strip():
        lines.append(f"Selected folder: {settings.master_folder.strip()}")
    else:
        lines.append("Selected folder: <empty>; automatic search from science/output folders")
    if searched:
        lines.append("Searched folders:")
        for folder in searched[:10]:
            lines.append(f"  - {folder}")
    else:
        lines.append("No plausible master folders were found.")
    lines.append("")

    selected = {
        "Bias": (settings.reuse_master_bias, "bias"),
        "Dark": (settings.reuse_master_dark, "dark_raw"),
        "Dark signal": (settings.reuse_master_dark, "dark_signal"),
        "Dark-flat": (settings.reuse_master_darkflat, "darkflat"),
        "Flat": (settings.reuse_master_flat, "flat_norm"),
    }
    for label, (enabled, key) in selected.items():
        if not enabled and label != "Dark signal":
            lines.append(f"{label}: not selected for reuse")
        elif key in found:
            lines.append(f"{label}: FOUND -> {Path(found[key]).name}")
        else:
            if label == "Dark signal" and not settings.reuse_master_dark:
                continue
            lines.append(f"{label}: not found")

    # Basic compatibility checks against the first science frame when possible.
    science_files = _list_fits(settings.science_folder, settings.science_pattern) if settings.science_folder.strip() else []
    if science_files:
        lines.append("")
        try:
            first_raw, first_hdr, _idx = _read_fits_image(science_files[0])
            science_shape = first_raw.shape
            exptime_keys = _split_keywords(settings.exptime_keywords) or ["EXPTIME"]
            science_exp = _median_exptime(science_files, exptime_keys)
            sci_filter = _header_get_text(first_hdr, ["FILTER", "FILTER1", "FILTER2", "INSFLNAM"], default="")
            lines.append(f"Science shape: {science_shape}")
            if np.isfinite(science_exp):
                lines.append(f"Science exposure median: {science_exp:.6g} s")
            if sci_filter:
                lines.append(f"Science filter keyword: {sci_filter}")

            for label, key in (("Bias", "bias"), ("Dark", "dark_raw"), ("Dark-flat", "darkflat"), ("Flat", "flat_norm")):
                if key not in found:
                    continue
                arr, hdr = _load_master_file(found[key], label)
                shape_status = "OK" if arr.shape == science_shape else f"MISMATCH {arr.shape}"
                msg = f"{label} shape: {shape_status}"
                if key == "dark_raw":
                    exp = _header_get_float(hdr, exptime_keys, default=math.nan)
                    if np.isfinite(exp) and np.isfinite(science_exp):
                        msg += f"; exposure {exp:.6g} s"
                if key == "flat_norm":
                    med = _master_median(arr)
                    msg += f"; median {med:.4g}" if np.isfinite(med) else "; median unavailable"
                lines.append(msg)
        except Exception as exc:
            lines.append(f"Compatibility check could not be completed: {exc}")
    else:
        lines.append("")
        lines.append("Select a science folder to run shape/exposure compatibility checks.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bayer extraction without debayer interpolation
# ---------------------------------------------------------------------------


def _infer_bayer_pattern(header, fallback: str) -> str:
    for key in ("BAYERPAT", "BAYER", "COLORTYP", "XBAYROFF", "BAYERPATTERN"):
        if header is not None and key in header:
            value = str(header.get(key)).strip().upper()
            for pat in ("RGGB", "BGGR", "GRBG", "GBRG"):
                if pat in value:
                    return pat
    fallback = str(fallback or "").strip().upper()
    if fallback in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        return fallback
    return "RGGB"


def _bayer_masks(shape: Tuple[int, int], pattern: str) -> Dict[str, np.ndarray]:
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    even_y = yy % 2 == 0
    even_x = xx % 2 == 0
    odd_y = ~even_y
    odd_x = ~even_x

    pattern = pattern.upper()
    if pattern == "RGGB":
        r = even_y & even_x
        g1 = even_y & odd_x
        g2 = odd_y & even_x
        b = odd_y & odd_x
    elif pattern == "BGGR":
        b = even_y & even_x
        g1 = even_y & odd_x
        g2 = odd_y & even_x
        r = odd_y & odd_x
    elif pattern == "GRBG":
        g1 = even_y & even_x
        r = even_y & odd_x
        b = odd_y & even_x
        g2 = odd_y & odd_x
    elif pattern == "GBRG":
        g1 = even_y & even_x
        b = even_y & odd_x
        r = odd_y & even_x
        g2 = odd_y & odd_x
    else:
        return _bayer_masks(shape, "RGGB")
    return {"R": r, "G1": g1, "G2": g2, "B": b, "Mean G": g1 | g2}


def _extract_bayer_channel(image: np.ndarray, header, camera_mode: str, bayer_pattern: str, channel: str) -> Tuple[np.ndarray, str]:
    if str(camera_mode).lower().startswith("mono"):
        return image, "Mono"

    pattern = _infer_bayer_pattern(header, bayer_pattern)
    channel = str(channel or "Mean G")
    if channel not in {"Mean G", "G1", "G2", "R", "B"}:
        channel = "Mean G"

    # CFA extraction: produce a compact half-resolution image made
    # only from real detector pixels.  This avoids RGB debayer interpolation and
    # also avoids writing NaN-valued checkerboard images that would break normal
    # aperture sums in the downstream photometry module.  For Mean G, the two
    # green pixels in each Bayer cell are averaged; for G1/G2/R/B a single real
    # pixel per Bayer cell is used.
    ny = image.shape[0] - (image.shape[0] % 2)
    nx = image.shape[1] - (image.shape[1] % 2)
    arr = image[:ny, :nx]

    if pattern == "RGGB":
        r = arr[0::2, 0::2]
        g1 = arr[0::2, 1::2]
        g2 = arr[1::2, 0::2]
        b = arr[1::2, 1::2]
    elif pattern == "BGGR":
        b = arr[0::2, 0::2]
        g1 = arr[0::2, 1::2]
        g2 = arr[1::2, 0::2]
        r = arr[1::2, 1::2]
    elif pattern == "GRBG":
        g1 = arr[0::2, 0::2]
        r = arr[0::2, 1::2]
        b = arr[1::2, 0::2]
        g2 = arr[1::2, 1::2]
    elif pattern == "GBRG":
        g1 = arr[0::2, 0::2]
        b = arr[0::2, 1::2]
        r = arr[1::2, 0::2]
        g2 = arr[1::2, 1::2]
    else:
        return _extract_bayer_channel(image, header, camera_mode, "RGGB", channel)

    if channel == "R":
        out = r
    elif channel == "B":
        out = b
    elif channel == "G1":
        out = g1
    elif channel == "G2":
        out = g2
    else:
        out = 0.5 * (g1 + g2)

    return np.asarray(out, dtype=np.float64), f"{pattern}:{channel}:half_resolution_no_debayer"


# ---------------------------------------------------------------------------
# Star detection and diagnostics
# ---------------------------------------------------------------------------


def _centroid_patch(image: np.ndarray, x0: int, y0: int, radius: int, sky: float) -> Optional[Tuple[float, float, float, float, float, float]]:
    ny, nx = image.shape
    x1 = max(0, x0 - radius)
    x2 = min(nx, x0 + radius + 1)
    y1 = max(0, y0 - radius)
    y2 = min(ny, y0 + radius + 1)
    patch = image[y1:y2, x1:x2]
    if patch.size < 9:
        return None
    finite = np.isfinite(patch)
    if not np.any(finite):
        return None
    signal = patch.astype(float) - sky
    signal[~finite] = 0.0
    signal[signal < 0] = 0.0
    total = float(np.sum(signal))
    if total <= 0:
        return None
    yy, xx = np.indices(patch.shape)
    xc = float(np.sum((xx + x1) * signal) / total)
    yc = float(np.sum((yy + y1) * signal) / total)
    dx = (xx + x1) - xc
    dy = (yy + y1) - yc
    var_x = float(np.sum(dx * dx * signal) / total)
    var_y = float(np.sum(dy * dy * signal) / total)
    cov = float(np.sum(dx * dy * signal) / total)
    trace = max(var_x + var_y, 0.0)
    det = max(var_x * var_y - cov * cov, 0.0)
    disc = max(trace * trace / 4.0 - det, 0.0)
    lam1 = max(trace / 2.0 + math.sqrt(disc), 0.0)
    lam2 = max(trace / 2.0 - math.sqrt(disc), 0.0)
    sigma_mean = math.sqrt(max((lam1 + lam2) / 2.0, 0.0))
    fwhm = 2.355 * sigma_mean if sigma_mean > 0 else math.nan
    elong = math.sqrt(lam1 / lam2) if lam2 > 0 else math.nan
    peak = float(np.nanmax(patch))
    return xc, yc, total, peak, fwhm, elong


def _star_patch_is_extended(image: np.ndarray, x0: int, y0: int, radius: int, sky: float, noise: float) -> bool:
    """Reject single/few-pixel spikes before using detections for alignment.

    Hot pixels and cosmic-ray residuals are local maxima but have too little
    spatial extent to be reliable registration sources.  This conservative
    check keeps detections whose flux is distributed over multiple pixels.
    """
    ny, nx = image.shape
    x1 = max(0, x0 - radius)
    x2 = min(nx, x0 + radius + 1)
    y1 = max(0, y0 - radius)
    y2 = min(ny, y0 + radius + 1)
    patch = np.asarray(image[y1:y2, x1:x2], dtype=float)
    if patch.size < 9 or not np.isfinite(patch).any():
        return False
    signal = patch - float(sky)
    signal[~np.isfinite(signal)] = 0.0
    signal[signal < 0.0] = 0.0
    total = float(np.nansum(signal))
    peak_signal = float(np.nanmax(signal)) if signal.size else math.nan
    if not np.isfinite(total) or total <= 0 or not np.isfinite(peak_signal) or peak_signal <= 0:
        return False
    effective_pixels = total / peak_signal
    local_noise = float(noise) if np.isfinite(noise) and noise > 0 else 1.0
    core_threshold = max(3.0 * local_noise, 0.20 * peak_signal)
    core_pixels = int(np.sum(signal >= core_threshold))
    flat = np.sort(signal[np.isfinite(signal)].ravel())
    second_signal = float(flat[-2]) if flat.size >= 2 else 0.0
    peak_ratio = peak_signal / max(second_signal, 1e-12)
    if effective_pixels < 2.2:
        return False
    if core_pixels < 3:
        return False
    if peak_ratio > 8.0 and core_pixels <= 4:
        return False
    return True


def detect_stars(image: np.ndarray, sigma: float = 6.0, max_stars: int = 120) -> StarDetection:
    """Detect relatively isolated stars with conservative local maxima.

    This is deliberately simple and deterministic; it is meant for registration
    and QC, not for final aperture photometry.
    """
    if ndimage is None:
        raise RuntimeError("scipy is required for automatic star detection and alignment.")
    arr = np.asarray(image, dtype=float)
    sky, noise = _robust_sky(arr)
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.nanstd(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 1.0
    threshold = sky + float(sigma) * noise

    finite_arr = np.where(np.isfinite(arr), arr, -np.inf)
    maxf = ndimage.maximum_filter(finite_arr, size=7, mode="nearest")
    candidates = (finite_arr == maxf) & np.isfinite(arr) & (arr > threshold)

    yy, xx = np.nonzero(candidates)
    if len(xx) == 0:
        empty = np.array([], dtype=float)
        return StarDetection(empty, empty, empty, empty, empty, empty)

    peaks = arr[yy, xx]
    order = np.argsort(peaks)[::-1]
    # Use a generous internal cap before centroiding to avoid long loops on hot pixels.
    order = order[: max(5 * int(max_stars), int(max_stars))]

    xs: List[float] = []
    ys: List[float] = []
    fluxes: List[float] = []
    peak_out: List[float] = []
    fwhms: List[float] = []
    elongations: List[float] = []
    min_sep2 = 6.0 * 6.0

    for idx in order:
        x0 = int(xx[idx])
        y0 = int(yy[idx])
        if x0 < 6 or y0 < 6 or x0 >= arr.shape[1] - 6 or y0 >= arr.shape[0] - 6:
            continue
        if xs:
            dist2 = (np.asarray(xs) - x0) ** 2 + (np.asarray(ys) - y0) ** 2
            if np.nanmin(dist2) < min_sep2:
                continue
        if not _star_patch_is_extended(arr, x0, y0, radius=5, sky=sky, noise=noise):
            continue
        result = _centroid_patch(arr, x0, y0, radius=5, sky=sky)
        if result is None:
            continue
        xc, yc, flux, peak, fwhm, elong = result
        if not np.isfinite(fwhm) or fwhm < 1.15 or fwhm > 20.0:
            continue
        if np.isfinite(elong) and elong > 5.0:
            continue
        xs.append(xc)
        ys.append(yc)
        fluxes.append(flux)
        peak_out.append(peak)
        fwhms.append(fwhm)
        elongations.append(elong)
        if len(xs) >= max_stars:
            break

    return StarDetection(
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        flux=np.asarray(fluxes, dtype=float),
        peak=np.asarray(peak_out, dtype=float),
        fwhm=np.asarray(fwhms, dtype=float),
        elongation=np.asarray(elongations, dtype=float),
    )


def _diagnostics_for_image(path: str, image: np.ndarray, header, detection: Optional[StarDetection], stage: str) -> ImageDiagnostics:
    sky, noise = _robust_sky(image)
    finite = image[np.isfinite(image)]
    max_pixel = float(np.nanmax(finite)) if finite.size else math.nan
    # A conservative saturation proxy when BITPIX/threshold is unavailable: top
    # 0.01% repeated maximum pixels.  It is only a diagnostic, not a rejection.
    if finite.size and np.isfinite(max_pixel):
        saturated_fraction = float(np.count_nonzero(finite >= max_pixel) / finite.size)
    else:
        saturated_fraction = math.nan
    if detection is not None and detection.n > 0:
        fwhm = float(np.nanmedian(detection.fwhm)) if detection.fwhm.size else math.nan
        elong = float(np.nanmedian(detection.elongation)) if detection.elongation.size else math.nan
        n_stars = detection.n
    else:
        fwhm = math.nan
        elong = math.nan
        n_stars = 0
    return ImageDiagnostics(
        filename=Path(path).name,
        stage=stage,
        date_obs=_header_get_text(header, ["DATE-OBS", "DATEOBS", "DATE"], default=""),
        exptime=_header_get_float(header, ["EXPTIME", "EXPOSURE", "EXPOSURE_TIME"], default=math.nan),
        filter_name=_header_get_text(header, ["FILTER", "FILTER1", "FILTER2", "INSFLNAM"], default=""),
        shape=f"{image.shape[0]}x{image.shape[1]}",
        sky_background=sky,
        sky_noise=noise,
        max_pixel=max_pixel,
        saturated_fraction=saturated_fraction,
        n_stars=n_stars,
        fwhm=fwhm,
        elongation=elong,
    )


# ---------------------------------------------------------------------------
# Registration / alignment
# ---------------------------------------------------------------------------


def _similarity_from_pair(src1: np.ndarray, src2: np.ndarray, dst1: np.ndarray, dst2: np.ndarray, reflected: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    z1 = complex(float(src1[0]), float(src1[1]))
    z2 = complex(float(src2[0]), float(src2[1]))
    w1 = complex(float(dst1[0]), float(dst1[1]))
    w2 = complex(float(dst2[0]), float(dst2[1]))
    denom = (np.conjugate(z2) - np.conjugate(z1)) if reflected else (z2 - z1)
    if abs(denom) < 1.0e-8:
        return None
    a = (w2 - w1) / denom
    if not np.isfinite(a.real) or not np.isfinite(a.imag):
        return None
    if reflected:
        b = w1 - a * np.conjugate(z1)
        A = np.array([[a.real, a.imag], [a.imag, -a.real]], dtype=float)
    else:
        b = w1 - a * z1
        A = np.array([[a.real, -a.imag], [a.imag, a.real]], dtype=float)
    offset = np.array([b.real, b.imag], dtype=float)
    return A, offset


def _apply_xy_transform(coords: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords.copy()
    return coords @ matrix.T + offset.reshape(1, 2)


def _match_with_transform(src: np.ndarray, dst: np.ndarray, matrix: np.ndarray, offset: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cKDTree is None:
        raise RuntimeError("scipy is required for image alignment.")
    transformed = _apply_xy_transform(src, matrix, offset)
    tree = cKDTree(dst)
    dist, idx = tree.query(transformed, distance_upper_bound=float(tolerance))
    ok = np.isfinite(dist) & (idx < len(dst))
    if not np.any(ok):
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    src_idx = np.nonzero(ok)[0]
    dst_idx = idx[ok]
    dist_ok = dist[ok]
    # Keep only the best source for each destination to avoid duplicate matches.
    best: Dict[int, Tuple[int, float]] = {}
    for s_i, d_i, dd in zip(src_idx, dst_idx, dist_ok):
        d_i = int(d_i)
        if d_i not in best or dd < best[d_i][1]:
            best[d_i] = (int(s_i), float(dd))
    src_unique = np.array([v[0] for v in best.values()], dtype=int)
    dst_unique = np.array(list(best.keys()), dtype=int)
    dist_unique = np.array([v[1] for v in best.values()], dtype=float)
    return src_unique, dst_unique, dist_unique


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(src) < 3:
        return _fit_similarity(src, dst, allow_reflection=True)
    X = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src))])
    coeff_x, *_ = np.linalg.lstsq(X, dst[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(X, dst[:, 1], rcond=None)
    A = np.array([[coeff_x[0], coeff_x[1]], [coeff_y[0], coeff_y[1]]], dtype=float)
    b = np.array([coeff_x[2], coeff_y[2]], dtype=float)
    return A, b


def _fit_similarity(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares similarity transform from src to dst."""
    if len(src) == 0:
        return np.eye(2), np.zeros(2)
    if len(src) == 1:
        return np.eye(2), dst[0] - src[0]
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var = np.sum(src_c * src_c)
    scale = float(np.sum(S) / var) if var > 0 else 1.0
    A = scale * R
    b = dst_mean - A @ src_mean
    return A, b


def _transform_properties(matrix: np.ndarray) -> Tuple[float, float, bool]:
    det = float(np.linalg.det(matrix))
    flip = det < 0
    # Characteristic scale from area.  This remains meaningful for affine fits.
    scale = math.sqrt(abs(det)) if np.isfinite(det) else math.nan
    # Approximate rotation angle from first column.  For mirrored transforms the
    # sign is only diagnostic, but useful in the QC table.
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    return scale, angle, flip


def estimate_transform(
    moving: StarDetection,
    reference: StarDetection,
    mode: str,
    tolerance: float = 5.0,
    max_stars: int = 50,
) -> TransformResult:
    """Estimate the transform from moving coordinates to reference coordinates."""
    moving_l = moving.limited(max_stars)
    ref_l = reference.limited(max_stars)
    src = moving_l.coords()
    dst = ref_l.coords()

    if src.shape[0] < 2 or dst.shape[0] < 2:
        return TransformResult(np.eye(2), np.zeros(2), mode, status="not enough stars")

    mode_l = str(mode or "").lower()
    allow_rotation = "rotation" in mode_l or "affine" in mode_l or "flip" in mode_l
    allow_reflection = "affine" in mode_l or "flip" in mode_l or "meridian" in mode_l
    full_affine = "affine" in mode_l

    if not allow_rotation:
        # Translation-only fallback.  Try all bright-star pair offsets and keep
        # the one with the largest match count.
        best_score = (-1, float("inf"))
        best_b = np.zeros(2)
        for p in src[: min(25, len(src))]:
            offsets = dst[: min(25, len(dst))] - p
            for b in offsets:
                sidx, didx, dist = _match_with_transform(src, dst, np.eye(2), b, tolerance)
                if len(sidx) == 0:
                    continue
                score = (len(sidx), -float(np.nanmedian(dist)))
                if score[0] > best_score[0] or (score[0] == best_score[0] and -score[1] < best_score[1]):
                    best_score = (score[0], -score[1])
                    best_b = b
        sidx, didx, dist = _match_with_transform(src, dst, np.eye(2), best_b, tolerance)
        if len(sidx) >= 1:
            refined_b = np.nanmedian(dst[didx] - src[sidx], axis=0)
            residual = np.linalg.norm(src[sidx] + refined_b - dst[didx], axis=1)
            return TransformResult(
                np.eye(2),
                refined_b,
                mode,
                n_inliers=int(len(sidx)),
                rms=float(np.sqrt(np.nanmean(residual**2))),
                median_residual=float(np.nanmedian(residual)),
                flip_detected=False,
                scale=1.0,
                rotation_deg=0.0,
            )
        return TransformResult(np.eye(2), np.zeros(2), mode, status="alignment failed")

    # Pair-based RANSAC.  Distances are invariant to translation, rotation and
    # reflection; using bright-star subsets keeps this deterministic and fast.
    n_src = min(len(src), max_stars)
    n_dst = min(len(dst), max_stars)
    src = src[:n_src]
    dst = dst[:n_dst]

    src_pairs: List[Tuple[int, int, float]] = []
    dst_pairs: List[Tuple[int, int, float]] = []
    for i in range(n_src - 1):
        for j in range(i + 1, n_src):
            d = float(np.linalg.norm(src[j] - src[i]))
            if d >= 8.0:
                src_pairs.append((i, j, d))
    for i in range(n_dst - 1):
        for j in range(i + 1, n_dst):
            d = float(np.linalg.norm(dst[j] - dst[i]))
            if d >= 8.0:
                dst_pairs.append((i, j, d))

    src_pairs = sorted(src_pairs, key=lambda x: x[2])
    dst_pairs = sorted(dst_pairs, key=lambda x: x[2])
    best: Optional[TransformResult] = None
    best_sidx: Optional[np.ndarray] = None
    best_didx: Optional[np.ndarray] = None

    # Avoid pathological runtimes on extremely rich fields.  Bright-star order
    # already carries useful information, so this cap mostly affects redundant
    # faint-star combinations.
    max_trials = 35000
    trials = 0
    length_tol = 0.18

    for si, sj, sd in src_pairs:
        for di, dj, dd in dst_pairs:
            ratio = dd / sd if sd > 0 else math.inf
            if ratio < 1.0 - length_tol or ratio > 1.0 + length_tol:
                continue
            for swap in (False, True):
                d1, d2 = (dj, di) if swap else (di, dj)
                for reflected in (False, True):
                    if reflected and not allow_reflection:
                        continue
                    candidate = _similarity_from_pair(src[si], src[sj], dst[d1], dst[d2], reflected=reflected)
                    if candidate is None:
                        continue
                    A, b = candidate
                    sidx, didx, dist = _match_with_transform(src, dst, A, b, tolerance)
                    if len(sidx) < 2:
                        trials += 1
                        if trials >= max_trials:
                            break
                        continue
                    rms = float(np.sqrt(np.nanmean(dist**2))) if len(dist) else math.nan
                    scale, angle, flip = _transform_properties(A)
                    cand_result = TransformResult(
                        A,
                        b,
                        mode,
                        n_inliers=int(len(sidx)),
                        rms=rms,
                        median_residual=float(np.nanmedian(dist)) if len(dist) else math.nan,
                        flip_detected=flip,
                        scale=scale,
                        rotation_deg=angle,
                    )
                    if best is None:
                        choose = True
                    else:
                        choose = cand_result.n_inliers > best.n_inliers or (
                            cand_result.n_inliers == best.n_inliers
                            and np.nan_to_num(cand_result.median_residual, nan=1.0e9)
                            < np.nan_to_num(best.median_residual, nan=1.0e9)
                        )
                    if choose:
                        best = cand_result
                        best_sidx = sidx
                        best_didx = didx
                    trials += 1
                    if best is not None and best.n_inliers >= min(12, len(src), len(dst)):
                        # Good enough; still deterministic but much faster for
                        # well-behaved image sequences.
                        break
                    if trials >= max_trials:
                        break
                if trials >= max_trials or (best is not None and best.n_inliers >= min(12, len(src), len(dst))):
                    break
            if trials >= max_trials or (best is not None and best.n_inliers >= min(12, len(src), len(dst))):
                break
        if trials >= max_trials or (best is not None and best.n_inliers >= min(12, len(src), len(dst))):
            break

    if best is None or best_sidx is None or best_didx is None or best.n_inliers < 2:
        return TransformResult(np.eye(2), np.zeros(2), mode, status="alignment failed")

    # Refine from all inliers found by the best candidate.
    matched_src = src[best_sidx]
    matched_dst = dst[best_didx]
    if full_affine and len(matched_src) >= 3:
        A_refined, b_refined = _fit_affine(matched_src, matched_dst)
    else:
        A_refined, b_refined = _fit_similarity(matched_src, matched_dst, allow_reflection=allow_reflection)
    sidx2, didx2, _dist2 = _match_with_transform(src, dst, A_refined, b_refined, tolerance)
    if len(sidx2) >= max(2, len(best_sidx)):
        matched_src = src[sidx2]
        matched_dst = dst[didx2]
        if full_affine and len(matched_src) >= 3:
            A_refined, b_refined = _fit_affine(matched_src, matched_dst)
        else:
            A_refined, b_refined = _fit_similarity(matched_src, matched_dst, allow_reflection=allow_reflection)

    residual = np.linalg.norm(_apply_xy_transform(matched_src, A_refined, b_refined) - matched_dst, axis=1)
    scale, angle, flip = _transform_properties(A_refined)
    return TransformResult(
        A_refined,
        b_refined,
        mode,
        n_inliers=int(len(matched_src)),
        rms=float(np.sqrt(np.nanmean(residual**2))) if len(residual) else math.nan,
        median_residual=float(np.nanmedian(residual)) if len(residual) else math.nan,
        flip_detected=flip,
        scale=scale,
        rotation_deg=angle,
        status="ok" if len(matched_src) >= 2 else "weak alignment",
    )


def _warp_to_reference(image: np.ndarray, transform: TransformResult, output_shape: Tuple[int, int], interpolation: str) -> np.ndarray:
    if ndimage is None:
        raise RuntimeError("scipy is required for image alignment.")
    A = np.asarray(transform.matrix, dtype=float)
    b = np.asarray(transform.offset, dtype=float)
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("The alignment transform is singular and cannot be applied.")
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrix_yx = swap @ invA @ swap
    offset_yx = -swap @ invA @ b
    interp_l = str(interpolation or "Linear").lower()
    if interp_l.startswith("nearest"):
        order = 0
    elif interp_l.startswith("linear"):
        order = 1
    else:
        order = 3
    return ndimage.affine_transform(
        np.asarray(image, dtype=float),
        matrix=matrix_yx,
        offset=offset_yx,
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=(order > 1),
    )


def _valid_overlap_mask(shape: Tuple[int, int], transforms: Sequence[TransformResult]) -> np.ndarray:
    """Return a boolean mask for pixels covered by all aligned frames."""
    if ndimage is None:
        return np.ones(shape, dtype=bool)
    base = np.ones(shape, dtype=float)
    masks: List[np.ndarray] = []
    for tr in transforms:
        warped = _warp_to_reference(base, tr, shape, "Nearest")
        masks.append(np.isfinite(warped) & (warped > 0.5))
    if not masks:
        return np.ones(shape, dtype=bool)
    return np.logical_and.reduce(masks)


def _crop_bbox(mask: np.ndarray) -> Optional[Tuple[slice, slice]]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0 or len(yy) == 0:
        return None
    return slice(int(np.min(yy)), int(np.max(yy)) + 1), slice(int(np.min(xx)), int(np.max(xx)) + 1)


# ---------------------------------------------------------------------------
# Main reduction worker
# ---------------------------------------------------------------------------


def _prepare_output_root(settings: ReductionSettings) -> Tuple[Path, Path, Path, Path, Path]:
    if settings.output_folder.strip():
        base = Path(settings.output_folder.strip())
    else:
        base = Path(settings.science_folder.strip()) / "ExoPhotoCurve_reduced"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base / f"reduction_{timestamp}"
    masters = root / "masters"
    calibrated = root / "calibrated"
    aligned = root / "aligned"
    reports = root / "reports"
    for p in (masters, calibrated, aligned, reports):
        _ensure_dir(p)
    return root, masters, calibrated, aligned, reports


def _save_master(path: Path, data: np.ndarray, header, name: str, overwrite: bool) -> None:
    hdr = _copy_header(header) if header is not None and fits is not None else fits.Header()
    hdr["EPCMSTR"] = (name, "ExoPhotoCurve master calibration frame")
    hdr.add_history(f"ExoPhotoCurve created {name}.")
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=hdr, overwrite=overwrite)


def _update_header_after_calibration(header, settings: ReductionSettings, channel_info: str) -> object:
    hdr = _copy_header(header) if header is not None and fits is not None else fits.Header()
    hdr["EPCRED"] = (True, "Reduced by ExoPhotoCurve")
    hdr["EPCCAL"] = (True, "Bias/dark/flat calibration applied")
    hdr["EPCBAYER"] = (channel_info, "Bayer channel extraction, if any")
    hdr.add_history("ExoPhotoCurve: calibrated with master frames.")
    hdr.add_history("No denoise/sharpen/background aesthetic processing was applied.")
    return hdr


def _update_header_after_alignment(header, transform: TransformResult) -> object:
    hdr = _copy_header(header) if header is not None and fits is not None else fits.Header()
    hdr["EPCALIGN"] = (True, "Aligned by ExoPhotoCurve")
    hdr["EPCINL"] = (int(transform.n_inliers), "Alignment inlier stars")
    if np.isfinite(transform.rms):
        hdr["EPCRMS"] = (float(transform.rms), "Alignment RMS in pixels")
    if np.isfinite(transform.rotation_deg):
        hdr["EPCROT"] = (float(transform.rotation_deg), "Approx. rotation in degrees")
    if np.isfinite(transform.scale):
        hdr["EPCSCALE"] = (float(transform.scale), "Approx. geometric scale")
    hdr["EPCFLIP"] = (bool(transform.flip_detected), "Reflection/meridian flip detected")
    hdr.add_history("ExoPhotoCurve: image registered to reference frame.")
    return hdr


def _calibrate_one(
    raw: np.ndarray,
    header,
    settings: ReductionSettings,
    master_bias: Optional[np.ndarray],
    master_dark_raw: Optional[np.ndarray],
    master_dark_signal: Optional[np.ndarray],
    master_flat_norm: Optional[np.ndarray],
    dark_exptime: float,
    exptime_keys: Sequence[str],
) -> np.ndarray:
    img = raw.astype(np.float64, copy=True)
    science_exp = _header_get_float(header, exptime_keys, default=math.nan)

    if master_dark_raw is not None:
        use_scaled = False
        scale = 1.0
        if str(settings.dark_scaling).lower() in {"on", "auto"} and np.isfinite(science_exp) and np.isfinite(dark_exptime) and dark_exptime > 0:
            if master_dark_signal is not None:
                # Scaling is safe only for the dark signal, not for a raw dark
                # that still contains bias/offset.
                scale = float(science_exp / dark_exptime)
                if str(settings.dark_scaling).lower() == "on" or not math.isclose(scale, 1.0, rel_tol=0.02, abs_tol=0.02):
                    use_scaled = True
        if master_bias is not None and master_dark_signal is not None:
            img = img - master_bias - (master_dark_signal * scale if use_scaled else master_dark_signal)
        else:
            img = img - master_dark_raw
    elif master_bias is not None:
        img = img - master_bias

    if master_flat_norm is not None:
        img = img / master_flat_norm

    return img


def reduce_sequence(settings: ReductionSettings, progress: ProgressCallback = None, cancel_token: Optional[CancellationToken] = None) -> ReductionResult:
    """Run calibration and optional alignment on a FITS sequence."""
    if fits is None:
        raise RuntimeError("Astropy is required for FITS reduction. Install it with: pip install astropy")
    if ndimage is None or cKDTree is None:
        raise RuntimeError("scipy is required for image alignment and diagnostics. Install it with: pip install scipy")
    _check_cancel(cancel_token)

    science_files = _list_fits(settings.science_folder, settings.science_pattern)
    if not science_files:
        raise ValueError("No science FITS files were found. Check the science folder and pattern.")
    bias_files = _list_fits(settings.bias_folder, settings.bias_pattern) if settings.bias_folder.strip() else []
    dark_files = _list_fits(settings.dark_folder, settings.dark_pattern) if settings.dark_folder.strip() else []
    flat_files = _list_fits(settings.flat_folder, settings.flat_pattern) if settings.flat_folder.strip() else []
    darkflat_files = _list_fits(settings.darkflat_folder, settings.darkflat_pattern) if settings.darkflat_folder.strip() else []
    exptime_keys = _split_keywords(settings.exptime_keywords) or ["EXPTIME"]

    root, masters_dir, calibrated_dir, aligned_dir, reports_dir = _prepare_output_root(settings)
    _log(progress, f"Output root: {root}", 1)
    _log(progress, f"Science frames: {len(science_files)}", 2)

    warnings: List[str] = []
    reused_master_files: Dict[str, str] = {}
    master_bias: Optional[np.ndarray] = None
    master_dark_raw: Optional[np.ndarray] = None
    master_dark_signal: Optional[np.ndarray] = None
    master_darkflat: Optional[np.ndarray] = None
    master_flat_norm: Optional[np.ndarray] = None
    master_headers: Dict[str, object] = {}

    reusable_master_files: Dict[str, str] = {}
    searched_master_dirs: List[str] = []
    if settings.reuse_masters:
        reusable_master_files, searched_master_dirs = find_reusable_master_files(settings)
        if settings.master_folder.strip():
            _log(progress, f"Master reuse enabled. Searching: {settings.master_folder.strip()}", 3)
        else:
            _log(progress, "Master reuse enabled. Auto-searching previous reductions from science/output folders.", 3)
        if searched_master_dirs:
            _log(progress, "Master search folders: " + "; ".join(searched_master_dirs[:4]), 3)
        else:
            warnings.append("Master reuse was enabled, but no plausible previous masters folder was found.")
        if reusable_master_files:
            for key, path in sorted(reusable_master_files.items()):
                _log(progress, f"Detected reusable {key}: {Path(path).name}", 3)
        else:
            warnings.append("Master reuse was enabled, but no reusable master files were detected.")

    if settings.reuse_masters and settings.reuse_master_bias:
        src = reusable_master_files.get("bias")
        if src:
            _log(progress, f"Reusing master bias: {Path(src).name}", 5)
            master_bias, hdr = _load_master_file(src, "bias")
            master_headers["bias"] = hdr
            reused_master_files["bias"] = src
            _copy_reused_master(src, masters_dir, "master_bias.fits", warnings)
            if bias_files:
                warnings.append("Master bias was reused; the selected bias frame folder was ignored.")
        else:
            warnings.append("Master bias was selected for reuse but was not found. The module will create it from bias frames if available.")

    if master_bias is None:
        if bias_files:
            _log(progress, f"Combining {len(bias_files)} bias frame(s)...", 5)
            master_bias, hdr, _rows = _combine_images(bias_files, settings.combine_method, settings.sigma_clip, progress, "bias frames", cancel_token)
            master_headers["bias"] = hdr
            _save_master(masters_dir / "master_bias.fits", master_bias, hdr, "master_bias", settings.overwrite)
        else:
            warnings.append("No bias frames selected. This is OK if dark frames/dark-flats include the detector offset.")

    if settings.reuse_masters and settings.reuse_master_dark:
        src = reusable_master_files.get("dark_raw")
        if src:
            _log(progress, f"Reusing master dark: {Path(src).name}", 12)
            master_dark_raw, hdr = _load_master_file(src, "dark")
            master_headers["dark"] = hdr
            reused_master_files["dark_raw"] = src
            _copy_reused_master(src, masters_dir, "master_dark_raw.fits", warnings)
            signal_src = reusable_master_files.get("dark_signal")
            if signal_src:
                try:
                    master_dark_signal, sig_hdr = _load_master_file(signal_src, "dark signal")
                    master_headers["dark_signal"] = sig_hdr
                    reused_master_files["dark_signal"] = signal_src
                    _copy_reused_master(signal_src, masters_dir, "master_dark_signal_minus_bias.fits", warnings)
                except Exception as exc:
                    warnings.append(f"The reusable dark-signal master could not be loaded and will be recomputed if possible: {exc}")
            if dark_files:
                warnings.append("Master dark was reused; the selected dark frame folder was ignored.")
        else:
            warnings.append("Master dark was selected for reuse but was not found. The module will create it from dark frames if available.")

    if master_dark_raw is None:
        if dark_files:
            _log(progress, f"Combining {len(dark_files)} dark frame(s)...", 12)
            master_dark_raw, hdr, _rows = _combine_images(dark_files, settings.combine_method, settings.sigma_clip, progress, "dark frames", cancel_token)
            master_headers["dark"] = hdr
            _save_master(masters_dir / "master_dark_raw.fits", master_dark_raw, hdr, "master_dark_raw", settings.overwrite)
            if master_bias is not None:
                master_dark_signal = master_dark_raw - master_bias
                _save_master(masters_dir / "master_dark_signal_minus_bias.fits", master_dark_signal, hdr, "master_dark_signal", settings.overwrite)
        else:
            warnings.append("No dark frames selected. Science frames will not receive a dark-current correction.")
    elif master_bias is not None and master_dark_signal is None:
        # With a reused raw dark and a valid bias, keep the existing safe logic:
        # subtract bias from the science and use a signal-only dark component.
        try:
            master_dark_signal = master_dark_raw - master_bias
            _save_master(masters_dir / "master_dark_signal_minus_bias.fits", master_dark_signal, master_headers.get("dark"), "master_dark_signal", settings.overwrite)
        except Exception as exc:
            warnings.append(f"Could not derive a signal-only dark from the reused dark and bias: {exc}")

    if settings.reuse_masters and settings.reuse_master_darkflat:
        src = reusable_master_files.get("darkflat")
        if src:
            _log(progress, f"Reusing master dark-flat: {Path(src).name}", 20)
            master_darkflat, hdr = _load_master_file(src, "dark-flat")
            master_headers["darkflat"] = hdr
            reused_master_files["darkflat"] = src
            _copy_reused_master(src, masters_dir, "master_darkflat.fits", warnings)
            if darkflat_files:
                warnings.append("Master dark-flat was reused; the selected dark-flat frame folder was ignored.")
        else:
            warnings.append("Master dark-flat was selected for reuse but was not found. The module will create it from dark-flat frames if available.")

    if master_darkflat is None:
        if darkflat_files:
            _log(progress, f"Combining {len(darkflat_files)} dark-flat frame(s)...", 20)
            master_darkflat, hdr, _rows = _combine_images(darkflat_files, settings.combine_method, settings.sigma_clip, progress, "dark-flat frames", cancel_token)
            master_headers["darkflat"] = hdr
            _save_master(masters_dir / "master_darkflat.fits", master_darkflat, hdr, "master_darkflat", settings.overwrite)
            if bias_files or "bias" in reused_master_files:
                warnings.append("Both bias and dark-flat masters are available. Flats are corrected with dark-flats; bias is still used for science/dark calibration.")
    elif master_bias is not None:
        warnings.append("Both bias and dark-flat masters are available. Flats are corrected with dark-flats if new flats are created; bias is still used for science/dark calibration.")

    if settings.reuse_masters and settings.reuse_master_flat:
        src = reusable_master_files.get("flat_norm")
        if src:
            _log(progress, f"Reusing normalized master flat: {Path(src).name}", 25)
            master_flat_norm, hdr = _load_master_file(src, "normalized flat")
            med_flat = _master_median(master_flat_norm)
            if not np.isfinite(med_flat) or abs(med_flat) < 1.0e-10:
                raise ValueError("The reused master flat has an invalid median and cannot be used.")
            if med_flat < 0.5 or med_flat > 1.5:
                warnings.append(f"The reused master flat median is {med_flat:.4g}, not close to 1. It was normalised internally before use.")
                master_flat_norm = _normalise_flat(master_flat_norm)
            master_headers["flat"] = hdr
            reused_master_files["flat_norm"] = src
            _copy_reused_master(src, masters_dir, "master_flat_normalized.fits", warnings)
            if flat_files:
                warnings.append("Normalized master flat was reused; the selected flat frame folder was ignored.")
        else:
            warnings.append("Normalized master flat was selected for reuse but was not found. The module will create it from flat frames if available.")

    if master_flat_norm is None:
        if flat_files:
            _log(progress, f"Combining {len(flat_files)} flat frame(s)...", 25)
            master_flat_raw, hdr, _rows = _combine_images(flat_files, settings.combine_method, settings.sigma_clip, progress, "flat frames", cancel_token)
            master_headers["flat"] = hdr
            if master_darkflat is not None:
                master_flat_cal = master_flat_raw - master_darkflat
                flat_method = "dark-flat corrected"
            elif master_bias is not None:
                master_flat_cal = master_flat_raw - master_bias
                flat_method = "bias corrected"
            else:
                master_flat_cal = master_flat_raw
                flat_method = "not bias/dark-flat corrected"
                warnings.append("Flats were provided without bias or dark-flats. The master flat was normalised but not offset-corrected.")
            master_flat_norm = _normalise_flat(master_flat_cal)
            _save_master(masters_dir / "master_flat_raw.fits", master_flat_raw, hdr, "master_flat_raw", settings.overwrite)
            _save_master(masters_dir / "master_flat_normalized.fits", master_flat_norm, hdr, f"master_flat_{flat_method}", settings.overwrite)
        else:
            warnings.append("No flat frames selected. Science frames will not receive a flat-field correction.")

    # Validate calibration shapes against the first science frame.
    first_raw, first_hdr, _idx = _read_fits_image(science_files[0])
    check_arrays: List[Tuple[str, np.ndarray]] = [("first science", first_raw)]
    for name, arr in (
        ("master bias", master_bias),
        ("master dark raw", master_dark_raw),
        ("master dark signal", master_dark_signal),
        ("master flat", master_flat_norm),
    ):
        if arr is not None:
            check_arrays.append((name, arr))
    _same_shape_or_fail(check_arrays)

    dark_exp = math.nan
    if dark_files and "dark_raw" not in reused_master_files:
        dark_exp = _median_exptime(dark_files, exptime_keys)
    elif master_headers.get("dark") is not None:
        dark_exp = _header_get_float(master_headers.get("dark"), exptime_keys, default=math.nan)
    science_exp = _median_exptime(science_files, exptime_keys)
    if master_dark_raw is not None and np.isfinite(dark_exp) and np.isfinite(science_exp):
        rel_diff = abs(dark_exp - science_exp) / max(abs(science_exp), 1.0e-9)
        if rel_diff > 0.02 and master_dark_signal is None:
            warnings.append("Dark and science exposure times differ, but no bias was available. Raw dark scaling was disabled to avoid scaling the detector offset.")
        elif rel_diff > 0.02 and str(settings.dark_scaling).lower() == "auto":
            warnings.append(f"Dark exposure ({dark_exp:.3f} s) differs from science exposure ({science_exp:.3f} s); dark signal will be scaled.")

    diagnostics: List[ImageDiagnostics] = []
    calibrated_files: List[str] = []
    calibrated_headers: List[object] = []
    detections: List[StarDetection] = []

    _log(progress, "Calibrating science frames...", 35)
    for i, src_path in enumerate(science_files, start=1):
        raw, hdr, _idx = _read_fits_image(src_path)
        calibrated = _calibrate_one(raw, hdr, settings, master_bias, master_dark_raw, master_dark_signal, master_flat_norm, dark_exp, exptime_keys)
        calibrated, channel_info = _extract_bayer_channel(calibrated, hdr, settings.camera_mode, settings.bayer_pattern, settings.bayer_channel)
        out_name = f"{Path(src_path).stem}_cal.fits"
        out_path = calibrated_dir / out_name
        out_hdr = _update_header_after_calibration(hdr, settings, channel_info)
        _write_fits(out_path, calibrated, out_hdr, settings.overwrite)

        try:
            det = detect_stars(calibrated, sigma=settings.detect_sigma, max_stars=max(settings.max_stars, 80))
        except Exception:
            det = StarDetection(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        diag = _diagnostics_for_image(src_path, calibrated, hdr, det, stage="calibrated")
        diag.output_file = str(out_path)
        diagnostics.append(diag)
        calibrated_files.append(str(out_path))
        calibrated_headers.append(out_hdr)
        detections.append(det)
        percent = 35 + int(30 * i / max(1, len(science_files)))
        _log(progress, f"Calibrated {i}/{len(science_files)}: {Path(src_path).name}", percent)

    aligned_files: List[str] = []
    transforms: List[TransformResult] = []

    if settings.run_alignment:
        _log(progress, "Selecting reference frame for alignment...", 68)
        if str(settings.reference_mode).startswith("Selected") and settings.reference_file.strip():
            selected = Path(settings.reference_file.strip())
            ref_idx = 0
            for idx, cal_path in enumerate(calibrated_files):
                cal = Path(cal_path)
                # Accept either the generated calibrated file or the original
                # science filename selected before the reduction starts.
                if cal == selected or cal.name == selected.name or cal.stem.startswith(selected.stem):
                    ref_idx = idx
                    break
        elif str(settings.reference_mode).startswith("First"):
            ref_idx = 0
        elif str(settings.reference_mode).startswith("Middle"):
            ref_idx = len(calibrated_files) // 2
        else:
            # Prefer many stars, low FWHM, low elongation.  Avoid using only the
            # first frame if seeing or tracking improves later in the sequence.
            scores: List[float] = []
            for det in detections:
                fwhm = float(np.nanmedian(det.fwhm)) if det.fwhm.size else 99.0
                elong = float(np.nanmedian(det.elongation)) if det.elongation.size else 9.0
                scores.append(det.n * 10.0 - fwhm * 3.0 - elong)
            ref_idx = int(np.nanargmax(scores)) if scores else 0

        ref_path = calibrated_files[ref_idx]
        ref_img, ref_hdr, _idx = _read_fits_image(ref_path)
        ref_det = detections[ref_idx]
        if ref_det.n < 2:
            warnings.append("Reference frame has too few detected stars. Alignment was skipped.")
            settings.run_alignment = False
        else:
            _log(progress, f"Reference frame: {Path(ref_path).name} with {ref_det.n} detected star(s).", 70)

    if settings.run_alignment:
        ref_img, ref_hdr, _idx = _read_fits_image(calibrated_files[ref_idx])
        output_shape = ref_img.shape
        for i, cal_path in enumerate(calibrated_files, start=1):
            img, hdr, _idx = _read_fits_image(cal_path)
            if i - 1 == ref_idx:
                tr = TransformResult(np.eye(2), np.zeros(2), settings.transform_mode, n_inliers=detections[ref_idx].n, rms=0.0, median_residual=0.0, flip_detected=False, scale=1.0, rotation_deg=0.0)
                aligned = img.copy()
            else:
                tr = estimate_transform(
                    detections[i - 1],
                    detections[ref_idx],
                    settings.transform_mode,
                    tolerance=float(settings.match_tolerance),
                    max_stars=int(settings.max_stars),
                )
                if tr.status not in {"ok", "weak alignment"}:
                    warnings.append(f"Alignment failed or weak for {Path(cal_path).name}: {tr.status}. Identity transform was used.")
                    tr = TransformResult(np.eye(2), np.zeros(2), settings.transform_mode, status=tr.status)
                    aligned = img.copy()
                else:
                    aligned = _warp_to_reference(img, tr, output_shape, settings.interpolation)
            transforms.append(tr)
            out_name = f"{Path(cal_path).stem}_aligned.fits"
            out_path = aligned_dir / out_name
            out_hdr = _update_header_after_alignment(hdr, tr)
            _write_fits(out_path, aligned, out_hdr, settings.overwrite)
            aligned_files.append(str(out_path))

            diag = _diagnostics_for_image(cal_path, aligned, hdr, detections[i - 1], stage="aligned")
            diag.output_file = str(out_path)
            diag.x_shift = float(tr.offset[0]) if tr.offset.size else math.nan
            diag.y_shift = float(tr.offset[1]) if tr.offset.size else math.nan
            diag.rotation_deg = tr.rotation_deg
            diag.scale = tr.scale
            diag.flip_detected = tr.flip_detected
            diag.alignment_rms = tr.rms
            diag.alignment_inliers = tr.n_inliers
            diag.status = tr.status
            diagnostics.append(diag)
            percent = 70 + int(25 * i / max(1, len(calibrated_files)))
            _log(progress, f"Aligned {i}/{len(calibrated_files)}: {Path(cal_path).name}", percent)

        if settings.crop_common and aligned_files and transforms:
            _log(progress, "Cropping aligned images to the common valid area...", 96)
            mask = _valid_overlap_mask(output_shape, transforms)
            bbox = _crop_bbox(mask)
            if bbox is not None:
                for path in aligned_files:
                    arr, hdr, _idx = _read_fits_image(path)
                    cropped = arr[bbox]
                    hdr.add_history("ExoPhotoCurve: cropped to common aligned overlap.")
                    _write_fits(path, cropped, hdr, settings.overwrite)
            else:
                warnings.append("Could not determine a common crop area. Aligned images were left uncropped.")
    else:
        # Keep the workflow convenient: if alignment is disabled, copy calibrated
        # files into the aligned folder so Build Light Curve can simply use the
        # final/reduced sequence path.
        for cal_path in calibrated_files:
            out_path = aligned_dir / Path(cal_path).name.replace("_cal.fits", "_final.fits")
            shutil.copy2(cal_path, out_path)
            aligned_files.append(str(out_path))
        _log(progress, "Alignment disabled: calibrated files copied to final folder.", 92)

    # Reports and settings.
    report_path = reports_dir / "reduction_report.csv"
    report_df = pd.DataFrame([asdict(d) for d in diagnostics])
    report_df.to_csv(report_path, index=False)

    settings_path = reports_dir / "reduction_settings.json"
    settings_dict = asdict(settings)
    settings_dict["created_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    settings_dict["science_files"] = [str(Path(f).name) for f in science_files]
    settings_dict["reused_master_files"] = reused_master_files
    settings_dict["searched_master_folders"] = searched_master_dirs
    settings_dict["warnings"] = warnings
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2, default=_json_safe)

    if warnings:
        with open(reports_dir / "reduction_warnings.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(warnings) + "\n")

    _check_cancel(cancel_token)
    _log(progress, "Reduction completed.", 100)
    return ReductionResult(
        output_root=str(root),
        calibrated_folder=str(calibrated_dir),
        aligned_folder=str(aligned_dir),
        report_path=str(report_path),
        settings_path=str(settings_path),
        n_science=len(science_files),
        n_calibrated=len(calibrated_files),
        n_aligned=len(aligned_files),
    )


# ---------------------------------------------------------------------------
# FreeSimpleGUI wrapper
# ---------------------------------------------------------------------------


def _folder_row(label: str, key: str, pattern_key: str, default_pattern: str = "*.fit*") -> List[sg.Element]:
    return [
        sg.Text(label, size=(11, 1)),
        sg.Input("", key=key, size=(38, 1)),
        sg.FolderBrowse("Browse"),
        sg.Text("Pattern"),
        sg.Input(default_pattern, key=pattern_key, size=(9, 1)),
    ]


def _make_reduction_layout() -> List[List[sg.Element]]:
    sg.set_options(font=("Helvetica", 12))
    input_frame = [
        _folder_row("Science", "-RED_SCI_FOLDER-", "-RED_SCI_PATTERN-"),
        _folder_row("Bias", "-RED_BIAS_FOLDER-", "-RED_BIAS_PATTERN-"),
        _folder_row("Dark", "-RED_DARK_FOLDER-", "-RED_DARK_PATTERN-"),
        _folder_row("Flat", "-RED_FLAT_FOLDER-", "-RED_FLAT_PATTERN-"),
        _folder_row("Dark-flat", "-RED_DARKFLAT_FOLDER-", "-RED_DARKFLAT_PATTERN-"),
        [
            sg.Text("Output", size=(11, 1)),
            sg.Input("", key="-RED_OUTPUT_FOLDER-", size=(38, 1)),
            sg.FolderBrowse("Browse"),
            sg.Text("empty = inside science folder", size=(27, 1)),
        ],
    ]

    calib_frame = [
        [
            sg.Text("Combine"),
            sg.Combo(["Median + sigma clipping", "Median", "Mean + sigma clipping", "Mean"], default_value="Median + sigma clipping", key="-RED_COMBINE-", size=(19, 1), readonly=True),
            sg.Text("sigma"),
            sg.Input("4.0", key="-RED_SIGMA-", size=(3, 1)),
            sg.Text("Dark scaling"),
            sg.Combo(["Auto", "Off", "On"], default_value="Auto", key="-RED_DARK_SCALING-", size=(4, 1), readonly=True),
            sg.Text("Exp. keys"),
            sg.Input("EXPTIME,EXPOSURE,EXPOSURE_TIME", key="-RED_EXPTIME_KEYS-", size=(19, 1)),
        ],
        [
            sg.Checkbox("Reuse existing masters", default=False, key="-RED_REUSE_MASTERS-", enable_events=True),
            sg.Text("Master folder"),
            sg.Input("", key="-RED_MASTER_FOLDER-", size=(28, 1), disabled=True),
            sg.FolderBrowse("Browse", target="-RED_MASTER_FOLDER-", key="-RED_MASTER_BROWSE-", disabled=True),
            sg.Button("Check masters", key="-RED_CHECK_MASTERS-", disabled=True),
        ],
        [
            sg.Text("Reuse:", size=(8, 1)),
            sg.Checkbox("Bias", default=True, key="-RED_REUSE_BIAS-", disabled=True),
            sg.Checkbox("Dark", default=True, key="-RED_REUSE_DARK-", disabled=True),
            sg.Checkbox("Dark-flat", default=True, key="-RED_REUSE_DARKFLAT-", disabled=True),
            sg.Checkbox("Flat", default=False, key="-RED_REUSE_FLAT-", disabled=True),
            # sg.Text("Flat reuse is usually best only when reducing the same sequence again.", text_color=('grey')),
        ],
        # [
        #     sg.Text("If dark-flats are provided, flats use dark-flats. Otherwise flats use bias when available.", text_color = ('grey')),
        # ],
    ]

    bayer_frame = [
        [
            sg.Text("Data source (mono or color)"),
            sg.Combo(["Mono", "Color Bayer"], default_value="Mono", key="-RED_CAMERA-", size=(13, 1), readonly=True, enable_events=True),
            sg.Text("Bayer pattern"),
            sg.Combo(["Auto", "RGGB", "BGGR", "GRBG", "GBRG"], default_value="Auto", key="-RED_BAYER_PATTERN-", size=(9, 1), readonly=True, disabled=True),
            sg.Text("Extract channel"),
            sg.Combo(["Mean G", "G1", "G2", "R", "B"], default_value="Mean G", key="-RED_BAYER_CHANNEL-", size=(10, 1), readonly=True, disabled=True),
        ],
        # [sg.Text("Bayer extraction keeps only real CFA pixels and does not perform RGB debayer interpolation.", text_color = ('grey'))],
    ]

    align_frame = [
        [
            sg.Checkbox("Align sequence", default=True, key="-RED_ALIGN-"),
            sg.Text("Reference"),
            sg.Combo(["Auto best frame", "First frame", "Middle frame", "Selected file"], default_value="Auto best frame", key="-RED_REFERENCE_MODE-", size=(13, 1), readonly=True),
        # ],
        # [
            sg.Text("Selected ref"),
            sg.Input("", key="-RED_REFERENCE_FILE-", size=(27, 1)),
            sg.FileBrowse("Browse", file_types=(("FITS", "*.fit *.fits *.fts"), ("All", "*.*"))),
        ],
        [
            sg.Text("Transform", size=(12, 1)),
            sg.Combo(["Shift only", "Shift + rotation/scale", "Full affine / meridian flip safe"], default_value="Full affine / meridian flip safe", key="-RED_TRANSFORM-", size=(28, 1), readonly=True),
            sg.Text("Interpolation"),
            sg.Combo(["Linear", "Cubic"], default_value="Linear", key="-RED_INTERP-", size=(8, 1), readonly=True),
        ],
        [
            sg.Text("Detect σ", size=(12, 1)),
            sg.Input("6.0", key="-RED_DETECT_SIGMA-", size=(6, 1)),
            sg.Text("Max stars"),
            sg.Input("60", key="-RED_MAX_STARS-", size=(6, 1)),
            sg.Text("Match tol px"),
            sg.Input("5.0", key="-RED_MATCH_TOL-", size=(6, 1)),
            sg.Checkbox("Crop common area", default=False, key="-RED_CROP_COMMON-"),
        ],
    ]

    run_frame = [
        [
            sg.Button("Run reduction", key="-RED_RUN-", button_color=("white", "#2d6cdf")),
            sg.Button("Stop", key="-RED_STOP-", disabled=True, button_color=("white", "#b00020")),
            sg.Button("Close"),
            sg.Text("Status:"),
            sg.Text("Ready.", key="-RED_STATUS-", size=(58, 1)),
        ],
        [sg.ProgressBar(100, orientation="h", size=(61, 14), key="-RED_PROGRESS-")],
        [sg.Multiline("", key="-RED_LOG-", size=(86, 14), disabled=True, autoscroll=True)],
    ]

    left = [
        [sg.Frame("1. Set calibration/science frames folders", input_frame, font=("Helvetica", 12, "bold"))],
        [sg.Frame("2. Calibration", calib_frame, font=("Helvetica", 12, "bold"))],
        [sg.Frame("3. Properties", bayer_frame, font=("Helvetica", 12, "bold"))],
        [sg.Frame("4. Alignment / registration", align_frame, font=("Helvetica", 12, "bold"))],
        [sg.Frame("5. Run", run_frame, font=("Helvetica", 12, "bold"))],
    ]

    return [[sg.Column(left, vertical_alignment="top")]]


def _settings_from_values(values: Dict[str, object]) -> ReductionSettings:
    sigma = parse_float(values.get("-RED_SIGMA-"), 4.0)
    detect_sigma = parse_float(values.get("-RED_DETECT_SIGMA-"), 6.0)
    max_stars = parse_int(values.get("-RED_MAX_STARS-"), 60)
    match_tol = parse_float(values.get("-RED_MATCH_TOL-"), 5.0)
    return ReductionSettings(
        science_folder=str(values.get("-RED_SCI_FOLDER-", "")).strip(),
        science_pattern=str(values.get("-RED_SCI_PATTERN-", "*.fit*")).strip() or "*.fit*",
        output_folder=str(values.get("-RED_OUTPUT_FOLDER-", "")).strip(),
        bias_folder=str(values.get("-RED_BIAS_FOLDER-", "")).strip(),
        bias_pattern=str(values.get("-RED_BIAS_PATTERN-", "*.fit*")).strip() or "*.fit*",
        dark_folder=str(values.get("-RED_DARK_FOLDER-", "")).strip(),
        dark_pattern=str(values.get("-RED_DARK_PATTERN-", "*.fit*")).strip() or "*.fit*",
        flat_folder=str(values.get("-RED_FLAT_FOLDER-", "")).strip(),
        flat_pattern=str(values.get("-RED_FLAT_PATTERN-", "*.fit*")).strip() or "*.fit*",
        darkflat_folder=str(values.get("-RED_DARKFLAT_FOLDER-", "")).strip(),
        darkflat_pattern=str(values.get("-RED_DARKFLAT_PATTERN-", "*.fit*")).strip() or "*.fit*",
        reuse_masters=bool(values.get("-RED_REUSE_MASTERS-", False)),
        master_folder=str(values.get("-RED_MASTER_FOLDER-", "")).strip(),
        reuse_master_bias=bool(values.get("-RED_REUSE_BIAS-", True)),
        reuse_master_dark=bool(values.get("-RED_REUSE_DARK-", True)),
        reuse_master_darkflat=bool(values.get("-RED_REUSE_DARKFLAT-", True)),
        reuse_master_flat=bool(values.get("-RED_REUSE_FLAT-", False)),
        combine_method=str(values.get("-RED_COMBINE-", "Median + sigma clipping")),
        sigma_clip=float(sigma if np.isfinite(sigma) and sigma > 0 else 4.0),
        dark_scaling=str(values.get("-RED_DARK_SCALING-", "Auto")),
        exptime_keywords=str(values.get("-RED_EXPTIME_KEYS-", "EXPTIME")),
        camera_mode=str(values.get("-RED_CAMERA-", "Mono")),
        bayer_pattern=str(values.get("-RED_BAYER_PATTERN-", "Auto")),
        bayer_channel=str(values.get("-RED_BAYER_CHANNEL-", "Mean G")),
        run_alignment=bool(values.get("-RED_ALIGN-", True)),
        reference_mode=str(values.get("-RED_REFERENCE_MODE-", "Auto best frame")),
        reference_file=str(values.get("-RED_REFERENCE_FILE-", "")).strip(),
        transform_mode=str(values.get("-RED_TRANSFORM-", "Full affine / meridian flip safe")),
        interpolation=str(values.get("-RED_INTERP-", "Linear")),
        crop_common=bool(values.get("-RED_CROP_COMMON-", False)),
        max_stars=max(10, int(max_stars if np.isfinite(max_stars) else 60)),
        detect_sigma=float(detect_sigma if np.isfinite(detect_sigma) and detect_sigma > 0 else 6.0),
        match_tolerance=float(match_tol if np.isfinite(match_tol) and match_tol > 0 else 5.0),
        overwrite=True,
    )



def _set_bayer_controls_state(window: sg.Window, camera_mode: str) -> None:
    """Enable Bayer pattern/channel selectors only for Color Bayer data."""
    is_color = str(camera_mode or "Mono").strip().lower().startswith("color")
    for key in ("-RED_BAYER_PATTERN-", "-RED_BAYER_CHANNEL-"):
        try:
            window[key].update(disabled=not is_color)
        except Exception:
            pass


def _set_master_reuse_controls_state(window: sg.Window, enabled: bool) -> None:
    """Enable master-reuse controls only when reuse is active."""
    for key in (
        "-RED_MASTER_FOLDER-",
        "-RED_MASTER_BROWSE-",
        "-RED_CHECK_MASTERS-",
        "-RED_REUSE_BIAS-",
        "-RED_REUSE_DARK-",
        "-RED_REUSE_DARKFLAT-",
        "-RED_REUSE_FLAT-",
    ):
        try:
            window[key].update(disabled=not bool(enabled))
        except Exception:
            pass


def _append_reduction_log(window: sg.Window, message: str) -> None:
    try:
        old = window["-RED_LOG-"].get()
        window["-RED_LOG-"].update(old + str(message).rstrip() + "\n")
    except Exception:
        pass


def _make_child_window_modal(window: sg.Window, parent_window: Optional[sg.Window]) -> None:
    """Keep the reduction window modal and prevent queued clicks on the parent."""
    if parent_window is None:
        try:
            window.TKroot.grab_set()
            window.TKroot.focus_force()
        except Exception:
            pass
        return

    try:
        window.TKroot.transient(parent_window.TKroot)
    except Exception:
        pass
    try:
        # Disable the parent explicitly.  This avoids the annoying Tk behavior
        # where clicks made on the inactive parent are queued and processed after
        # the child window closes.
        if hasattr(parent_window, "disable"):
            parent_window.disable()
        elif hasattr(parent_window, "Disable"):
            parent_window.Disable()
    except Exception:
        pass
    try:
        # Tk's -disabled attribute is mainly supported on Windows.  It is safe to
        # try it and ignore it on platforms where it is unavailable.
        parent_window.TKroot.attributes("-disabled", True)
    except Exception:
        pass
    try:
        window.TKroot.grab_set()
        window.TKroot.lift()
        window.TKroot.focus_force()
    except Exception:
        pass


def _restore_parent_window(parent_window: Optional[sg.Window]) -> None:
    """Re-enable the parent window after the reduction tool is closed."""
    if parent_window is None:
        return
    try:
        parent_window.TKroot.attributes("-disabled", False)
    except Exception:
        pass
    try:
        if hasattr(parent_window, "enable"):
            parent_window.enable()
        elif hasattr(parent_window, "Enable"):
            parent_window.Enable()
    except Exception:
        pass
    try:
        parent_window.TKroot.lift()
        parent_window.TKroot.focus_force()
    except Exception:
        pass


def run_image_reduction_tool(parent_window: Optional[sg.Window] = None) -> Optional[ReductionResult]:
    """Open the image-reduction GUI and return the last run result."""
    if fits is None:
        sg.popup_error("Astropy is required for FITS image reduction.\nInstall it with: pip install astropy")
        return None
    if ndimage is None or cKDTree is None:
        sg.popup_error("SciPy is required for image alignment and diagnostics.\nInstall it with: pip install scipy")
        return None

    window = sg.Window(
        "ExoPhotoCurve - Image reduction",
        _make_reduction_layout(),
        finalize=True,
        resizable=True,
        icon=icon_path if Path(icon_path).exists() else None,
        modal=True,
    )
    center_window(window)
    apply_preferences_to_window(window, "image_reduction", IMAGE_REDUCTION_PREFERENCE_KEYS)
    try:
        window.TKroot.update_idletasks()
        screen_w = window.TKroot.winfo_screenwidth()
        screen_h = window.TKroot.winfo_screenheight()
        win_w = max(window.TKroot.winfo_width(), window.TKroot.winfo_reqwidth())
        win_h = max(window.TKroot.winfo_height(), window.TKroot.winfo_reqheight())
        x = max(0, int((screen_w - win_w) / 2))
        y = max(0, int((screen_h - win_h) / 2))
        window.move(x, y)
    except Exception:
        pass

    try:
        _set_bayer_controls_state(window, str(window["-RED_CAMERA-"].get()))
    except Exception:
        _set_bayer_controls_state(window, "Mono")
    try:
        _set_master_reuse_controls_state(window, bool(window["-RED_REUSE_MASTERS-"].get()))
    except Exception:
        _set_master_reuse_controls_state(window, False)
    _make_child_window_modal(window, parent_window)

    last_result: Optional[ReductionResult] = None
    reduction_running = False
    cancel_token: Optional[CancellationToken] = None

    def set_running_state(running: bool) -> None:
        """Enable/disable controls while a reduction worker is active."""
        try:
            window["-RED_RUN-"].update(disabled=running)
            window["-RED_STOP-"].update(disabled=not running)
            window["Close"].update(disabled=running)
        except Exception:
            pass

    def gui_progress(message: str, percent: Optional[int] = None) -> None:
        """Thread-safe progress callback used by the reduction worker."""
        try:
            if cancel_token is not None and cancel_token.is_cancelled():
                raise ReductionCancelled("Image reduction stopped by user.")
            window.write_event_value("-RED_THREAD_PROGRESS-", (str(message), percent))
        except ReductionCancelled:
            raise
        except Exception:
            pass

    def handle_progress(message: str, percent: Optional[int] = None) -> None:
        try:
            old = window["-RED_LOG-"].get()
            window["-RED_LOG-"].update(old + str(message) + "\n")
            window["-RED_STATUS-"].update(str(message)[:80])
            if percent is not None:
                window["-RED_PROGRESS-"].update(max(0, min(100, int(percent))))
        except Exception:
            pass

    def reduction_worker(settings: ReductionSettings, token: CancellationToken) -> None:
        try:
            result = reduce_sequence(settings, progress=gui_progress, cancel_token=token)
            window.write_event_value("-RED_THREAD_DONE-", result)
        except ReductionCancelled as exc:
            window.write_event_value("-RED_THREAD_CANCELLED-", str(exc))
        except Exception:
            window.write_event_value("-RED_THREAD_ERROR-", traceback.format_exc(limit=8))


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
                if reduction_running:
                    if cancel_token is not None:
                        cancel_token.cancel()
                    handle_progress("Stop requested. Waiting for the current safe checkpoint before closing...", None)
                    continue
                break
            if event == "-RED_THREAD_PROGRESS-":
                message, percent = values.get("-RED_THREAD_PROGRESS-", ("", None))
                handle_progress(str(message), percent)
                continue
            if event == "-RED_THREAD_DONE-":
                reduction_running = False
                cancel_token = None
                set_running_state(False)
                last_result = values.get("-RED_THREAD_DONE-")
                handle_progress("Reduction completed.", 100)
                if last_result is not None:
                    msg = (
                        "Reduction completed.\n\n"
                        f"Science frames: {last_result.n_science}\n"
                        f"Calibrated frames: {last_result.n_calibrated}\n"
                        f"Final/aligned frames: {last_result.n_aligned}\n\n"
                        f"Final sequence folder:\n{last_result.aligned_folder}\n\n"
                        f"Report:\n{last_result.report_path}"
                    )
                    sg.popup_ok(msg, title="ExoPhotoCurve reduction completed")
                    if parent_window is not None:
                        try:
                            parent_window["-STATUS-"].update(f"Reduced sequence ready: {last_result.aligned_folder}")
                        except Exception:
                            pass
                continue
            if event == "-RED_THREAD_CANCELLED-":
                reduction_running = False
                cancel_token = None
                set_running_state(False)
                handle_progress(str(values.get("-RED_THREAD_CANCELLED-", "Reduction stopped by user.")), None)
                window["-RED_STATUS-"].update("Reduction stopped.")
                continue
            if event == "-RED_THREAD_ERROR-":
                reduction_running = False
                cancel_token = None
                set_running_state(False)
                tb = str(values.get("-RED_THREAD_ERROR-", ""))
                window["-RED_LOG-"].update(window["-RED_LOG-"].get() + "\nERROR:\n" + tb + "\n")
                sg.popup_error("Reduction failed. See the reduction log for details.")
                continue
            if event == "-RED_STOP-":
                if reduction_running and cancel_token is not None:
                    cancel_token.cancel()
                    window["-RED_STOP-"].update(disabled=True)
                    handle_progress("Stop requested. The reduction will stop at the next safe checkpoint...", None)
                continue
            if reduction_running:
                continue
            if event == "-RED_CAMERA-":
                _set_bayer_controls_state(window, values.get("-RED_CAMERA-", "Mono"))
                continue
            if event == "-RED_REUSE_MASTERS-":
                _set_master_reuse_controls_state(window, bool(values.get("-RED_REUSE_MASTERS-", False)))
                continue
            if event == "-RED_CHECK_MASTERS-":
                try:
                    settings = _settings_from_values(values)
                    report = build_master_check_report(settings)
                    _append_reduction_log(window, "\n" + report + "\n")
                    try:
                        sg.popup_scrolled(report, title="Reusable master check", size=(90, 28))
                    except Exception:
                        sg.popup_ok(report, title="Reusable master check")
                except Exception as exc:
                    sg.popup_error(f"Master check failed:\n{exc}")
                continue
            if event == "-RED_RUN-":
                try:
                    settings = _settings_from_values(values)
                    if not settings.science_folder or not Path(settings.science_folder).is_dir():
                        sg.popup_error("Please select a valid science-frame folder.")
                        continue
                    if settings.reuse_masters and not any((settings.reuse_master_bias, settings.reuse_master_dark, settings.reuse_master_darkflat, settings.reuse_master_flat)):
                        sg.popup_error("Master reuse is enabled, but no master type is selected. Select at least one master type or disable master reuse.")
                        continue
                    window["-RED_LOG-"].update("")
                    window["-RED_PROGRESS-"].update(0)
                    handle_progress("Starting reduction...", 0)
                    cancel_token = CancellationToken()
                    reduction_running = True
                    set_running_state(True)
                    threading.Thread(target=reduction_worker, args=(settings, cancel_token), daemon=True).start()
                except Exception as exc:
                    reduction_running = False
                    cancel_token = None
                    set_running_state(False)
                    tb = traceback.format_exc(limit=5)
                    window["-RED_LOG-"].update(window["-RED_LOG-"].get() + "\nERROR:\n" + tb + "\n")
                    sg.popup_error(f"Reduction failed:\n{exc}")
    finally:
        if not reduction_running:
            _save_reduction_preferences(window)
        try:
            window.TKroot.grab_release()
        except Exception:
            pass
        window.close()
        _restore_parent_window(parent_window)

    return last_result


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
