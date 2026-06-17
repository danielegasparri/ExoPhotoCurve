"""Time-system conversion utilities for transit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TimeConversionResult:
    """Result of converting input photometric times to BJD_TDB."""

    time_bjd_tdb: np.ndarray
    applied: bool
    correction_minutes: float
    note: str


def _parse_float(value: object, name: str) -> float:
    """Parse a required floating-point GUI value."""
    try:
        out = float(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Invalid {name}: {value!r}")
    return out


def get_observatory_parameters(values: dict) -> Tuple[float, float, float]:
    """Return observatory latitude, longitude and altitude from GUI values.

    Longitude follows the usual astronomical/geographical convention used by
    Astropy: east-positive and west-negative.
    """
    lat = _parse_float(values.get("-TR_OBS_LAT-", ""), "observatory latitude")
    lon = _parse_float(values.get("-TR_OBS_LON-", ""), "observatory longitude")
    alt = _parse_float(values.get("-TR_OBS_ALT-", "0"), "observatory altitude")

    if not (-90.0 <= lat <= 90.0):
        raise ValueError("Observatory latitude must be between -90 and +90 degrees.")
    if not (-360.0 <= lon <= 360.0):
        raise ValueError("Observatory longitude must be in degrees, east-positive, west-negative.")
    return lat, lon, alt


def convert_input_times_to_bjd_tdb(
    time_days: np.ndarray,
    input_time_system: str,
    ra_deg: float,
    dec_deg: float,
    observatory_lat_deg: float,
    observatory_lon_deg: float,
    observatory_alt_m: float,
) -> TimeConversionResult:
    """Convert mid-exposure input times to BJD_TDB.

    Supported input systems are:
    - BJD_TDB: returned unchanged;
    - BJD_UTC: converted from UTC scale to TDB scale without a new barycentric
      light-time correction;
    - JD_UTC: converted to BJD_TDB using Astropy, target coordinates and the
      observatory location.

    HJD_UTC and Other are left unchanged with a warning note, because converting
    HJD to BJD robustly is not a simple offset.
    """
    t = np.asarray(time_days, dtype=float)
    system = str(input_time_system).strip().upper()

    if system == "BJD_TDB":
        return TimeConversionResult(
            time_bjd_tdb=t.copy(),
            applied=False,
            correction_minutes=0.0,
            note="Input times were assumed to be already in BJD_TDB.",
        )

    try:
        from astropy import units as u
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        from astropy.utils import iers
    except Exception as exc:
        raise RuntimeError(
            "Time conversion to BJD_TDB requires astropy. Install it with: pip install astropy"
        ) from exc

    # Avoid long network stalls when Astropy tries to update IERS tables.
    # Astropy normally refuses to use IERS-A predictive values older than
    # ``auto_max_age`` days.  That is sensible for high-precision work, but it
    # can make an offline diagnostic tool fail completely when the local IERS
    # table is slightly stale.  For our purpose, using the local table with a
    # warning is preferable to aborting the BJD_TDB conversion.  Users who need
    # maximum timing precision can update ``astropy-iers-data`` periodically.
    try:
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        iers.conf.iers_degraded_accuracy = "warn"
    except Exception:
        pass

    if system == "BJD_UTC":
        time_utc = Time(t, format="jd", scale="utc")
        out = np.asarray(time_utc.tdb.jd, dtype=float)
        corr_min = float(np.nanmedian((out - t) * 24.0 * 60.0))
        return TimeConversionResult(
            time_bjd_tdb=out,
            applied=True,
            correction_minutes=corr_min,
            note="Converted BJD_UTC to BJD_TDB by changing the time scale from UTC to TDB.",
        )

    if system != "JD_UTC":
        return TimeConversionResult(
            time_bjd_tdb=t.copy(),
            applied=False,
            correction_minutes=0.0,
            note=(
                "Input time system is not fully supported for automatic barycentric correction; "
                "times were used as provided. O-C may be biased."
            ),
        )

    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        raise ValueError(
            "JD_UTC to BJD_TDB conversion requires RA/Dec in the planet catalogue. "
            "Regenerate the catalogue with the updated NASA builder or add ra_deg and dec_deg."
        )

    location = EarthLocation(
        lat=float(observatory_lat_deg) * u.deg,
        lon=float(observatory_lon_deg) * u.deg,
        height=float(observatory_alt_m) * u.m,
    )
    target = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")

    time_utc = Time(t, format="jd", scale="utc", location=location)
    ltt_bary = time_utc.light_travel_time(target, kind="barycentric")
    out = np.asarray((time_utc.tdb + ltt_bary).jd, dtype=float)
    corr_min = float(np.nanmedian((out - t) * 24.0 * 60.0))

    return TimeConversionResult(
        time_bjd_tdb=out,
        applied=True,
        correction_minutes=corr_min,
        note="Converted JD_UTC mid-exposure times to BJD_TDB using Astropy barycentric correction.",
    )
