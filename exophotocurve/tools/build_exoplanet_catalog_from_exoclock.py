#!/usr/bin/env python3
"""Build an ExoPhotoCurve transit catalogue from the ExoClock planets database.

The default output is an **ExoClock-pure** catalogue: ephemerides, duration,
transit depth, Rp/Rs when available, coordinates and ExoClock observability
metadata are taken from the ExoClock JSON endpoint only.  No NASA geometry is
mixed in unless explicitly requested.

This is important for transparency.  ExoClock currently provides excellent and
frequently updated transit timing information, but its public JSON endpoint may
not contain every physical parameter needed by the optional ``batman`` physical
model for every planet.  When a/Rs or inclination are missing, ExoPhotoCurve's
``Auto`` model mode will fall back to its empirical flux-depth template instead
of silently creating an ExoClock+NASA hybrid model.

Typical pure-catalogue usage from the project root:

    python tools/build_exoplanet_catalog_from_exoclock.py

This writes:

    exophotocurve/catalogs/exoclock_transit_catalog.csv

To deliberately create an ExoClock+NASA hybrid catalogue, pass a supplement
catalogue explicitly:

    python tools/build_exoplanet_catalog_from_nasa.py
    python tools/build_exoplanet_catalog_from_exoclock.py \
        --supplement-catalogue exophotocurve/catalogs/exoplanet_transit_catalog.csv \
        --output exophotocurve/catalogs/exoclock_hybrid_transit_catalog.csv

Hybrid rows are marked in the output columns and in the notes field.  Use such a
catalogue only when you explicitly want ExoClock timing/depth combined with
NASA-style geometry for the physical transit engine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from exophotocurve.exoplanet_catalog import CATALOGUE_COLUMNS, default_catalogue_path, default_exoclock_catalogue_path
except Exception:  # pragma: no cover - fallback for unusual standalone use
    CATALOGUE_COLUMNS = [
        "planet_name", "aliases", "period_days", "period_err_days",
        "t0_bjd_tdb", "t0_err_days", "duration_hours", "duration_err_hours",
        "depth_ppt", "depth_err_ppt", "rp_rs", "rp_rs_err",
        "model_depth_ppt", "a_rs", "a_rs_err", "inclination_deg",
        "inclination_err_deg", "ecc", "ecc_err", "omega_deg",
        "omega_err_deg", "stellar_teff_k", "stellar_teff_err_k",
        "stellar_logg", "stellar_logg_err", "stellar_feh",
        "stellar_feh_err", "ra_deg", "dec_deg", "time_reference",
        "source", "notes",
    ]

    def default_catalogue_path() -> Path:
        return PROJECT_ROOT / "exophotocurve" / "catalogs" / "exoplanet_transit_catalog.csv"

    def default_exoclock_catalogue_path() -> Path:
        return PROJECT_ROOT / "exophotocurve" / "catalogs" / "exoclock_transit_catalog.csv"


EXOCLOCK_JSON_URL = "https://www.exoclock.space/database/planets_json"

PROVENANCE_COLUMNS = [
    "catalogue_type",
    "depth_source",
    "rprs_source",
    "geometry_source",
    "stellar_source",
]

EXTRA_COLUMNS = PROVENANCE_COLUMNS + [
    "exoclock_priority",
    "exoclock_total_observations",
    "exoclock_recent_observations",
    "exoclock_current_oc_min",
    "exoclock_min_telescope_inches",
    "v_mag",
    "r_mag",
    "gaia_g_mag",
]

SUPPLEMENT_COLUMNS = [
    # Geometry and stellar parameters used by the batman physical model.
    "a_rs",
    "a_rs_err",
    "inclination_deg",
    "inclination_err_deg",
    "ecc",
    "ecc_err",
    "omega_deg",
    "omega_err_deg",
    "stellar_teff_k",
    "stellar_teff_err_k",
    "stellar_logg",
    "stellar_logg_err",
    "stellar_feh",
    "stellar_feh_err",
    # Useful uncertainties if present in the supplement catalogue.
    "rp_rs_err",
]


def _first_from_container(value: Any) -> Any:
    """Return the most likely numeric value from simple containers."""
    if isinstance(value, dict):
        for key in ("value", "val", "nominal", "mean", "median", "best", "data"):
            if key in value:
                return value[key]
        for candidate in value.values():
            if not isinstance(candidate, (dict, list, tuple)):
                return candidate
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _get_first(item: dict[str, Any], *keys: str) -> Any:
    """Return the first available value among several possible JSON keys."""
    value, _key = _get_first_with_key(item, *keys)
    return value


def _get_first_with_key(item: dict[str, Any], *keys: str) -> tuple[Any, str]:
    """Return the first available value and the JSON key that supplied it."""
    for key in keys:
        if key in item:
            value = item.get(key)
            if value not in (None, "", "--"):
                return value, key
    return None, ""


def _to_float(value: Any) -> float:
    """Convert a value to float, returning NaN for empty or invalid entries.

    The parser accepts strings containing units or uncertainties, e.g.
    '1.234 +/- 0.005', and simple containers such as [value, uncertainty] or
    {'value': value}.
    """
    value = _first_from_container(value)
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "--"}:
            return float("nan")
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        text = text.replace("−", "-")
        try:
            return float(text)
        except ValueError:
            import re

            match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?", text)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return float("nan")
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalise_key(text: Any) -> str:
    """Return a compact name key for cross-matching planet names."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _normalise_alias(name: str) -> str:
    """Return a compact alias useful for filename matching."""
    return str(name).replace(" ", "").replace("-", "").replace("_", "")


def _depth_mmag_to_ppt(depth_mmag: Any) -> float:
    """Convert transit depth from millimagnitudes to relative-flux ppt.

    For a dimming of dm mag, F_transit / F_out = 10^(-dm/2.5).  The relative
    flux depth is therefore 1 - 10^(-dm/2.5), expressed here in ppt.
    """
    mmag = _to_float(depth_mmag)
    if not np.isfinite(mmag) or mmag <= 0:
        return float("nan")
    dm_mag = mmag / 1000.0
    return (1.0 - 10.0 ** (-dm_mag / 2.5)) * 1000.0


def _relative_or_ppt_depth(value: Any) -> float:
    """Parse a direct flux-depth value that may be fractional or already in ppt."""
    direct_depth = _to_float(value)
    if not np.isfinite(direct_depth) or direct_depth <= 0:
        return float("nan")
    return direct_depth * 1000.0 if direct_depth < 0.2 else direct_depth


def _split_angle_parts(text: str) -> list[float]:
    """Split a sexagesimal coordinate string into numeric parts."""
    cleaned = text.strip().lower().replace("h", ":").replace("d", ":")
    cleaned = cleaned.replace("m", ":").replace("s", "")
    cleaned = cleaned.replace(",", " ").replace(";", " ")
    if ":" in cleaned:
        parts = [part for part in cleaned.split(":") if part.strip()]
    else:
        parts = [part for part in cleaned.split() if part.strip()]
    return [float(part) for part in parts]


def _parse_ra_deg(value: Any) -> float:
    """Parse right ascension as decimal degrees."""
    val = _to_float(value)
    text = str(value).strip()
    if np.isfinite(val) and not any(sep in text for sep in [":", "h", "m", "s", " "]):
        return val
    if not text or text.lower() in {"nan", "none", "null", "--"}:
        return float("nan")
    try:
        parts = _split_angle_parts(text)
        if not parts:
            return float("nan")
        hours = abs(parts[0]) + (parts[1] if len(parts) > 1 else 0.0) / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0
        return hours * 15.0
    except Exception:
        return float("nan")


def _parse_dec_deg(value: Any) -> float:
    """Parse declination as decimal degrees."""
    val = _to_float(value)
    text = str(value).strip()
    if np.isfinite(val) and not any(sep in text for sep in [":", "d", "m", "s", " "]):
        return val
    if not text or text.lower() in {"nan", "none", "null", "--"}:
        return float("nan")
    try:
        sign = -1.0 if text.lstrip().startswith("-") else 1.0
        parts = _split_angle_parts(text.replace("+", "").replace("-", ""))
        if not parts:
            return float("nan")
        deg = abs(parts[0]) + (parts[1] if len(parts) > 1 else 0.0) / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0
        return sign * deg
    except Exception:
        return float("nan")


def _looks_like_planet_row(item: dict[str, Any]) -> bool:
    """Return True if a dictionary looks like one ExoClock planet row."""
    keys = set(item.keys())
    essential_any = {"name", "planet", "planet_name", "pl_name"}
    ephemeris_any = {"period_days", "period", "t0_bjd_tdb", "mid_time", "duration_hours", "depth_mmag"}
    return bool(keys & essential_any) or len(keys & ephemeris_any) >= 2


def _iter_exoclock_planets(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield planet dictionaries from the ExoClock JSON payload."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield dict(item)
        return

    if not isinstance(payload, dict):
        return

    for container_key in ("planets", "data", "results", "objects", "targets", "planet_data"):
        if container_key in payload:
            container = payload[container_key]
            yielded = False
            for row in _iter_exoclock_planets(container):
                yielded = True
                yield row
            if yielded:
                return

    planet_like_values = []
    for key, value in payload.items():
        if isinstance(value, dict):
            row = dict(value)
            row.setdefault("name", key)
            if _looks_like_planet_row(row):
                planet_like_values.append(row)

    if planet_like_values:
        for row in planet_like_values:
            yield row
        return

    for value in payload.values():
        if isinstance(value, (dict, list)):
            yield from _iter_exoclock_planets(value)


def download_exoclock_json(url: str = EXOCLOCK_JSON_URL) -> Any:
    """Download and decode the ExoClock planets JSON database."""
    print("Downloading ExoClock planets database:")
    print(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ExoPhotoCurve-ExoClock-catalogue-builder/1.4"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace")
        print("\nExoClock returned an HTTP error.")
        print(f"HTTP status: {exc.code} {exc.reason}")
        print(error_payload[:4000])
        raise
    except urllib.error.URLError as exc:
        print("\nCould not connect to ExoClock.")
        print(exc)
        raise

    return json.loads(payload)


def _get_float_with_source(item: dict[str, Any], source_label: str, *keys: str) -> tuple[float, str]:
    value, key = _get_first_with_key(item, *keys)
    parsed = _to_float(value)
    if np.isfinite(parsed):
        return parsed, f"{source_label}:{key}"
    return float("nan"), "missing"


def _source_if_any(prefix: str, values: list[float]) -> str:
    return prefix if any(np.isfinite(v) for v in values) else "missing"


def build_exoclock_dataframe(payload: Any, generated: str) -> pd.DataFrame:
    """Convert the ExoClock JSON payload to ExoPhotoCurve catalogue rows."""
    rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    raw_items = list(_iter_exoclock_planets(payload))
    print(f"Found {len(raw_items)} candidate planet rows in the ExoClock JSON payload.")

    for item in raw_items:
        planet_name = str(_get_first(item, "name", "planet_name", "planet", "pl_name", "target") or "").strip()
        if not planet_name:
            continue

        period_days = _to_float(_get_first(item, "period_days", "period", "orbital_period", "pl_orbper"))
        t0_bjd_tdb = _to_float(_get_first(item, "t0_bjd_tdb", "mid_time", "t0", "epoch", "tc", "pl_tranmid"))
        duration_hours = _to_float(_get_first(item, "duration_hours", "duration", "transit_duration", "pl_trandur"))

        # Depth is stored internally in relative-flux ppt, never in mmag.
        # ExoClock has historically exposed the millimagnitude depth under
        # slightly different names.  In some JSON dumps the generic key is just
        # ``depth``; for ExoClock this is treated as a magnitude depth and is
        # converted immediately to relative-flux ppt.
        depth_value, depth_key = _get_first_with_key(
            item,
            "depth_mmag", "transit_depth_mmag", "depth_mmags",
            "depth_R_mmag", "depth_r_mmag", "depth_r", "depth",
        )
        depth_ppt = _depth_mmag_to_ppt(depth_value)
        depth_source = f"ExoClock:{depth_key} converted mmag to flux ppt" if np.isfinite(depth_ppt) else "missing"

        if not np.isfinite(depth_ppt):
            direct_value, direct_key = _get_first_with_key(item, "depth_ppt", "depth_ppthousand", "depth_relative", "transit_depth")
            depth_ppt = _relative_or_ppt_depth(direct_value)
            if np.isfinite(depth_ppt):
                depth_source = f"ExoClock:{direct_key} interpreted as flux depth"

        rp_rs_direct, rp_key = _get_float_with_source(item, "ExoClock", "rp_rs", "rp_over_rs", "rprs", "pl_ratror")
        if np.isfinite(rp_rs_direct) and rp_rs_direct > 0:
            rp_rs = rp_rs_direct
            rprs_source = rp_key
            if not np.isfinite(depth_ppt):
                depth_ppt = rp_rs * rp_rs * 1000.0
                depth_source = "derived from ExoClock Rp/Rs squared because no flux depth was available"
        else:
            rp_rs = math.sqrt(depth_ppt / 1000.0) if np.isfinite(depth_ppt) and depth_ppt > 0 else float("nan")
            rprs_source = "derived from ExoClock flux depth" if np.isfinite(rp_rs) else "missing"

        a_rs, a_rs_source = _get_float_with_source(
            item,
            "ExoClock",
            "a_rs", "a_over_rs", "aors", "scaled_semimajor_axis",
            "semimajor_axis_over_stellar_radius", "sma_over_rs", "pl_ratdor",
        )
        a_rs_err = _to_float(_get_first(item, "a_rs_err", "a_over_rs_unc", "a_over_rs_err", "pl_ratdorerr1"))

        inclination_deg, inc_source = _get_float_with_source(
            item,
            "ExoClock",
            "inclination_deg", "inclination", "inclination_degrees",
            "orbital_inclination", "pl_orbincl",
        )
        inclination_err_deg = _to_float(_get_first(item, "inclination_err", "inclination_unc", "inclination_deg_err", "pl_orbinclerr1"))

        ecc, ecc_source = _get_float_with_source(item, "ExoClock", "ecc", "eccentricity", "pl_orbeccen")
        if not np.isfinite(ecc):
            ecc = 0.0
            ecc_source = "assumed circular default"
        ecc_err = _to_float(_get_first(item, "ecc_err", "eccentricity_unc", "pl_orbeccenerr1"))

        omega_deg, omega_source = _get_float_with_source(
            item,
            "ExoClock",
            "omega_deg", "omega", "argument_of_periastron", "periastron_argument", "pl_orblper",
        )
        if not np.isfinite(omega_deg):
            omega_deg = 90.0
            omega_source = "default for circular orbit"
        omega_err_deg = _to_float(_get_first(item, "omega_err", "omega_unc", "argument_of_periastron_unc", "pl_orblpererr1"))

        stellar_teff_k, teff_source = _get_float_with_source(item, "ExoClock", "stellar_teff_k", "stellar_teff", "teff", "teff_k", "st_teff")
        stellar_teff_err_k = _to_float(_get_first(item, "stellar_teff_err_k", "teff_err", "teff_unc", "st_tefferr1"))
        stellar_logg, logg_source = _get_float_with_source(item, "ExoClock", "stellar_logg", "logg", "st_logg")
        stellar_logg_err = _to_float(_get_first(item, "stellar_logg_err", "logg_err", "logg_unc", "st_loggerr1"))
        stellar_feh, feh_source = _get_float_with_source(item, "ExoClock", "stellar_feh", "feh", "metallicity", "st_met")
        stellar_feh_err = _to_float(_get_first(item, "stellar_feh_err", "feh_err", "feh_unc", "st_meterr1"))

        geometry_source = "ExoClock JSON" if np.isfinite(a_rs) and np.isfinite(inclination_deg) else "missing"
        stellar_source = _source_if_any("ExoClock JSON", [stellar_teff_k, stellar_logg, stellar_feh])

        # For ExoClock-pure empirical models, keep the model depth tied to the
        # ExoClock flux depth.  Do not silently replace it with Rp/Rs^2 when a
        # flux depth exists; that was the source of confusing expected models.
        if np.isfinite(depth_ppt) and depth_ppt > 0:
            model_depth_ppt = depth_ppt
        elif np.isfinite(rp_rs):
            model_depth_ppt = rp_rs * rp_rs * 1000.0
        else:
            model_depth_ppt = float("nan")

        row = {
            "planet_name": planet_name,
            "aliases": _normalise_alias(planet_name),
            "period_days": period_days,
            "period_err_days": _to_float(_get_first(item, "period_unc", "period_err", "period_error", "pl_orbpererr1")),
            "t0_bjd_tdb": t0_bjd_tdb,
            "t0_err_days": _to_float(_get_first(item, "t0_unc", "t0_err", "mid_time_unc", "epoch_unc", "pl_tranmiderr1")),
            "duration_hours": duration_hours,
            "duration_err_hours": _to_float(_get_first(item, "duration_unc", "duration_err", "duration_error", "pl_trandurerr1")),
            "depth_ppt": depth_ppt,
            "depth_err_ppt": _relative_or_ppt_depth(_get_first(item, "depth_err_ppt", "depth_ppt_err", "depth_unc", "depth_error")),
            "rp_rs": rp_rs,
            "rp_rs_err": _to_float(_get_first(item, "rp_rs_unc", "rp_rs_err", "rp_over_rs_unc", "rp_over_rs_err", "pl_ratrorerr1")),
            "model_depth_ppt": model_depth_ppt,
            "a_rs": a_rs,
            "a_rs_err": a_rs_err,
            "inclination_deg": inclination_deg,
            "inclination_err_deg": inclination_err_deg,
            "ecc": ecc,
            "ecc_err": ecc_err,
            "omega_deg": omega_deg,
            "omega_err_deg": omega_err_deg,
            "stellar_teff_k": stellar_teff_k,
            "stellar_teff_err_k": stellar_teff_err_k,
            "stellar_logg": stellar_logg,
            "stellar_logg_err": stellar_logg_err,
            "stellar_feh": stellar_feh,
            "stellar_feh_err": stellar_feh_err,
            "ra_deg": _parse_ra_deg(_get_first(item, "ra_j2000", "ra", "ra_deg", "target_ra")),
            "dec_deg": _parse_dec_deg(_get_first(item, "dec_j2000", "dec", "dec_deg", "target_dec")),
            "time_reference": "BJD_TDB",
            "source": "ExoClock planets_json pure",
            "notes": (
                f"Generated {generated} with tools/build_exoplanet_catalog_from_exoclock.py; "
                "pure ExoClock row; no NASA supplement applied by default; "
                "all depths are stored in relative-flux ppt for ExoPhotoCurve. "
                f"Depth source: {depth_source}. Rp/Rs source: {rprs_source}. "
                f"Geometry source: {geometry_source}."
            ),
            "catalogue_type": "ExoClock pure",
            "depth_source": depth_source,
            "rprs_source": rprs_source,
            "geometry_source": geometry_source,
            "stellar_source": stellar_source,
            "exoclock_priority": str(_get_first(item, "priority") or ""),
            "exoclock_total_observations": _to_float(_get_first(item, "total_observations")),
            "exoclock_recent_observations": _to_float(_get_first(item, "recent_observations")),
            "exoclock_current_oc_min": _to_float(_get_first(item, "current_oc_min")),
            "exoclock_min_telescope_inches": _to_float(_get_first(item, "min_telescope_inches")),
            "v_mag": _to_float(_get_first(item, "v_mag")),
            "r_mag": _to_float(_get_first(item, "r_mag")),
            "gaia_g_mag": _to_float(_get_first(item, "gaia_g_mag")),
        }
        debug_rows.append(
            {
                "planet_name": planet_name,
                "period_days": period_days,
                "t0_bjd_tdb": t0_bjd_tdb,
                "duration_hours": duration_hours,
                "depth_ppt": depth_ppt,
                "depth_source": depth_source,
                "available_keys": ", ".join(sorted(map(str, item.keys()))[:35]),
            }
        )
        rows.append(row)

    if not rows:
        raise RuntimeError("The ExoClock JSON payload did not contain any usable planet rows.")

    out = pd.DataFrame(rows)
    output_columns = list(dict.fromkeys(CATALOGUE_COLUMNS + EXTRA_COLUMNS))
    for col in output_columns:
        if col not in out.columns:
            out[col] = np.nan if col not in {"planet_name", "aliases", "time_reference", "source", "notes", "exoclock_priority", *PROVENANCE_COLUMNS} else ""

    before = len(out)
    numeric_required = ["period_days", "t0_bjd_tdb", "duration_hours"]
    for col in numeric_required + ["depth_ppt"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep rows with valid ephemeris and duration.  Depth/RpRs can be missing
    # only for unusual rows; ExoPhotoCurve needs depth for empirical modelling,
    # so rows without any positive depth are rejected below.
    out = out.dropna(subset=numeric_required)
    out = out[(out["period_days"] > 0) & (out["duration_hours"] > 0)]
    out = out[pd.to_numeric(out["depth_ppt"], errors="coerce") > 0]
    after_clean = len(out)

    out = out.sort_values("planet_name", kind="mergesort").reset_index(drop=True)

    if out.empty:
        print("\nDiagnostic sample of parsed ExoClock rows before cleaning:")
        for row in debug_rows[:10]:
            print(row)
        raise RuntimeError(
            "All ExoClock rows were rejected after cleaning. "
            "This usually means that the JSON schema has changed or the endpoint returned a wrapper page. "
            "The diagnostic sample above shows the keys that were found."
        )

    print(f"Accepted {len(out)} rows after cleaning ({before - after_clean} rejected).")
    n_missing_geometry = int(np.sum(~(np.isfinite(pd.to_numeric(out["a_rs"], errors="coerce")) & np.isfinite(pd.to_numeric(out["inclination_deg"], errors="coerce")))))
    if n_missing_geometry:
        print(f"Info: {n_missing_geometry} rows have no complete ExoClock geometry; Auto model will use the empirical template for them.")
    return out[output_columns].copy()


def supplement_with_catalogue(exoclock_df: pd.DataFrame, supplement_path: Path, *, fill_rprs: bool = False) -> tuple[pd.DataFrame, int]:
    """Fill missing physical parameters from a NASA-style catalogue explicitly.

    This creates a transparent ExoClock+NASA hybrid catalogue.  ExoClock timing
    and flux depths remain primary; the supplement is used only when the row is
    missing geometry/stellar parameters.
    """
    if not supplement_path.exists():
        return exoclock_df, 0

    supplement = pd.read_csv(supplement_path)
    if "planet_name" not in supplement.columns:
        return exoclock_df, 0

    supplement = supplement.copy()
    supplement["_match_key"] = supplement["planet_name"].map(_normalise_key)
    exoclock_df = exoclock_df.copy()
    exoclock_df["_match_key"] = exoclock_df["planet_name"].map(_normalise_key)

    supplement_index = supplement.drop_duplicates("_match_key").set_index("_match_key")
    matched_rows = 0

    columns_to_fill = list(SUPPLEMENT_COLUMNS)
    if fill_rprs:
        columns_to_fill.extend(["rp_rs", "rp_rs_err"])

    geometry_cols = {"a_rs", "a_rs_err", "inclination_deg", "inclination_err_deg", "ecc", "ecc_err", "omega_deg", "omega_err_deg"}
    stellar_cols = {"stellar_teff_k", "stellar_teff_err_k", "stellar_logg", "stellar_logg_err", "stellar_feh", "stellar_feh_err"}

    for idx, row in exoclock_df.iterrows():
        key = row["_match_key"]
        if key not in supplement_index.index:
            continue
        src = supplement_index.loc[key]
        row_had_update = False
        geometry_updated = False
        stellar_updated = False
        rprs_updated = False

        for col in columns_to_fill:
            if col not in exoclock_df.columns or col not in supplement_index.columns:
                continue
            current = _to_float(exoclock_df.at[idx, col])
            candidate = _to_float(src.get(col, np.nan))
            if (not np.isfinite(current) or (col in {"ecc", "omega_deg"} and not np.isfinite(current))) and np.isfinite(candidate):
                exoclock_df.at[idx, col] = candidate
                row_had_update = True
                if col in geometry_cols:
                    geometry_updated = True
                if col in stellar_cols:
                    stellar_updated = True
                if col in {"rp_rs", "rp_rs_err"}:
                    rprs_updated = True

        if rprs_updated:
            rp_now = _to_float(exoclock_df.at[idx, "rp_rs"]) if "rp_rs" in exoclock_df.columns else float("nan")
            depth_now = _to_float(exoclock_df.at[idx, "depth_ppt"]) if "depth_ppt" in exoclock_df.columns else float("nan")
            if np.isfinite(rp_now) and (not np.isfinite(depth_now) or depth_now <= 0):
                exoclock_df.at[idx, "depth_ppt"] = rp_now * rp_now * 1000.0
                exoclock_df.at[idx, "model_depth_ppt"] = rp_now * rp_now * 1000.0
                exoclock_df.at[idx, "depth_source"] = f"derived from Rp/Rs supplied by {supplement_path.name}"

        if row_had_update:
            matched_rows += 1
            exoclock_df.at[idx, "catalogue_type"] = "ExoClock+supplement hybrid"
            if geometry_updated:
                exoclock_df.at[idx, "geometry_source"] = f"supplement:{supplement_path.name}"
            if stellar_updated:
                exoclock_df.at[idx, "stellar_source"] = f"supplement:{supplement_path.name}"
            if rprs_updated:
                exoclock_df.at[idx, "rprs_source"] = f"supplement:{supplement_path.name}"
            notes = str(exoclock_df.at[idx, "notes"])
            exoclock_df.at[idx, "notes"] = (
                notes
                + f" Hybrid warning: missing physical parameters were supplemented from {supplement_path.name}; "
                "the physical expected model may not match the ExoClock online model."
            )

    exoclock_df = exoclock_df.drop(columns=["_match_key"])
    return exoclock_df, matched_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build an ExoPhotoCurve catalogue from ExoClock.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_exoclock_catalogue_path(),
        help="Output CSV path. Default: exophotocurve/catalogs/exoclock_transit_catalog.csv",
    )
    parser.add_argument(
        "--url",
        default=EXOCLOCK_JSON_URL,
        help="ExoClock JSON endpoint.",
    )
    parser.add_argument(
        "--supplement-catalogue",
        type=Path,
        default=None,
        help=(
            "Optional NASA-style catalogue used to fill missing a/Rs, inclination and stellar parameters. "
            "If omitted, the output is ExoClock-pure."
        ),
    )
    parser.add_argument(
        "--no-supplement",
        action="store_true",
        help="Compatibility option. Supplementation is already disabled by default unless --supplement-catalogue is given.",
    )
    parser.add_argument(
        "--fill-rprs-from-supplement",
        action="store_true",
        help="Also fill Rp/Rs from the supplement catalogue when ExoClock-derived Rp/Rs is missing.",
    )
    return parser.parse_args()


def main() -> None:
    """Download the ExoClock catalogue and write a PhotoCurve-compatible CSV."""
    args = parse_args()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    payload = download_exoclock_json(args.url)
    out = build_exoclock_dataframe(payload, generated)

    supplemented_rows = 0
    supplement_label = "disabled; ExoClock-pure catalogue"
    if args.supplement_catalogue is not None and not args.no_supplement:
        supplement_path = Path(args.supplement_catalogue)
        supplement_label = str(supplement_path)
        if supplement_path.exists():
            out, supplemented_rows = supplement_with_catalogue(
                out,
                supplement_path,
                fill_rprs=args.fill_rprs_from_supplement,
            )
            print(f"Supplemented {supplemented_rows} rows with geometry/stellar parameters from {supplement_path}")
        else:
            print(f"Supplement catalogue not found, continuing with ExoClock-pure rows: {supplement_path}")
            supplement_label = f"requested but not found: {supplement_path}"
    elif args.no_supplement:
        supplement_label = "disabled by --no-supplement"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    n_batman_ready = int(np.sum(np.isfinite(pd.to_numeric(out["a_rs"], errors="coerce")) & np.isfinite(pd.to_numeric(out["inclination_deg"], errors="coerce"))))
    n_hybrid = int(np.sum(out.get("catalogue_type", pd.Series("", index=out.index)).astype(str).str.contains("hybrid", case=False, na=False)))
    metadata_path = args.output.with_suffix(".metadata.txt")
    metadata_path.write_text(
        "ExoPhotoCurve ExoClock transit catalogue\n"
        "Source: ExoClock planets_json endpoint\n"
        f"Endpoint: {args.url}\n"
        f"Generated: {generated}\n"
        f"Rows: {len(out)}\n"
        f"Catalogue type: {'ExoClock+supplement hybrid' if n_hybrid else 'ExoClock pure'}\n"
        f"Rows with a/Rs and inclination for batman: {n_batman_ready}\n"
        f"Supplement catalogue: {supplement_label}\n"
        f"Supplemented rows: {supplemented_rows}\n"
        "Depth convention: ExoClock millimagnitude depths, when provided, are converted to relative flux ppt using "
        "depth = (1 - 10^(-depth_mmag/2500)) * 1000. Direct ExoClock flux depths are stored in ppt.\n"
        "Default behavior: no NASA-style physical geometry is mixed into the ExoClock catalogue. "
        "Rows without complete ExoClock geometry should be modelled empirically in ExoPhotoCurve Auto mode.\n"
        "Hybrid behavior: if --supplement-catalogue is explicitly provided, missing geometry/stellar parameters are filled "
        "from that supplement and affected rows are marked as ExoClock+supplement hybrid.\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"Rows with a/Rs and inclination for batman: {n_batman_ready}")
    print(f"Hybrid rows: {n_hybrid}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nExoClock catalogue generation failed: {exc}")
        sys.exit(1)
