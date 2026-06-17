#!/usr/bin/env python3
"""Build a PhotoCurve Lab transit catalogue from the ExoClock planets database.

The ExoClock endpoint provides updated transit ephemerides and useful
observability information.  The JSON endpoint does not currently provide all
physical parameters needed by the batman transit engine, especially a/Rs and
inclination.  For this reason the builder can optionally supplement the
ExoClock ephemerides with physical parameters from an existing NASA-style
PhotoCurve Lab catalogue.

Typical usage from the project root:

    python tools/build_exoplanet_catalog_from_exoclock.py

This writes:

    photocurve_lab/catalogs/exoclock_transit_catalog.csv

To use a full NASA catalogue as a source of physical parameters:

    python tools/build_exoplanet_catalog_from_nasa.py
    python tools/build_exoplanet_catalog_from_exoclock.py \
        --supplement-catalogue photocurve_lab/catalogs/exoplanet_transit_catalog.csv

The ExoClock values remain primary for T0, period, duration, depth and target
coordinates. The supplement catalogue is used only to fill missing geometry and
stellar parameters required for physical modelling.
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
    from photocurve_lab.exoplanet_catalog import CATALOGUE_COLUMNS, default_catalogue_path, default_exoclock_catalogue_path
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
        return PROJECT_ROOT / "photocurve_lab" / "catalogs" / "exoplanet_transit_catalog.csv"

    def default_exoclock_catalogue_path() -> Path:
        return PROJECT_ROOT / "photocurve_lab" / "catalogs" / "exoclock_transit_catalog.csv"


EXOCLOCK_JSON_URL = "https://www.exoclock.space/database/planets_json"

EXTRA_COLUMNS = [
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
        # Fall back to the first scalar-looking value.
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
    for key in keys:
        if key in item:
            value = item.get(key)
            if value not in (None, "", "--"):
                return value
    return None


def _to_float(value: Any) -> float:
    """Convert a value to float, returning NaN for empty or invalid entries.

    The ExoClock JSON normally stores plain numeric values, but this parser is
    intentionally permissive: it also accepts strings containing units or
    uncertainties, e.g. '1.234 +/- 0.005', and simple containers such as
    [value, uncertainty] or {'value': value}.
    """
    value = _first_from_container(value)
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "--"}:
            return float("nan")
        # Accept decimal commas only when there is no decimal point.
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
    """Convert transit depth from millimagnitudes to ppt.

    ExoClock gives the depth in millimagnitudes. For a dimming of dm mag,
    F_transit / F_out = 10^(-dm/2.5), therefore the relative depth is
    1 - 10^(-dm/2.5).  The result is returned in ppt.
    """
    mmag = _to_float(depth_mmag)
    if not np.isfinite(mmag) or mmag <= 0:
        return float("nan")
    dm_mag = mmag / 1000.0
    return (1.0 - 10.0 ** (-dm_mag / 2.5)) * 1000.0


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
    """Parse right ascension as decimal degrees.

    Decimal values are assumed to already be degrees. Sexagesimal values such
    as '17:55:33.8' are interpreted as hours, minutes and seconds.
    """
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
    """Yield planet dictionaries from the ExoClock JSON payload.

    The documented endpoint returns a Python dictionary, but some deployments
    wrap the actual planet table inside keys such as 'planets' or 'data'.  We
    therefore look for known containers before treating the top-level mapping
    as a planet-name -> row dictionary. This avoids accidentally interpreting
    metadata dictionaries as planets.
    """
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield dict(item)
        return

    if not isinstance(payload, dict):
        return

    # Known wrapper forms. Check these before generic dictionary iteration.
    for container_key in ("planets", "data", "results", "objects", "targets", "planet_data"):
        if container_key in payload:
            container = payload[container_key]
            yielded = False
            for row in _iter_exoclock_planets(container):
                yielded = True
                yield row
            if yielded:
                return

    # Documented/common form: {"WASP-12 b": { ... planet fields ... }, ...}
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

    # Last-resort recursive search for a nested planet table.
    for value in payload.values():
        if isinstance(value, (dict, list)):
            yield from _iter_exoclock_planets(value)


def download_exoclock_json(url: str = EXOCLOCK_JSON_URL) -> Any:
    """Download and decode the ExoClock planets JSON database."""
    print("Downloading ExoClock planets database:")
    print(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PhotoCurveLab-ExoClock-catalogue-builder/0.4.10"},
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


def build_exoclock_dataframe(payload: Any, generated: str) -> pd.DataFrame:
    """Convert the ExoClock JSON payload to PhotoCurve Lab catalogue rows."""
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

        # ExoClock documentation gives depth_mmag.  Keep a few aliases for
        # robustness in case the endpoint labels the column slightly differently.
        depth_value = _get_first(item, "depth_mmag", "transit_depth_mmag", "depth", "depth_mmags")
        depth_ppt = _depth_mmag_to_ppt(depth_value)

        # If a future endpoint gives a direct relative-flux depth, use it only as
        # a fallback.  Values below ~0.2 are interpreted as fractional depth;
        # larger values are interpreted as ppt.
        if not np.isfinite(depth_ppt):
            direct_depth = _to_float(_get_first(item, "depth_ppt", "depth_ppthousand", "depth_relative", "transit_depth"))
            if np.isfinite(direct_depth) and direct_depth > 0:
                depth_ppt = direct_depth * 1000.0 if direct_depth < 0.2 else direct_depth

        rp_rs_direct = _to_float(_get_first(item, "rp_rs", "rp_over_rs", "rprs", "pl_ratror"))
        if np.isfinite(rp_rs_direct) and rp_rs_direct > 0:
            rp_rs = rp_rs_direct
            if not np.isfinite(depth_ppt):
                depth_ppt = rp_rs * rp_rs * 1000.0
        else:
            rp_rs = math.sqrt(depth_ppt / 1000.0) if np.isfinite(depth_ppt) and depth_ppt > 0 else float("nan")

        row = {
            "planet_name": planet_name,
            "aliases": _normalise_alias(planet_name),
            "period_days": period_days,
            "period_err_days": _to_float(_get_first(item, "period_unc", "period_err", "period_error", "pl_orbpererr1")),
            "t0_bjd_tdb": t0_bjd_tdb,
            "t0_err_days": _to_float(_get_first(item, "t0_unc", "t0_err", "mid_time_unc", "epoch_unc", "pl_tranmiderr1")),
            "duration_hours": duration_hours,
            "duration_err_hours": _to_float(_get_first(item, "duration_unc", "duration_err", "duration_error")),
            "depth_ppt": depth_ppt,
            "depth_err_ppt": float("nan"),
            "rp_rs": rp_rs,
            "rp_rs_err": float("nan"),
            "model_depth_ppt": (rp_rs * rp_rs * 1000.0) if np.isfinite(rp_rs) else depth_ppt,
            "a_rs": float("nan"),
            "a_rs_err": float("nan"),
            "inclination_deg": float("nan"),
            "inclination_err_deg": float("nan"),
            "ecc": 0.0,
            "ecc_err": float("nan"),
            "omega_deg": 90.0,
            "omega_err_deg": float("nan"),
            "stellar_teff_k": float("nan"),
            "stellar_teff_err_k": float("nan"),
            "stellar_logg": float("nan"),
            "stellar_logg_err": float("nan"),
            "stellar_feh": float("nan"),
            "stellar_feh_err": float("nan"),
            "ra_deg": _parse_ra_deg(_get_first(item, "ra_j2000", "ra", "ra_deg", "target_ra")),
            "dec_deg": _parse_dec_deg(_get_first(item, "dec_j2000", "dec", "dec_deg", "target_dec")),
            "time_reference": "BJD_TDB",
            "source": "ExoClock planets_json",
            "notes": (
                f"Generated {generated} with tools/build_exoplanet_catalog_from_exoclock.py; "
                "depth_mmag converted to depth_ppt using the exact magnitude-to-flux relation; "
                "Rp/Rs estimated as sqrt(depth) unless provided directly or later curated."
            ),
            "exoclock_priority": str(_get_first(item, "priority") or ""),
            "exoclock_total_observations": _to_float(_get_first(item, "total_observations")),
            "exoclock_recent_observations": _to_float(_get_first(item, "recent_observations")),
            "exoclock_current_oc_min": _to_float(_get_first(item, "current_oc_min")),
            "exoclock_min_telescope_inches": _to_float(_get_first(item, "min_telescope_inches")),
            "v_mag": _to_float(_get_first(item, "v_mag")),
            "r_mag": _to_float(_get_first(item, "r_mag")),
            "gaia_g_mag": _to_float(_get_first(item, "gaia_g_mag")),
        }
        debug_rows.append({
            "planet_name": planet_name,
            "period_days": period_days,
            "t0_bjd_tdb": t0_bjd_tdb,
            "duration_hours": duration_hours,
            "depth_ppt": depth_ppt,
            "available_keys": ", ".join(sorted(map(str, item.keys()))[:25]),
        })
        rows.append(row)

    if not rows:
        raise RuntimeError("The ExoClock JSON payload did not contain any usable planet rows.")

    out = pd.DataFrame(rows)
    for col in CATALOGUE_COLUMNS + EXTRA_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan if col not in {"planet_name", "aliases", "time_reference", "source", "notes", "exoclock_priority"} else ""

    before = len(out)
    numeric_required = ["period_days", "t0_bjd_tdb", "duration_hours"]
    for col in numeric_required + ["depth_ppt"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep rows with valid ephemeris and duration.  Depth/RpRs can be supplied
    # later by the NASA supplement catalogue, so do not reject rows only because
    # the ExoClock depth field was missing or formatted differently.
    out = out.dropna(subset=numeric_required)
    out = out[(out["period_days"] > 0) & (out["duration_hours"] > 0)]
    after_ephem = len(out)

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

    missing_depth = int(np.sum(~np.isfinite(pd.to_numeric(out["depth_ppt"], errors="coerce"))))
    print(f"Accepted {len(out)} rows after ephemeris cleaning ({before - after_ephem} rejected).")
    if missing_depth:
        print(f"Warning: {missing_depth} accepted rows have no valid ExoClock depth/RpRs yet; a supplement catalogue may fill them.")
    return out[CATALOGUE_COLUMNS + EXTRA_COLUMNS].copy()


def supplement_with_catalogue(exoclock_df: pd.DataFrame, supplement_path: Path, *, fill_rprs: bool = False) -> tuple[pd.DataFrame, int]:
    """Fill missing physical parameters from a NASA-style catalogue."""
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
        columns_to_fill.extend(["rp_rs", "rp_rs_err", "model_depth_ppt"])

    for idx, row in exoclock_df.iterrows():
        key = row["_match_key"]
        if key not in supplement_index.index:
            continue
        src = supplement_index.loc[key]
        row_had_update = False
        for col in columns_to_fill:
            if col not in exoclock_df.columns or col not in supplement_index.columns:
                continue
            current = _to_float(exoclock_df.at[idx, col])
            candidate = _to_float(src.get(col, np.nan))
            if (not np.isfinite(current) or (col in {"ecc", "omega_deg"} and not np.isfinite(current))) and np.isfinite(candidate):
                exoclock_df.at[idx, col] = candidate
                row_had_update = True
        # If Rp/Rs has become available, keep model depth consistent when needed.
        rp_now = _to_float(exoclock_df.at[idx, "rp_rs"]) if "rp_rs" in exoclock_df.columns else float("nan")
        depth_now = _to_float(exoclock_df.at[idx, "depth_ppt"]) if "depth_ppt" in exoclock_df.columns else float("nan")
        if np.isfinite(rp_now) and (not np.isfinite(depth_now) or depth_now <= 0):
            exoclock_df.at[idx, "depth_ppt"] = rp_now * rp_now * 1000.0
            exoclock_df.at[idx, "model_depth_ppt"] = rp_now * rp_now * 1000.0
            row_had_update = True

        if row_had_update:
            matched_rows += 1
            notes = str(exoclock_df.at[idx, "notes"])
            exoclock_df.at[idx, "notes"] = notes + f" Supplemented with geometry/stellar parameters from {supplement_path.name}."

    exoclock_df = exoclock_df.drop(columns=["_match_key"])
    return exoclock_df, matched_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a PhotoCurve Lab catalogue from ExoClock.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_exoclock_catalogue_path(),
        help="Output CSV path. Default: photocurve_lab/catalogs/exoclock_transit_catalog.csv",
    )
    parser.add_argument(
        "--url",
        default=EXOCLOCK_JSON_URL,
        help="ExoClock JSON endpoint.",
    )
    parser.add_argument(
        "--supplement-catalogue",
        type=Path,
        default=default_catalogue_path(),
        help="Optional NASA-style catalogue used to fill a/Rs, inclination and stellar parameters.",
    )
    parser.add_argument(
        "--no-supplement",
        action="store_true",
        help="Do not supplement ExoClock rows with physical parameters from another catalogue.",
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
    supplement_path = Path(args.supplement_catalogue)
    if not args.no_supplement:
        if supplement_path.exists():
            out, supplemented_rows = supplement_with_catalogue(
                out,
                supplement_path,
                fill_rprs=args.fill_rprs_from_supplement,
            )
            print(f"Supplemented {supplemented_rows} rows with geometry/stellar parameters from {supplement_path}")
        else:
            print(f"Supplement catalogue not found, continuing without geometry supplement: {supplement_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    n_batman_ready = int(np.sum(np.isfinite(pd.to_numeric(out["a_rs"], errors="coerce")) & np.isfinite(pd.to_numeric(out["inclination_deg"], errors="coerce"))))
    metadata_path = args.output.with_suffix(".metadata.txt")
    metadata_path.write_text(
        "PhotoCurve Lab ExoClock transit catalogue\n"
        "Source: ExoClock planets_json endpoint\n"
        f"Endpoint: {args.url}\n"
        f"Generated: {generated}\n"
        f"Rows: {len(out)}\n"
        f"Rows with a/Rs and inclination for batman: {n_batman_ready}\n"
        f"Supplement catalogue: {supplement_path if not args.no_supplement else 'disabled'}\n"
        f"Supplemented rows: {supplemented_rows}\n"
        "Depth convention: depth_mmag was converted to relative flux depth in ppt using "
        "depth = 1 - 10^(-depth_mmag/2500).\n"
        "Important: ExoClock currently supplies ephemerides, duration, depth and coordinates, "
        "but not all geometric parameters used by batman. Use a NASA-style supplement catalogue "
        "or manually curated rows to obtain physical-model fits for more targets.\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"Rows with a/Rs and inclination for batman: {n_batman_ready}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nExoClock catalogue generation failed: {exc}")
        sys.exit(1)
