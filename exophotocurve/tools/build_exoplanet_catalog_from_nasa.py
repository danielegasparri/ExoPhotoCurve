#!/usr/bin/env python3
"""Build PhotoCurve Lab's offline transit catalogue from NASA Exoplanet Archive.

This script is intended for the maintainer/developer when preparing a release.
Normal users should receive PhotoCurve Lab with the generated CSV already bundled.

The builder first asks the TAP service which columns exist in the PSCompPars
schema and then builds a compatible query dynamically. This avoids fragile
failures when optional columns differ between archive releases.
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TABLE_NAME = "pscomppars"

CORE_COLUMNS = [
    "pl_name",
    "hostname",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandur",
    "ra",
    "dec",
    "disc_facility",
    "discoverymethod",
]

OPTIONAL_COLUMNS = [
    "pl_tranmid_systemref",
    "pl_trandep",
    "pl_trandeperr1",
    "pl_trandeperr2",
    "pl_ratror",
    "pl_ratrorerr1",
    "pl_ratrorerr2",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_tranmiderr1",
    "pl_tranmiderr2",
    "pl_trandurerr1",
    "pl_trandurerr2",
    "pl_ratdor",
    "pl_ratdorerr1",
    "pl_ratdorerr2",
    "pl_orbincl",
    "pl_orbinclerr1",
    "pl_orbinclerr2",
    "pl_orbeccen",
    "pl_orbeccenerr1",
    "pl_orbeccenerr2",
    "pl_orblper",
    "pl_orblpererr1",
    "pl_orblpererr2",
    "st_teff",
    "st_tefferr1",
    "st_tefferr2",
    "st_logg",
    "st_loggerr1",
    "st_loggerr2",
    "st_met",
    "st_meterr1",
    "st_meterr2",
]


def normalise_query(query: str) -> str:
    """Collapse whitespace in an ADQL query."""
    return " ".join(query.split())


def build_tap_url(query: str) -> str:
    """Build a NASA Exoplanet Archive TAP synchronous-query URL."""
    params = urllib.parse.urlencode({"query": normalise_query(query), "format": "csv"})
    return f"{TAP_SYNC_URL}?{params}"


def download_tap_csv(query: str) -> pd.DataFrame:
    """Download a CSV result from the TAP service, showing useful errors."""
    url = build_tap_url(query)
    print("Downloading from NASA Exoplanet Archive TAP service:")
    print(url)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PhotoCurveLab-catalogue-builder/0.3.7"},
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace")
        print("\nNASA TAP service returned an HTTP error.")
        print(f"HTTP status: {exc.code} {exc.reason}")
        print("\nServer response:")
        print(error_payload[:4000])
        print("\nThe most common cause is an invalid ADQL query or a column that is not present in the selected table.")
        raise
    except urllib.error.URLError as exc:
        print("\nCould not connect to the NASA TAP service.")
        print(exc)
        raise

    return pd.read_csv(io.StringIO(payload))


def available_pscomppars_columns() -> set[str]:
    """Return the column names currently available in PSCompPars."""
    schema_queries = [
        "select column_name from TAP_SCHEMA.columns where table_name = 'pscomppars'",
        "select column_name from tap_schema.columns where table_name = 'pscomppars'",
        "select column_name from TAP_SCHEMA.columns where table_name = 'PSCompPars'",
    ]

    last_error: Exception | None = None
    for query in schema_queries:
        try:
            df = download_tap_csv(query)
            if "column_name" in df.columns:
                return {str(name).lower() for name in df["column_name"].dropna()}
            if "COLUMN_NAME" in df.columns:
                return {str(name).lower() for name in df["COLUMN_NAME"].dropna()}
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

    raise RuntimeError(f"Could not read TAP schema for {TABLE_NAME}: {last_error}")


def build_catalogue_query(available_columns: set[str]) -> str:
    """Build a PSCompPars query using only columns that are available."""
    missing_core = [col for col in CORE_COLUMNS if col.lower() not in available_columns]
    if missing_core:
        raise RuntimeError("The TAP table is missing required columns: " + ", ".join(missing_core))

    selected = list(CORE_COLUMNS)
    for col in OPTIONAL_COLUMNS:
        if col.lower() in available_columns:
            selected.append(col)

    for needed in ["tran_flag", "pl_trandep", "pl_ratror"]:
        if needed not in available_columns:
            raise RuntimeError(f"The TAP table is missing required filtering/model column: {needed}")

    return f"""
    select
      {', '.join(selected)}
    from {TABLE_NAME}
    where tran_flag = 1
      and pl_orbper is not null
      and pl_tranmid is not null
      and pl_trandur is not null
      and (pl_trandep is not null or pl_ratror is not null)
    order by pl_name
    """


def _normalise_alias(name: str) -> str:
    """Return a compact alias useful for filename matching."""
    return str(name).replace(" ", "").replace("-", "").replace("_", "")


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series, or a NaN Series if the column is absent."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _symmetric_error(df: pd.DataFrame, plus_col: str, minus_col: str) -> pd.Series:
    """Return a symmetric uncertainty from plus/minus archive columns."""
    plus = _numeric_column(df, plus_col).abs()
    minus = _numeric_column(df, minus_col).abs()
    both = pd.concat([plus, minus], axis=1)
    return both.mean(axis=1, skipna=True)


def _first_string_column(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    """Return the first available string column from a list of candidate names."""
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=str)


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[1]
        / "photocurve_lab"
        / "catalogs"
        / "exoplanet_transit_catalog.csv"
    )

    available = available_pscomppars_columns()
    query = build_catalogue_query(available)
    raw = download_tap_csv(query)

    if raw.empty:
        raise RuntimeError("The TAP query returned no rows. The catalogue was not created.")

    # NASA's pl_trandep is expressed as a percentage. PhotoCurve Lab stores
    # depths in ppt, where 1% = 10 ppt.
    trandep_percent = _numeric_column(raw, "pl_trandep")
    depth_from_percent = trandep_percent * 10.0
    depth_err_from_percent = _symmetric_error(raw, "pl_trandeperr1", "pl_trandeperr2") * 10.0

    rp_rs = _numeric_column(raw, "pl_ratror")
    rp_rs_err = _symmetric_error(raw, "pl_ratrorerr1", "pl_ratrorerr2")
    depth_from_rprs = (rp_rs**2) * 1000.0
    depth_err_from_rprs = 2.0 * 1000.0 * rp_rs * rp_rs_err

    # Keep the published transit-depth column, when available, but use Rp/Rs^2
    # as the model depth when Rp/Rs exists. This keeps the displayed expected
    # Rp/Rs and the empirical-model depth mutually consistent.
    depth_ppt = depth_from_percent.fillna(depth_from_rprs)
    depth_err_ppt = depth_err_from_percent.fillna(depth_err_from_rprs)
    rp_rs = rp_rs.fillna(np.sqrt(depth_ppt / 1000.0))
    model_depth_ppt = depth_from_rprs.fillna(depth_ppt)

    planet_name = raw["pl_name"].astype(str)
    hostname = raw.get("hostname", pd.Series("", index=raw.index)).astype(str)
    aliases = planet_name.map(_normalise_alias) + ";" + hostname.map(_normalise_alias)

    time_reference = _first_string_column(raw, ["pl_tranmid_systemref"])

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    out = pd.DataFrame(
        {
            "planet_name": planet_name,
            "aliases": aliases,
            "period_days": _numeric_column(raw, "pl_orbper"),
            "period_err_days": _symmetric_error(raw, "pl_orbpererr1", "pl_orbpererr2"),
            "t0_bjd_tdb": _numeric_column(raw, "pl_tranmid"),
            "t0_err_days": _symmetric_error(raw, "pl_tranmiderr1", "pl_tranmiderr2"),
            "duration_hours": _numeric_column(raw, "pl_trandur"),
            "duration_err_hours": _symmetric_error(raw, "pl_trandurerr1", "pl_trandurerr2"),
            "depth_ppt": depth_ppt,
            "depth_err_ppt": depth_err_ppt,
            "rp_rs": rp_rs,
            "rp_rs_err": rp_rs_err,
            "model_depth_ppt": model_depth_ppt,
            "a_rs": _numeric_column(raw, "pl_ratdor"),
            "a_rs_err": _symmetric_error(raw, "pl_ratdorerr1", "pl_ratdorerr2"),
            "inclination_deg": _numeric_column(raw, "pl_orbincl"),
            "inclination_err_deg": _symmetric_error(raw, "pl_orbinclerr1", "pl_orbinclerr2"),
            "ecc": _numeric_column(raw, "pl_orbeccen"),
            "ecc_err": _symmetric_error(raw, "pl_orbeccenerr1", "pl_orbeccenerr2"),
            "omega_deg": _numeric_column(raw, "pl_orblper"),
            "omega_err_deg": _symmetric_error(raw, "pl_orblpererr1", "pl_orblpererr2"),
            "stellar_teff_k": _numeric_column(raw, "st_teff"),
            "stellar_teff_err_k": _symmetric_error(raw, "st_tefferr1", "st_tefferr2"),
            "stellar_logg": _numeric_column(raw, "st_logg"),
            "stellar_logg_err": _symmetric_error(raw, "st_loggerr1", "st_loggerr2"),
            "stellar_feh": _numeric_column(raw, "st_met"),
            "stellar_feh_err": _symmetric_error(raw, "st_meterr1", "st_meterr2"),
            "ra_deg": _numeric_column(raw, "ra"),
            "dec_deg": _numeric_column(raw, "dec"),
            "time_reference": time_reference,
            "source": "NASA Exoplanet Archive PSCompPars TAP export",
            "notes": (
                f"Generated {generated} with tools/build_exoplanet_catalog_from_nasa.py; "
                "pl_trandep converted from percent to ppt; model_depth_ppt uses Rp/Rs^2 when Rp/Rs is available."
            ),
        }
    )

    out = out.dropna(subset=["period_days", "t0_bjd_tdb", "duration_hours", "depth_ppt"])
    out = out[(out["period_days"] > 0) & (out["duration_hours"] > 0) & (out["depth_ppt"] > 0)]
    out = out.sort_values("planet_name", kind="mergesort").reset_index(drop=True)

    if out.empty:
        raise RuntimeError("All rows were rejected after cleaning. The catalogue was not created.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    metadata_path = out_path.with_suffix(".metadata.txt")
    metadata_path.write_text(
        "PhotoCurve Lab exoplanet transit catalogue\n"
        "Source: NASA Exoplanet Archive PSCompPars TAP service\n"
        f"Generated: {generated}\n"
        f"Rows: {len(out)}\n"
        f"TAP URL: {build_tap_url(query)}\n"
        "Depth convention: depth_ppt stores published pl_trandep where available; "
        "model_depth_ppt uses Rp/Rs^2 when Rp/Rs is available for internal consistency.\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(out)} rows to {out_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCatalogue generation failed: {exc}")
        sys.exit(1)
