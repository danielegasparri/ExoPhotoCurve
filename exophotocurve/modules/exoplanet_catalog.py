"""Offline exoplanet transit catalogue utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import re

import numpy as np
import pandas as pd


CATALOGUE_COLUMNS = [
    "planet_name",
    "aliases",
    "period_days",
    "period_err_days",
    "t0_bjd_tdb",
    "t0_err_days",
    "duration_hours",
    "duration_err_hours",
    "depth_ppt",
    "depth_err_ppt",
    "rp_rs",
    "rp_rs_err",
    "model_depth_ppt",
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
    "ra_deg",
    "dec_deg",
    "time_reference",
    "source",
    "notes",
    "catalogue_type",
    "depth_source",
    "rprs_source",
    "geometry_source",
    "stellar_source",
]

REQUIRED_CATALOGUE_COLUMNS = [
    "planet_name",
    "period_days",
    "t0_bjd_tdb",
    "duration_hours",
    "depth_ppt",
]

OPTIONAL_CATALOGUE_COLUMNS = [
    "aliases",
    "period_err_days",
    "t0_err_days",
    "duration_err_hours",
    "depth_err_ppt",
    "rp_rs",
    "rp_rs_err",
    "model_depth_ppt",
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
    "ra_deg",
    "dec_deg",
    "time_reference",
    "source",
    "notes",
    "catalogue_type",
    "depth_source",
    "rprs_source",
    "geometry_source",
    "stellar_source",
]


@dataclass
class PlanetTransitParameters:
    """Minimal transit parameters needed for diagnostic fitting."""

    planet_name: str
    aliases: str
    period_days: float
    t0_bjd_tdb: float
    duration_hours: float
    depth_ppt: float
    rp_rs: float
    period_err_days: float = np.nan
    t0_err_days: float = np.nan
    duration_err_hours: float = np.nan
    depth_err_ppt: float = np.nan
    rp_rs_err: float = np.nan
    model_depth_ppt: float = np.nan
    a_rs: float = np.nan
    a_rs_err: float = np.nan
    inclination_deg: float = np.nan
    inclination_err_deg: float = np.nan
    ecc: float = np.nan
    ecc_err: float = np.nan
    omega_deg: float = np.nan
    omega_err_deg: float = np.nan
    stellar_teff_k: float = np.nan
    stellar_teff_err_k: float = np.nan
    stellar_logg: float = np.nan
    stellar_logg_err: float = np.nan
    stellar_feh: float = np.nan
    stellar_feh_err: float = np.nan
    ra_deg: float = np.nan
    dec_deg: float = np.nan
    time_reference: str = ""
    source: str = ""
    notes: str = ""
    catalogue_type: str = ""
    depth_source: str = ""
    rprs_source: str = ""
    geometry_source: str = ""
    stellar_source: str = ""

    @property
    def depth_relative(self) -> float:
        """Return the expected model depth in relative-flux units.

        If a model depth is present, it is used for the empirical model.
        Otherwise, PhotoCurve Lab falls back to the catalogue transit depth.
        """
        if np.isfinite(self.model_depth_ppt) and self.model_depth_ppt > 0:
            return self.model_depth_ppt / 1000.0
        return self.depth_ppt / 1000.0

    @property
    def expected_model_depth_ppt(self) -> float:
        """Return the depth used by the diagnostic transit model."""
        if np.isfinite(self.model_depth_ppt) and self.model_depth_ppt > 0:
            return float(self.model_depth_ppt)
        return float(self.depth_ppt)


def default_catalogue_path() -> Path:
    """Return the path to the bundled/default NASA-style exoplanet catalogue."""
    return Path(__file__).resolve().parent.parent / "catalogs" / "exoplanet_transit_catalog.csv"


def default_exoclock_catalogue_path() -> Path:
    """Return the standard path for an ExoClock-derived catalogue.

    The CSV is not necessarily bundled in every release because it is generated
    from the online ExoClock database by tools/build_exoplanet_catalog_from_exoclock.py.
    """
    return Path(__file__).resolve().parent.parent / "catalogs" / "exoclock_transit_catalog.csv"


def load_exoplanet_catalogue(path: str | Path) -> pd.DataFrame:
    """Load an offline exoplanet transit catalogue from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalogue not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_CATALOGUE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The catalogue is missing required columns: " + ", ".join(missing)
        )

    numeric_optional = {
        "period_err_days", "t0_err_days", "duration_err_hours",
        "depth_err_ppt", "rp_rs", "rp_rs_err", "model_depth_ppt",
        "a_rs", "a_rs_err", "inclination_deg", "inclination_err_deg",
        "ecc", "ecc_err", "omega_deg", "omega_err_deg",
        "stellar_teff_k", "stellar_teff_err_k", "stellar_logg",
        "stellar_logg_err", "stellar_feh", "stellar_feh_err",
        "ra_deg", "dec_deg",
    }

    for col in OPTIONAL_CATALOGUE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in numeric_optional else ""

    numeric_columns = [
        "period_days", "period_err_days", "t0_bjd_tdb", "t0_err_days",
        "duration_hours", "duration_err_hours", "depth_ppt", "depth_err_ppt",
        "rp_rs", "rp_rs_err", "model_depth_ppt", "a_rs", "a_rs_err",
        "inclination_deg", "inclination_err_deg", "ecc", "ecc_err",
        "omega_deg", "omega_err_deg", "stellar_teff_k", "stellar_teff_err_k",
        "stellar_logg", "stellar_logg_err", "stellar_feh", "stellar_feh_err",
        "ra_deg", "dec_deg",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ``depth_ppt`` is the catalogue flux depth used by the empirical model.
    # ``model_depth_ppt`` is an optional explicit model-depth override.  Older
    # ExoClock-derived catalogues sometimes ended up with depth_ppt = Rp/Rs^2
    # while model_depth_ppt contained the intended flux depth converted from the
    # ExoClock millimagnitude depth.  For ExoClock-pure rows, keep the two
    # quantities consistent with the flux/model depth so that the report does
    # not show a misleading "Catalogue depth = Rp/Rs^2" value.
    missing_model_depth = ~np.isfinite(df["model_depth_ppt"]) | (df["model_depth_ppt"] <= 0)
    df.loc[missing_model_depth, "model_depth_ppt"] = df.loc[missing_model_depth, "depth_ppt"]

    is_exoclock_pure = (
        df["catalogue_type"].astype(str).str.contains("ExoClock pure", case=False, na=False)
        | df["source"].astype(str).str.contains("ExoClock", case=False, na=False)
    )
    has_model_flux_depth = np.isfinite(df["model_depth_ppt"]) & (df["model_depth_ppt"] > 0)
    has_depth = np.isfinite(df["depth_ppt"]) & (df["depth_ppt"] > 0)
    differs_substantially = (
        has_model_flux_depth
        & has_depth
        & (np.abs(df["model_depth_ppt"] - df["depth_ppt"]) > np.maximum(0.25, 0.03 * df["depth_ppt"]))
    )
    df.loc[is_exoclock_pure & differs_substantially, "depth_ppt"] = df.loc[
        is_exoclock_pure & differs_substantially, "model_depth_ppt"
    ]

    # For legacy non-ExoClock catalogues without a model-depth column, keep a
    # fallback based on Rp/Rs only when no flux depth/model depth exists.
    missing_model_depth = ~np.isfinite(df["model_depth_ppt"]) | (df["model_depth_ppt"] <= 0)
    has_rprs = np.isfinite(df["rp_rs"]) & (df["rp_rs"] > 0)
    df.loc[missing_model_depth & has_rprs, "model_depth_ppt"] = (df.loc[missing_model_depth & has_rprs, "rp_rs"] ** 2) * 1000.0
    df.loc[~np.isfinite(df["model_depth_ppt"]) | (df["model_depth_ppt"] <= 0), "model_depth_ppt"] = df["depth_ppt"]

    # Keep only rows that contain the minimum needed for the empirical model.
    required = ["planet_name", "period_days", "t0_bjd_tdb", "duration_hours", "depth_ppt"]
    df = df.dropna(subset=[c for c in required if c != "planet_name"]).copy()
    df["planet_name"] = df["planet_name"].astype(str)
    df = df.sort_values("planet_name", kind="mergesort").reset_index(drop=True)
    return df


def planet_names(catalogue: pd.DataFrame) -> List[str]:
    """Return sorted planet names for GUI combo boxes."""
    if catalogue is None or catalogue.empty or "planet_name" not in catalogue.columns:
        return []
    return [str(name) for name in catalogue["planet_name"].tolist()]


def _normalise_name_for_matching(text: str) -> str:
    """Return a simplified planet name for matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _split_aliases(aliases: object) -> List[str]:
    """Split the aliases field into individual candidate names.

    Catalogues produced by different sources may use semicolons, commas or
    pipes. Empty values and literal NaNs are ignored.
    """
    text = str(aliases).strip()
    if not text or text.lower() == "nan":
        return []

    parts = re.split(r"[;,|]", text)
    return [part.strip() for part in parts if part.strip()]


def _target_match_keys(text: str) -> set[str]:
    """Return robust match keys extracted from a filename or free text.

    The previous matcher used simple substring matching on a fully compacted
    string. That is dangerous for planet names: for example ``WASP-1 b`` can be
    found inside ``wasp-15b`` if the separators are removed. Here we keep the
    original token boundaries and also add short joined-token combinations, so
    filenames such as ``wasp-15b_lightcurve.txt``, ``wasp15b.txt`` and
    ``HAT-P-24_b.txt`` can all be matched without allowing unsafe partial
    matches.
    """
    raw = str(text).lower()
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", raw) if tok]

    keys: set[str] = set(tokens)

    # Join a few neighbouring tokens to recover names split by hyphens,
    # underscores or spaces, e.g. wasp + 15b -> wasp15b, hat + p + 24b -> hatp24b.
    max_join = 5
    for i in range(len(tokens)):
        current = ""
        for j in range(i, min(len(tokens), i + max_join)):
            current += tokens[j]
            if current:
                keys.add(current)

    # Also add the complete compact string only for exact full-name comparisons.
    # It is not used as a substring search target.
    compact = _normalise_name_for_matching(text)
    if compact:
        keys.add(compact)

    return keys


def _is_useful_planet_key(key: str) -> bool:
    """Return True for compact keys that look like real planet identifiers.

    Very short keys such as ``b`` or ``15b`` are deliberately rejected because
    they would create many false matches.  Keys such as ``wasp15b``,
    ``hats24b`` and ``55cnce`` are accepted.
    """
    key = str(key).strip().lower()
    if len(key) < 4:
        return False
    has_letter = any(ch.isalpha() for ch in key)
    has_digit = any(ch.isdigit() for ch in key)
    return has_letter and has_digit


def _candidate_match_keys(name: str) -> set[str]:
    """Return normalised candidate keys for one planet name or alias.

    ExoClock and NASA sometimes use slightly different display names.  For
    example, a planet can appear as ``WASP-15 b`` in one catalogue and
    ``WASP-15b`` in another.  Some catalogues may also append a readable alias
    or note in brackets.  We therefore generate compact keys from the full name
    and from contiguous token combinations.

    The matching remains conservative because only keys that look like real
    planet identifiers are kept.  This avoids matching ``WASP-1 b`` when the
    filename contains ``WASP-15b``.
    """
    name = str(name).strip()
    if not name or name.lower() == "nan":
        return set()

    keys: set[str] = set()

    full_key = _normalise_name_for_matching(name)
    if full_key:
        keys.add(full_key)

    tokens = [tok for tok in re.split(r"[^a-z0-9]+", name.lower()) if tok]

    # Full joined-token form: ``WASP 15 b`` -> ``wasp15b``.
    if tokens:
        keys.add("".join(tokens))

    # Short joined-token combinations help when a catalogue appends extra text,
    # for example ``WASP-15b (Nyamien)``.  The useful key is then ``wasp15b``,
    # not the full compact string ``wasp15bnyamien``.
    max_join = 5
    for i in range(len(tokens)):
        current = ""
        for j in range(i, min(len(tokens), i + max_join)):
            current += tokens[j]
            if current:
                keys.add(current)

    return {key for key in keys if _is_useful_planet_key(key)}


def guess_planet_from_text(catalogue: pd.DataFrame, text: str) -> Optional[str]:
    """Guess the planet name from a filename or free text.

    The match is deliberately conservative. It requires an exact match between
    a normalised planet name/alias and one of the filename-derived keys. When
    several planets match, the longest match wins. This avoids selecting
    ``HATS-2 b`` for a file containing ``hats24b`` or ``WASP-1 b`` for a file
    containing ``wasp-15b``.
    """
    if catalogue is None or catalogue.empty:
        return None

    target_keys = _target_match_keys(text)
    if not target_keys:
        return None

    best_name: Optional[str] = None
    best_score = -1

    for row_index, row in catalogue.iterrows():
        planet_name = str(row.get("planet_name", "")).strip()
        if not planet_name:
            continue

        candidates = [planet_name]
        candidates.extend(_split_aliases(row.get("aliases", "")))

        for candidate in candidates:
            candidate_keys = _candidate_match_keys(candidate)
            for candidate_key in candidate_keys:
                if candidate_key in target_keys:
                    # Exact key match. Prefer longer/more specific names and the
                    # official planet_name over aliases when scores are similar.
                    is_primary_name = candidate.strip().lower() == planet_name.strip().lower()
                    score = len(candidate_key) * 10 + (5 if is_primary_name else 0)

                    # Stable tie-breaker: keep the first catalogue occurrence.
                    if score > best_score:
                        best_score = score
                        best_name = planet_name

    return best_name


def find_planet(catalogue: pd.DataFrame, name: str) -> PlanetTransitParameters:
    """Find a planet by name or alias and return its parameters."""
    if catalogue is None or catalogue.empty:
        raise ValueError("The exoplanet catalogue is empty.")

    query = str(name).strip().lower()
    if not query:
        raise ValueError("Select a planet from the catalogue.")

    for _, row in catalogue.iterrows():
        planet_name = str(row.get("planet_name", "")).strip()
        aliases = str(row.get("aliases", "")).strip()
        candidates = [planet_name.lower()]
        if aliases and aliases.lower() != "nan":
            candidates.extend([part.strip().lower() for part in aliases.split(";") if part.strip()])

        if query in candidates:
            rp_rs = row.get("rp_rs", np.nan)
            if not np.isfinite(rp_rs):
                rp_rs = np.sqrt(float(row["depth_ppt"]) / 1000.0)

            def f(column: str) -> float:
                return float(row.get(column, np.nan))

            return PlanetTransitParameters(
                planet_name=planet_name,
                aliases=aliases if aliases.lower() != "nan" else "",
                period_days=float(row["period_days"]),
                t0_bjd_tdb=float(row["t0_bjd_tdb"]),
                duration_hours=float(row["duration_hours"]),
                depth_ppt=float(row["depth_ppt"]),
                rp_rs=float(rp_rs),
                period_err_days=f("period_err_days"),
                t0_err_days=f("t0_err_days"),
                duration_err_hours=f("duration_err_hours"),
                depth_err_ppt=f("depth_err_ppt"),
                rp_rs_err=f("rp_rs_err"),
                model_depth_ppt=f("model_depth_ppt"),
                a_rs=f("a_rs"),
                a_rs_err=f("a_rs_err"),
                inclination_deg=f("inclination_deg"),
                inclination_err_deg=f("inclination_err_deg"),
                ecc=f("ecc"),
                ecc_err=f("ecc_err"),
                omega_deg=f("omega_deg"),
                omega_err_deg=f("omega_err_deg"),
                stellar_teff_k=f("stellar_teff_k"),
                stellar_teff_err_k=f("stellar_teff_err_k"),
                stellar_logg=f("stellar_logg"),
                stellar_logg_err=f("stellar_logg_err"),
                stellar_feh=f("stellar_feh"),
                stellar_feh_err=f("stellar_feh_err"),
                ra_deg=f("ra_deg"),
                dec_deg=f("dec_deg"),
                time_reference=str(row.get("time_reference", "")),
                source=str(row.get("source", "")),
                notes=str(row.get("notes", "")),
                catalogue_type=str(row.get("catalogue_type", "")),
                depth_source=str(row.get("depth_source", "")),
                rprs_source=str(row.get("rprs_source", "")),
                geometry_source=str(row.get("geometry_source", "")),
                stellar_source=str(row.get("stellar_source", "")),
            )

    raise ValueError(f"Planet not found in catalogue: {name}")
