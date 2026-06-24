# Building exoplanet transit catalogues

Normal ExoPhotoCurve users do not need to run catalogue-generation scripts.
Each release can be bundled with one or more ready-to-use CSV catalogues inside:

```text
exophotocurve/catalogs/
```

The Transit modeling tab can load any CSV that follows the ExoPhotoCurve catalogue format.
The GUI provides quick buttons for the default NASA-style catalogue and for the
standard ExoClock-derived catalogue.

---

## NASA Exoplanet Archive catalogue

The default NASA-style catalogue is written to:

```text
exophotocurve/catalogs/exoplanet_transit_catalog.csv
```

Regenerate it with:

```bash
python tools/build_exoplanet_catalog_from_nasa.py
```

The script queries the NASA Exoplanet Archive TAP service, specifically the
`pscomppars` table, and maps the following columns into the ExoPhotoCurve format:

- `pl_name` -> `planet_name`
- `pl_orbper` -> `period_days`
- `pl_tranmid` -> `t0_bjd_tdb`
- `pl_trandur` -> `duration_hours`
- `pl_trandep` -> `depth_ppt` after converting percent to ppt
- `pl_ratror` -> `rp_rs`
- `pl_ratdor` -> `a_rs`
- `pl_orbincl` -> `inclination_deg`
- `ra`, `dec` -> `ra_deg`, `dec_deg`

If `pl_trandep` is missing, the depth is estimated from `(Rp/Rs)^2`.
If `pl_ratror` is missing, `Rp/Rs` is estimated from the transit depth.

For the NASA catalogue, `model_depth_ppt` is kept internally consistent with
`Rp/Rs` when `Rp/Rs` is available. This is useful because the NASA catalogue also
usually provides the geometry needed by the physical `batman` model.

---

## ExoClock pure catalogue

The ExoClock-derived catalogue is written to:

```text
exophotocurve/catalogs/exoclock_transit_catalog.csv
```

Regenerate it with:

```bash
python tools/build_exoplanet_catalog_from_exoclock.py
```

The default ExoClock catalogue is now **ExoClock pure**. This means that the
script uses only the information available in the ExoClock JSON endpoint. It no
longer fills missing physical geometry from the NASA catalogue unless this is
explicitly requested.

The script downloads:

```text
https://www.exoclock.space/database/planets_json
```

and maps ExoClock fields into the same CSV structure used by the NASA catalogue.

Important ExoClock mappings include:

- `name` -> `planet_name`
- `period_days` -> `period_days`
- `period_unc` -> `period_err_days`
- `t0_bjd_tdb` -> `t0_bjd_tdb`
- `t0_unc` -> `t0_err_days`
- `duration_hours` -> `duration_hours`
- `depth_mmag` / `depth` -> `depth_ppt` after conversion from ExoClock millimagnitudes
- `rp_rs`, when available -> `rp_rs`
- `ra_j2000`, `dec_j2000` -> `ra_deg`, `dec_deg`

ExoPhotoCurve works internally in relative flux, not magnitudes. For this reason
ExoClock millimagnitude depths are converted to flux depth in ppt with. Some ExoClock JSON payloads expose this quantity with the generic key `depth`; for the ExoClock builder this key is treated as a millimagnitude depth, not as a flux depth:

```text
depth_ppt = (1 - 10^(-depth_mmag / 2500)) * 1000
```

Direct ExoClock flux-depth fields, if provided by the endpoint in the future, are
also stored in ppt. Values below about 0.2 are interpreted as fractional flux
depths and converted to ppt; larger positive values are interpreted as already in
ppt.

For the ExoClock pure catalogue, `model_depth_ppt` is set to the ExoClock flux
`depth_ppt` when available. This prevents the empirical expected model from being
silently replaced by `Rp/Rs^2` when ExoClock provides both an observed/catalogue
flux depth and an `Rp/Rs` value.

Rows lacking `a_rs` or `inclination_deg` can still be used. In those cases,
ExoPhotoCurve's `Auto` model mode falls back to the empirical transit template,
because the optional `batman` physical model requires the missing geometry.

The generated CSV includes provenance columns such as:

- `catalogue_type`
- `depth_source`
- `rprs_source`
- `geometry_source`
- `stellar_source`

These columns make it clear whether a row is ExoClock pure or has been
supplemented with external physical parameters.

---

## Explicit ExoClock + NASA hybrid catalogue

A hybrid catalogue can still be useful when you deliberately want ExoClock timing
and depth, but you also want to run the physical `batman` model for targets whose
ExoClock JSON row does not provide `a_rs` and `inclination_deg`.

Create such a catalogue explicitly with:

```bash
python tools/build_exoplanet_catalog_from_nasa.py
python tools/build_exoplanet_catalog_from_exoclock.py \
    --supplement-catalogue exophotocurve/catalogs/exoplanet_transit_catalog.csv \
    --output exophotocurve/catalogs/exoclock_hybrid_transit_catalog.csv
```

This keeps the ExoClock ephemerides and flux-depth values as the primary timing
and depth reference, while using the supplement catalogue only to fill missing
parameters such as:

- `a_rs`
- `inclination_deg`
- `ecc`
- `omega_deg`
- stellar parameters, when present

Hybrid rows are marked as:

```text
catalogue_type = ExoClock+supplement hybrid
```

and the notes field contains a warning that the physical expected model may not
match the ExoClock online model. This is intentional: hybrid catalogues are
advanced diagnostic products, not a replacement for the ExoClock pure catalogue.

Useful options:

```bash
python tools/build_exoplanet_catalog_from_exoclock.py --output my_exoclock_catalog.csv
python tools/build_exoplanet_catalog_from_exoclock.py --supplement-catalogue exophotocurve/catalogs/exoplanet_transit_catalog.csv
python tools/build_exoplanet_catalog_from_exoclock.py --supplement-catalogue exophotocurve/catalogs/exoplanet_transit_catalog.csv --fill-rprs-from-supplement
```

Use `--fill-rprs-from-supplement` only when ExoClock-derived `Rp/Rs` is missing
and you explicitly want to fill it from the supplement catalogue.
