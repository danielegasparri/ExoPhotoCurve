# Building exoplanet transit catalogues

Normal PhotoCurve Lab users do not need to run catalogue-generation scripts.
Each release can be bundled with one or more ready-to-use CSV catalogues inside:

```text
photocurve_lab/catalogs/
```

The Transit tab can load any CSV that follows the PhotoCurve Lab catalogue format.
The GUI provides quick buttons for the default NASA-style catalogue and for the
standard ExoClock-derived catalogue.

---

## NASA Exoplanet Archive catalogue

The default NASA-style catalogue is written to:

```text
photocurve_lab/catalogs/exoplanet_transit_catalog.csv
```

Regenerate it with:

```bash
python tools/build_exoplanet_catalog_from_nasa.py
```

The script queries the NASA Exoplanet Archive TAP service, specifically the
`pscomppars` table, and maps the following columns into the PhotoCurve Lab format:

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

---

## ExoClock catalogue

The ExoClock-derived catalogue is written to:

```text
photocurve_lab/catalogs/exoclock_transit_catalog.csv
```

Regenerate it with:

```bash
python tools/build_exoplanet_catalog_from_exoclock.py
```

The script downloads:

```text
https://www.exoclock.space/database/planets_json
```

and maps the ExoClock fields into the same PhotoCurve Lab CSV structure used by
the NASA catalogue.

Important ExoClock mappings:

- `name` -> `planet_name`
- `period_days` -> `period_days`
- `period_unc` -> `period_err_days`
- `t0_bjd_tdb` -> `t0_bjd_tdb`
- `t0_unc` -> `t0_err_days`
- `duration_hours` -> `duration_hours`
- `depth_mmag` -> `depth_ppt`
- `ra_j2000`, `dec_j2000` -> `ra_deg`, `dec_deg`

The `depth_mmag` value is converted to flux depth with:

```text
depth_ppt = (1 - 10^(-depth_mmag / 2500)) * 1000
```

Since the ExoClock JSON endpoint provides ephemerides, depth, duration and target
coordinates but not all geometric parameters used by `batman`, the builder can
also supplement the ExoClock rows with geometry from an existing NASA-style
catalogue:

```bash
python tools/build_exoplanet_catalog_from_nasa.py
python tools/build_exoplanet_catalog_from_exoclock.py \
    --supplement-catalogue photocurve_lab/catalogs/exoplanet_transit_catalog.csv
```

This keeps the ExoClock ephemerides as the primary timing reference, while using
the supplement catalogue to fill missing parameters such as:

- `a_rs`
- `inclination_deg`
- `ecc`
- `omega_deg`
- stellar parameters, when present

Rows lacking `a_rs` or `inclination_deg` can still be used, but PhotoCurve Lab's
`Auto` model mode will fall back to the empirical transit template for those
objects because the `batman` physical model needs the missing geometry.

Useful options:

```bash
python tools/build_exoplanet_catalog_from_exoclock.py --no-supplement
python tools/build_exoplanet_catalog_from_exoclock.py --output my_exoclock_catalog.csv
python tools/build_exoplanet_catalog_from_exoclock.py --fill-rprs-from-supplement
```
