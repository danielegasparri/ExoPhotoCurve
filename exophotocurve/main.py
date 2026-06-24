"""
    Copyright (C) 2026, Daniele Gasparri
    E-mail: daniele.gasparri@gmail.com

    ExoPhotoCurve (EPC) is a GUI software for generating, modeling and analyzing transiting 
    extrasolar planet light curves.
    It is intended particularly for advanced amateur astronomers for follow-up studies of known 
    transiting extrasolar planets.
    EPC does not provide the required tools and accuracy for a complete scientific transit analysis and modeling.

    1. This software is licensed for non-commercial, academic and personal use only.
    2. The source code may be used and modified for research and educational purposes, 
    but any modifications must remain for private use unless explicitly authorized 
    in writing by the original author.
    3. Redistribution of the software in its original, unmodified form is permitted 
    for non-commercial purposes, provided that this license notice is always included.
    4. Redistribution or public release of modified versions of the source code 
    is prohibited without prior written permission from the author.
    5. Any user of this software must properly attribute the original author 
    in any academic work, research, or derivative project.
    6. Commercial use of this software is strictly prohibited without prior 
    written permission from the author.

    DISCLAIMER:
    THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ctypes

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#local import
try:
    from modules import misc

    from modules.axis_utils import transform_x_axis
    from modules.aperture_photometry_tool import run_aperture_photometry_tool
    from modules.image_reduction_tool import run_image_reduction_tool
    from modules.binning_utils import bin_light_curve
    from modules.cleaning_utils import compute_auto_sigma_clip_reject_indices, compute_cleaning_mask
    from modules.config_utils import apply_config, load_config, save_config
    from modules.comparison_optimizer import (
        AijFluxDetection,
        ComparisonDiagnosticCurve,
        ComparisonOptimisationResult,
        build_comparison_diagnostics,
        build_manual_comparison_result,
        detect_aij_flux_columns,
        format_comparison_diagnostics_report,
        optimise_comparison_stars,
    )
    from modules.detrending_utils import (
        DetrendRegressorDetection,
        PhotometricDetrendResult,
        apply_photometric_detrending,
        detect_detrending_regressors,
    )
    from modules.constants import NONE_COL
    from modules.exoplanet_catalog import (
        default_catalogue_path,
        default_exoclock_catalogue_path,
        find_planet,
        guess_planet_from_text,
        load_exoplanet_catalogue,
        planet_names,
    )
    from modules.gui_layout import make_layout
    from modules.io_utils import guess_column, numeric_columns, read_ascii_table
    from modules.numeric_utils import parse_float, parse_int, to_numeric_array
    from modules.plot_utils import build_plot, delete_figure_agg, draw_figure
    from modules.sg_loader import sg
    from modules.transit_diagnostics import (
        format_transit_report,
        result_to_dict,
        run_transit_diagnostics,
    )
    from modules.statistics_utils import (
        SeriesStatistics,
        compute_series_statistics,
        format_statistics_report,
        statistics_to_dataframe_rows,
    )

# Import for script execution
except ModuleNotFoundError: #local import if executed as package
    from .modules import misc

    from .modules.axis_utils import transform_x_axis
    from .modules.aperture_photometry_tool import run_aperture_photometry_tool
    from .modules.image_reduction_tool import run_image_reduction_tool
    from .modules.binning_utils import bin_light_curve
    from .modules.cleaning_utils import compute_auto_sigma_clip_reject_indices, compute_cleaning_mask
    from .modules.config_utils import apply_config, load_config, save_config
    from .modules.comparison_optimizer import (
        AijFluxDetection,
        ComparisonDiagnosticCurve,
        ComparisonOptimisationResult,
        build_comparison_diagnostics,
        build_manual_comparison_result,
        detect_aij_flux_columns,
        format_comparison_diagnostics_report,
        optimise_comparison_stars,
    )
    from .modules.detrending_utils import (
        DetrendRegressorDetection,
        PhotometricDetrendResult,
        apply_photometric_detrending,
        detect_detrending_regressors,
    )
    from .modules.constants import NONE_COL
    from .modules.exoplanet_catalog import (
        default_catalogue_path,
        default_exoclock_catalogue_path,
        find_planet,
        guess_planet_from_text,
        load_exoplanet_catalogue,
        planet_names,
    )
    from .modules.gui_layout import make_layout
    from .modules.io_utils import guess_column, numeric_columns, read_ascii_table
    from .modules.numeric_utils import parse_float, parse_int, to_numeric_array
    from .modules.plot_utils import build_plot, delete_figure_agg, draw_figure
    from .modules.sg_loader import sg
    from .modules.transit_diagnostics import (
        format_transit_report,
        result_to_dict,
        run_transit_diagnostics,
    )
    from .modules.statistics_utils import (
        SeriesStatistics,
        compute_series_statistics,
        format_statistics_report,
        statistics_to_dataframe_rows,
    )

#Define the base dir of SPAN in your device
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(BASE_DIR, "ExoPhotoCurve.ico")

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

# Simple function to open the PDF manual
def open_manual():
    try:
        manual_path = os.path.join(BASE_DIR, "manual/ExoPhotoCurve_User_Guide.pdf")

        # if sys.platform.startswith("darwin"):  # macOS
        #     subprocess.run(["open", manual_path])
        if os.name == "nt":  # Windows
            os.startfile(manual_path) 
        # elif os.name == "posix":  # Linux/Unix
        #     subprocess.run(["xdg-open", manual_path])
        else:
            raise RuntimeError("Unsupported platform")
    except Exception:
        sg.popup('Manual not found, sorry.')
        
        
def _column_has_usable_numeric_data(df: Optional[pd.DataFrame], column: str) -> bool:
    """Return True when *column* contains at least one finite numeric value."""
    if df is None or not column or column not in df.columns:
        return True
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return bool(np.count_nonzero(np.isfinite(values)) > 0)


def _guess_time_column(columns: List[str], df: Optional[pd.DataFrame] = None) -> str:
    """Guess the safest time column, avoiding empty AstroImageJ BJD_TDB columns.

    Some raw AstroImageJ exports contain a ``BJD_TDB`` column that is present
    but entirely NaN.  Selecting it automatically is inconvenient and can also
    force the Transit tab back to ``BJD_TDB`` after a diagnostic run.  Prefer
    AIJ's usable JD_UTC columns when available, and only select BJD_TDB if it
    contains finite values.
    """
    exact_priority = [
        "JD_UTC_B",
        "JD_UTC",
        "BJD_TDB",
        "BJD_UTC",
        "HJD_UTC",
        "J.D.-2400000",
        "J.D.-2400000_1",
    ]
    lower_to_column = {str(col).lower(): col for col in columns}

    for name in exact_priority:
        column = lower_to_column.get(name.lower())
        if column and _column_has_usable_numeric_data(df, column):
            return column

    # Fall back to substring matching, still rejecting columns that have no
    # finite values when the DataFrame is available.
    for keyword in ["jd_utc", "bjd_tdb", "bjd", "hjd", "jd", "time"]:
        for column in columns:
            if keyword in str(column).lower() and _column_has_usable_numeric_data(df, column):
                return column

    return NONE_COL


def _time_system_from_column(column: str) -> Optional[str]:
    """Infer the Transit-tab time-system selector from a time column name."""
    text = str(column).strip().lower()
    if not text or text == str(NONE_COL).lower():
        return None
    if "bjd_tdb" in text:
        return "BJD_TDB"
    if "bjd_utc" in text:
        return "BJD_UTC"
    if "jd_utc" in text or "j.d." in text or text == "jd":
        return "JD_UTC"
    if "hjd" in text:
        return "HJD_UTC"
    return None


def update_combo_values(
    window: sg.Window,
    columns: List[str],
    df: Optional[pd.DataFrame] = None,
    sync_transit_time_system: bool = False,
) -> Dict[str, str]:
    """Update column combo boxes after table changes and return the guessed mapping.

    ``sync_transit_time_system`` should be True only when loading/restoring an
    original photometry table.  It must remain False after adding PhotoCurve
    diagnostic columns, otherwise a user-selected ``JD_UTC`` setting can be
    overwritten by a generated or empty BJD column.
    """
    choices = [NONE_COL] + columns

    guessed = {
        "-XCOL-": _guess_time_column(columns, df),
        "-YCOL-": guess_column(
            columns,
            ["rel_flux_T1_dfn", "rel_flux_T1", "rel_flux", "flux", "mag"],
        ),
        "-YERRCOL-": guess_column(
            columns,
            ["rel_flux_err_T1_dfn", "rel_flux_err_T1", "rel_flux_err", "flux_err", "err", "error", "sigma"],
        ),
        "-MODEL_COL-": guess_column(
            columns,
            ["rel_flux_T1_dfn_model", "dfn_model", "model"], # "fit"],
        ),
        "-RES_COL-": guess_column(
            columns,
            ["rel_flux_T1_dfn_residual", "dfn_residual", "residual", "resid", "o-c", "oc"],
        ),
        "-RESERR_COL-": guess_column(
            columns,
            ["rel_flux_err_T1_dfn_residual", "dfn_residual_err", "residual_err", "resid_err", "res_err", "sigma"],
        ),
    }

    for key in [
        "-XCOL-", "-YCOL-", "-YERRCOL-",
        "-MODEL_COL-", "-RES_COL-", "-RESERR_COL-",
    ]:
        window[key].update(values=choices, value=guessed[key])

    if sync_transit_time_system:
        inferred = _time_system_from_column(guessed.get("-XCOL-", ""))
        if inferred is not None:
            try:
                window["-TR_TIME_SYSTEM-"].update(value=inferred)
            except Exception:
                pass

    return guessed


def current_column_selection(values: Dict[str, object]) -> Dict[str, str]:
    """Return the currently selected data/model columns."""
    keys = [
        "-XCOL-", "-YCOL-", "-YERRCOL-",
        "-MODEL_COL-", "-RES_COL-", "-RESERR_COL-",
    ]
    return {key: str(values.get(key, NONE_COL)) for key in keys}






def resolve_meridian_flip_time_from_fraction(x_values: np.ndarray, fraction_value: object) -> Optional[float]:
    """Resolve an AIJ-style fractional meridian-flip time to a full JD-like time.

    AstroImageJ asks the user to enter only the fractional part of the time.
    For example, if the meridian flip occurred at JD_UTC = 2461203.771, the
    user enters .771.  We recover the integer part from the median input time
    of the light curve.  Full JD/MJD values are also accepted for safety.
    """
    value = parse_float(fraction_value, None)
    if value is None or not np.isfinite(value):
        return None
    value = float(value)
    if abs(value) >= 1000.0:
        return value
    x_arr = np.asarray(x_values, dtype=float)
    finite = x_arr[np.isfinite(x_arr)]
    if finite.size == 0:
        return None
    integer_part = float(np.floor(np.nanmedian(finite)))
    return integer_part + value

def _selection_uses_prefix(selection: Optional[Dict[str, str]], prefixes: Tuple[str, ...]) -> bool:
    """Return True if any selected column starts with one of *prefixes*."""
    if not selection:
        return False
    return any(str(value).startswith(prefix) for value in selection.values() for prefix in prefixes)


def resolve_detrending_input_selection(
    values: Dict[str, object],
    pre_transit_column_selection: Optional[Dict[str, str]],
    original_column_selection: Optional[Dict[str, str]],
    last_detrend_input_selection: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], str]:
    """Return the column mapping that should be used as detrending input.

    Detrending must be repeatable.  If the current GUI selection points to a
    previous transit diagnostic display or to the previous detrended output, we
    go back to the real source columns that produced that detrended curve.
    Otherwise repeated runs can accidentally decorrelate already-detrended data
    or, worse, pass already BJD-corrected times to the transit fitter as if they
    were still JD_UTC values.
    """
    current = current_column_selection(values)

    if contains_photocurve_columns(current):
        candidate = pre_transit_column_selection or original_column_selection or current
        if _selection_uses_prefix(candidate, ("PhotoCurve_det_",)) and last_detrend_input_selection:
            return last_detrend_input_selection.copy(), "previous detrending input columns"
        return candidate.copy(), "pre-transit input columns"

    if _selection_uses_prefix(current, ("PhotoCurve_det_",)) and last_detrend_input_selection:
        return last_detrend_input_selection.copy(), "previous detrending input columns"

    return current.copy(), "current Data-tab columns"


def contains_photocurve_columns(selection: Dict[str, str]) -> bool:
    """Return True if a selection points to transient diagnostic columns.

    Comparison-star optimizer columns are intentionally treated as user-facing
    input columns.  This lets the user fit a transit to
    ``PhotoCurve_compopt_flux`` and later hide the transit model while keeping
    the optimised light curve selected.
    """
    return any(
        str(value).startswith("PhotoCurve_")
        and not str(value).startswith("PhotoCurve_compopt_")
        and not str(value).startswith("PhotoCurve_compdiag_")
        and not str(value).startswith("PhotoCurve_det_")
        for value in selection.values()
    )


def restore_column_selection(window: sg.Window, selection: Optional[Dict[str, str]]) -> None:
    """Restore a previous column selection if available."""
    if not selection:
        return
    for key, value in selection.items():
        try:
            window[key].update(value=value)
        except Exception:
            pass


def update_values_with_selection(values: Dict[str, object], selection: Optional[Dict[str, str]]) -> Dict[str, object]:
    """Return a values dictionary with an explicit column selection applied."""
    updated = dict(values)
    if selection:
        updated.update(selection)
    return updated


def remove_photocurve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* without transient diagnostic PhotoCurve columns.

    The comparison-star optimizer produces ``PhotoCurve_compopt_*`` columns
    that are valid light-curve inputs for the Transit tab.  They should survive
    a transit diagnostic run; Reset view/data is the action that removes them.
    """
    keep_columns = [
        column
        for column in df.columns
        if not str(column).startswith("PhotoCurve_")
        or str(column).startswith("PhotoCurve_compopt_")
        or str(column).startswith("PhotoCurve_compdiag_")
        or str(column).startswith("PhotoCurve_det_")
    ]
    return df.loc[:, keep_columns].copy()




def remove_detrending_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* without photometric-detrending output columns.

    This is intentionally narrower than Reset view/data: comparison-star
    products, manual point masks and the original imported table are kept.
    It lets the user undo only the Detrend-tab output and go back to the
    real light-curve source that was used before the detrending step.
    """
    det_prefixes = (
        "PhotoCurve_det_",
    )
    keep_columns = [
        column for column in df.columns
        if not any(str(column).startswith(prefix) for prefix in det_prefixes)
    ]
    return df.loc[:, keep_columns].copy()


def clear_transit_diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove transit-diagnostic display columns while keeping upstream products.

    Clearing detrending invalidates any transit fit that was computed on the
    detrended curve.  The comparison-star output remains available because it is
    an upstream photometric product, not a transit diagnostic product.
    """
    keep_columns = [
        column
        for column in df.columns
        if not str(column).startswith("PhotoCurve_")
        or str(column).startswith("PhotoCurve_compopt_")
        or str(column).startswith("PhotoCurve_compdiag_")
    ]
    return df.loc[:, keep_columns].copy()


def clear_downstream_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove products that become invalid when the light curve changes.

    A new comparison-star selection changes the differential light curve itself.
    Therefore any later products computed from the previous curve are no longer
    scientifically valid: photometric detrending, transit-model columns,
    residuals, baseline aliases, expected/calculated transit markers and other
    downstream PhotoCurve diagnostics are removed.  The upstream comparison-star
    columns are preserved because they are regenerated immediately after the
    selection change.
    """
    return clear_transit_diagnostic_columns(df)

def transit_display_selection(display_mode: str) -> Dict[str, str]:
    """Return the column mapping used to display the current transit result."""
    if display_mode == "Detrended flux":
        return {
            "-XCOL-": "PhotoCurve_time_corrected",
            "-YCOL-": "PhotoCurve_detrended_flux",
            "-YERRCOL-": "PhotoCurve_detrended_err",
            "-MODEL_COL-": "PhotoCurve_fit_model",
            "-RES_COL-": "PhotoCurve_fit_residual",
            "-RESERR_COL-": "PhotoCurve_detrended_err",
            "-YLABEL-": "Detrended relative flux",
        }

    return {
        "-XCOL-": "PhotoCurve_time_corrected",
        "-MODEL_COL-": "PhotoCurve_fit_model",
        "-RES_COL-": "PhotoCurve_fit_residual",
    }


def apply_transit_display_aliases(df: pd.DataFrame, display_mode: str) -> pd.DataFrame:
    """Point generic PhotoCurve columns to the selected transit display mode.

    This lets the user switch between the publication-style detrended display
    and the raw-flux diagnostic display without rerunning the fit.  The alias
    columns are replaced in one concatenation step, rather than by repeated
    column assignment, to keep the pandas DataFrame compact and avoid
    fragmentation warnings after several diagnostic runs.
    """
    required = {
        "PhotoCurve_expected_transit_model",
        "PhotoCurve_fit_transit_model",
        "PhotoCurve_fit_detrended_residual",
        "PhotoCurve_expected_full_model",
        "PhotoCurve_fit_full_model",
        "PhotoCurve_fit_full_residual",
    }
    if not required.issubset(set(df.columns)):
        return df

    if display_mode == "Detrended flux":
        alias_data = {
            "PhotoCurve_expected_model": df["PhotoCurve_expected_transit_model"].to_numpy(),
            "PhotoCurve_fit_model": df["PhotoCurve_fit_transit_model"].to_numpy(),
            "PhotoCurve_fit_residual": df["PhotoCurve_fit_detrended_residual"].to_numpy(),
        }
    else:
        alias_data = {
            "PhotoCurve_expected_model": df["PhotoCurve_expected_full_model"].to_numpy(),
            "PhotoCurve_fit_model": df["PhotoCurve_fit_full_model"].to_numpy(),
            "PhotoCurve_fit_residual": df["PhotoCurve_fit_full_residual"].to_numpy(),
        }

    alias_columns = list(alias_data)
    base = df.drop(columns=alias_columns, errors="ignore")
    aliases = pd.DataFrame(alias_data, index=df.index)
    return pd.concat([base, aliases], axis=1).copy()


def set_transit_plot_columns(
    window: sg.Window,
    values: Dict[str, object],
    display_mode: str,
    original_selection: Optional[Dict[str, str]],
    show_transit_result: bool,
) -> Dict[str, object]:
    """Update GUI columns for transit display and return matching plot values."""
    if not show_transit_result:
        restore_column_selection(window, original_selection)
        return update_values_with_selection(values, original_selection)

    selection = transit_display_selection(display_mode)
    if display_mode != "Detrended flux" and original_selection:
        # Raw diagnostic display: keep the original photometry on the Y axis,
        # but show the full baseline-multiplied transit models and residuals.
        selection["-YCOL-"] = original_selection.get("-YCOL-", NONE_COL)
        selection["-YERRCOL-"] = original_selection.get("-YERRCOL-", NONE_COL)
        selection["-RESERR_COL-"] = original_selection.get("-YERRCOL-", NONE_COL)

    for key, value in selection.items():
        if key.startswith("-") and key.endswith("-") and key != "-YLABEL-":
            try:
                window[key].update(value=value)
            except Exception:
                pass

    plot_values = dict(values)
    plot_values.update(selection)
    return plot_values

def collect_statistics(df: pd.DataFrame, values: Dict[str, object]) -> List[SeriesStatistics]:
    """Collect the requested photometric statistics from the current GUI state."""
    x_col = str(values.get("-XCOL-", NONE_COL))
    y_col = str(values.get("-YCOL-", NONE_COL))
    yerr_col = str(values.get("-YERRCOL-", NONE_COL))
    res_col = str(values.get("-RES_COL-", NONE_COL))
    reserr_col = str(values.get("-RESERR_COL-", NONE_COL))

    x = to_numeric_array(df, x_col)
    y = to_numeric_array(df, y_col)
    yerr = to_numeric_array(df, yerr_col)
    residuals = to_numeric_array(df, res_col)
    residual_err = to_numeric_array(df, reserr_col)
    model = to_numeric_array(df, str(values.get("-MODEL_COL-", NONE_COL)))

    if x is None:
        raise ValueError("Select an X/time column before computing statistics.")

    use_transformed_x = bool(values.get("-STATS_USE_TRANSFORMED_X-", True))
    if use_transformed_x:
        x_for_stats, _, _ = transform_x_axis(x, values, x_col, "")
    else:
        x_for_stats = x

    if bool(values.get("-CLEAN_APPLY_STATS-", True)):
        cleaning = compute_cleaning_mask(x, y, model, residuals, values)
        stats_mask = cleaning.keep_mask
    else:
        stats_mask = None

    if stats_mask is not None and stats_mask.shape == x_for_stats.shape:
        x_for_stats = x_for_stats[stats_mask]
        if y is not None:
            y = y[stats_mask]
        if yerr is not None:
            yerr = yerr[stats_mask]
        if residuals is not None:
            residuals = residuals[stats_mask]
        if residual_err is not None:
            residual_err = residual_err[stats_mask]

    y_type = str(values.get("-STATS_YTYPE-", "Relative flux"))
    target = str(values.get("-STATS_TARGET-", "Both"))

    blocks: List[SeriesStatistics] = []

    if target in ("Light curve", "Both"):
        if y is None:
            raise ValueError("Select a light-curve column before computing light-curve statistics.")

        blocks.append(
            compute_series_statistics(
                x_for_stats,
                y,
                yerr,
                name=f"Light curve: {y_col}",
                y_type=y_type,
            )
        )

        include_binned = bool(values.get("-STATS_INCLUDE_BINNED-", True))
        bin_active = bool(values.get("-BIN_ACTIVE-", False))
        if include_binned and bin_active:
            bin_n = max(1, parse_int(values.get("-BIN_N-", 4), 4))
            x_bin, y_bin, yerr_bin, _ = bin_light_curve(x_for_stats, y, yerr, bin_n)
            if x_bin.size > 0:
                blocks.append(
                    compute_series_statistics(
                        x_bin,
                        y_bin,
                        yerr_bin,
                        name=f"Binned light curve: {bin_n:d} points/bin",
                        y_type=y_type,
                    )
                )

    if target in ("Residuals", "Both"):
        if residuals is not None:
            blocks.append(
                compute_series_statistics(
                    x_for_stats,
                    residuals,
                    residual_err,
                    name=f"Residuals: {res_col}",
                    y_type="Relative flux",
                )
            )
        elif target == "Residuals":
            raise ValueError("Select a residual column before computing residual statistics.")

    return blocks


def build_statistics_report(df: pd.DataFrame, values: Dict[str, object]) -> Tuple[str, List[SeriesStatistics]]:
    """Compute and format the statistics report."""
    blocks = collect_statistics(df, values)
    report = format_statistics_report(blocks)
    return report, blocks


def save_statistics_file(path: str, report: str, blocks: List[SeriesStatistics]) -> None:
    """Save the statistics report as TXT, CSV or JSON depending on extension."""
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        rows = statistics_to_dataframe_rows(blocks)
        pd.DataFrame(rows).to_csv(path, index=False)
    elif suffix == ".json":
        rows = statistics_to_dataframe_rows(blocks)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=4)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)


def save_transit_diagnostic_file(path: str, report: str, result) -> None:
    """Save a transit diagnostics report as TXT, CSV or JSON."""
    suffix = Path(path).suffix.lower()
    row = result_to_dict(result)
    row["text_report"] = report

    if suffix == ".csv":
        pd.DataFrame([row]).to_csv(path, index=False)
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=4)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)


def save_reproducibility_recipe_file(path: str, recipe: str) -> None:
    """Save the current reduction/analysis recipe as TXT or JSON."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"PhotoCurve_recipe": recipe}, f, indent=4)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(recipe)


def _is_selected_column(column: object) -> bool:
    """Return True when a GUI column selection points to a real column."""
    text = str(column).strip()
    return bool(text) and text != str(NONE_COL)


def _add_numeric_column_if_available(
    output: Dict[str, object],
    df: pd.DataFrame,
    output_name: str,
    source_column: object,
) -> None:
    """Append a numeric column to an export table if it exists and is usable."""
    source = str(source_column).strip()
    if not _is_selected_column(source) or source not in df.columns:
        return
    arr = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
    output[output_name] = arr


def _add_numeric_column_by_name(
    output: Dict[str, object],
    df: pd.DataFrame,
    output_name: str,
    source_column: str,
) -> bool:
    """Append a numeric column by its dataframe name and report success."""
    source = str(source_column).strip()
    if not source or source not in df.columns:
        return False
    arr = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
    output[output_name] = arr
    return True


def _describe_time_column(column_name: str, values: Dict[str, object]) -> str:
    """Return a human-readable description of a time column in exports."""
    name = str(column_name).strip()
    time_system = str(values.get("-TR_TIME_SYSTEM-", "")).strip()
    timestamp_ref = str(values.get("-TR_TIMESTAMP_REF-", "")).strip()
    if name == "PhotoCurve_time_corrected":
        return "BJD_TDB, mid-exposure corrected, after barycentric correction"
    if name == "PhotoCurve_time_input":
        return f"input time before BJD_TDB conversion ({time_system}, {timestamp_ref})"
    if name.startswith("PhotoCurve_compopt_time"):
        return f"comparison-star optimizer time, inherited from the source column ({time_system})"
    if name.startswith("PhotoCurve_det_time"):
        return f"photometric-detrending time, inherited from the source column ({time_system})"
    if "BJD" in name.upper() and "TDB" in name.upper():
        return "input BJD_TDB column from the photometry table"
    if "JD_UTC" in name.upper() or name.upper().startswith("JD"):
        return f"input JD_UTC-like column from the photometry table ({timestamp_ref})"
    return "current X/time column selected in the Data tab"


def build_processed_light_curve_export(
    df: pd.DataFrame,
    values: Dict[str, object],
    photometry_file_path: Optional[str],
    catalogue_path: object,
    last_transit_result,
    auto_reject_indices: set[int],
    manual_reject_indices: set[int],
    manual_keep_indices: set[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build the curated full ExoPhotoCurve light-curve export.

    The full export is intentionally not a dump of all internal ``PhotoCurve_*``
    columns.  It writes a stable, documented schema containing the original
    differential/undetrended light curve, the detrended analysis curve, the
    current/fit model, residuals and the masks used by the GUI.  This keeps the
    file useful for external checks without exposing duplicate aliases produced
    by repeated fitting.
    """
    if df is None or df.empty:
        raise ValueError("There is no table to export.")

    n_rows = len(df)
    selection = current_column_selection(values)
    x_col = str(selection.get("-XCOL-", NONE_COL))
    y_col = str(selection.get("-YCOL-", NONE_COL))
    yerr_col = str(selection.get("-YERRCOL-", NONE_COL))
    model_col = str(selection.get("-MODEL_COL-", NONE_COL))
    res_col = str(selection.get("-RES_COL-", NONE_COL))
    reserr_col = str(selection.get("-RESERR_COL-", NONE_COL))

    output: Dict[str, object] = {"row_index": np.arange(n_rows, dtype=int)}
    source_columns: Dict[str, str] = {}

    def add_numeric(output_name: str, source_column: str) -> bool:
        source = str(source_column).strip()
        if not source or source == NONE_COL or source not in df.columns:
            return False
        arr = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
        if arr.size != n_rows:
            return False
        output[output_name] = arr
        source_columns[output_name] = source
        return True

    def add_first(output_name: str, candidates: list[str]) -> str:
        for candidate in candidates:
            if add_numeric(output_name, candidate):
                return candidate
        return ""

    # Canonical time columns.  ``time`` is the current analysis/plotting time;
    # the other columns document the original input time and the BJD_TDB timing
    # correction when available.
    input_time_column_for_export = ""
    if add_numeric("time_input", "PhotoCurve_time_input"):
        input_time_column_for_export = "PhotoCurve_time_input"
    elif add_numeric("time_input", x_col):
        input_time_column_for_export = x_col

    jd_utc_column_written = False
    time_system_upper = str(values.get("-TR_TIME_SYSTEM-", "")).strip().upper()
    if time_system_upper == "JD_UTC" and input_time_column_for_export:
        jd_utc_column_written = add_numeric("JD_UTC", input_time_column_for_export)
    elif x_col in df.columns and "JD_UTC" in x_col.upper():
        jd_utc_column_written = add_numeric("JD_UTC", x_col)

    add_numeric("time_bjd_tdb", "PhotoCurve_time_corrected")
    add_numeric("time", x_col)

    # Current analysis/light-curve columns.  These generic names are kept for
    # easy plotting and for the simple ExoClock/HOPS companion export.
    add_numeric("flux", y_col)
    add_numeric("flux_error", yerr_col)

    # Non-detrended, non-fitted light curve.  Prefer the comparison-star output,
    # because it is the scientifically useful relative light curve before later
    # decorrelation or transit-model baseline fitting.  Fall back gracefully for
    # externally loaded light curves or analyses without the comparison module.
    undetrended_flux_source = add_first(
        "flux_undetrended",
        ["PhotoCurve_compopt_flux", "PhotoCurve_flux_input", y_col],
    )
    undetrended_err_source = add_first(
        "flux_undetrended_error",
        ["PhotoCurve_compopt_err", "PhotoCurve_flux_err_input", yerr_col],
    )
    add_numeric("comparison_ensemble", "PhotoCurve_compopt_ensemble")

    # Detrended/analysis curve.  Prefer the final transit-diagnostics detrended
    # curve, then the photometric-detrending output, then the current flux.
    detrended_flux_source = add_first(
        "flux_detrended",
        ["PhotoCurve_detrended_flux", "PhotoCurve_det_flux", y_col],
    )
    detrended_err_source = add_first(
        "flux_detrended_error",
        ["PhotoCurve_detrended_err", "PhotoCurve_det_err", yerr_col],
    )
    baseline_source = add_first(
        "detrending_baseline",
        ["PhotoCurve_baseline", "PhotoCurve_det_baseline"],
    )

    # Transit model and residuals.  The alias columns already respect the
    # selected display mode, while the stable output names avoid duplicate
    # PhotoCurve internal columns in the saved file.
    expected_model_source = add_first(
        "expected_model",
        ["PhotoCurve_expected_model", "PhotoCurve_expected_transit_model"],
    )
    fit_model_source = add_first(
        "fit_model",
        ["PhotoCurve_fit_model", "PhotoCurve_fit_transit_model", model_col],
    )
    residual_source = add_first(
        "residual",
        ["PhotoCurve_fit_residual", "PhotoCurve_fit_detrended_residual", res_col],
    )
    residual_err_source = add_first(
        "residual_error",
        [reserr_col, "PhotoCurve_detrended_err", "PhotoCurve_det_err", yerr_col],
    )

    # Current cleaning/manual mask.  Keep this independent of whether the user
    # applies cleaning to statistics; it documents what the GUI treats as valid.
    x = to_numeric_array(df, x_col)
    y = to_numeric_array(df, y_col)
    model = to_numeric_array(df, model_col)
    residuals = to_numeric_array(df, res_col)

    export_values = dict(values)
    export_values["-AUTO_REJECT_INDICES-"] = _format_index_set(auto_reject_indices)
    export_values["-MANUAL_REJECT_INDICES-"] = _format_index_set(manual_reject_indices)
    export_values["-MANUAL_KEEP_INDICES-"] = _format_index_set(manual_keep_indices)
    export_values["-CLEAN_ACTIVE-"] = False

    if x is not None and len(x) == n_rows:
        cleaning = compute_cleaning_mask(x, y, model, residuals, export_values)
        keep_current_mask = cleaning.keep_mask.astype(bool)
        rejected_current_mask = cleaning.rejected_mask.astype(bool)
    else:
        keep_current_mask = np.ones(n_rows, dtype=bool)
        rejected_current_mask = np.zeros(n_rows, dtype=bool)

    output["keep_current_mask"] = keep_current_mask.astype(int)
    output["rejected_current_mask"] = rejected_current_mask.astype(int)

    auto_rejected = np.zeros(n_rows, dtype=int)
    for index in auto_reject_indices:
        if 0 <= index < n_rows:
            auto_rejected[index] = 1
    output["auto_sigma_rejected"] = auto_rejected

    manual_rejected = np.zeros(n_rows, dtype=int)
    manual_restored = np.zeros(n_rows, dtype=int)
    for index in manual_reject_indices:
        if 0 <= index < n_rows:
            manual_rejected[index] = 1
    for index in manual_keep_indices:
        if 0 <= index < n_rows:
            manual_restored[index] = 1
    output["manual_rejected"] = manual_rejected
    output["manual_restored"] = manual_restored

    latest_transit_keep: Optional[np.ndarray] = None
    if last_transit_result is not None:
        try:
            transit_keep = np.asarray(last_transit_result.keep_mask, dtype=bool)
            if transit_keep.size == n_rows:
                latest_transit_keep = transit_keep
                output["keep_latest_transit_fit"] = transit_keep.astype(int)
                output["rejected_latest_transit_fit"] = (~transit_keep).astype(int)
        except Exception:
            latest_transit_keep = None

    export_keep_mask = keep_current_mask.copy()
    if latest_transit_keep is not None:
        export_keep_mask &= latest_transit_keep
    output["exported_row_mask"] = export_keep_mask.astype(int)

    exported = pd.DataFrame(output)

    # By default the full export is complete and keeps rejected points with mask
    # columns, similar to an audit table.  The existing GUI checkbox still lets
    # advanced users write only the kept points if desired.
    include_rejected_points = bool(values.get("-EXPORT_REJECTED_POINTS-", True))
    if not include_rejected_points:
        exported = exported.loc[export_keep_mask].reset_index(drop=True)

    current_time_description = _describe_time_column(x_col, values)
    input_time_description = (
        _describe_time_column("PhotoCurve_time_input", values)
        if "PhotoCurve_time_input" in df.columns
        else _describe_time_column(x_col, values)
    )
    bjd_time_description = (
        _describe_time_column("PhotoCurve_time_corrected", values)
        if "PhotoCurve_time_corrected" in df.columns
        else "not available"
    )

    metadata = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "export_schema_version": 2,
        "full_export_content": (
            "curated light-curve table: input time, analysis time, undetrended differential flux, "
            "detrended flux, model, residuals and rejection masks"
        ),
        "photometry_file": photometry_file_path or "",
        "catalogue_file": str(catalogue_path or ""),
        "planet": str(values.get("-TR_PLANET-", "")),
        "filter": str(values.get("-TR_FILTER-", "")),
        "time_system": str(values.get("-TR_TIME_SYSTEM-", "")),
        "timestamp_reference": str(values.get("-TR_TIMESTAMP_REF-", "")),
        "display_mode": str(values.get("-TR_DISPLAY_MODE-", "")),
        "x_column": x_col,
        "x_column_meaning": current_time_description,
        "time_input_meaning": input_time_description,
        "time_bjd_tdb_meaning": bjd_time_description,
        "JD_UTC_column_written": bool(jd_utc_column_written),
        "JD_UTC_meaning": (
            "original JD_UTC input time before mid-exposure and barycentric corrections"
            if jd_utc_column_written
            else "not written; original input time system was not JD_UTC or no JD_UTC-like column was available"
        ),
        "flux_column": y_col,
        "error_column": yerr_col,
        "flux_undetrended_source": undetrended_flux_source,
        "flux_undetrended_meaning": (
            "differential light curve after comparison-star correction when available; "
            "before photometric detrending and transit-model fitting"
        ),
        "flux_undetrended_error_source": undetrended_err_source,
        "flux_detrended_source": detrended_flux_source,
        "flux_detrended_meaning": "light curve after detrending, used for transit-model analysis when available",
        "flux_detrended_error_source": detrended_err_source,
        "detrending_baseline_source": baseline_source,
        "expected_model_source": expected_model_source,
        "fit_model_source": fit_model_source,
        "residual_source": residual_source,
        "residual_error_source": residual_err_source,
        "auto_sigma_rejected_indices": sorted(auto_reject_indices),
        "manual_rejected_indices": sorted(manual_reject_indices),
        "manual_restored_indices": sorted(manual_keep_indices),
        "include_rejected_points_in_full_export": include_rejected_points,
        "n_rows_input_table": int(n_rows),
        "n_rows_exported": int(len(exported)),
        "n_rows_kept_for_analysis_or_simple_export": int(np.count_nonzero(export_keep_mask)),
        "n_rows_rejected_or_not_used": int(n_rows - np.count_nonzero(export_keep_mask)),
    }

    if last_transit_result is not None:
        metadata.update(
            {
                "transit_tmid_observed_bjd_tdb": float(last_transit_result.tmid_observed),
                "transit_tmid_reference_bjd_tdb": float(last_transit_result.tmid_predicted),
                "transit_oc_minutes": float(last_transit_result.oc_minutes),
                "transit_rp_rs": float(last_transit_result.observed_rp_rs),
                "transit_depth_ppt": float(last_transit_result.observed_depth_ppt),
                "transit_residual_rms_ppt": float(last_transit_result.residual_rms_ppt),
            }
        )

    return exported, metadata


def save_processed_light_curve_file(path: str, table: pd.DataFrame, metadata: Dict[str, object]) -> None:
    """Save a processed light curve as CSV, TXT/TSV or JSON."""
    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        payload = {
            "metadata": metadata,
            "data": table.replace({np.nan: None}).to_dict(orient="records"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
    elif suffix in (".txt", ".dat", ".tsv"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# PhotoCurve Lab processed light curve export\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")
            table.to_csv(f, index=False, sep="\t")
    else:
        table.to_csv(path, index=False)


def _safe_export_token(text: object, default: str = "time") -> str:
    """Return a compact filename/header token safe for simple exports."""
    token = str(text or "").strip() or default
    token = token.replace("/", "_").replace("\\", "_").replace(" ", "_")
    token = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-", "."})
    return token or default


def build_simple_exoclock_hops_export(
    processed_table: pd.DataFrame,
    metadata: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build a compact three-column light curve for ExoClock/HOPS-style use.

    The full processed export remains the authoritative, traceable product.  This
    companion table is intentionally minimal: time, relative flux and flux error.
    It is labelled ``JD_UTC`` only when ExoPhotoCurve can identify the original
    observing time as JD_UTC.  Otherwise a simple file can still be written, but
    it is explicitly marked as not ExoClock-ready to avoid silently mixing time
    systems such as BJD_TDB and JD_UTC.
    """
    if processed_table is None or processed_table.empty:
        raise ValueError("There are no exported points available for the simple light-curve file.")
    if "flux" not in processed_table.columns:
        raise ValueError("No flux column is available in the processed export.")
    if "flux_error" not in processed_table.columns:
        raise ValueError("No flux_error column is available in the processed export.")

    time_system = str(metadata.get("time_system", "")).strip() or "unknown"
    time_system_upper = time_system.upper()
    jd_written = bool(metadata.get("JD_UTC_column_written", False))

    exoclock_ready = False
    time_source_column = ""
    output_time_column = ""
    if jd_written and "JD_UTC" in processed_table.columns:
        time_source_column = "JD_UTC"
        output_time_column = "JD_UTC"
        exoclock_ready = True
    elif "time_input" in processed_table.columns:
        time_source_column = "time_input"
        output_time_column = _safe_export_token(time_system_upper, "time")
    elif "time" in processed_table.columns:
        time_source_column = "time"
        output_time_column = _safe_export_token(time_system_upper, "time")
    else:
        raise ValueError("No usable time column is available in the processed export.")

    time_values = pd.to_numeric(processed_table[time_source_column], errors="coerce").to_numpy(dtype=float)
    flux_values = pd.to_numeric(processed_table["flux"], errors="coerce").to_numpy(dtype=float)
    err_values = pd.to_numeric(processed_table["flux_error"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(time_values) & np.isfinite(flux_values) & np.isfinite(err_values) & (err_values > 0)
    if "exported_row_mask" in processed_table.columns:
        # If the full export was requested with rejected points included, the
        # simple external-use file still remains clean by default.
        exported_mask = pd.to_numeric(processed_table["exported_row_mask"], errors="coerce").to_numpy(dtype=float)
        valid &= exported_mask > 0.5

    if np.count_nonzero(valid) == 0:
        raise ValueError("No finite points with positive flux_error are available for the simple light-curve file.")

    simple_table = pd.DataFrame(
        {
            output_time_column: time_values[valid],
            "flux": flux_values[valid],
            "flux_error": err_values[valid],
        }
    )

    simple_metadata = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_product": "ExoPhotoCurve processed light curve export",
        "target": str(metadata.get("planet", "")),
        "filter": str(metadata.get("filter", "")),
        "time_system": time_system,
        "time_source_column": time_source_column,
        "time_output_column": output_time_column,
        "flux_source_column": str(metadata.get("flux_column", "flux")),
        "error_source_column": str(metadata.get("error_column", "flux_error")),
        "exoclock_hops_ready": bool(exoclock_ready),
        "warning": "" if exoclock_ready else (
            "Time was not identified as JD_UTC. The file is a simple flux table, "
            "but it is not labelled as ExoClock JD_UTC-ready."
        ),
        "rejected_points_excluded": True,
        "finite_positive_error_filter_applied": True,
        "n_rows_processed_export": int(len(processed_table)),
        "n_rows_simple_export": int(len(simple_table)),
        "n_rows_removed_for_simple_export": int(len(processed_table) - len(simple_table)),
    }
    return simple_table, simple_metadata


def simple_exoclock_hops_export_path(processed_path: str, simple_metadata: Dict[str, object]) -> str:
    """Return the companion path for the compact external-use light curve."""
    path = Path(processed_path)
    if bool(simple_metadata.get("exoclock_hops_ready", False)):
        suffix = "_ExoClock_HOPS.txt"
    else:
        time_token = _safe_export_token(simple_metadata.get("time_output_column", "time"), "time")
        suffix = f"_simple_{time_token}_flux.txt"
    return str(path.with_name(path.stem + suffix))


def save_simple_exoclock_hops_curve_file(path: str, table: pd.DataFrame, metadata: Dict[str, object]) -> None:
    """Save the compact external-use light curve as a commented TSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ExoPhotoCurve simple light curve export\n")
        f.write("# Intended use: quick import into ExoClock/HOPS-style workflows when time is JD_UTC.\n")
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")
        # ExoClock-style uploads are safest when every non-numeric line is a
        # comment.  Therefore the column-name line is written explicitly with
        # a leading # and pandas writes only numeric data rows below it.
        f.write("# " + "\t".join(str(col) for col in table.columns) + "\n")
        table.to_csv(f, index=False, header=False, sep="\t", float_format="%.10f")


def _yes_no(value: object) -> str:
    """Return a compact yes/no string for reports."""
    return "yes" if bool(value) else "no"


def _compact_list(items: object, empty: str = "none", max_items: int = 80) -> str:
    """Return a readable list while avoiding extremely long single lines."""
    if items is None:
        return empty
    try:
        values = list(items)
    except Exception:
        values = [items]
    values = [str(item) for item in values if str(item).strip()]
    if not values:
        return empty
    if len(values) > max_items:
        shown = ", ".join(values[:max_items])
        return f"{shown}, ... ({len(values)} total)"
    return ", ".join(values)


def _selection_lines(title: str, selection: Optional[Dict[str, str]]) -> List[str]:
    """Return formatted column-selection lines for a recipe report."""
    lines = [title]
    if not selection:
        lines.append("  none recorded")
        return lines
    labels = {
        "-XCOL-": "time column",
        "-YCOL-": "flux column",
        "-YERRCOL-": "error column",
        "-MODEL_COL-": "model column",
        "-RES_COL-": "residual column",
        "-RESERR_COL-": "residual error column",
    }
    for key, label in labels.items():
        lines.append(f"  {label}: {selection.get(key, NONE_COL)}")
    return lines


def _metric_summary(metric) -> str:
    """Return a compact summary for a comparison-star quality metric."""
    if metric is None:
        return "not available"
    try:
        return (
            f"RMS={metric.rms_ppt:.3f} ppt, MAD={metric.mad_ppt:.3f} ppt, "
            f"beta={metric.beta_factor:.3f}, lag1={metric.autocorr_lag1:.3f}, N={metric.n_points}"
        )
    except Exception:
        return "not available"


def build_reproducibility_recipe(
    df: Optional[pd.DataFrame],
    values: Dict[str, object],
    photometry_file_path: Optional[str],
    original_column_selection: Optional[Dict[str, str]],
    pre_transit_column_selection: Optional[Dict[str, str]],
    aij_flux_detection: Optional[AijFluxDetection],
    comp_active_stars: set[str],
    last_comp_result: Optional[ComparisonOptimisationResult],
    detrend_detection: Optional[DetrendRegressorDetection],
    detrend_active_regressors: set[str],
    last_detrend_result: Optional[PhotometricDetrendResult],
    last_detrend_input_selection: Optional[Dict[str, str]],
    auto_reject_indices: set[int],
    manual_reject_indices: set[int],
    manual_keep_indices: set[int],
    last_transit_result,
) -> str:
    """Build a complete, human-readable recipe for reproducing an analysis."""
    current_selection = current_column_selection(values)
    lines: List[str] = []
    lines.append("PhotoCurve Lab - Reproducibility recipe")
    lines.append("=" * 43)
    lines.append("")

    lines.append("Input files and table")
    lines.append(f"Photometry file: {photometry_file_path or 'not recorded'}")
    lines.append(f"Catalogue file: {values.get('-TR_CATALOG-', '')}")
    lines.append(f"Delimiter: {values.get('-DELIM-', 'Auto')}")
    lines.append(f"Header row: {_yes_no(values.get('-HEADER-', True))}")
    if df is not None:
        lines.append(f"Rows: {len(df)}")
        lines.append(f"Columns: {len(df.columns)}")
    lines.append("")

    lines.extend(_selection_lines("Original column selection after loading", original_column_selection))
    lines.append("")
    lines.extend(_selection_lines("Current Data-tab column selection", current_selection))
    lines.append("")
    lines.extend(_selection_lines("Input columns used by the latest transit fit", pre_transit_column_selection))
    lines.append("")

    lines.append("Comparison-star stage")
    if aij_flux_detection is None:
        lines.append("Raw flux columns detected: not checked")
    else:
        lines.append(f"Raw flux columns detected: {_yes_no(aij_flux_detection.compatible)}")
        if aij_flux_detection.compatible:
            lines.append(f"Detected targets: {_compact_list(aij_flux_detection.target_ids)}")
            lines.append(f"Detected comparison stars: {_compact_list(aij_flux_detection.comparison_ids)}")
            lines.append(f"Comparison time column: {aij_flux_detection.time_column or 'not found'}")
        elif aij_flux_detection.warning:
            lines.append(f"Detection warning: {aij_flux_detection.warning}")
    lines.append(f"Comp-star target: {values.get('-COMP_TARGET-', '')}")
    lines.append(f"Comp-star mode: {values.get('-COMP_MODE-', '')}")
    lines.append(f"Check star: {values.get('-COMP_CHECK-', '') or 'none'}")
    lines.append(f"Mask expected transit during comp-star optimisation: {_yes_no(values.get('-COMP_MASK_TRANSIT-', True))}")
    lines.append(f"Comp-star polynomial order: {values.get('-COMP_POLY_ORDER-', '')}")
    lines.append(f"Manual/current active comparison stars: {_compact_list(sorted(comp_active_stars))}")
    if last_comp_result is not None:
        lines.append(f"Automatic optimizer was run: yes")
        lines.append(f"Optimizer initial comparison stars: {_compact_list(last_comp_result.initial_comparisons)}")
        lines.append(f"Optimizer rejected comparison stars: {_compact_list(last_comp_result.rejected_comparisons)}")
        lines.append(f"Optimizer removed sequence: {_compact_list(last_comp_result.removed_sequence)}")
        lines.append(f"Optimizer selected comparison stars: {_compact_list(last_comp_result.selected_comparisons)}")
        lines.append(f"All-comparison metric: {_metric_summary(last_comp_result.all_comparisons_metric)}")
        lines.append(f"Optimized metric: {_metric_summary(last_comp_result.optimised_metric)}")
        if last_comp_result.improvement_vs_current_percent is not None:
            lines.append(f"Improvement vs current input: {last_comp_result.improvement_vs_current_percent:+.2f} %")
        lines.append(f"Improvement vs all accepted comparisons: {last_comp_result.improvement_vs_all_percent:+.2f} %")
    else:
        lines.append("Automatic optimizer was run: no")
    lines.append("")

    lines.append("Cleaning and manual point editing")
    lines.append(f"Auto sigma clipping applied: {_yes_no(bool(auto_reject_indices))}")
    lines.append(f"Auto sigma clipping target: {values.get('-CLEAN_TARGET-', '')}")
    lines.append(f"Sigma threshold: {values.get('-CLEAN_SIGMA-', '')}")
    lines.append(f"Sigma max iterations per Apply: {values.get('-CLEAN_MAXITER-', '')}")
    lines.append(f"Sigma center: {values.get('-CLEAN_CENTRE-', '')}")
    lines.append(f"Sigma scale: {values.get('-CLEAN_SCALE-', '')}")
    lines.append(f"Auto sigma-clipped point indices: {_compact_list(sorted(auto_reject_indices))}")
    lines.append(f"Manual rejects enabled: {_yes_no(values.get('-MANUAL_CLEAN_ACTIVE-', True))}")
    lines.append(f"Manual rejected point indices: {_compact_list(sorted(manual_reject_indices))}")
    lines.append(f"Manual restored/kept point indices: {_compact_list(sorted(manual_keep_indices))}")
    if last_transit_result is not None:
        try:
            rejected_by_final_mask = int(last_transit_result.keep_mask.size - np.count_nonzero(last_transit_result.keep_mask))
            lines.append(f"Points excluded from latest transit fit: {rejected_by_final_mask}")
        except Exception:
            pass
    lines.append("")

    lines.append("Photometric detrending stage")
    if detrend_detection is None:
        lines.append("Detrending regressors detected: not checked")
    else:
        lines.append(f"Detrending regressors detected: {_yes_no(detrend_detection.compatible)}")
        if detrend_detection.compatible:
            lines.append(f"Detected regressor candidates: {_compact_list(detrend_detection.columns)}")
            lines.append(f"Suggested regressors: {_compact_list(detrend_detection.suggested)}")
        elif detrend_detection.warning:
            lines.append(f"Detection warning: {detrend_detection.warning}")
    lines.append(f"Active detrending regressors in GUI: {_compact_list(sorted(detrend_active_regressors))}")
    lines.append(f"Mask expected transit during detrending: {_yes_no(values.get('-DET_MASK_TRANSIT-', True))}")
    lines.append(f"Use cleaning mask during detrending: {_yes_no(values.get('-DET_USE_CLEANING_MASK-', True))}")
    lines.append(f"Consider transit fit model during detrending: {_yes_no(values.get('-DET_USE_TRANSIT_MODEL-', False))}")
    lines.append(f"Detrending polynomial order: {values.get('-DET_POLY_ORDER-', '')}")
    lines.append(f"Detrending robust sigma: {values.get('-DET_ROBUST_SIGMA-', '')}")
    lines.append(f"Detrending robust iterations: {values.get('-DET_ROBUST_ITER-', '')}")
    lines.append(f"Meridian flip detrending enabled: {_yes_no(values.get('-DET_FLIP_ACTIVE-', False))}")
    lines.append(f"Meridian flip time fraction: {values.get('-DET_FLIP_FRAC-', '')}")
    lines.append(f"Meridian flip mode: {values.get('-DET_FLIP_MODE-', '')}")
    lines.append(f"Show meridian-flip marker: {_yes_no(values.get('-DET_SHOW_FLIP_MARKER-', True))}")
    if last_detrend_result is not None:
        lines.append("Photometric detrending was run: yes")
        lines.append(f"Regressors used: {_compact_list(last_detrend_result.selected_regressors)}")
        lines.append(f"Detrending fit points used: {int(np.count_nonzero(last_detrend_result.fit_mask))} / {last_detrend_result.fit_mask.size}")
        lines.append(f"Detrending robust keep points: {int(np.count_nonzero(last_detrend_result.robust_keep_mask))} / {last_detrend_result.robust_keep_mask.size}")
        lines.append(f"RMS before detrending: {last_detrend_result.rms_before_ppt:.3f} ppt")
        lines.append(f"RMS after detrending: {last_detrend_result.rms_after_ppt:.3f} ppt")
        lines.append(f"Detrending improvement: {last_detrend_result.improvement_percent:+.2f} %")
        lines.append(f"Transit fit model used by detrending: {_yes_no(getattr(last_detrend_result, 'transit_model_used', False))}")
        if getattr(last_detrend_result, 'transit_model_used', False):
            lines.append(f"Model-aware detrending mode: {getattr(last_detrend_result, 'model_aware_mode', 'Single pass')}")
            lines.append(f"Model-aware iterations: {getattr(last_detrend_result, 'model_aware_iterations', 1)}")
            lines.append(f"Model-aware converged: {_yes_no(getattr(last_detrend_result, 'model_aware_converged', False))}")
            lines.append(f"Model-aware stop reason: {getattr(last_detrend_result, 'model_aware_stop_reason', '')}")
            try:
                tol_pct = float(getattr(last_detrend_result, 'model_aware_tolerance_percent', float('nan')))
                if np.isfinite(tol_pct):
                    lines.append(f"Model-aware tolerance: {tol_pct:.3f} %")
            except Exception:
                pass
            history = getattr(last_detrend_result, 'model_aware_history', []) or []
            if history:
                lines.append("Model-aware iteration history:")
                for row in history:
                    try:
                        lines.append(
                            f"  iter {int(row.get('iteration', 0))}: "
                            f"dTmid={float(row.get('tmid_delta_min', np.nan)):.4f} min, "
                            f"dRp/Rs={float(row.get('rp_rs_delta_pct', np.nan)):.4f} %, "
                            f"dRMS={float(row.get('rms_delta_pct', np.nan)):.4f} %, "
                            f"dBaseline={float(row.get('baseline_delta_ppt', np.nan)):.4f} ppt"
                        )
                    except Exception:
                        pass
        if getattr(last_detrend_result, 'meridian_flip_enabled', False):
            lines.append(f"Meridian flip full time used: {last_detrend_result.meridian_flip_time:.8f}")
            lines.append(f"Meridian flip model used: {last_detrend_result.meridian_flip_mode}")
        if last_detrend_input_selection:
            lines.extend(_selection_lines("Detrending input columns", last_detrend_input_selection))
        if last_detrend_result.coefficients:
            lines.append("Detrending baseline coefficients:")
            for name, coefficient in last_detrend_result.coefficients.items():
                try:
                    lines.append(f"  {name}: {float(coefficient):+.8e}")
                except Exception:
                    lines.append(f"  {name}: {coefficient}")
    else:
        lines.append("Photometric detrending was run: no")
    lines.append("")

    lines.append("Transit diagnostic settings")
    lines.append(f"Planet: {values.get('-TR_PLANET-', '')}")
    lines.append(f"Catalogue source/path: {values.get('-TR_CATALOG-', '')}")
    lines.append(f"Filter: {values.get('-TR_FILTER-', '')}")
    lines.append(f"Exposure time: {values.get('-TR_EXPTIME-', '')} s")
    lines.append(f"Input time system: {values.get('-TR_TIME_SYSTEM-', '')}")
    lines.append(f"Timestamp reference: {values.get('-TR_TIMESTAMP_REF-', '')}")
    lines.append(f"Observatory latitude: {values.get('-TR_OBS_LAT-', '')}")
    lines.append(f"Observatory longitude: {values.get('-TR_OBS_LON-', '')}")
    lines.append(f"Observatory altitude: {values.get('-TR_OBS_ALT-', '')} m")
    lines.append(f"Tmid reference override: {values.get('-TR_TMID_OVERRIDE-', '') or 'none'}")
    lines.append(f"Transit baseline model: {values.get('-TR_BASELINE-', '')}")
    lines.append(f"Transit model engine: {values.get('-TR_MODEL_ENGINE-', '')}")
    lines.append(f"Display mode: {values.get('-TR_DISPLAY_MODE-', '')}")
    lines.append(f"Fit Tmid: {_yes_no(values.get('-TR_FIT_TMID-', True))}")
    lines.append(f"Fit depth: {_yes_no(values.get('-TR_FIT_DEPTH-', True))}")
    lines.append(f"Fit duration: {_yes_no(values.get('-TR_FIT_DURATION-', True))}")
    lines.append(f"Show transit fit on plot: {_yes_no(values.get('-TR_SET_MODEL_COLUMNS-', True))}")
    lines.append(f"Show predicted-time labels: {_yes_no(values.get('-TR_SHOW_PREDICTED_TIMES-', False))}")
    lines.append(f"Show calculated-time labels: {_yes_no(values.get('-TR_SHOW_CALCULATED_TIMES-', False))}")
    lines.append("")

    lines.append("Plot and visual processing")
    lines.append(f"Plot layout: {values.get('-PLOT_LAYOUT-', '')}")
    lines.append(f"X mode: {values.get('-XMODE-', '')}")
    lines.append(f"X offset/T0: {values.get('-XOFFSET-', '')}")
    lines.append(f"Binning enabled: {_yes_no(values.get('-BIN_ACTIVE-', False))}")
    lines.append(f"Binning points/bin: {values.get('-BIN_N-', '')}")
    lines.append(f"Grid: {_yes_no(values.get('-GRID-', True))}")
    lines.append(f"Legend: {_yes_no(values.get('-LEGEND-', True))}")
    lines.append("")

    if last_transit_result is not None:
        lines.append("Latest transit diagnostic summary")
        lines.append(f"Observed Tmid: {last_transit_result.tmid_observed:.8f} BJD_TDB")
        lines.append(f"Reference Tmid: {last_transit_result.tmid_predicted:.8f} BJD_TDB")
        lines.append(f"O-C: {last_transit_result.oc_minutes:+.3f} min")
        lines.append(f"Rp/Rs: {last_transit_result.observed_rp_rs:.5f} expected {last_transit_result.expected_rp_rs:.5f}")
        lines.append(f"Depth: {last_transit_result.observed_depth_ppt:.3f} ppt expected model {last_transit_result.expected_depth_ppt:.3f} ppt")
        lines.append(f"Duration: {last_transit_result.observed_duration_hours:.4f} h expected {last_transit_result.expected_duration_hours:.4f} h")
        lines.append(f"Residual RMS: {last_transit_result.residual_rms_ppt:.3f} ppt")
    else:
        lines.append("Latest transit diagnostic summary")
        lines.append("No transit diagnostic has been run in the current session.")

    return "\n".join(lines)


def load_catalogue_into_window(
    window: sg.Window,
    path: str,
    current_planet: str = "",
    autodetect_text: str = "",
    source_label: str = "",
):
    """Load an exoplanet catalogue and update the planet combo box.

    The selected planet is chosen conservatively:

    1. keep the currently selected planet if the new catalogue contains it;
    2. otherwise try to auto-detect the planet from the loaded photometry file;
    3. otherwise fall back to the first catalogue row.

    This prevents quick catalogue switches, for example NASA -> ExoClock, from
    losing the target that was already identified from the filename.
    """
    catalogue = load_exoplanet_catalogue(path)
    names = planet_names(catalogue)
    name_set = set(names)

    selected = ""
    selection_reason = ""

    if current_planet and current_planet in name_set:
        selected = current_planet
        selection_reason = f"kept selected planet: {selected}"
    elif current_planet:
        # NASA and ExoClock can write the same planet with slightly different
        # display names, e.g. ``WASP-15 b`` versus ``WASP-15b``.  Treat the
        # current selection as a text query before falling back to the filename.
        matched_current = guess_planet_from_text(catalogue, current_planet)
        if matched_current:
            selected = matched_current
            selection_reason = f"matched selected planet in new catalogue: {selected}"

    if not selected and autodetect_text:
        guessed_planet = guess_planet_from_text(catalogue, autodetect_text)
        if guessed_planet:
            selected = guessed_planet
            selection_reason = f"auto-selected planet: {selected}"

    if not selected and names:
        selected = names[0]
        selection_reason = f"selected first catalogue entry: {selected}"

    window["-TR_PLANET-"].update(values=names, value=selected)

    label = f" {source_label}" if source_label else ""
    if selection_reason:
        window["-STATUS-"].update(f"Loaded{label} transit catalogue: {len(names)} planets | {selection_reason}")
    else:
        window["-STATUS-"].update(f"Loaded{label} transit catalogue: {len(names)} planets")

    return catalogue


def autodetect_planet_in_window(window: sg.Window, catalogue, photometry_file_path: Optional[str]) -> Optional[str]:
    """Auto-select a planet from the current photometry filename, if possible."""
    if catalogue is None:
        window["-STATUS-"].update("No transit catalogue is loaded.")
        return None

    if not photometry_file_path:
        window["-STATUS-"].update("Load a photometry file before auto-detecting the planet.")
        return None

    guessed_planet = guess_planet_from_text(catalogue, Path(photometry_file_path).name)
    if guessed_planet:
        window["-TR_PLANET-"].update(value=guessed_planet)
        window["-STATUS-"].update(
            f"Auto-selected planet from filename: {guessed_planet}"
        )
        return guessed_planet

    window["-STATUS-"].update(
        f"No planet match found in filename: {Path(photometry_file_path).name}"
    )
    return None



def _format_index_set(indices: set[int]) -> str:
    """Return a compact comma-separated representation of point indices."""
    return ",".join(str(index) for index in sorted(indices))


def _parse_index_set_text(text: object) -> set[int]:
    """Parse a comma-separated index list stored in a hidden GUI input."""
    if text is None:
        return set()
    indices: set[int] = set()
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(token)
        except Exception:
            continue
        if index >= 0:
            indices.add(index)
    return indices


def update_manual_point_controls(window: sg.Window, reject_indices: set[int], keep_indices: set[int]) -> None:
    """Synchronise the hidden/manual point-editing GUI fields."""
    try:
        window["-MANUAL_REJECT_INDICES-"].update(_format_index_set(reject_indices))
        window["-MANUAL_KEEP_INDICES-"].update(_format_index_set(keep_indices))
        window["-MANUAL_REJECT_COUNT-"].update(f"reject {len(reject_indices)}")
        window["-MANUAL_KEEP_COUNT-"].update(f"keep {len(keep_indices)}")
    except Exception:
        pass


def update_auto_clip_controls(window: sg.Window, reject_indices: set[int]) -> None:
    """Synchronise the persistent automatic sigma-clipping GUI fields."""
    try:
        window["-AUTO_REJECT_INDICES-"].update(_format_index_set(reject_indices))
        window["-AUTO_REJECT_COUNT-"].update(f"auto {len(reject_indices)}")
        # The old dynamic checkbox is intentionally kept disabled in the new
        # workflow. The actual auto mask is stored in -AUTO_REJECT_INDICES-.
        if "-CLEAN_ACTIVE-" in window.AllKeysDict:
            window["-CLEAN_ACTIVE-"].update(value=False)
    except Exception:
        pass


def _with_current_masks(
    values: Dict[str, object],
    auto_reject_indices: set[int],
    manual_reject_indices: set[int],
    manual_keep_indices: set[int],
) -> Dict[str, object]:
    """Return a values copy with the in-memory point masks injected."""
    output = dict(values)
    output["-AUTO_REJECT_INDICES-"] = _format_index_set(auto_reject_indices)
    output["-MANUAL_REJECT_INDICES-"] = _format_index_set(manual_reject_indices)
    output["-MANUAL_KEEP_INDICES-"] = _format_index_set(manual_keep_indices)
    # Prevent legacy dynamic sigma clipping from changing points on each redraw/refit.
    output["-CLEAN_ACTIVE-"] = False
    return output


def connect_plot_click(figure_canvas_agg: Optional[FigureCanvasTkAgg], window: sg.Window) -> None:
    """Forward matplotlib click events to the GUI event loop."""
    if figure_canvas_agg is None:
        return

    def _on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        window.write_event_value(
            "-PLOT_POINT_CLICK-",
            {
                "xdata": float(event.xdata),
                "ydata": float(event.ydata),
                "button": int(event.button) if event.button is not None else 1,
            },
        )

    figure_canvas_agg.mpl_connect("button_press_event", _on_click)


def redraw_plot(window: sg.Window, old_fig_agg, df: pd.DataFrame, values: Dict[str, object]):
    """Redraw the plot and reconnect interactive point editing."""
    delete_figure_agg(old_fig_agg)
    fig = build_plot(df, values)
    fig_agg = draw_figure(window["-CANVAS-"], fig)
    connect_plot_click(fig_agg, window)
    return fig, fig_agg


def _parse_comp_star_from_display(item: str) -> str:
    """Extract an AIJ comparison-star identifier from the listbox display text."""
    text = str(item).strip()
    for prefix in ("[x]", "[ ]", "☑", "☐"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.split()[0].strip()


def _comparison_star_list_values(detection: AijFluxDetection, active_stars: set[str]) -> List[str]:
    """Return listbox labels for active/inactive comparison stars."""
    labels: List[str] = []
    for star_id in detection.comparison_ids:
        marker = "[x]" if star_id in active_stars else "[ ]"
        labels.append(f"{marker} {star_id}")
    return labels


def update_comparison_star_list(
    window: sg.Window,
    detection: Optional[AijFluxDetection],
    active_stars: set[str],
    disabled: bool = False,
) -> None:
    """Update the manual comparison-star listbox.

    Some PySimpleGUI/FreeSimpleGUI Listbox implementations do not accept the
    ``value`` keyword in ``update``.  Passing it can silently leave the list
    visually empty if the exception is caught by the GUI event loop.  Update
    only the supported fields and clear the Tk selection separately when the
    underlying widget is available.
    """
    try:
        if detection is None or not detection.compatible:
            window["-COMP_STAR_LIST-"].update(values=[], disabled=True)
            return
        labels = _comparison_star_list_values(detection, active_stars)
        element = window["-COMP_STAR_LIST-"]

        # Preserve the user's scroll position.  Rebuilding the listbox after a
        # toggle is necessary because the [x]/[ ] labels change, but Tk resets
        # the view to the first row unless we explicitly restore it.
        first_visible_fraction = 0.0
        try:
            first_visible_fraction = float(element.Widget.yview()[0])
        except Exception:
            first_visible_fraction = 0.0

        element.update(values=labels, disabled=disabled)

        try:
            element.Widget.yview_moveto(first_visible_fraction)
        except Exception:
            pass

        # Clear the selection after every refresh so the next click is a clean
        # toggle operation and repeated clicks on the same item still generate
        # a clear state change.
        try:
            element.Widget.selection_clear(0, "end")
        except Exception:
            pass
    except Exception:
        pass


def _get_comp_x_column(values: Dict[str, object], detection: AijFluxDetection, df: pd.DataFrame) -> str:
    """Return the original table time column for comparison-star operations.

    The comparison-star tools rebuild a new differential light curve from the
    raw AstroImageJ flux columns.  Therefore they must use a real time column
    from the imported table, not a generated PhotoCurve diagnostic column such
    as ``PhotoCurve_time_corrected``.  Otherwise, if the user changes the
    comparison-star subset after a transit fit, the already BJD-corrected time
    can be treated again as JD_UTC and shifted a second time during the next
    transit diagnostic run.
    """
    x_col = str(values.get("-XCOL-", NONE_COL))

    # Generated PhotoCurve columns are downstream products.  They are useful for
    # plotting, but are not valid inputs when reconstructing an AIJ comparison
    # ensemble from raw Source-Sky_* fluxes.
    if (
        x_col == NONE_COL
        or x_col not in df.columns
        or x_col.startswith("PhotoCurve_")
    ):
        x_col = detection.time_column

    if not x_col or x_col not in df.columns:
        raise ValueError("Select a valid original X/time column before using comparison-star tools.")

    x_values = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    if np.count_nonzero(np.isfinite(x_values)) == 0:
        raise ValueError(f"The comparison-star time column {x_col!r} contains no valid numeric values.")
    return x_col


def run_manual_comparison_selection(
    window: sg.Window,
    df: pd.DataFrame,
    detection: AijFluxDetection,
    active_stars: set[str],
    values: Dict[str, object],
    exoplanet_catalogue,
):
    """Generate and plot a light curve from the current manual comparison subset."""
    if not detection.compatible:
        raise ValueError(detection.warning or "The loaded table is not compatible with comparison-star selection.")
    if not active_stars:
        raise ValueError("Select at least one comparison star.")

    x_col = _get_comp_x_column(values, detection, df)
    x = to_numeric_array(df, x_col)
    if x is None:
        raise ValueError("The selected X/time column is not numeric.")
    current_flux = to_numeric_array(df, str(values.get("-YCOL-", NONE_COL)))

    planet = None
    if exoplanet_catalogue is not None and str(values.get("-TR_PLANET-", "")).strip():
        try:
            planet = find_planet(exoplanet_catalogue, str(values.get("-TR_PLANET-", "")))
        except Exception:
            planet = None

    polynomial_order = max(0, min(2, parse_int(values.get("-COMP_POLY_ORDER-", 1), 1)))
    result = build_manual_comparison_result(
        df,
        detection,
        x,
        x_col,
        current_flux,
        target_id=str(values.get("-COMP_TARGET-", "")).strip(),
        selected_comparisons=sorted(active_stars),
        mode=str(values.get("-COMP_MODE-", "Target light curve")),
        check_id=str(values.get("-COMP_CHECK-", "")).strip(),
        planet=planet,
        mask_expected_transit=bool(values.get("-COMP_MASK_TRANSIT-", True)),
        polynomial_order=polynomial_order,
    )
    return result


def nearest_point_index_for_click(
    df: pd.DataFrame,
    values: Dict[str, object],
    click_x: float,
    click_y: float,
) -> Optional[int]:
    """Find the nearest plotted light-curve point to a mouse click."""
    x_col = str(values.get("-XCOL-", NONE_COL))
    y_col = str(values.get("-YCOL-", NONE_COL))
    x = to_numeric_array(df, x_col)
    y = to_numeric_array(df, y_col)
    if x is None or y is None:
        return None

    x_plot, _, _ = transform_x_axis(x, values, x_col, str(values.get("-XLABEL-", "")))
    y_plot = np.asarray(y, dtype=float).copy()
    lc_offset = parse_float(values.get("-LC_OFFSET-", 0.0), 0.0) or 0.0
    y_plot = y_plot + lc_offset

    finite = np.isfinite(x_plot) & np.isfinite(y_plot)
    if np.count_nonzero(finite) == 0:
        return None

    x_range = float(np.nanmax(x_plot[finite]) - np.nanmin(x_plot[finite]))
    y_range = float(np.nanmax(y_plot[finite]) - np.nanmin(y_plot[finite]))
    if not np.isfinite(x_range) or x_range <= 0:
        x_range = 1.0
    if not np.isfinite(y_range) or y_range <= 0:
        y_range = 1.0

    dx = (x_plot - click_x) / x_range
    dy = (y_plot - click_y) / y_range
    dist2 = dx * dx + dy * dy
    dist2[~finite] = np.inf
    index = int(np.nanargmin(dist2))
    if not np.isfinite(dist2[index]):
        return None

    # Avoid surprising edits when the user clicks far from all data points.
    if dist2[index] > 0.050 ** 2:
        return None
    return index


def update_comparison_tab(window: sg.Window, detection: Optional[AijFluxDetection], active_stars: Optional[set[str]] = None) -> None:
    """Enable or disable the comparison-star optimizer controls."""
    control_keys = [
        "-COMP_TARGET-",
        "-COMP_CHECK-",
        "-COMP_MODE-",
        "-COMP_MASK_TRANSIT-",
        "-COMP_POLY_ORDER-",
        "-COMP_MIN_STARS-",
        "-COMP_MAX_STARS-",
        "-COMP_IMPROVE_THRESHOLD-",
        "-COMP_SEND_TO_DATA-",
        "-COMP_SHOW_POPUP-",
        "-COMP_STAR_LIST-",
        "-COMP_DIAG_STAR-",
        "-COMP_PLOT_DIAG-",
        "-COMP_SELECT_ALL-",
        "-COMP_SELECT_NONE-",
        "Run comp optimizer",
    ]

    if detection is None or not detection.compatible:
        warning = detection.warning if detection is not None else (
            "Comparison-star optimizer inactive: load an AstroImageJ table with Source-Sky_T*/C* flux columns."
        )
        try:
            window["-COMP_STATUS-"].update(warning, text_color="firebrick")
            window["-COMP_TARGET-"].update(values=[], value="")
            window["-COMP_CHECK-"].update(values=[""], value="")
            window["-COMP_DIAG_STAR-"].update(values=[], value="")
            window["-COMP_PLOT_DIAG-"].update(value=False)
            window["-COMP_REPORT-"].update(warning)
            update_comparison_star_list(window, detection, set(), disabled=True)
            for key in control_keys:
                window[key].update(disabled=True)
        except Exception:
            pass
        return

    try:
        target_value = detection.target_ids[0] if detection.target_ids else ""
        window["-COMP_STATUS-"].update(
            f"Raw fluxes detected: {len(detection.target_ids)} target-like star(s), "
            f"{len(detection.comparison_ids)} comparison star(s). Optimizer active.",
            text_color="darkgreen",
        )
        window["-COMP_TARGET-"].update(values=detection.target_ids, value=target_value, disabled=False)
        window["-COMP_CHECK-"].update(values=[""] + detection.comparison_ids, value="", disabled=False)
        if active_stars is None:
            active_stars = set(detection.comparison_ids)
        update_comparison_star_list(window, detection, active_stars, disabled=False)
        update_comparison_diagnostic_controls(window, {}, "")
        for key in control_keys:
            window[key].update(disabled=False)
        window["-COMP_DIAG_STAR-"].update(disabled=True)
        window["-COMP_PLOT_DIAG-"].update(disabled=True)
        window["-COMP_REPORT-"].update(
            "Ready. Choose a target/check star and run the comparison-star optimizer.\n"
            "For exoplanet work, Target light curve mode masks the expected transit when a catalogue ephemeris is available."
        )
    except Exception:
        pass


def add_comparison_optimisation_columns(
    df: pd.DataFrame,
    result: ComparisonOptimisationResult,
) -> pd.DataFrame:
    """Add optimizer output columns without removing existing data."""
    comp_columns = [
        "PhotoCurve_compopt_time",
        "PhotoCurve_compopt_flux",
        "PhotoCurve_compopt_err",
        "PhotoCurve_compopt_ensemble",
    ]
    base = df.drop(columns=comp_columns, errors="ignore")
    output = pd.DataFrame(
        {
            "PhotoCurve_compopt_time": np.asarray(result.x, dtype=float),
            "PhotoCurve_compopt_flux": np.asarray(result.optimised_flux, dtype=float),
            "PhotoCurve_compopt_err": np.asarray(result.optimised_flux_err, dtype=float),
            "PhotoCurve_compopt_ensemble": np.asarray(result.comparison_ensemble, dtype=float),
        },
        index=base.index,
    )
    return pd.concat([base, output], axis=1).copy()


def set_comparison_output_columns(window: sg.Window, values: Dict[str, object]) -> Dict[str, object]:
    """Select the optimised light curve in the Data tab and return plot values."""
    selection = {
        "-XCOL-": "PhotoCurve_compopt_time",
        "-YCOL-": "PhotoCurve_compopt_flux",
        "-YERRCOL-": "PhotoCurve_compopt_err",
        "-MODEL_COL-": NONE_COL,
        "-RES_COL-": NONE_COL,
        "-RESERR_COL-": NONE_COL,
        "-YLABEL-": "Optimised relative flux",
    }
    for key, value in selection.items():
        if key.startswith("-") and key.endswith("-") and key != "-YLABEL-":
            try:
                window[key].update(value=value)
            except Exception:
                pass
    plot_values = dict(values)
    plot_values.update(selection)
    return plot_values




def _comp_diag_flux_column(star_id: str) -> str:
    """Return the generated flux column name for one comparison diagnostic."""
    safe_id = str(star_id).strip().replace(" ", "_")
    return f"PhotoCurve_compdiag_{safe_id}_flux"


def _comp_diag_err_column(star_id: str) -> str:
    """Return the generated error column name for one comparison diagnostic."""
    safe_id = str(star_id).strip().replace(" ", "_")
    return f"PhotoCurve_compdiag_{safe_id}_err"


def _comp_diag_ensemble_column(star_id: str) -> str:
    """Return the generated ensemble column name for one comparison diagnostic."""
    safe_id = str(star_id).strip().replace(" ", "_")
    return f"PhotoCurve_compdiag_{safe_id}_ensemble"


def add_comparison_diagnostic_columns(
    df: pd.DataFrame,
    x: np.ndarray,
    diagnostics: Dict[str, ComparisonDiagnosticCurve],
) -> pd.DataFrame:
    """Add leave-one-out comparison-star diagnostic columns to the table."""
    remove_columns = [
        column for column in df.columns
        if str(column) == "PhotoCurve_compdiag_time" or str(column).startswith("PhotoCurve_compdiag_")
    ]
    base = df.drop(columns=remove_columns, errors="ignore")
    if not diagnostics:
        return base.copy()

    output: Dict[str, np.ndarray] = {"PhotoCurve_compdiag_time": np.asarray(x, dtype=float)}
    for star_id, diagnostic in diagnostics.items():
        output[_comp_diag_flux_column(star_id)] = np.asarray(diagnostic.flux, dtype=float)
        output[_comp_diag_err_column(star_id)] = np.asarray(diagnostic.flux_err, dtype=float)
        output[_comp_diag_ensemble_column(star_id)] = np.asarray(diagnostic.ensemble, dtype=float)
    return pd.concat([base, pd.DataFrame(output, index=base.index)], axis=1).copy()


def update_comparison_diagnostic_controls(
    window: sg.Window,
    diagnostics: Dict[str, ComparisonDiagnosticCurve],
    selected_star: str = "",
) -> str:
    """Update diagnostic-star controls and return the selected diagnostic star."""
    stars = sorted(diagnostics.keys())
    selected_star = str(selected_star or "").strip()
    if selected_star not in stars:
        selected_star = stars[0] if stars else ""
    try:
        window["-COMP_DIAG_STAR-"].update(values=stars, value=selected_star, disabled=not bool(stars))
        window["-COMP_PLOT_DIAG-"].update(disabled=not bool(stars))
    except Exception:
        pass
    return selected_star


def set_comparison_diagnostic_output_columns(
    window: sg.Window,
    values: Dict[str, object],
    star_id: str,
) -> Dict[str, object]:
    """Select one comparison-star diagnostic curve in the Data tab."""
    star_id = str(star_id).strip()
    selection = {
        "-XCOL-": "PhotoCurve_compdiag_time",
        "-YCOL-": _comp_diag_flux_column(star_id),
        "-YERRCOL-": _comp_diag_err_column(star_id),
        "-MODEL_COL-": NONE_COL,
        "-RES_COL-": NONE_COL,
        "-RESERR_COL-": NONE_COL,
        "-YLABEL-": f"{star_id} relative flux",
    }
    for key, value in selection.items():
        if key.startswith("-") and key.endswith("-") and key != "-YLABEL-":
            try:
                window[key].update(value=value)
            except Exception:
                pass
        elif key == "-YLABEL-":
            try:
                window[key].update(value=value)
            except Exception:
                pass
    plot_values = dict(values)
    plot_values.update(selection)
    return plot_values


def build_current_comparison_diagnostics(
    df: pd.DataFrame,
    detection: AijFluxDetection,
    active_stars: set[str],
    values: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, ComparisonDiagnosticCurve]]:
    """Build comparison-star diagnostics using the current time column and subset."""
    x_col = _get_comp_x_column(values, detection, df)
    x = to_numeric_array(df, x_col)
    if x is None:
        raise ValueError("The selected comparison-star time column is not numeric.")
    polynomial_order = max(0, min(2, parse_int(values.get("-COMP_POLY_ORDER-", 1), 1)))
    diagnostics = build_comparison_diagnostics(
        df,
        detection,
        x,
        selected_comparisons=sorted(active_stars),
        polynomial_order=polynomial_order,
    )
    return x, diagnostics


def compose_comparison_report(
    main_report: str,
    diagnostics: Dict[str, ComparisonDiagnosticCurve],
) -> str:
    """Append comparison-star diagnostic information to the main report."""
    if main_report:
        return main_report.rstrip() + "\n\n" + format_comparison_diagnostics_report(diagnostics)
    return format_comparison_diagnostics_report(diagnostics)

def _parse_detrend_regressor_from_display(item: str) -> str:
    """Extract a detrending regressor name from a listbox label."""
    text = str(item).strip()
    for prefix in ("[x]", "[ ]", "☑", "☐"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _detrend_regressor_list_values(
    detection: DetrendRegressorDetection,
    active_regressors: set[str],
) -> List[str]:
    """Return listbox labels for active/inactive detrending regressors."""
    labels: List[str] = []
    for column in detection.columns:
        marker = "[x]" if column in active_regressors else "[ ]"
        labels.append(f"{marker} {column}")
    return labels


def update_detrend_regressor_list(
    window: sg.Window,
    detection: Optional[DetrendRegressorDetection],
    active_regressors: set[str],
    disabled: bool = False,
) -> None:
    """Update the detrending-regressor listbox while preserving scroll position."""
    try:
        if detection is None or not detection.compatible:
            window["-DET_REGRESSOR_LIST-"].update(values=[], disabled=True)
            return

        element = window["-DET_REGRESSOR_LIST-"]
        first_visible_fraction = 0.0
        try:
            first_visible_fraction = float(element.Widget.yview()[0])
        except Exception:
            first_visible_fraction = 0.0

        labels = _detrend_regressor_list_values(detection, active_regressors)
        element.update(values=labels, disabled=disabled)

        try:
            element.Widget.yview_moveto(first_visible_fraction)
            element.Widget.selection_clear(0, "end")
        except Exception:
            pass
    except Exception:
        pass


def update_detrending_tab(
    window: sg.Window,
    detection: Optional[DetrendRegressorDetection],
    active_regressors: Optional[set[str]] = None,
    transit_fit_available: bool = False,
) -> None:
    """Enable or disable the photometric detrending controls."""
    control_keys = [
        "-DET_REGRESSOR_LIST-",
        "-DET_SELECT_ALL-",
        "-DET_SELECT_NONE-",
        "-DET_SELECT_SUGGESTED-",
        "-DET_MASK_TRANSIT-",
        "-DET_USE_CLEANING_MASK-",
        "-DET_POLY_ORDER-",
        "-DET_ROBUST_SIGMA-",
        "-DET_ROBUST_ITER-",
        "-DET_FLIP_ACTIVE-",
        "-DET_FLIP_FRAC-",
        # "-DET_UPDATE_FLIP_MARKER-",
        "-DET_FLIP_MODE-",
        "-DET_SHOW_FLIP_MARKER-",
        "-DET_USE_TRANSIT_MODEL-",
        "-DET_MODEL_ITER_MODE-",
        "-DET_MODEL_MAX_ITER-",
        "-DET_MODEL_TOL_PCT-",
        "-DET_SEND_TO_DATA-",
        "-DET_SHOW_POPUP-",
        "Run detrending",
        "Clear detrending",
    ]

    if detection is None or not detection.compatible:
        warning = detection.warning if detection is not None else (
            "Photometric detrending inactive: load a table with usable time/airmass/FWHM/sky/centroid columns."
        )
        try:
            window["-DET_STATUS-"].update(warning, text_color="firebrick")
            window["-DET_REPORT-"].update(warning)
            update_detrend_regressor_list(window, detection, set(), disabled=True)
            for key in control_keys:
                window[key].update(disabled=True)
        except Exception:
            pass
        return

    if active_regressors is None:
        # Keep detrending conservative by default. The user can still press
        # Suggested to activate the automatically recommended regressors.
        active_regressors = set()

    try:
        window["-DET_STATUS-"].update(
            f"Detrending regressors detected: {len(detection.columns)} candidate column(s). "
            "Select one or more and run detrending.",
            text_color="darkgreen",
        )
        update_detrend_regressor_list(window, detection, active_regressors, disabled=False)
        for key in control_keys:
            window[key].update(disabled=False)
        if not bool(transit_fit_available):
            try:
                window["-DET_USE_TRANSIT_MODEL-"].update(value=False, disabled=True)
            except Exception:
                pass
        try:
            update_model_aware_detrending_controls(
                window,
                bool(transit_fit_available),
                bool(window["-DET_USE_TRANSIT_MODEL-"].get()) if bool(transit_fit_available) else False,
            )
        except Exception:
            pass
        window["-DET_REPORT-"].update(
            "Ready. No detrending regressor is active by default. "
            "Select columns manually, or press Suggested to use the automatically recommended regressors."
        )
    except Exception:
        pass


def update_model_aware_detrending_controls(
    window: sg.Window,
    available: bool,
    active: Optional[bool] = None,
) -> None:
    """Enable/disable model-aware iteration controls consistently."""
    if active is None:
        try:
            active = bool(window["-DET_USE_TRANSIT_MODEL-"].get())
        except Exception:
            active = False
    disabled = not (bool(available) and bool(active))
    for key in ("-DET_MODEL_ITER_MODE-", "-DET_MODEL_MAX_ITER-", "-DET_MODEL_TOL_PCT-"):
        try:
            window[key].update(disabled=disabled)
        except Exception:
            pass


def update_detrend_fit_model_control(window: sg.Window, detection: Optional[DetrendRegressorDetection], last_transit_result) -> None:
    """Enable model-aware detrending only after a valid transit fit exists."""
    available = bool(detection is not None and detection.compatible and last_transit_result is not None)
    active = False
    try:
        active = bool(window["-DET_USE_TRANSIT_MODEL-"].get())
        window["-DET_USE_TRANSIT_MODEL-"].update(disabled=not available)
        if not available:
            window["-DET_USE_TRANSIT_MODEL-"].update(value=False)
            active = False
    except Exception:
        pass
    update_model_aware_detrending_controls(window, available, active)


def _relative_delta_percent(new_value: float, old_value: float) -> float:
    """Return |new-old|/|old| in percent, guarding against zero values."""
    try:
        new_f = float(new_value)
        old_f = float(old_value)
    except Exception:
        return float("nan")
    if not np.isfinite(new_f) or not np.isfinite(old_f):
        return float("nan")
    denom = max(abs(old_f), 1.0e-12)
    return 100.0 * abs(new_f - old_f) / denom


def _baseline_delta_ppt(new_baseline: Optional[np.ndarray], old_baseline: Optional[np.ndarray]) -> float:
    """Return the median multiplicative baseline change in ppt."""
    if new_baseline is None or old_baseline is None:
        return float("nan")
    new_arr = np.asarray(new_baseline, dtype=float)
    old_arr = np.asarray(old_baseline, dtype=float)
    if new_arr.shape != old_arr.shape:
        return float("nan")
    good = np.isfinite(new_arr) & np.isfinite(old_arr) & (np.abs(old_arr) > 1.0e-12)
    if np.count_nonzero(good) < 3:
        return float("nan")
    ratio = new_arr[good] / old_arr[good]
    return float(1000.0 * np.nanmedian(np.abs(ratio - 1.0)))


def _model_aware_convergence_row(
    iteration: int,
    previous_transit,
    current_transit,
    previous_detrend: Optional[PhotometricDetrendResult],
    current_detrend: PhotometricDetrendResult,
) -> Dict[str, object]:
    """Build one convergence-history row for model-aware detrending."""
    tmid_delta_min = float("nan")
    rp_rs_delta_pct = float("nan")
    rms_delta_pct = float("nan")
    depth_delta_pct = float("nan")
    if previous_transit is not None and current_transit is not None:
        try:
            tmid_delta_min = abs(float(current_transit.tmid_observed) - float(previous_transit.tmid_observed)) * 24.0 * 60.0
        except Exception:
            tmid_delta_min = float("nan")
        rp_rs_delta_pct = _relative_delta_percent(
            getattr(current_transit, "observed_rp_rs", float("nan")),
            getattr(previous_transit, "observed_rp_rs", float("nan")),
        )
        rms_delta_pct = _relative_delta_percent(
            getattr(current_transit, "residual_rms_ppt", float("nan")),
            getattr(previous_transit, "residual_rms_ppt", float("nan")),
        )
        depth_delta_pct = _relative_delta_percent(
            getattr(current_transit, "observed_depth_ppt", float("nan")),
            getattr(previous_transit, "observed_depth_ppt", float("nan")),
        )
    return {
        "iteration": int(iteration),
        "tmid_delta_min": tmid_delta_min,
        "rp_rs_delta_pct": rp_rs_delta_pct,
        "depth_delta_pct": depth_delta_pct,
        "rms_delta_pct": rms_delta_pct,
        "baseline_delta_ppt": _baseline_delta_ppt(
            getattr(current_detrend, "baseline", None),
            getattr(previous_detrend, "baseline", None),
        ),
        "tmid_observed": float(getattr(current_transit, "tmid_observed", float("nan"))) if current_transit is not None else float("nan"),
        "rp_rs": float(getattr(current_transit, "observed_rp_rs", float("nan"))) if current_transit is not None else float("nan"),
        "rms_ppt": float(getattr(current_transit, "residual_rms_ppt", float("nan"))) if current_transit is not None else float("nan"),
    }


def _model_aware_has_converged(row: Dict[str, object], tol_pct: float, tmid_tol_min: float, baseline_tol_ppt: float) -> bool:
    """Return True when all model-aware convergence criteria are satisfied."""
    required = [
        float(row.get("tmid_delta_min", float("nan"))) <= float(tmid_tol_min),
        float(row.get("rp_rs_delta_pct", float("nan"))) <= float(tol_pct),
        float(row.get("rms_delta_pct", float("nan"))) <= float(tol_pct),
    ]
    baseline_delta = float(row.get("baseline_delta_ppt", float("nan")))
    if np.isfinite(baseline_delta):
        required.append(baseline_delta <= float(baseline_tol_ppt))
    return all(required)


def _format_model_aware_iteration_report(
    mode: str,
    history: Sequence[Dict[str, object]],
    converged: bool,
    reason: str,
    tol_pct: float,
    tmid_tol_min: float,
    baseline_tol_ppt: float,
) -> str:
    """Return a transparent report for model-aware detrending/refit iterations."""
    lines = [
        "Model-aware detrending convergence",
        f"Mode: {mode}",
        f"Converged: {'yes' if converged else 'no'}",
        f"Stop reason: {reason}",
        "Objective criteria for convergence:",
        f"  |ΔTmid| <= {tmid_tol_min:.3f} min",
        f"  |ΔRp/Rs| <= {tol_pct:.3f} %",
        f"  |ΔRMS| <= {tol_pct:.3f} %",
        f"  median |Δbaseline/baseline| <= {baseline_tol_ppt:.3f} ppt, when measurable",
        "",
        "Iteration history:",
        "iter  ΔTmid[min]  ΔRp/Rs[%]  ΔDepth[%]  ΔRMS[%]  Δbaseline[ppt]  Tmid[BJD_TDB]  Rp/Rs  RMS[ppt]",
    ]
    if not history:
        lines.append("none")
    for row in history:
        def fmt(value, width=10, precision=4):
            try:
                val = float(value)
            except Exception:
                return "n/a".rjust(width)
            if not np.isfinite(val):
                return "n/a".rjust(width)
            return f"{val:{width}.{precision}f}"
        lines.append(
            f"{int(row.get('iteration', 0)):>4d}"
            f"  {fmt(row.get('tmid_delta_min'), 10, 4)}"
            f"  {fmt(row.get('rp_rs_delta_pct'), 10, 4)}"
            f"  {fmt(row.get('depth_delta_pct'), 10, 4)}"
            f"  {fmt(row.get('rms_delta_pct'), 8, 4)}"
            f"  {fmt(row.get('baseline_delta_ppt'), 14, 4)}"
            f"  {fmt(row.get('tmid_observed'), 13, 8)}"
            f"  {fmt(row.get('rp_rs'), 7, 5)}"
            f"  {fmt(row.get('rms_ppt'), 8, 3)}"
        )
    return "\n".join(lines)


def add_detrending_columns(
    df: pd.DataFrame,
    result: PhotometricDetrendResult,
) -> pd.DataFrame:
    """Add photometric-detrending output columns without removing user data."""
    det_columns = [
        "PhotoCurve_det_time",
        "PhotoCurve_det_flux",
        "PhotoCurve_det_err",
        "PhotoCurve_det_baseline",
        "PhotoCurve_det_residual",
        "PhotoCurve_det_fit_mask",
    ]
    base = df.drop(columns=det_columns, errors="ignore")
    output = pd.DataFrame(
        {
            "PhotoCurve_det_time": np.asarray(result.x, dtype=float),
            "PhotoCurve_det_flux": np.asarray(result.detrended_flux, dtype=float),
            "PhotoCurve_det_err": np.asarray(result.detrended_flux_err, dtype=float),
            "PhotoCurve_det_baseline": np.asarray(result.baseline, dtype=float),
            "PhotoCurve_det_residual": np.asarray(result.residuals, dtype=float),
            "PhotoCurve_det_fit_mask": np.asarray(result.robust_keep_mask, dtype=float),
        },
        index=base.index,
    )
    return pd.concat([base, output], axis=1).copy()


def set_detrending_output_columns(window: sg.Window, values: Dict[str, object]) -> Dict[str, object]:
    """Select the detrended light curve in the Data tab and return plot values."""
    selection = {
        "-XCOL-": "PhotoCurve_det_time",
        "-YCOL-": "PhotoCurve_det_flux",
        "-YERRCOL-": "PhotoCurve_det_err",
        "-MODEL_COL-": NONE_COL,
        "-RES_COL-": NONE_COL,
        "-RESERR_COL-": NONE_COL,
        "-YLABEL-": "Decorrelated relative flux",
    }
    for key, value in selection.items():
        if key.startswith("-") and key.endswith("-") and key != "-YLABEL-":
            try:
                window[key].update(value=value)
            except Exception:
                pass
    plot_values = dict(values)
    plot_values.update(selection)
    return plot_values

def add_transit_diagnostic_columns(
    df: pd.DataFrame,
    result,
    input_time_days,
    input_flux,
    input_flux_err,
) -> pd.DataFrame:
    """Add model, residual and baseline-corrected columns.

    The raw input time used for the diagnostic fit is stored separately from
    the mid-exposure corrected time. This prevents cumulative timestamp
    corrections when the user presses ``Run transit diag`` more than once.

    The transit diagnostic computes both full models, which include the fitted
    baseline, and detrended models, where the photometry has been divided by
    the fitted baseline.  The generic ``PhotoCurve_fit_model`` and
    ``PhotoCurve_expected_model`` aliases point to the display mode selected in
    the Transit tab.

    All PhotoCurve columns are replaced in one concatenation step.  This avoids
    pandas ``PerformanceWarning: DataFrame is highly fragmented`` messages that
    can appear when many columns are inserted one by one.
    """
    n_rows = len(df)
    base = remove_photocurve_columns(df)

    if input_flux_err is None:
        flux_err_input = np.full(n_rows, np.nan, dtype=float)
    else:
        flux_err_input = np.asarray(input_flux_err, dtype=float)

    diagnostic_data = {
        "PhotoCurve_time_input": np.asarray(input_time_days, dtype=float),
        "PhotoCurve_flux_input": np.asarray(input_flux, dtype=float),
        "PhotoCurve_flux_err_input": flux_err_input,
        "PhotoCurve_time_corrected": np.asarray(result.corrected_time, dtype=float),
        "PhotoCurve_baseline": np.asarray(result.baseline, dtype=float),
        "PhotoCurve_detrended_flux": np.asarray(result.detrended_flux, dtype=float),
        "PhotoCurve_detrended_err": np.asarray(result.detrended_flux_err, dtype=float),
        "PhotoCurve_fit_transit_model": np.asarray(result.fit_transit_model, dtype=float),
        "PhotoCurve_expected_transit_model": np.asarray(result.expected_transit_model, dtype=float),
        "PhotoCurve_fit_full_model": np.asarray(result.fit_full_model, dtype=float),
        "PhotoCurve_expected_full_model": np.asarray(result.expected_full_model, dtype=float),
        "PhotoCurve_fit_full_residual": np.asarray(result.full_residuals, dtype=float),
        "PhotoCurve_fit_detrended_residual": np.asarray(result.detrended_residuals, dtype=float),
        "PhotoCurve_expected_model": np.asarray(result.expected_model, dtype=float),
        "PhotoCurve_fit_model": np.asarray(result.model, dtype=float),
        "PhotoCurve_fit_residual": np.asarray(result.residuals, dtype=float),
        "PhotoCurve_predicted_start_time": np.full(n_rows, result.tmid_predicted - result.expected_duration_hours / 48.0, dtype=float),
        "PhotoCurve_predicted_tmid_time": np.full(n_rows, result.tmid_predicted, dtype=float),
        "PhotoCurve_predicted_end_time": np.full(n_rows, result.tmid_predicted + result.expected_duration_hours / 48.0, dtype=float),
        "PhotoCurve_calculated_start_time": np.full(n_rows, result.tmid_observed - result.observed_duration_hours / 48.0, dtype=float),
        "PhotoCurve_calculated_tmid_time": np.full(n_rows, result.tmid_observed, dtype=float),
        "PhotoCurve_calculated_end_time": np.full(n_rows, result.tmid_observed + result.observed_duration_hours / 48.0, dtype=float),
        "PhotoCurve_oc_minutes": np.full(n_rows, result.oc_minutes, dtype=float),
    }

    diagnostic_columns = pd.DataFrame(diagnostic_data, index=base.index)
    return pd.concat([base, diagnostic_columns], axis=1).copy()


def main() -> None:
    """Run the GUI."""
    sg.theme("SystemDefault")
    
    # Handling the extreme scaling (>150%) on Windows systems
    if os.name == "nt":
        # OS-reported scale used only as initial HINT
        try:
            dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
        except Exception:
            dpi_scale = 1.0

        # If OS scale < 1.5, start from 1.5 as a comfortable default
        scale_win = 1.3 if dpi_scale < 1.5 else float(dpi_scale)

        if dpi_scale < 1.5:
            window = sg.Window(
                "ExoPhotoCurve - Exoplanet transit lightcurve diagnostics and analysis - Daniele Gasparri",
                make_layout(),
                resizable=True,
                finalize=True,
                icon=icon_path,
            )
        else:
            window = sg.Window(
                "ExoPhotoCurve - Exoplanet transit lightcurve diagnostics and analysis - Daniele Gasparri",
                make_layout(),
                resizable=True,
                finalize=True,
                icon=icon_path,
                scaling =scale_win,
            )       
        
        #Mouse over button on Windows
        misc.enable_hover_effect(window)
        
    else:
        window = sg.Window(
            "ExoPhotoCurve - Exoplanet transit lightcurve diagnostics and analysis - Daniele Gasparri",
            make_layout(),
            resizable=True,
            finalize=True,
            icon=icon_path,
        )
        
    center_window(window)
    df: Optional[pd.DataFrame] = None
    original_df: Optional[pd.DataFrame] = None
    original_column_selection: Optional[Dict[str, str]] = None
    pre_transit_column_selection: Optional[Dict[str, str]] = None
    fig_agg: Optional[FigureCanvasTkAgg] = None
    current_fig = None
    last_stats_report: Optional[str] = None
    last_stats_blocks: List[SeriesStatistics] = []
    exoplanet_catalogue = None
    last_transit_report: Optional[str] = None
    last_transit_result = None
    aij_flux_detection: Optional[AijFluxDetection] = None
    last_comp_report: Optional[str] = None
    last_comp_result: Optional[ComparisonOptimisationResult] = None
    last_comp_diagnostics: Dict[str, ComparisonDiagnosticCurve] = {}
    current_photometry_file_path: Optional[str] = None
    detrend_detection: Optional[DetrendRegressorDetection] = None
    detrend_active_regressors: set[str] = set()
    last_detrend_report: Optional[str] = None
    last_detrend_result: Optional[PhotometricDetrendResult] = None
    last_detrend_input_selection: Optional[Dict[str, str]] = None
    last_recipe_report: Optional[str] = None
    comp_active_stars: set[str] = set()
    auto_reject_indices: set[int] = set()
    manual_reject_indices: set[int] = set()
    manual_keep_indices: set[int] = set()

    try:
        exoplanet_catalogue = load_catalogue_into_window(
            window,
            str(default_exoclock_catalogue_path()),
            source_label="ExoClock",
        )
    except Exception as exc:
        window["-STATUS-"].update(f"Transit catalogue not loaded: {exc}")

    update_comparison_tab(window, None)
    update_detrending_tab(window, None)
    update_detrend_fit_model_control(window, None, None)
    update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
    update_auto_clip_controls(window, auto_reject_indices)

    # Keep DPI awareness on Windows for crisp rendering
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        
    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "Exit"):
            break


        if event == 'User manual':
            open_manual()

        if event == "Load":
            file_path = str(values["-FILE-"]).strip()

            if not file_path or not os.path.exists(file_path):
                sg.popup_error("Please select a valid ASCII file.")
                continue

            try:
                current_photometry_file_path = file_path
                df = read_ascii_table(
                    file_path,
                    str(values["-DELIM-"]),
                    bool(values["-HEADER-"]),
                )

                original_df = df.copy(deep=True)
                cols = numeric_columns(df)
                original_column_selection = update_combo_values(window, cols, df=df, sync_transit_time_system=True)
                pre_transit_column_selection = original_column_selection.copy()
                last_transit_result = None
                last_transit_report = None
                last_stats_report = None
                last_stats_blocks = []
                last_comp_report = None
                last_comp_result = None
                last_comp_diagnostics = {}
                last_detrend_report = None
                last_detrend_result = None
                last_detrend_input_selection = None
                last_recipe_report = None
                auto_reject_indices.clear()
                manual_reject_indices.clear()
                manual_keep_indices.clear()
                update_auto_clip_controls(window, auto_reject_indices)
                update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
                window["-TR_SET_MODEL_COLUMNS-"].update(value=True)

                aij_flux_detection = detect_aij_flux_columns(df)
                comp_active_stars = set(aij_flux_detection.comparison_ids) if aij_flux_detection.compatible else set()
                update_comparison_tab(window, aij_flux_detection, comp_active_stars)

                detrend_detection = detect_detrending_regressors(
                    df,
                    x_column=original_column_selection.get("-XCOL-", "") if original_column_selection else "",
                    y_column=original_column_selection.get("-YCOL-", "") if original_column_selection else "",
                    yerr_column=original_column_selection.get("-YERRCOL-", "") if original_column_selection else "",
                )
                detrend_active_regressors = set()
                update_detrending_tab(window, detrend_detection, detrend_active_regressors)
                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)

                window["-NROWS-"].update(str(len(df)))
                window["-NCOLS-"].update(str(len(df.columns)))

                # Convenience auto-selection: if the filename contains a planet
                # name or alias present in the offline catalogue, select it in
                # the Transit diagnostics tab. This keeps the ExoClock-like
                # workflow fast but remains fully editable by the user.
                guessed_planet = None
                if exoplanet_catalogue is not None:
                    guessed_planet = autodetect_planet_in_window(
                        window,
                        exoplanet_catalogue,
                        current_photometry_file_path,
                    )

                if guessed_planet:
                    window["-STATUS-"].update(
                        f"Loaded: {Path(file_path).name} | Auto-selected planet: {guessed_planet}"
                    )
                else:
                    window["-STATUS-"].update(f"Loaded: {Path(file_path).name}")

            except Exception as exc:
                sg.popup_error(f"Could not read file:\n{exc}")

        if event == "Reduce img":
            reduction_result = run_image_reduction_tool(window)
            if reduction_result:
                try:
                    window["-STATUS-"].update(
                        f"Reduced sequence ready for Build light curve: {reduction_result.aligned_folder}"
                    )
                except Exception:
                    pass

        if event == "Build LC":
            generated_path = run_aperture_photometry_tool(window)
            if generated_path:
                try:
                    current_photometry_file_path = generated_path
                    window["-FILE-"].update(generated_path)
                    window["-DELIM-"].update(value="Tab")
                    window["-HEADER-"].update(value=True)

                    df = read_ascii_table(generated_path, "Tab", True)
                    original_df = df.copy(deep=True)
                    cols = numeric_columns(df)
                    original_column_selection = update_combo_values(window, cols, df=df, sync_transit_time_system=True)
                    pre_transit_column_selection = original_column_selection.copy()
                    last_transit_result = None
                    last_transit_report = None
                    last_stats_report = None
                    last_stats_blocks = []
                    last_comp_report = None
                    last_comp_result = None
                    last_comp_diagnostics = {}
                    last_detrend_report = None
                    last_detrend_result = None
                    last_detrend_input_selection = None
                    last_recipe_report = None
                    auto_reject_indices.clear()
                    manual_reject_indices.clear()
                    manual_keep_indices.clear()
                    update_auto_clip_controls(window, auto_reject_indices)
                    update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
                    window["-TR_SET_MODEL_COLUMNS-"].update(value=True)

                    aij_flux_detection = detect_aij_flux_columns(df)
                    comp_active_stars = set(aij_flux_detection.comparison_ids) if aij_flux_detection.compatible else set()
                    update_comparison_tab(window, aij_flux_detection, comp_active_stars)

                    detrend_detection = detect_detrending_regressors(
                        df,
                        x_column=original_column_selection.get("-XCOL-", "") if original_column_selection else "",
                        y_column=original_column_selection.get("-YCOL-", "") if original_column_selection else "",
                        yerr_column=original_column_selection.get("-YERRCOL-", "") if original_column_selection else "",
                    )
                    detrend_active_regressors = set()
                    update_detrending_tab(window, detrend_detection, detrend_active_regressors)
                    update_detrend_fit_model_control(window, detrend_detection, last_transit_result)

                    window["-NROWS-"].update(str(len(df)))
                    window["-NCOLS-"].update(str(len(df.columns)))

                    guessed_planet = None
                    if exoplanet_catalogue is not None:
                        guessed_planet = autodetect_planet_in_window(
                            window,
                            exoplanet_catalogue,
                            current_photometry_file_path,
                        )

                    if guessed_planet:
                        window["-STATUS-"].update(
                            f"Generated and loaded: {Path(generated_path).name} | Auto-selected planet: {guessed_planet}"
                        )
                    else:
                        window["-STATUS-"].update(f"Generated and loaded: {Path(generated_path).name}")

                except Exception as exc:
                    sg.popup_error(f"The photometry table was generated but could not be loaded:\n{exc}")

        if event in ("Use NASA", "Use ExoClock"):
            try:
                current_planet = str(values.get("-TR_PLANET-", "")).strip()
                autodetect_text = Path(current_photometry_file_path).name if current_photometry_file_path else ""

                if event == "Use NASA":
                    catalog_path = default_catalogue_path()
                    source_label = "NASA"
                    missing_message = (
                        "The default NASA-style catalogue was not found.\n"
                        "Run tools/build_exoplanet_catalog_from_nasa.py or browse to an existing CSV."
                    )
                else:
                    catalog_path = default_exoclock_catalogue_path()
                    source_label = "ExoClock"
                    missing_message = (
                        "The ExoClock catalogue was not found.\n"
                        "Run tools/build_exoplanet_catalog_from_exoclock.py first, then try again."
                    )

                window["-TR_CATALOG-"].update(str(catalog_path))
                if not Path(catalog_path).exists():
                    sg.popup_error(missing_message)
                    continue

                exoplanet_catalogue = load_catalogue_into_window(
                    window,
                    str(catalog_path),
                    current_planet=current_planet,
                    autodetect_text=autodetect_text,
                    source_label=source_label,
                )
            except Exception as exc:
                sg.popup_error(f"Could not load transit catalogue:\n{exc}")

        if event == "Load catalogue":
            catalog_path = str(values.get("-TR_CATALOG-", "")).strip()
            if not catalog_path:
                sg.popup_error("Please select a valid catalogue CSV file.")
                continue
            try:
                exoplanet_catalogue = load_catalogue_into_window(
                    window,
                    catalog_path,
                    current_planet=str(values.get("-TR_PLANET-", "")).strip(),
                    autodetect_text=Path(current_photometry_file_path).name if current_photometry_file_path else "",
                    source_label="custom",
                )
            except Exception as exc:
                sg.popup_error(f"Could not load transit catalogue:\n{exc}")

        if event in (
            "-TR_SET_MODEL_COLUMNS-",
            "-TR_DISPLAY_MODE-",
            "-TR_SHOW_PREDICTED_TIMES-",
            "-TR_SHOW_CALCULATED_TIMES-",
        ):
            if df is None:
                continue

            try:
                show_transit_result = bool(values.get("-TR_SET_MODEL_COLUMNS-", True))

                if show_transit_result and last_transit_result is None:
                    window["-STATUS-"].update("Run transit diagnostics model to show the transit fit.")
                    continue

                if show_transit_result:
                    df = apply_transit_display_aliases(
                        df,
                        str(values.get("-TR_DISPLAY_MODE-", "Detrended flux")),
                    )

                plot_values = set_transit_plot_columns(
                    window,
                    values,
                    str(values.get("-TR_DISPLAY_MODE-", "Detrended flux")),
                    pre_transit_column_selection or original_column_selection,
                    show_transit_result,
                )

                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)

                if show_transit_result:
                    window["-STATUS-"].update("Transit fit display enabled.")
                else:
                    window["-STATUS-"].update("Transit fit display disabled; original column selection restored.")

            except Exception as exc:
                sg.popup_error(f"Could not update transit display:\n{exc}")

        if event == "Reset view/data":
            if original_df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                df = original_df.copy(deep=True)
                cols = numeric_columns(df)
                guessed = update_combo_values(window, cols, df=df, sync_transit_time_system=True)
                if original_column_selection is None:
                    original_column_selection = guessed
                pre_transit_column_selection = original_column_selection.copy()
                restore_column_selection(window, original_column_selection)

                # Disable all plot-level manipulations that can change the
                # visual interpretation of the original data. Style, labels and
                # observatory/planet settings are intentionally preserved.
                window["-BIN_ACTIVE-"].update(value=False)
                window["-CLEAN_ACTIVE-"].update(value=False)
                window["-TR_SET_MODEL_COLUMNS-"].update(value=True)
                try:
                    window["-DET_FLIP_ACTIVE-"].update(value=False)
                    window["-DET_FLIP_FRAC-"].update(value="")
                    window["-DET_FLIP_MODE-"].update(value="Robust level matching")
                    window["-DET_SHOW_FLIP_MARKER-"].update(value=True)
                except Exception:
                    pass
                auto_reject_indices.clear()
                manual_reject_indices.clear()
                manual_keep_indices.clear()
                update_auto_clip_controls(window, auto_reject_indices)
                update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)

                reset_values = dict(values)
                reset_values.update(original_column_selection)
                reset_values["-BIN_ACTIVE-"] = False
                reset_values["-CLEAN_ACTIVE-"] = False
                reset_values["-TR_SET_MODEL_COLUMNS-"] = True
                reset_values["-DET_FLIP_ACTIVE-"] = False
                reset_values["-DET_FLIP_FRAC-"] = ""
                reset_values["-DET_FLIP_MODE-"] = "Robust level matching"
                reset_values["-DET_SHOW_FLIP_MARKER-"] = True
                reset_values["-AUTO_REJECT_INDICES-"] = ""
                reset_values["-MANUAL_REJECT_INDICES-"] = ""
                reset_values["-MANUAL_KEEP_INDICES-"] = ""

                last_transit_report = None
                last_transit_result = None
                last_stats_report = None
                last_stats_blocks = []
                last_comp_report = None
                last_comp_result = None
                last_comp_diagnostics = {}
                last_detrend_report = None
                last_detrend_result = None
                last_detrend_input_selection = None
                last_recipe_report = None
                aij_flux_detection = detect_aij_flux_columns(df)
                comp_active_stars = set(aij_flux_detection.comparison_ids) if aij_flux_detection.compatible else set()
                update_comparison_tab(window, aij_flux_detection, comp_active_stars)
                detrend_detection = detect_detrending_regressors(
                    df,
                    x_column=original_column_selection.get("-XCOL-", "") if original_column_selection else "",
                    y_column=original_column_selection.get("-YCOL-", "") if original_column_selection else "",
                    yerr_column=original_column_selection.get("-YERRCOL-", "") if original_column_selection else "",
                )
                detrend_active_regressors = set()
                update_detrending_tab(window, detrend_detection, detrend_active_regressors)
                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)
                try:
                    window["-DET_FLIP_ACTIVE-"].update(value=False)
                    window["-DET_FLIP_FRAC-"].update(value="")
                    window["-DET_FLIP_MODE-"].update(value="Robust level matching")
                    window["-DET_SHOW_FLIP_MARKER-"].update(value=True)
                except Exception:
                    pass

                current_fig, fig_agg = redraw_plot(window, fig_agg, df, reset_values)
                window["-STATUS-"].update("Reset complete: original data restored; clipping, binning and detrending disabled. Transit-fit display remains enabled by default.")

            except Exception as exc:
                sg.popup_error(f"Could not reset the view/data:\n{exc}")

        if event == "Plot / update":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                plot_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
                window["-STATUS-"].update("Plot updated.")

            except Exception as exc:
                sg.popup_error(f"Could not create plot:\n{exc}")

        if event == "Apply sigma clipping":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                auto_reject_indices = _parse_index_set_text(values.get("-AUTO_REJECT_INDICES-", _format_index_set(auto_reject_indices)))
                manual_reject_indices = _parse_index_set_text(values.get("-MANUAL_REJECT_INDICES-", _format_index_set(manual_reject_indices)))
                manual_keep_indices = _parse_index_set_text(values.get("-MANUAL_KEEP_INDICES-", _format_index_set(manual_keep_indices)))

                x_col = str(values.get("-XCOL-", NONE_COL))
                y_col = str(values.get("-YCOL-", NONE_COL))
                model_col = str(values.get("-MODEL_COL-", NONE_COL))
                res_col = str(values.get("-RES_COL-", NONE_COL))

                x = to_numeric_array(df, x_col)
                y = to_numeric_array(df, y_col)
                model = to_numeric_array(df, model_col)
                residuals = to_numeric_array(df, res_col)
                if x is None or y is None:
                    raise ValueError("Select X/time and light-curve columns before applying sigma clipping.")

                clip_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                new_indices, target_label, n_iter = compute_auto_sigma_clip_reject_indices(
                    x,
                    y,
                    model,
                    residuals,
                    clip_values,
                )

                before = len(auto_reject_indices)
                auto_reject_indices |= new_indices
                update_auto_clip_controls(window, auto_reject_indices)
                update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)

                plot_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)

                added = len(auto_reject_indices) - before
                if added > 0:
                    window["-STATUS-"].update(
                        f"Auto sigma clipping applied on {target_label}: {added} new point(s) locked "
                        f"({len(auto_reject_indices)} total)."
                    )
                else:
                    window["-STATUS-"].update(
                        f"Auto sigma clipping applied on {target_label}: no new points rejected "
                        f"({len(auto_reject_indices)} already locked)."
                    )

            except Exception as exc:
                sg.popup_error(f"Could not apply sigma clipping:\n{exc}")

        if event == "Reset auto clipping":
            auto_reject_indices.clear()
            update_auto_clip_controls(window, auto_reject_indices)
            if df is not None:
                try:
                    plot_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                    current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
                except Exception as exc:
                    sg.popup_error(f"Could not reset auto clipping plot:\n{exc}")
                    continue
            window["-STATUS-"].update("Auto sigma clipping mask cleared.")

        if event == "Compute stats":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                stats_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                last_stats_report, last_stats_blocks = build_statistics_report(df, stats_values)
                window["-STATUS-"].update("Statistics computed.")

                if bool(values.get("-STATS_SHOW_POPUP-", True)):
                    sg.popup_scrolled(
                        last_stats_report,
                        title="Photometric statistics",
                        size=(90, 34),
                    )

            except Exception as exc:
                sg.popup_error(f"Could not compute statistics:\n{exc}")




        if event in ("-DET_REGRESSOR_LIST-", "-DET_SELECT_ALL-", "-DET_SELECT_NONE-", "-DET_SELECT_SUGGESTED-"):
            if df is None or detrend_detection is None or not detrend_detection.compatible:
                continue

            try:
                if event == "-DET_REGRESSOR_LIST-":
                    selected_items = values.get("-DET_REGRESSOR_LIST-", []) or []
                    if not selected_items:
                        continue
                    clicked = _parse_detrend_regressor_from_display(str(selected_items[0]))
                    if clicked in detrend_active_regressors:
                        detrend_active_regressors.remove(clicked)
                    elif clicked in detrend_detection.columns:
                        detrend_active_regressors.add(clicked)
                elif event == "-DET_SELECT_ALL-":
                    detrend_active_regressors = set(detrend_detection.columns)
                elif event == "-DET_SELECT_NONE-":
                    detrend_active_regressors = set()
                elif event == "-DET_SELECT_SUGGESTED-":
                    detrend_active_regressors = set(detrend_detection.suggested)

                update_detrend_regressor_list(window, detrend_detection, detrend_active_regressors, disabled=False)
                if detrend_active_regressors:
                    window["-DET_REPORT-"].update(
                        "Active regressors:\n" + "\n".join(sorted(detrend_active_regressors))
                    )
                    window["-STATUS-"].update(f"Selected {len(detrend_active_regressors)} detrending regressor(s).")
                else:
                    window["-DET_REPORT-"].update("No active detrending regressor. Select at least one column.")
                    window["-STATUS-"].update("No active detrending regressor selected.")
            except Exception as exc:
                window["-STATUS-"].update(f"Detrending regressor update failed: {exc}")

        if event in ("-DET_FLIP_ACTIVE-", "-DET_SHOW_FLIP_MARKER-"):
            if df is not None:
                try:
                    current_fig, fig_agg = redraw_plot(window, fig_agg, df, values)
                    if event == "-DET_FLIP_ACTIVE-" and not bool(values.get("-DET_FLIP_ACTIVE-", False)):
                        if getattr(last_detrend_result, "meridian_flip_enabled", False):
                            window["-STATUS-"].update(
                                "Meridian flip disabled for the next detrending run. "
                                "Press Run detrending to regenerate the curve without it, or Clear detrending."
                            )
                        else:
                            window["-STATUS-"].update("Meridian-flip marker disabled.")
                    elif bool(values.get("-DET_FLIP_ACTIVE-", False)) and str(values.get("-DET_FLIP_FRAC-", "")).strip():
                        window["-STATUS-"].update("Meridian-flip marker updated on the plot. Press Run detrending to apply the correction.")
                    else:
                        window["-STATUS-"].update("Meridian-flip option updated. Enter the time fraction and press Update line or Run detrending.")
                except Exception as exc:
                    window["-STATUS-"].update(f"Could not update meridian-flip marker: {exc}")

        if event == "Clear detrending":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                # Restore the column selection that was used as the input for
                # the most recent Detrend-tab run.  If no detrending has been
                # run yet, fall back to the original imported-column selection.
                restore_selection = (
                    last_detrend_input_selection
                    or pre_transit_column_selection
                    or original_column_selection
                    or current_column_selection(values)
                ).copy()

                # Remove detrending output columns.  A transit diagnostic run
                # performed on the detrended curve is also cleared from the
                # display state, because otherwise the plot can still show a
                # model/residual tied to the removed detrended curve.
                df = remove_detrending_columns(df)
                if last_transit_result is not None and _selection_uses_prefix(
                    current_column_selection(values),
                    ("PhotoCurve_det_", "PhotoCurve_time_corrected", "PhotoCurve_detrended_"),
                ):
                    df = clear_transit_diagnostic_columns(df)
                    last_transit_result = None
                    last_transit_report = None
                    window["-TR_SET_MODEL_COLUMNS-"].update(value=True)

                cols = numeric_columns(df)
                update_combo_values(window, cols, df=df, sync_transit_time_system=False)
                restore_column_selection(window, restore_selection)
                restored_y = str(restore_selection.get("-YCOL-", ""))
                restored_ylabel = "Optimised relative flux" if restored_y.startswith("PhotoCurve_compopt_") else "Relative flux"
                try:
                    window["-YLABEL-"].update(value=restored_ylabel)
                except Exception:
                    pass

                last_detrend_report = None
                last_detrend_result = None
                last_detrend_input_selection = None
                last_recipe_report = None

                # Re-detect regressors on the remaining table and keep the GUI
                # ready for a new detrending run.
                detrend_detection = detect_detrending_regressors(
                    df,
                    x_column=restore_selection.get("-XCOL-", ""),
                    y_column=restore_selection.get("-YCOL-", ""),
                    yerr_column=restore_selection.get("-YERRCOL-", ""),
                )
                detrend_active_regressors = set()
                update_detrending_tab(window, detrend_detection, detrend_active_regressors)
                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)
                window["-DET_REPORT-"].update(
                    "Detrending cleared. The light curve has been restored to the source columns used before detrending."
                )

                plot_values = update_values_with_selection(values, restore_selection)
                plot_values["-YLABEL-"] = restored_ylabel
                plot_values["-TR_SET_MODEL_COLUMNS-"] = True
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
                window["-STATUS-"].update("Detrending cleared; source light curve restored.")

            except Exception as exc:
                sg.popup_error(f"Could not clear photometric detrending:\n{exc}")


        if event == "-DET_USE_TRANSIT_MODEL-":
            try:
                available = bool(detrend_detection is not None and detrend_detection.compatible and last_transit_result is not None)
                update_model_aware_detrending_controls(window, available, bool(values.get("-DET_USE_TRANSIT_MODEL-", False)))
                if bool(values.get("-DET_USE_TRANSIT_MODEL-", False)) and available:
                    window["-STATUS-"].update(
                        "Model-aware detrending enabled. Choose Single pass or Iterate to convergence before running detrending."
                    )
            except Exception:
                pass

        if event == "Run detrending":
            if df is None:
                sg.popup_error("Load a table first.")
                continue
            if detrend_detection is None or not detrend_detection.compatible:
                message = (
                    detrend_detection.warning
                    if detrend_detection is not None
                    else "The loaded table does not contain usable detrending regressors."
                )
                sg.popup_error(message)
                continue
            if not detrend_active_regressors and not bool(values.get("-DET_FLIP_ACTIVE-", False)):
                sg.popup_error("Select at least one detrending regressor or enable meridian-flip detrending.")
                continue

            try:
                detrend_input_selection, detrend_source_note = resolve_detrending_input_selection(
                    values,
                    pre_transit_column_selection,
                    original_column_selection,
                    last_detrend_input_selection,
                )

                x_col = str(detrend_input_selection.get("-XCOL-", NONE_COL))
                y_col = str(detrend_input_selection.get("-YCOL-", NONE_COL))
                yerr_col = str(detrend_input_selection.get("-YERRCOL-", NONE_COL))
                model_col = str(detrend_input_selection.get("-MODEL_COL-", NONE_COL))
                res_col = str(detrend_input_selection.get("-RES_COL-", NONE_COL))

                x = to_numeric_array(df, x_col)
                y = to_numeric_array(df, y_col)
                yerr = to_numeric_array(df, yerr_col)
                model = to_numeric_array(df, model_col)
                residuals = to_numeric_array(df, res_col)

                if x is None or y is None:
                    raise ValueError("Select X/time and light-curve columns before running detrending.")

                planet = None
                if exoplanet_catalogue is not None and str(values.get("-TR_PLANET-", "")).strip():
                    try:
                        planet = find_planet(exoplanet_catalogue, str(values.get("-TR_PLANET-", "")))
                    except Exception:
                        planet = None
                if planet is None and bool(values.get("-DET_USE_TRANSIT_MODEL-", False)):
                    if exoplanet_catalogue is None:
                        catalog_path = str(values.get("-TR_CATALOG-", "")).strip()
                        exoplanet_catalogue = load_catalogue_into_window(
                            window,
                            catalog_path,
                            current_planet=str(values.get("-TR_PLANET-", "")).strip(),
                            autodetect_text=Path(current_photometry_file_path).name if current_photometry_file_path else "",
                            source_label="custom",
                        )
                    planet = find_planet(exoplanet_catalogue, str(values.get("-TR_PLANET-", "")))

                external_keep_mask = None
                if bool(values.get("-DET_USE_CLEANING_MASK-", True)):
                    clean_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                    cleaning = compute_cleaning_mask(x, y, model, residuals, clean_values)
                    external_keep_mask = cleaning.keep_mask

                polynomial_order = max(1, min(2, parse_int(values.get("-DET_POLY_ORDER-", 1), 1)))
                robust_sigma = parse_float(values.get("-DET_ROBUST_SIGMA-", 4.0), 4.0)
                if robust_sigma is None or robust_sigma <= 0:
                    robust_sigma = 4.0
                robust_iterations = max(0, parse_int(values.get("-DET_ROBUST_ITER-", 3), 3))

                flip_time = None
                if bool(values.get("-DET_FLIP_ACTIVE-", False)):
                    flip_time = resolve_meridian_flip_time_from_fraction(x, values.get("-DET_FLIP_FRAC-", ""))
                    if flip_time is None:
                        raise ValueError(
                            "Meridian flip is enabled but the flip time fraction is invalid. "
                            "Enter a value such as .771 for JD_UTC = 2461203.771."
                        )

                use_fit_model_for_detrending = bool(values.get("-DET_USE_TRANSIT_MODEL-", False))
                model_iter_mode = str(values.get("-DET_MODEL_ITER_MODE-", "Single pass"))
                max_model_iterations = max(1, min(30, parse_int(values.get("-DET_MODEL_MAX_ITER-", 10), 10)))
                convergence_tol_pct = parse_float(values.get("-DET_MODEL_TOL_PCT-", 0.10), 0.10)
                if convergence_tol_pct is None or not np.isfinite(convergence_tol_pct) or convergence_tol_pct <= 0:
                    convergence_tol_pct = 0.10
                convergence_tol_pct = float(max(0.001, min(10.0, convergence_tol_pct)))
                tmid_tol_min = 0.02
                baseline_tol_ppt = max(0.05, 0.5 * convergence_tol_pct)

                convergence_history: List[Dict[str, object]] = []
                convergence_report = ""
                convergence_reason = "standard detrending; model-aware convergence not used"
                convergence_converged = False
                n_model_iterations_done = 0

                current_transit_for_model = last_transit_result
                previous_detrend_result: Optional[PhotometricDetrendResult] = None
                final_transit_result = None

                if use_fit_model_for_detrending and current_transit_for_model is None:
                    raise ValueError(
                        "Model-aware detrending requires a previous transit fit. "
                        "Press Run transit model first, then run detrending again."
                    )

                n_requested_iterations = max_model_iterations if (
                    use_fit_model_for_detrending and model_iter_mode == "Iterate to convergence"
                ) else 1

                for iteration in range(1, n_requested_iterations + 1):
                    transit_model_for_detrending = None
                    if use_fit_model_for_detrending:
                        candidate_model = np.asarray(getattr(current_transit_for_model, "fit_transit_model", []), dtype=float)
                        if candidate_model.shape != np.asarray(y, dtype=float).shape:
                            raise ValueError(
                                "The latest transit model does not match the current light curve length. "
                                "Run the transit model again before using model-aware detrending."
                            )
                        transit_model_for_detrending = candidate_model

                    candidate_detrend_result = apply_photometric_detrending(
                        df,
                        x,
                        y,
                        yerr,
                        selected_regressors=sorted(detrend_active_regressors),
                        planet=planet,
                        mask_expected_transit=bool(values.get("-DET_MASK_TRANSIT-", True)),
                        external_keep_mask=external_keep_mask,
                        polynomial_order=polynomial_order,
                        robust_sigma=float(robust_sigma),
                        robust_iterations=robust_iterations,
                        meridian_flip_time=flip_time,
                        meridian_flip_mode=str(values.get("-DET_FLIP_MODE-", "Robust level matching")),
                        transit_model=transit_model_for_detrending,
                    )

                    last_detrend_result = candidate_detrend_result
                    n_model_iterations_done = iteration

                    if not use_fit_model_for_detrending:
                        break

                    # In model-aware mode the detrended curve must be refitted
                    # immediately and internally.  This makes the procedure
                    # reproducible and prevents the result from depending on how
                    # many times the user manually presses two GUI buttons.
                    clean_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                    det_cleaning = compute_cleaning_mask(
                        candidate_detrend_result.x,
                        candidate_detrend_result.detrended_flux,
                        None,
                        None,
                        clean_values,
                    )
                    candidate_transit_result = run_transit_diagnostics(
                        candidate_detrend_result.x,
                        candidate_detrend_result.detrended_flux,
                        candidate_detrend_result.detrended_flux_err,
                        planet,
                        values,
                        keep_mask=det_cleaning.keep_mask,
                    )
                    final_transit_result = candidate_transit_result

                    row = _model_aware_convergence_row(
                        iteration,
                        current_transit_for_model,
                        candidate_transit_result,
                        previous_detrend_result,
                        candidate_detrend_result,
                    )
                    convergence_history.append(row)

                    if model_iter_mode == "Single pass":
                        convergence_reason = "single pass selected by user"
                        convergence_converged = False
                        current_transit_for_model = candidate_transit_result
                        previous_detrend_result = candidate_detrend_result
                        break

                    if _model_aware_has_converged(row, convergence_tol_pct, tmid_tol_min, baseline_tol_ppt):
                        convergence_reason = f"objective criteria met at iteration {iteration}"
                        convergence_converged = True
                        current_transit_for_model = candidate_transit_result
                        previous_detrend_result = candidate_detrend_result
                        break

                    current_transit_for_model = candidate_transit_result
                    previous_detrend_result = candidate_detrend_result
                    convergence_reason = f"maximum iterations reached ({max_model_iterations})"

                if use_fit_model_for_detrending:
                    last_transit_result = final_transit_result
                    last_detrend_result.model_aware_mode = model_iter_mode
                    last_detrend_result.model_aware_iterations = int(n_model_iterations_done)
                    last_detrend_result.model_aware_converged = bool(convergence_converged)
                    last_detrend_result.model_aware_stop_reason = convergence_reason
                    last_detrend_result.model_aware_tolerance_percent = float(convergence_tol_pct)
                    last_detrend_result.model_aware_history = convergence_history
                    convergence_report = _format_model_aware_iteration_report(
                        model_iter_mode,
                        convergence_history,
                        convergence_converged,
                        convergence_reason,
                        convergence_tol_pct,
                        tmid_tol_min,
                        baseline_tol_ppt,
                    )
                else:
                    last_detrend_result.model_aware_mode = "Off"
                    last_detrend_result.model_aware_iterations = 0
                    last_detrend_result.model_aware_converged = False
                    last_detrend_result.model_aware_stop_reason = convergence_reason
                    last_detrend_result.model_aware_history = []

                last_detrend_report = (
                    last_detrend_result.report
                    + (("\n\n" + convergence_report) if convergence_report else "")
                    + "\n\nInput columns used"
                    + f"\nSource: {detrend_source_note}"
                    + f"\nTime column: {x_col}"
                    + f"\nFlux column: {y_col}"
                    + ("\nTransit fit model considered for baseline: yes" if use_fit_model_for_detrending else "\nTransit fit model considered for baseline: no")
                    + f"\nError column: {yerr_col if yerr_col != NONE_COL else 'none'}"
                    + (f"\nMeridian flip full time used: {last_detrend_result.meridian_flip_time:.8f}" if getattr(last_detrend_result, 'meridian_flip_enabled', False) else "")
                )
                last_detrend_input_selection = detrend_input_selection.copy()
                df = add_detrending_columns(df, last_detrend_result)
                cols = numeric_columns(df)
                update_combo_values(window, cols)

                plot_values = dict(values)
                if bool(values.get("-DET_SEND_TO_DATA-", True)):
                    plot_values = set_detrending_output_columns(window, values)
                    pre_transit_column_selection = current_column_selection(plot_values)

                if use_fit_model_for_detrending and last_transit_result is not None:
                    last_recipe_report = build_reproducibility_recipe(
                        df,
                        values,
                        current_photometry_file_path,
                        original_column_selection,
                        pre_transit_column_selection,
                        aij_flux_detection,
                        comp_active_stars,
                        last_comp_result,
                        detrend_detection,
                        detrend_active_regressors,
                        last_detrend_result,
                        last_detrend_input_selection,
                        auto_reject_indices,
                        manual_reject_indices,
                        manual_keep_indices,
                        last_transit_result,
                    )
                    last_transit_report = format_transit_report(last_transit_result) + "\n\n" + last_recipe_report
                    df = add_transit_diagnostic_columns(
                        df,
                        last_transit_result,
                        last_detrend_result.x,
                        last_detrend_result.detrended_flux,
                        last_detrend_result.detrended_flux_err,
                    )
                    df = apply_transit_display_aliases(
                        df,
                        str(values.get("-TR_DISPLAY_MODE-", "Detrended flux")),
                    )
                    cols = numeric_columns(df)
                    update_combo_values(window, cols)

                    show_transit_result = bool(values.get("-TR_SET_MODEL_COLUMNS-", True))
                    plot_values = set_transit_plot_columns(
                        window,
                        values,
                        str(values.get("-TR_DISPLAY_MODE-", "Detrended flux")),
                        pre_transit_column_selection or original_column_selection,
                        show_transit_result,
                    )
                    plot_values = _with_current_masks(plot_values, auto_reject_indices, manual_reject_indices, manual_keep_indices)

                    last_detrend_report += (
                        "\n\nFinal model-aware transit summary"
                        f"\nObserved Tmid: {last_transit_result.tmid_observed:.8f} BJD_TDB"
                        f"\nO-C: {last_transit_result.oc_minutes:+.3f} min"
                        f"\nRp/Rs: {last_transit_result.observed_rp_rs:.5f}"
                        f"\nDepth: {last_transit_result.observed_depth_ppt:.3f} ppt"
                        f"\nResidual RMS: {last_transit_result.residual_rms_ppt:.3f} ppt"
                    )

                window["-DET_REPORT-"].update(last_detrend_report)

                if use_fit_model_for_detrending and last_transit_result is not None:
                    if model_iter_mode == "Iterate to convergence":
                        status_prefix = "Model-aware detrending converged" if convergence_converged else "Model-aware detrending stopped"
                        window["-STATUS-"].update(
                            f"{status_prefix} after {n_model_iterations_done} iteration(s): "
                            f"RMS = {last_transit_result.residual_rms_ppt:.3f} ppt. {convergence_reason}."
                        )
                    else:
                        window["-STATUS-"].update(
                            f"Model-aware single-pass detrending complete and transit refitted: "
                            f"RMS = {last_transit_result.residual_rms_ppt:.3f} ppt."
                        )
                else:
                    window["-STATUS-"].update(
                        f"Detrending complete: RMS {last_detrend_result.rms_before_ppt:.2f} -> "
                        f"{last_detrend_result.rms_after_ppt:.2f} ppt "
                        f"({last_detrend_result.improvement_percent:+.1f}%)."
                    )

                if bool(values.get("-DET_SHOW_POPUP-", True)):
                    sg.popup_scrolled(
                        last_detrend_report,
                        title="Photometric detrending",
                        size=(96, 38),
                    )

                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)

            except Exception as exc:
                sg.popup_error(f"Could not run photometric detrending:\n{exc}")


        if event in ("-COMP_DIAG_STAR-", "-COMP_PLOT_DIAG-"):
            if df is None or not last_comp_diagnostics:
                continue
            try:
                selected_diag_star = str(values.get("-COMP_DIAG_STAR-", "")).strip()
                if selected_diag_star not in last_comp_diagnostics:
                    selected_diag_star = update_comparison_diagnostic_controls(window, last_comp_diagnostics, selected_diag_star)
                if bool(values.get("-COMP_PLOT_DIAG-", False)) and selected_diag_star in last_comp_diagnostics:
                    plot_values = set_comparison_diagnostic_output_columns(window, values, selected_diag_star)
                elif last_comp_result is not None and bool(values.get("-COMP_SEND_TO_DATA-", True)):
                    plot_values = set_comparison_output_columns(window, values)
                    pre_transit_column_selection = current_column_selection(plot_values)
                else:
                    plot_values = dict(values)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
                if bool(values.get("-COMP_PLOT_DIAG-", False)) and selected_diag_star:
                    window["-STATUS-"].update(f"Showing comparison diagnostic curve for {selected_diag_star}.")
                else:
                    window["-STATUS-"].update("Comparison diagnostic display disabled.")
            except Exception as exc:
                window["-STATUS-"].update(f"Comparison diagnostic display failed: {exc}")

        if event in ("-COMP_STAR_LIST-", "-COMP_SELECT_ALL-", "-COMP_SELECT_NONE-", "-COMP_TARGET-", "-COMP_CHECK-", "-COMP_MODE-"):
            if df is None or aij_flux_detection is None or not aij_flux_detection.compatible:
                continue

            try:
                rebuild_manual_curve = True

                if event == "-COMP_STAR_LIST-":
                    selected_items = values.get("-COMP_STAR_LIST-", []) or []
                    if not selected_items:
                        # This can happen when the listbox selection is cleared
                        # after a refresh.  In that case there is no star to
                        # toggle and the light curve should not be recomputed.
                        continue
                    clicked_star = _parse_comp_star_from_display(str(selected_items[0]))
                    if clicked_star in comp_active_stars:
                        comp_active_stars.remove(clicked_star)
                    elif clicked_star in aij_flux_detection.comparison_ids:
                        comp_active_stars.add(clicked_star)
                    else:
                        rebuild_manual_curve = False
                elif event == "-COMP_SELECT_ALL-":
                    comp_active_stars = set(aij_flux_detection.comparison_ids)
                elif event == "-COMP_SELECT_NONE-":
                    comp_active_stars = set()

                update_comparison_star_list(window, aij_flux_detection, comp_active_stars, disabled=False)

                if not comp_active_stars:
                    last_comp_diagnostics = {}
                    update_comparison_diagnostic_controls(window, last_comp_diagnostics, "")
                    window["-COMP_REPORT-"].update(
                        "No comparison star is active. Select at least one star or press All."
                    )
                    window["-STATUS-"].update("No active comparison stars selected.")
                    continue

                if not rebuild_manual_curve:
                    continue

                last_comp_result = run_manual_comparison_selection(
                    window,
                    df,
                    aij_flux_detection,
                    comp_active_stars,
                    values,
                    exoplanet_catalogue,
                )
                last_comp_report = last_comp_result.report

                downstream_cleared = (
                    last_detrend_result is not None
                    or last_transit_result is not None
                    or bool(auto_reject_indices)
                    or last_stats_report is not None
                    or last_recipe_report is not None
                    or any(
                        str(column).startswith("PhotoCurve_det_")
                        or str(column).startswith("PhotoCurve_time_")
                        or str(column).startswith("PhotoCurve_fit_")
                        or str(column).startswith("PhotoCurve_expected_")
                        or str(column).startswith("PhotoCurve_detrended_")
                        or str(column).startswith("PhotoCurve_baseline")
                        or str(column).startswith("PhotoCurve_predicted_")
                        or str(column).startswith("PhotoCurve_calculated_")
                        or str(column).startswith("PhotoCurve_oc_")
                        for column in df.columns
                    )
                )
                df = clear_downstream_analysis_columns(df)
                last_detrend_report = None
                last_detrend_result = None
                last_detrend_input_selection = None
                last_transit_report = None
                last_transit_result = None
                last_stats_report = None
                last_stats_blocks = []
                last_recipe_report = None
                auto_reject_indices.clear()
                update_auto_clip_controls(window, auto_reject_indices)
                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)
                try:
                    window["-DET_REPORT-"].update(
                        "Comparison stars changed. Previous detrending and transit-fit products were cleared "
                        "because the light curve changed."
                    )
                except Exception:
                    pass

                diag_x, last_comp_diagnostics = build_current_comparison_diagnostics(
                    df,
                    aij_flux_detection,
                    comp_active_stars,
                    values,
                )
                selected_diag_star = update_comparison_diagnostic_controls(
                    window,
                    last_comp_diagnostics,
                    str(values.get("-COMP_DIAG_STAR-", "")),
                )

                df = add_comparison_optimisation_columns(df, last_comp_result)
                df = add_comparison_diagnostic_columns(df, diag_x, last_comp_diagnostics)
                update_combo_values(window, numeric_columns(df))
                window["-COMP_REPORT-"].update(compose_comparison_report(last_comp_report, last_comp_diagnostics))

                plot_values = dict(values)
                if bool(values.get("-COMP_PLOT_DIAG-", False)) and selected_diag_star in last_comp_diagnostics:
                    plot_values = set_comparison_diagnostic_output_columns(window, values, selected_diag_star)
                elif bool(values.get("-COMP_SEND_TO_DATA-", True)):
                    plot_values = set_comparison_output_columns(window, values)
                    pre_transit_column_selection = current_column_selection(plot_values)

                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
                status_message = (
                    f"Manual comparison subset: {len(last_comp_result.selected_comparisons)} star(s); "
                    f"RMS = {last_comp_result.optimised_metric.rms_ppt:.2f} ppt"
                )
                if downstream_cleared:
                    status_message += (
                        " | Comparison stars changed: downstream analysis products "
                        "(detrending, transit fit, statistics and auto sigma clipping) "
                        "were cleared because the light curve changed."
                    )
                window["-STATUS-"].update(status_message)

            except Exception as exc:
                window["-STATUS-"].update(f"Manual comparison update failed: {exc}")

        if event == "-MANUAL_CLEAR_POINTS-":
            manual_reject_indices.clear()
            manual_keep_indices.clear()
            update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
            if df is not None:
                plot_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
            window["-STATUS-"].update("Manual point mask cleared.")

        if event in ("-MANUAL_CLEAN_ACTIVE-",):
            if df is not None:
                plot_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)

        if event == "-PLOT_POINT_CLICK-":
            if df is None:
                continue
            if not bool(values.get("-MANUAL_POINT_EDIT-", False)):
                continue

            try:
                payload = values.get("-PLOT_POINT_CLICK-", {}) or {}
                index = nearest_point_index_for_click(
                    df,
                    values,
                    float(payload.get("xdata")),
                    float(payload.get("ydata")),
                )
                if index is None:
                    window["-STATUS-"].update("Click was not close enough to a plotted data point.")
                    continue

                auto_reject_indices = _parse_index_set_text(values.get("-AUTO_REJECT_INDICES-", _format_index_set(auto_reject_indices)))
                manual_reject_indices = _parse_index_set_text(values.get("-MANUAL_REJECT_INDICES-", ""))
                manual_keep_indices = _parse_index_set_text(values.get("-MANUAL_KEEP_INDICES-", ""))

                # Determine whether the point is currently kept. This makes the
                # default toggle mode intuitive even when sigma clipping is also
                # active: clicking a rejected point forces it back in; clicking a
                # kept point rejects it manually.
                x_col = str(values.get("-XCOL-", NONE_COL))
                y_col = str(values.get("-YCOL-", NONE_COL))
                x = to_numeric_array(df, x_col)
                y = to_numeric_array(df, y_col)
                model = to_numeric_array(df, str(values.get("-MODEL_COL-", NONE_COL)))
                residuals = to_numeric_array(df, str(values.get("-RES_COL-", NONE_COL)))
                if x is None:
                    continue
                clean_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                cleaning = compute_cleaning_mask(x, y, model, residuals, clean_values)
                currently_kept = bool(index < len(cleaning.keep_mask) and cleaning.keep_mask[index])

                mode = str(values.get("-MANUAL_POINT_MODE-", "Toggle nearest"))
                if mode == "Reject nearest" or (mode == "Toggle nearest" and currently_kept):
                    manual_reject_indices.add(index)
                    manual_keep_indices.discard(index)
                    action = "rejected"
                else:
                    manual_keep_indices.add(index)
                    manual_reject_indices.discard(index)
                    action = "restored"

                window["-MANUAL_CLEAN_ACTIVE-"].update(value=True)
                update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)

                plot_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                plot_values["-MANUAL_CLEAN_ACTIVE-"] = True
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)
                window["-STATUS-"].update(f"Point {index} manually {action}.")

            except Exception as exc:
                sg.popup_error(f"Could not edit point manually:\n{exc}")

        if event == "Run comp optimizer":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            if aij_flux_detection is None or not aij_flux_detection.compatible:
                message = (
                    aij_flux_detection.warning
                    if aij_flux_detection is not None
                    else "The loaded table does not contain compatible AstroImageJ raw flux columns."
                )
                sg.popup_error(message)
                continue

            try:
                x_col = _get_comp_x_column(values, aij_flux_detection, df)
                x = to_numeric_array(df, x_col)
                if x is None:
                    raise ValueError("The selected comparison-star time column is not numeric.")

                current_flux = to_numeric_array(df, str(values.get("-YCOL-", NONE_COL)))

                planet = None
                if exoplanet_catalogue is not None and str(values.get("-TR_PLANET-", "")).strip():
                    try:
                        planet = find_planet(exoplanet_catalogue, str(values.get("-TR_PLANET-", "")))
                    except Exception:
                        planet = None

                max_stars_text = str(values.get("-COMP_MAX_STARS-", "auto")).strip().lower()
                if max_stars_text in ("", "auto", "all", "none"):
                    max_stars = None
                else:
                    max_stars = max(1, parse_int(max_stars_text, 0))

                min_stars = max(1, parse_int(values.get("-COMP_MIN_STARS-", 2), 2))
                polynomial_order = max(0, min(2, parse_int(values.get("-COMP_POLY_ORDER-", 1), 1)))
                improvement_threshold = parse_float(values.get("-COMP_IMPROVE_THRESHOLD-", 0.5), 0.5)
                if improvement_threshold is None:
                    improvement_threshold = 0.5

                last_comp_result = optimise_comparison_stars(
                    df,
                    aij_flux_detection,
                    x,
                    x_col,
                    current_flux,
                    target_id=str(values.get("-COMP_TARGET-", "")).strip(),
                    mode=str(values.get("-COMP_MODE-", "Target light curve")),
                    check_id=str(values.get("-COMP_CHECK-", "")).strip(),
                    planet=planet,
                    mask_expected_transit=bool(values.get("-COMP_MASK_TRANSIT-", True)),
                    min_stars=min_stars,
                    max_stars=max_stars,
                    polynomial_order=polynomial_order,
                    improvement_threshold_percent=float(improvement_threshold),
                    allowed_comparisons=sorted(comp_active_stars) if comp_active_stars else None,
                )
                last_comp_report = last_comp_result.report
                comp_active_stars = set(last_comp_result.selected_comparisons)

                downstream_cleared = (
                    last_detrend_result is not None
                    or last_transit_result is not None
                    or bool(auto_reject_indices)
                    or last_stats_report is not None
                    or last_recipe_report is not None
                    or any(
                        str(column).startswith("PhotoCurve_det_")
                        or str(column).startswith("PhotoCurve_time_")
                        or str(column).startswith("PhotoCurve_fit_")
                        or str(column).startswith("PhotoCurve_expected_")
                        or str(column).startswith("PhotoCurve_detrended_")
                        or str(column).startswith("PhotoCurve_baseline")
                        or str(column).startswith("PhotoCurve_predicted_")
                        or str(column).startswith("PhotoCurve_calculated_")
                        or str(column).startswith("PhotoCurve_oc_")
                        for column in df.columns
                    )
                )
                df = clear_downstream_analysis_columns(df)
                last_detrend_report = None
                last_detrend_result = None
                last_detrend_input_selection = None
                last_transit_report = None
                last_transit_result = None
                last_stats_report = None
                last_stats_blocks = []
                last_recipe_report = None
                auto_reject_indices.clear()
                update_auto_clip_controls(window, auto_reject_indices)
                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)
                try:
                    window["-DET_REPORT-"].update(
                        "Comparison stars changed. Previous detrending and transit-fit products were cleared "
                        "because the light curve changed."
                    )
                except Exception:
                    pass

                last_comp_diagnostics = build_comparison_diagnostics(
                    df,
                    aij_flux_detection,
                    x,
                    selected_comparisons=sorted(comp_active_stars),
                    polynomial_order=polynomial_order,
                )
                selected_diag_star = update_comparison_diagnostic_controls(
                    window,
                    last_comp_diagnostics,
                    str(values.get("-COMP_DIAG_STAR-", "")),
                )

                df = add_comparison_optimisation_columns(df, last_comp_result)
                df = add_comparison_diagnostic_columns(df, x, last_comp_diagnostics)
                cols = numeric_columns(df)
                update_combo_values(window, cols)
                update_comparison_star_list(window, aij_flux_detection, comp_active_stars, disabled=False)
                window["-COMP_REPORT-"].update(compose_comparison_report(last_comp_report, last_comp_diagnostics))

                plot_values = dict(values)
                if bool(values.get("-COMP_PLOT_DIAG-", False)) and selected_diag_star in last_comp_diagnostics:
                    plot_values = set_comparison_diagnostic_output_columns(window, values, selected_diag_star)
                elif bool(values.get("-COMP_SEND_TO_DATA-", True)):
                    plot_values = set_comparison_output_columns(window, values)
                    pre_transit_column_selection = current_column_selection(plot_values)

                status_message = (
                    f"Comparison optimizer: selected {len(last_comp_result.selected_comparisons)} star(s); "
                    f"RMS = {last_comp_result.optimised_metric.rms_ppt:.2f} ppt"
                )
                if downstream_cleared:
                    status_message += (
                        " | Comparison stars changed: downstream analysis products "
                        "(detrending, transit fit, statistics and auto sigma clipping) "
                        "were cleared because the light curve changed."
                    )
                window["-STATUS-"].update(status_message)

                if bool(values.get("-COMP_SHOW_POPUP-", True)):
                    sg.popup_scrolled(
                        compose_comparison_report(last_comp_report, last_comp_diagnostics),
                        title="Comparison-star optimizer",
                        size=(92, 36),
                    )

                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)

            except Exception as exc:
                sg.popup_error(f"Could not run comparison-star optimizer:\n{exc}")

        if event == "Run transit model":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                if exoplanet_catalogue is None:
                    catalog_path = str(values.get("-TR_CATALOG-", "")).strip()
                    exoplanet_catalogue = load_catalogue_into_window(
                        window,
                        catalog_path,
                        current_planet=str(values.get("-TR_PLANET-", "")).strip(),
                        autodetect_text=Path(current_photometry_file_path).name if current_photometry_file_path else "",
                        source_label="custom",
                    )

                planet = find_planet(exoplanet_catalogue, str(values.get("-TR_PLANET-", "")))

                current_selection = current_column_selection(values)
                if not contains_photocurve_columns(current_selection):
                    # This is the user's real input-column mapping. Keep it so
                    # the transit fit can be hidden later without reloading the
                    # photometry file.
                    pre_transit_column_selection = current_selection.copy()
                elif pre_transit_column_selection is None:
                    pre_transit_column_selection = (
                        original_column_selection.copy() if original_column_selection is not None else current_selection.copy()
                    )

                x_col = str(values.get("-XCOL-", NONE_COL))
                y_col = str(values.get("-YCOL-", NONE_COL))
                yerr_col = str(values.get("-YERRCOL-", NONE_COL))
                model_col = str(values.get("-MODEL_COL-", NONE_COL))
                res_col = str(values.get("-RES_COL-", NONE_COL))

                x = to_numeric_array(df, x_col)
                y = to_numeric_array(df, y_col)
                yerr = to_numeric_array(df, yerr_col)
                model = to_numeric_array(df, model_col)
                residuals = to_numeric_array(df, res_col)

                if x is None or y is None:
                    raise ValueError("Select X/time and light-curve columns before running transit diagnostics.")

                # If a previous diagnostic run selected PhotoCurve_time_corrected
                # as the current X column, do not use it as the new input time.
                # Otherwise the mid-exposure correction would be applied again
                # at every click, shifting the expected model run after run.
                x_for_diagnostics = x
                if x_col == "PhotoCurve_time_corrected" and "PhotoCurve_time_input" in df.columns:
                    stored_input_time = to_numeric_array(df, "PhotoCurve_time_input")
                    if stored_input_time is not None:
                        x_for_diagnostics = stored_input_time

                # Likewise, if the current GUI selection is a detrended
                # PhotoCurve column from a previous run, go back to the stored
                # original photometry before fitting again. This prevents
                # repeated baseline divisions when the user presses Run transit
                # diag multiple times.
                y_for_diagnostics = y
                yerr_for_diagnostics = yerr
                if y_col == "PhotoCurve_detrended_flux" and "PhotoCurve_flux_input" in df.columns:
                    stored_flux = to_numeric_array(df, "PhotoCurve_flux_input")
                    if stored_flux is not None:
                        y_for_diagnostics = stored_flux
                if yerr_col == "PhotoCurve_detrended_err" and "PhotoCurve_flux_err_input" in df.columns:
                    stored_flux_err = to_numeric_array(df, "PhotoCurve_flux_err_input")
                    if stored_flux_err is not None:
                        yerr_for_diagnostics = stored_flux_err

                clean_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                cleaning = compute_cleaning_mask(x_for_diagnostics, y_for_diagnostics, model, residuals, clean_values)

                last_transit_result = run_transit_diagnostics(
                    x_for_diagnostics,
                    y_for_diagnostics,
                    yerr_for_diagnostics,
                    planet,
                    values,
                    keep_mask=cleaning.keep_mask,
                )
                last_recipe_report = build_reproducibility_recipe(
                    df,
                    values,
                    current_photometry_file_path,
                    original_column_selection,
                    pre_transit_column_selection,
                    aij_flux_detection,
                    comp_active_stars,
                    last_comp_result,
                    detrend_detection,
                    detrend_active_regressors,
                    last_detrend_result,
                    last_detrend_input_selection,
                    auto_reject_indices,
                    manual_reject_indices,
                    manual_keep_indices,
                    last_transit_result,
                )
                last_transit_report = format_transit_report(last_transit_result) + "\n\n" + last_recipe_report

                # Always store the diagnostic columns after a run. The
                # checkbox controls whether they are displayed, not whether the
                # result exists. This makes it possible to turn the fit on and
                # off without rerunning the analysis.
                df = add_transit_diagnostic_columns(
                    df,
                    last_transit_result,
                    x_for_diagnostics,
                    y_for_diagnostics,
                    yerr_for_diagnostics,
                )
                df = apply_transit_display_aliases(
                    df,
                    str(values.get("-TR_DISPLAY_MODE-", "Detrended flux")),
                )
                cols = numeric_columns(df)
                update_combo_values(window, cols)

                show_transit_result = bool(values.get("-TR_SET_MODEL_COLUMNS-", True))
                plot_values = set_transit_plot_columns(
                    window,
                    values,
                    str(values.get("-TR_DISPLAY_MODE-", "Detrended flux")),
                    pre_transit_column_selection or original_column_selection,
                    show_transit_result,
                )
                plot_values = _with_current_masks(plot_values, auto_reject_indices, manual_reject_indices, manual_keep_indices)

                window["-STATUS-"].update(
                    f"Transit diagnostics: O-C = {last_transit_result.oc_minutes:+.2f} min, "
                    f"SNR = {last_transit_result.transit_snr:.1f}"
                )

                if bool(values.get("-TR_SHOW_POPUP-", True)):
                    sg.popup_scrolled(
                        last_transit_report,
                        title="Transit diagnostics",
                        size=(92, 36),
                    )

                # Refresh the plot automatically. If the transit-fit display
                # switch is off, the original column mapping is used instead.
                update_detrend_fit_model_control(window, detrend_detection, last_transit_result)
                current_fig, fig_agg = redraw_plot(window, fig_agg, df, plot_values)

            except Exception as exc:
                sg.popup_error(f"Could not run transit diagnostics:\n{exc}")

        if event == "Save curve":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            save_path = sg.popup_get_file(
                "Save processed light curve",
                save_as=True,
                no_window=True,
                default_extension=".txt",
                file_types=(
                    ("Tab-separated text", "*.txt;*.dat;*.tsv"),
                    ("CSV table", "*.csv"),
                    ("JSON", "*.json"),
                    ("All files", "*.*"),
                ),
            )

            if save_path:
                try:
                    curve_table, curve_metadata = build_processed_light_curve_export(
                        df,
                        values,
                        current_photometry_file_path,
                        values.get("-TR_CATALOG-", ""),
                        last_transit_result,
                        auto_reject_indices,
                        manual_reject_indices,
                        manual_keep_indices,
                    )
                    save_processed_light_curve_file(save_path, curve_table, curve_metadata)

                    saved_paths = [f"Processed light curve: {save_path}"]
                    simple_warning = ""
                    if bool(values.get("-EXPORT_SIMPLE_EXOCLOCK-", True)):
                        try:
                            simple_table, simple_metadata = build_simple_exoclock_hops_export(curve_table, curve_metadata)
                            simple_path = simple_exoclock_hops_export_path(save_path, simple_metadata)
                            save_simple_exoclock_hops_curve_file(simple_path, simple_table, simple_metadata)
                            if bool(simple_metadata.get("exoclock_hops_ready", False)):
                                saved_paths.append(f"ExoClock/HOPS simple light curve: {simple_path}")
                            else:
                                saved_paths.append(f"Simple light curve (not JD_UTC-labelled): {simple_path}")
                                simple_warning = str(simple_metadata.get("warning", ""))
                        except Exception as simple_exc:
                            simple_warning = f"Simple ExoClock/HOPS export was not created: {simple_exc}"

                    status_text = "Saved " + "; ".join(saved_paths)
                    if simple_warning:
                        status_text += f" | {simple_warning}"
                    window["-STATUS-"].update(status_text[:220])
                    if simple_warning:
                        sg.popup_scrolled(status_text, title="Save curve", size=(88, 12))
                except Exception as exc:
                    sg.popup_error(f"Could not save processed light curve:\n{exc}")


        if event == "Save stats":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            if last_stats_report is None:
                try:
                    stats_values = _with_current_masks(values, auto_reject_indices, manual_reject_indices, manual_keep_indices)
                    last_stats_report, last_stats_blocks = build_statistics_report(df, stats_values)
                except Exception as exc:
                    sg.popup_error(f"Could not compute statistics:\n{exc}")
                    continue

            save_path = sg.popup_get_file(
                "Save statistics report",
                save_as=True,
                no_window=True,
                default_extension=".txt",
                file_types=(
                    ("Text report", "*.txt"),
                    ("CSV table", "*.csv"),
                    ("JSON", "*.json"),
                    ("All files", "*.*"),
                ),
            )

            if save_path:
                try:
                    save_statistics_file(save_path, last_stats_report, last_stats_blocks)
                    window["-STATUS-"].update(f"Statistics saved: {save_path}")
                except Exception as exc:
                    sg.popup_error(f"Could not save statistics:\n{exc}")

        if event == "Save recipe":
            if df is None:
                sg.popup_error("Load a table first.")
                continue

            try:
                recipe_to_save = build_reproducibility_recipe(
                    df,
                    values,
                    current_photometry_file_path,
                    original_column_selection,
                    pre_transit_column_selection,
                    aij_flux_detection,
                    comp_active_stars,
                    last_comp_result,
                    detrend_detection,
                    detrend_active_regressors,
                    last_detrend_result,
                    last_detrend_input_selection,
                    auto_reject_indices,
                    manual_reject_indices,
                    manual_keep_indices,
                    last_transit_result,
                )
            except Exception as exc:
                sg.popup_error(f"Could not build reproducibility recipe:\n{exc}")
                continue

            save_path = sg.popup_get_file(
                "Save reproducibility recipe",
                save_as=True,
                no_window=True,
                default_extension=".txt",
                file_types=(
                    ("Text recipe", "*.txt"),
                    ("JSON", "*.json"),
                    ("All files", "*.*"),
                ),
            )

            if save_path:
                try:
                    save_reproducibility_recipe_file(save_path, recipe_to_save)
                    last_recipe_report = recipe_to_save
                    window["-STATUS-"].update(f"Recipe saved: {save_path}")
                except Exception as exc:
                    sg.popup_error(f"Could not save recipe:\n{exc}")

        if event == "Save model results":
            if last_transit_report is None or last_transit_result is None:
                sg.popup_error("Run transit model first.")
                continue

            save_path = sg.popup_get_file(
                "Save transit diagnostics report",
                save_as=True,
                no_window=True,
                default_extension=".txt",
                file_types=(
                    ("Text report", "*.txt"),
                    ("CSV table", "*.csv"),
                    ("JSON", "*.json"),
                    ("All files", "*.*"),
                ),
            )

            if save_path:
                try:
                    save_transit_diagnostic_file(save_path, last_transit_report, last_transit_result)
                    window["-STATUS-"].update(f"Transit diagnostics and recipe saved: {save_path}")
                except Exception as exc:
                    sg.popup_error(f"Could not save transit diagnostics:\n{exc}")

        if event == "Save figure":
            if current_fig is None:
                sg.popup_error("Create a plot first.")
                continue

            save_path = sg.popup_get_file(
                "Save figure as",
                save_as=True,
                no_window=True,
                default_extension=".png",
                file_types=(
                    ("PNG", "*.png"),
                    ("PDF", "*.pdf"),
                    ("SVG", "*.svg"),
                    ("All files", "*.*"),
                ),
            )

            if save_path:
                try:
                    current_fig.savefig(save_path, bbox_inches="tight")
                    window["-STATUS-"].update(f"Figure saved: {save_path}")
                except Exception as exc:
                    sg.popup_error(f"Could not save figure:\n{exc}")

        # if event == "Save settings":
        #     save_path = sg.popup_get_file(
        #         "Save settings as JSON",
        #         save_as=True,
        #         no_window=True,
        #         default_extension=".json",
        #         file_types=(
        #             ("JSON", "*.json"),
        #             ("All files", "*.*"),
        #         ),
        #     )
        # 
        #     if save_path:
        #         try:
        #             save_config(save_path, values)
        #             window["-STATUS-"].update(f"Settings saved: {save_path}")
        #         except Exception as exc:
        #             sg.popup_error(f"Could not save settings:\n{exc}")
        # 
        # if event == "Load settings":
        #     config_path = sg.popup_get_file(
        #         "Load settings JSON",
        #         no_window=True,
        #         file_types=(
        #             ("JSON", "*.json"),
        #             ("All files", "*.*"),
        #         ),
        #     )
        # 
        #     if config_path:
        #         try:
        #             config = load_config(config_path)
        #             apply_config(window, config)
        #             auto_reject_indices = _parse_index_set_text(config.get("-AUTO_REJECT_INDICES-", ""))
        #             manual_reject_indices = _parse_index_set_text(config.get("-MANUAL_REJECT_INDICES-", ""))
        #             manual_keep_indices = _parse_index_set_text(config.get("-MANUAL_KEEP_INDICES-", ""))
        #             update_auto_clip_controls(window, auto_reject_indices)
        #             update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
        # 
        #             file_path = str(config.get("-FILE-", "")).strip()
        # 
        #             if file_path and os.path.exists(file_path):
        #                 df = read_ascii_table(
        #                     file_path,
        #                     str(config.get("-DELIM-", "Auto")),
        #                     bool(config.get("-HEADER-", True)),
        #                 )
        # 
        #                 original_df = df.copy(deep=True)
        #                 cols = numeric_columns(df)
        #                 original_column_selection = update_combo_values(window, cols, df=df, sync_transit_time_system=True)
        #                 pre_transit_column_selection = original_column_selection.copy()
        #                 last_detrend_input_selection = None
        #                 aij_flux_detection = detect_aij_flux_columns(df)
        #                 comp_active_stars = set(aij_flux_detection.comparison_ids) if aij_flux_detection.compatible else set()
        #                 update_comparison_tab(window, aij_flux_detection, comp_active_stars)
        #                 auto_reject_indices.clear()
        #                 manual_reject_indices.clear()
        #                 manual_keep_indices.clear()
        #                 update_auto_clip_controls(window, auto_reject_indices)
        #                 update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
        #                 apply_config(window, config)
        #                 auto_reject_indices = _parse_index_set_text(config.get("-AUTO_REJECT_INDICES-", ""))
        #                 manual_reject_indices = _parse_index_set_text(config.get("-MANUAL_REJECT_INDICES-", ""))
        #                 manual_keep_indices = _parse_index_set_text(config.get("-MANUAL_KEEP_INDICES-", ""))
        #                 update_auto_clip_controls(window, auto_reject_indices)
        #                 update_manual_point_controls(window, manual_reject_indices, manual_keep_indices)
        # 
        #                 window["-NROWS-"].update(str(len(df)))
        #                 window["-NCOLS-"].update(str(len(df.columns)))
        # 
        #             window["-STATUS-"].update(f"Settings loaded: {config_path}")
        # 
        #         except Exception as exc:
        #             sg.popup_error(f"Could not load settings:\n{exc}")

    delete_figure_agg(fig_agg)
    window.close()


if __name__ == "__main__":
    main()
