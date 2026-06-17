"""Matplotlib plotting utilities for PhotoCurve Lab."""

from __future__ import annotations

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .axis_utils import transform_x_axis
from .binning_utils import bin_light_curve
from .cleaning_utils import compute_cleaning_mask
from .numeric_utils import format_number, finite_mask, parse_float, parse_int, to_numeric_array
from .statistics_utils import compute_residual_statistics
from .sg_loader import sg


def draw_figure(canvas_elem: sg.Canvas, figure: plt.Figure) -> FigureCanvasTkAgg:
    """Draw a matplotlib figure inside a Tk canvas."""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas_elem.TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=True)
    return figure_canvas_agg


def delete_figure_agg(figure_canvas_agg: Optional[FigureCanvasTkAgg]) -> None:
    """Remove the old matplotlib figure from the Tk canvas."""
    if figure_canvas_agg is not None:
        figure_canvas_agg.get_tk_widget().destroy()
        plt.close(figure_canvas_agg.figure)


def legend_label(enabled: bool, text: str) -> str:
    """Return a matplotlib legend label or suppress the legend entry."""
    return text if enabled else "_nolegend_"


def legend_location(value: object) -> str:
    """Translate the GUI legend-location choice into a matplotlib location."""
    location_map = {
        "Auto": "best",
        "Lower left corner": "lower left",
        "Lower right corner": "lower right",
        "Upper left corner": "upper left",
        "Upper right corner": "upper right",
    }

    return location_map.get(str(value), "best")


def set_errorbar_alpha(container, alpha: float) -> None:
    """Set alpha for the error-bar lines without changing the point alpha."""
    try:
        for barlinecol in container[2]:
            barlinecol.set_alpha(alpha)
        for capline in container[1]:
            capline.set_alpha(alpha)
    except Exception:
        pass


def apply_grid(ax: plt.Axes, enabled: bool, grid_alpha: float) -> None:
    """Apply a thin grid to one axis."""
    if enabled:
        ax.grid(True, which="both", lw=0.4, alpha=grid_alpha)


def apply_scientific_ticks(ax: plt.Axes) -> None:
    """Apply inward ticks and minor ticks for a cleaner scientific look."""
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, which="both")


def add_residual_statistics_text(
    ax: plt.Axes,
    residuals: np.ndarray,
    keep_mask: Optional[np.ndarray] = None,
) -> None:
    """Add residual RMS and N to the lower-right corner of an axis.

    If a cleaning mask is supplied, the statistics refer to the cleaned
    residuals actually shown in the plot.
    """
    if keep_mask is not None and keep_mask.shape == residuals.shape:
        residuals_for_stats = residuals[keep_mask]
    else:
        residuals_for_stats = residuals

    rms, n_points = compute_residual_statistics(residuals_for_stats)

    if n_points <= 0 or not np.isfinite(rms):
        return

    rms_ppt = rms * 1000.0
    rms_text = f"RMS = {rms_ppt:.2f} ppt\nN = {n_points:d}"

    ax.text(
        0.97,
        0.04,
        rms_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
        ),
        zorder=10,
    )


def add_residual_offset_text(ax: plt.Axes, res_offset: float) -> None:
    """Annotate the residual vertical offset used in single-panel mode."""
    if abs(res_offset) < 1e-15:
        text = "Residuals not shifted"
    elif res_offset > 0:
        text = f"Residuals + {format_number(res_offset)}"
    else:
        text = f"Residuals - {format_number(abs(res_offset))}"

    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.25",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.65,
        ),
        zorder=10,
    )


def first_finite_column_value(df: pd.DataFrame, column_name: str) -> float:
    """Return the first finite value in a column, or NaN if unavailable."""
    if column_name not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column_name], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[0])


def transform_time_marker(time_value: float, x_mode: str, x_offset: float) -> float:
    """Transform a BJD/JD time marker in the same way as the plot X axis."""
    if not np.isfinite(time_value):
        return float("nan")
    if x_mode == "X - offset":
        return float(time_value - x_offset)
    if x_mode == "Hours from offset":
        return float((time_value - x_offset) * 24.0)
    return float(time_value)


def format_timing_marker_value(x_value: float, x_mode: str) -> str:
    """Return a compact label value for a timing marker."""
    if not np.isfinite(x_value):
        return ""
    if x_mode == "Hours from offset":
        return f"{x_value:.2f} h"
    if abs(x_value) >= 1000:
        return f"{x_value:.5f}"
    return f"{x_value:.5f}"


def draw_timing_marker(
    ax: plt.Axes,
    x_value: float,
    label: str,
    colour: str,
    linestyle: str,
    y_text: float,
    alpha: float = 0.60,
    linewidth: float = 0.9,
) -> None:
    """Draw one vertical timing marker with a small rotated label."""
    if not np.isfinite(x_value):
        return

    ax.axvline(
        x_value,
        color=colour,
        ls=linestyle,
        lw=linewidth,
        alpha=alpha,
        zorder=1.8,
    )
    ax.text(
        x_value,
        y_text,
        label,
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=7.5,
        color=colour,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.65,
        ),
        clip_on=True,
        zorder=10,
    )




def _resolve_fractional_time_from_axis(x_values: np.ndarray, fraction_value: object) -> float:
    """Resolve a fractional JD-like time entered by the user to a full time.

    AstroImageJ asks for meridian-flip times as the fractional part of the JD.
    For example, for JD_UTC = 2461203.771 the user enters .771.  If the
    supplied value already looks like a full JD/MJD, it is left unchanged.
    """
    fraction = parse_float(fraction_value, float("nan"))
    if fraction is None or not np.isfinite(fraction):
        return float("nan")
    if abs(float(fraction)) >= 1000.0:
        return float(fraction)

    finite = np.asarray(x_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    reference = float(np.nanmedian(finite))
    integer_part = np.floor(reference)
    return float(integer_part + float(fraction))


def add_meridian_flip_marker(
    ax: plt.Axes,
    x_raw: np.ndarray,
    values: Dict[str, object],
    x_offset: float,
) -> None:
    """Add an optional live meridian-flip marker to the light-curve panel."""
    if not bool(values.get("-DET_FLIP_ACTIVE-", False)):
        return
    if not bool(values.get("-DET_SHOW_FLIP_MARKER-", True)):
        return

    flip_time = _resolve_fractional_time_from_axis(x_raw, values.get("-DET_FLIP_FRAC-", ""))
    if not np.isfinite(flip_time):
        return

    x_mode = str(values.get("-XMODE-", "Raw X"))
    x_value = transform_time_marker(flip_time, x_mode, x_offset)
    if not np.isfinite(x_value):
        return

    frac = flip_time - np.floor(flip_time)
    draw_timing_marker(
        ax,
        x_value,
        f"Meridian flip\n.{frac:.5f}".replace(".0.", "."),
        "0.25",
        "-.",
        0.52,
        alpha=0.55,
        linewidth=0.9,
    )
def add_transit_timing_markers(
    ax: plt.Axes,
    df: pd.DataFrame,
    values: Dict[str, object],
    x_offset: float,
    expected_colour: str,
    calculated_colour: str,
) -> None:
    """Add optional ExoClock-style timing markers to the light-curve panel.

    The markers are shown only after a PhotoCurve transit diagnostic run and
    only when the transit-fit display switch is enabled.  The vertical lines are
    deliberately thin and semi-transparent so that they guide the eye without
    hiding the photometry.
    """
    show_transit_result = bool(values.get("-TR_SET_MODEL_COLUMNS-", True))
    if not show_transit_result:
        return

    show_predicted = bool(values.get("-TR_SHOW_PREDICTED_TIMES-", False))
    show_calculated = bool(values.get("-TR_SHOW_CALCULATED_TIMES-", False))
    if not (show_predicted or show_calculated):
        return

    required = [
        "PhotoCurve_predicted_start_time",
        "PhotoCurve_predicted_tmid_time",
        "PhotoCurve_predicted_end_time",
        "PhotoCurve_calculated_start_time",
        "PhotoCurve_calculated_tmid_time",
        "PhotoCurve_calculated_end_time",
    ]
    if not any(column in df.columns for column in required):
        return

    x_mode = str(values.get("-XMODE-", "Raw X"))

    predicted = {
        "Predicted start": first_finite_column_value(df, "PhotoCurve_predicted_start_time"),
        "Prediction Tmid": first_finite_column_value(df, "PhotoCurve_predicted_tmid_time"),
        "Predicted end": first_finite_column_value(df, "PhotoCurve_predicted_end_time"),
    }
    calculated = {
        "Calculated start": first_finite_column_value(df, "PhotoCurve_calculated_start_time"),
        "Calculated Tmid": first_finite_column_value(df, "PhotoCurve_calculated_tmid_time"),
        "Calculated end": first_finite_column_value(df, "PhotoCurve_calculated_end_time"),
    }

    predicted_x = {name: transform_time_marker(value, x_mode, x_offset) for name, value in predicted.items()}
    calculated_x = {name: transform_time_marker(value, x_mode, x_offset) for name, value in calculated.items()}

    if show_predicted:
        for name, linestyle, y_text, lw, alpha in [
            ("Predicted start", "--", 0.98, 0.8, 0.50),
            ("Prediction Tmid", "--", 0.86, 1.1, 0.75),
            ("Predicted end", "--", 0.98, 0.8, 0.50),
        ]:
            x_value = predicted_x[name]
            compact_value = format_timing_marker_value(x_value, x_mode)
            draw_timing_marker(
                ax,
                x_value,
                f"{name}\n{compact_value}",
                expected_colour,
                linestyle,
                y_text,
                alpha=alpha,
                linewidth=lw,
            )

    if show_calculated:
        for name, linestyle, y_text, lw, alpha in [
            ("Calculated start", ":", 0.74, 0.8, 0.50),
            ("Calculated Tmid", ":", 0.62, 1.1, 0.75),
            ("Calculated end", ":", 0.74, 0.8, 0.50),
        ]:
            x_value = calculated_x[name]
            compact_value = format_timing_marker_value(x_value, x_mode)
            draw_timing_marker(
                ax,
                x_value,
                f"{name}\n{compact_value}",
                calculated_colour,
                linestyle,
                y_text,
                alpha=alpha,
                linewidth=lw,
            )

    # If both timing families are visible, add a compact O-C annotation between
    # the predicted and calculated mid-transit markers.  This avoids a crowded
    # extra vertical label while still making the timing offset obvious.
    if show_predicted and show_calculated:
        x_pred = predicted_x.get("Prediction Tmid", float("nan"))
        x_calc = calculated_x.get("Calculated Tmid", float("nan"))
        oc_minutes = first_finite_column_value(df, "PhotoCurve_oc_minutes")
        if np.isfinite(x_pred) and np.isfinite(x_calc) and np.isfinite(oc_minutes):
            x_mid = 0.5 * (x_pred + x_calc)
            ax.annotate(
                "",
                xy=(x_calc, 0.93),
                xytext=(x_pred, 0.93),
                xycoords=ax.get_xaxis_transform(),
                textcoords=ax.get_xaxis_transform(),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="0.25",
                    lw=0.8,
                    alpha=0.65,
                ),
                zorder=9,
            )
            ax.text(
                x_mid,
                0.935,
                f"O-C = {oc_minutes:+.2f} min",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="0.20",
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
                clip_on=True,
                zorder=10,
            )


def build_plot(df: pd.DataFrame, values: Dict[str, object]) -> plt.Figure:
    """Create the light-curve plot from the current GUI settings."""
    fig_w = parse_float(values.get("-FIG_W-", 8.0), 8.0) or 8.0
    fig_h = parse_float(values.get("-FIG_H-", 5.0), 5.0) or 5.0
    dpi = parse_int(values.get("-DPI-", 120), 120)

    x_col = str(values.get("-XCOL-", "-- None --"))
    y_col = str(values.get("-YCOL-", "-- None --"))

    x = to_numeric_array(df, x_col)
    y = to_numeric_array(df, y_col)
    yerr = to_numeric_array(df, str(values.get("-YERRCOL-", "-- None --")))
    model_col = str(values.get("-MODEL_COL-", "-- None --"))
    model = to_numeric_array(df, model_col)
    expected_model = None
    # The expected transit model is an auxiliary PhotoCurve product. Show it
    # only when the transit-fit display switch is active and the PhotoCurve fit
    # model is the selected model column. This lets the user hide the transit
    # diagnostic result without having to reload the photometry table.
    show_transit_result = bool(values.get("-TR_SET_MODEL_COLUMNS-", True))
    if show_transit_result and model_col == "PhotoCurve_fit_model" and "PhotoCurve_expected_model" in df.columns:
        expected_model = to_numeric_array(df, "PhotoCurve_expected_model")
    residuals = to_numeric_array(df, str(values.get("-RES_COL-", "-- None --")))
    residual_err = to_numeric_array(df, str(values.get("-RESERR_COL-", "-- None --")))

    if x is None or y is None:
        raise ValueError("Select at least X and light-curve Y columns.")

    xlabel_text = str(values.get("-XLABEL-", "")).strip()
    x_plot, xlabel, x_offset = transform_x_axis(x, values, x_col, xlabel_text)

    cleaning = compute_cleaning_mask(x, y, model, residuals, values)
    keep_mask = cleaning.keep_mask
    rejected_mask = cleaning.rejected_mask

    plot_layout = str(values.get("-PLOT_LAYOUT-", "Single panel"))
    two_panels = plot_layout == "Two panels" and residuals is not None

    if two_panels:
        fig, (ax_lc, ax_res) = plt.subplots(
            2,
            1,
            figsize=(fig_w, fig_h),
            dpi=dpi,
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        )
    else:
        fig, ax_lc = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax_res = ax_lc

    lc_offset = parse_float(values.get("-LC_OFFSET-", 0.0), 0.0) or 0.0
    res_offset = parse_float(values.get("-RES_OFFSET-", -0.03), -0.03) or 0.0
    zero_offset = parse_float(values.get("-ZERO_OFFSET-", res_offset), res_offset) or 0.0

    marker = str(values.get("-MARKER-", "o"))
    res_marker = str(values.get("-RES_MARKER-", "o"))

    marker_fmt = "none" if marker == "none" else marker
    res_marker_fmt = "none" if res_marker == "none" else res_marker

    marker_size = parse_float(values.get("-MSIZE-", 4.0), 4.0) or 4.0
    point_alpha = parse_float(values.get("-ALPHA-", 0.85), 0.85) or 0.85
    err_alpha = parse_float(values.get("-ERR_ALPHA-", 0.55), 0.55) or 0.55
    line_width = parse_float(values.get("-LW-", 1.2), 1.2) or 1.2

    lc_colour = str(values.get("-LC_COLOUR-", "black"))
    model_colour = str(values.get("-MODEL_COLOUR-", "red"))
    expected_model_colour = str(values.get("-EXPECTED_MODEL_COLOUR-", "tab:cyan"))
    res_colour = str(values.get("-RES_COLOUR-", "blue"))
    zero_colour = str(values.get("-ZERO_COLOUR-", "grey"))
    err_colour = str(values.get("-ERR_COLOUR-", "grey"))

    show_lc_legend = bool(values.get("-LEG_LC-", True))
    show_model_legend = bool(values.get("-LEG_MODEL-", True))
    show_res_legend = bool(values.get("-LEG_RES-", True))
    show_zero_legend = bool(values.get("-LEG_ZERO-", False))

    mask_lc = finite_mask(x_plot, y, yerr)
    if yerr is None:
        mask_lc = finite_mask(x_plot, y)
    mask_lc &= keep_mask

    yerr_plot = None if yerr is None else np.abs(yerr[mask_lc])

    lc_container = ax_lc.errorbar(
        x_plot[mask_lc],
        y[mask_lc] + lc_offset,
        yerr=yerr_plot,
        fmt=marker_fmt,
        ms=marker_size,
        color=lc_colour,
        ecolor=err_colour,
        elinewidth=0.8,
        capsize=0,
        alpha=point_alpha,
        label=legend_label(show_lc_legend, "Light curve"),
        linestyle="none",
        zorder=3,
    )
    set_errorbar_alpha(lc_container, err_alpha)

    if cleaning.enabled and bool(values.get("-CLEAN_SHOW_REJECTED-", True)):
        rej_lc_mask = rejected_mask & finite_mask(x_plot, y)
        if np.any(rej_lc_mask):
            rej_marker = str(values.get("-CLEAN_REJ_MARKER-", "x"))
            rej_marker_fmt = "none" if rej_marker == "none" else rej_marker
            rej_colour = str(values.get("-CLEAN_REJ_COLOUR-", "tab:red"))
            rej_size = parse_float(values.get("-CLEAN_REJ_SIZE-", 6.0), 6.0) or 6.0
            rej_alpha = parse_float(values.get("-CLEAN_REJ_ALPHA-", 0.9), 0.9) or 0.9
            rej_legend = bool(values.get("-CLEAN_REJ_LEGEND-", True))

            ax_lc.plot(
                x_plot[rej_lc_mask],
                y[rej_lc_mask] + lc_offset,
                linestyle="none",
                marker=rej_marker_fmt,
                ms=rej_size,
                color=rej_colour,
                alpha=rej_alpha,
                label=legend_label(rej_legend, f"Rejected ({cleaning.n_rejected:d})"),
                zorder=7,
            )

    # Plot the expected catalogue model, when available, separately from the
    # fitted diagnostic model. This mirrors the ExoClock-style comparison:
    # expected model versus best-fit model.
    if expected_model is not None:
        mask_expected = finite_mask(x_plot, expected_model)
        if np.any(mask_expected):
            order_expected = np.argsort(x_plot[mask_expected])
            ax_lc.plot(
                x_plot[mask_expected][order_expected],
                expected_model[mask_expected][order_expected] + lc_offset,
                color=expected_model_colour,
                lw=max(0.8, line_width),
                ls="--",
                label=legend_label(show_model_legend, "Expected model"),
                zorder=3.5,
            )

    if model is not None:
        mask_model = finite_mask(x_plot, model)
        order = np.argsort(x_plot[mask_model])

        model_label = "Best-fit model" if model_col == "PhotoCurve_fit_model" else "Model"
        ax_lc.plot(
            x_plot[mask_model][order],
            model[mask_model][order] + lc_offset,
            color=model_colour,
            lw=line_width,
            label=legend_label(show_model_legend, model_label),
            zorder=4,
        )

    add_transit_timing_markers(
        ax_lc,
        df,
        values,
        x_offset,
        expected_model_colour,
        model_colour,
    )
    add_meridian_flip_marker(ax_lc, x, values, x_offset)

    if bool(values.get("-BIN_ACTIVE-", False)):
        bin_n = max(1, parse_int(values.get("-BIN_N-", 4), 4))
        yerr_clean = None if yerr is None else yerr[keep_mask]
        x_bin, y_bin, yerr_bin, n_bin = bin_light_curve(x_plot[keep_mask], y[keep_mask], yerr_clean, bin_n)

        if x_bin.size > 0:
            bin_marker = str(values.get("-BIN_MARKER-", "s"))
            bin_marker_fmt = "none" if bin_marker == "none" else bin_marker
            bin_colour = str(values.get("-BIN_COLOUR-", "tab:orange"))
            bin_err_colour = str(values.get("-BIN_ERR_COLOUR-", bin_colour))
            bin_marker_size = parse_float(values.get("-BIN_MARKER_SIZE-", 6.0), 6.0) or 6.0
            bin_alpha = parse_float(values.get("-BIN_ALPHA-", 0.95), 0.95) or 0.95
            show_bin_errors = bool(values.get("-BIN_SHOW_ERR-", True))
            show_bin_legend = bool(values.get("-BIN_LEGEND-", True))

            if show_bin_errors and yerr_bin is not None:
                yerr_bin_plot = np.abs(yerr_bin)
            else:
                yerr_bin_plot = None

            bin_label = f"Binned data ({bin_n:d} pts/bin)"
            if np.any(n_bin != bin_n):
                bin_label = f"Binned data ({bin_n:d} pts/bin; last partial)"

            bin_container = ax_lc.errorbar(
                x_bin,
                y_bin + lc_offset,
                yerr=yerr_bin_plot,
                fmt=bin_marker_fmt,
                ms=bin_marker_size,
                color=bin_colour,
                ecolor=bin_err_colour,
                elinewidth=1.0,
                capsize=0,
                alpha=bin_alpha,
                label=legend_label(show_bin_legend, bin_label),
                linestyle="none",
                zorder=6,
            )
            set_errorbar_alpha(bin_container, bin_alpha)

    if residuals is not None:
        mask_res = finite_mask(x_plot, residuals, residual_err)
        if residual_err is None:
            mask_res = finite_mask(x_plot, residuals)
        mask_res &= keep_mask

        if two_panels:
            residual_plot = residuals[mask_res] * 1000.0
            res_err_plot = None if residual_err is None else np.abs(residual_err[mask_res]) * 1000.0
            zero_line_y = 0.0
            res_ylabel = "Residuals [ppt]"
        else:
            residual_plot = residuals[mask_res] + res_offset
            res_err_plot = None if residual_err is None else np.abs(residual_err[mask_res])
            zero_line_y = zero_offset
            res_ylabel = None

        res_container = ax_res.errorbar(
            x_plot[mask_res],
            residual_plot,
            yerr=res_err_plot,
            fmt=res_marker_fmt,
            ms=marker_size,
            color=res_colour,
            ecolor=res_colour,
            elinewidth=0.8,
            capsize=0,
            alpha=point_alpha,
            label=legend_label(show_res_legend and not two_panels, "Residuals"),
            linestyle="none",
            zorder=2,
        )
        set_errorbar_alpha(res_container, err_alpha)

        if cleaning.enabled and bool(values.get("-CLEAN_SHOW_REJECTED-", True)):
            rej_res_mask = rejected_mask & finite_mask(x_plot, residuals)
            if np.any(rej_res_mask):
                rej_marker = str(values.get("-CLEAN_REJ_MARKER-", "x"))
                rej_marker_fmt = "none" if rej_marker == "none" else rej_marker
                rej_colour = str(values.get("-CLEAN_REJ_COLOUR-", "tab:red"))
                rej_size = parse_float(values.get("-CLEAN_REJ_SIZE-", 6.0), 6.0) or 6.0
                rej_alpha = parse_float(values.get("-CLEAN_REJ_ALPHA-", 0.9), 0.9) or 0.9

                if two_panels:
                    rejected_residual_plot = residuals[rej_res_mask] * 1000.0
                    rejected_label = "_nolegend_"
                else:
                    rejected_residual_plot = residuals[rej_res_mask] + res_offset
                    rejected_label = "_nolegend_"

                ax_res.plot(
                    x_plot[rej_res_mask],
                    rejected_residual_plot,
                    linestyle="none",
                    marker=rej_marker_fmt,
                    ms=rej_size,
                    color=rej_colour,
                    alpha=rej_alpha,
                    label=rejected_label,
                    zorder=7,
                )

        ax_res.axhline(
            zero_line_y,
            color=zero_colour,
            lw=1.0,
            ls="--",
            label=legend_label(show_zero_legend and not two_panels, "Residual zero line"),
            zorder=1,
        )

        if bool(values.get("-SHOW_RMS-", True)):
            add_residual_statistics_text(ax_res, residuals, keep_mask=keep_mask)

        if (not two_panels) and bool(values.get("-SHOW_RES_OFFSET_TEXT-", True)):
            add_residual_offset_text(ax_res, res_offset)

        if two_panels:
            ax_res.set_ylabel(res_ylabel)

    title = str(values.get("-TITLE-", "")).strip()
    ylabel = str(values.get("-YLABEL-", "")).strip()

    if title:
        ax_lc.set_title(title)

    ax_lc.set_ylabel(ylabel if ylabel else "Relative flux")

    if two_panels:
        ax_lc.tick_params(labelbottom=False)
        ax_res.set_xlabel(xlabel)
    else:
        ax_lc.set_xlabel(xlabel)

    xmin = parse_float(values.get("-XMIN-", None), None)
    xmax = parse_float(values.get("-XMAX-", None), None)
    ymin = parse_float(values.get("-YMIN-", None), None)
    ymax = parse_float(values.get("-YMAX-", None), None)

    target_x_axis = ax_res if two_panels else ax_lc
    if xmin is not None or xmax is not None:
        target_x_axis.set_xlim(left=xmin, right=xmax)

    # In two-panel mode, Y min/max refer to the light-curve panel only.
    if ymin is not None or ymax is not None:
        ax_lc.set_ylim(bottom=ymin, top=ymax)

    if bool(values.get("-INVERT_Y-", False)):
        ax_lc.invert_yaxis()

    grid_enabled = bool(values.get("-GRID-", True))
    grid_alpha = parse_float(values.get("-GRID_ALPHA-", 0.25), 0.25) or 0.25
    apply_grid(ax_lc, grid_enabled, grid_alpha)
    apply_scientific_ticks(ax_lc)
    if two_panels:
        apply_grid(ax_res, grid_enabled, grid_alpha)
        apply_scientific_ticks(ax_res)

    if bool(values.get("-LEGEND-", True)):
        leg_loc = legend_location(values.get("-LEGEND_LOC-", "Auto"))

        if not two_panels and residuals is not None:
            handles, labels = ax_lc.get_legend_handles_labels()
            filtered = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
            if filtered:
                handles, labels = zip(*filtered)
                ax_lc.legend(handles, labels, loc=leg_loc, frameon=False, fontsize=9)
        else:
            ax_lc.legend(loc=leg_loc, frameon=False, fontsize=9)

    # ``tight_layout`` occasionally raises a harmless warning with embedded Tk
    # figures or shared-axis two-panel plots. Keep the automatic spacing, but
    # silence only this specific Matplotlib layout warning so the console remains
    # clean for real diagnostic messages.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()

    return fig
