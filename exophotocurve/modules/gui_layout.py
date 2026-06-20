"""FreeSimpleGUI layout definition."""

from __future__ import annotations

from typing import List

from .constants import (
    CLEANING_CENTRES,
    CLEANING_SCALES,
    CLEANING_TARGETS,
    COLOURS,
    LEGEND_LOCATIONS,
    MARKERS,
    NONE_COL,
    PLOT_LAYOUTS,
    STATS_TARGETS,
    TRANSIT_BASELINES,
    TRANSIT_FILTERS,
    TRANSIT_TIME_SYSTEMS,
    TRANSIT_TIMESTAMP_REFERENCES,
    TRANSIT_MODEL_ENGINES,
    TRANSIT_DISPLAY_MODES,
    X_MODES,
    Y_DATA_TYPES,
)
from .sg_loader import sg
from .exoplanet_catalog import default_catalogue_path, default_exoclock_catalogue_path


def make_layout() -> List[List[sg.Element]]:
    """Create the FreeSimpleGUI layout.

    Compact v0.4.1 layout: all controls live inside tabs, including the
    column-selection controls. This keeps the left panel narrow and gives more
    horizontal space to the plot, which is now the main working area.
    """
    sg.set_options(font=("Helvetica", 13)) # setting uo the default font size
    # ------------------------------------------------------------------
    # File loader. Kept outside the tabs because it is always needed.
    # ------------------------------------------------------------------
    file_frame = [
        [
            sg.Button("Build light curve", button_color= ('black','light blue'),tooltip='Construct your lightcurve here starting from a calibrated and aligned image sequence in fit format'),
            sg.Text("Or Load a light curve"),
            sg.Input(key="-FILE-", size=(17, 1)),
            sg.FileBrowse(
                button_text="Browse",
                file_types=(
                    ("Text files", "*.txt *.dat *.csv *.tsv"),
                    ("All files", "*.*"),
                ), tooltip='If you already have a lightcurve file, browse it here, click Load table\nand proceed to visualization and analysis\n You can directly upload an AstroImageJ table or any table containing the data and names of the columns, WITHOUT the # character'),
            sg.Button("Load table", tooltip='Load your lightcurve text file that you have just browsed'),
        ],
        [
            sg.Text("Delimiter", size=(13, 1)),
            sg.Combo(
                ["Auto", "Whitespace", "Tab", "Comma", "Semicolon"],
                default_value="Auto",
                key="-DELIM-",
                readonly=True,
                size=(12, 1),
            ),
            sg.Checkbox("Header", default=True, key="-HEADER-"),
            sg.Text("Rows"),
            sg.Text("-", key="-NROWS-", size=(6, 1)),
            sg.Text("Cols"),
            sg.Text("-", key="-NCOLS-", size=(6, 1)),
        ],
    ]

    # ------------------------------------------------------------------
    # Data tab: column mapping. It used to be permanently visible, but this
    # made the control panel too wide once transit diagnostics were added.
    # ------------------------------------------------------------------
    data_tab = [
        [sg.Text("X / time", size=(15, 1)), sg.Combo([NONE_COL], key="-XCOL-", size=(31, 1), readonly=True)],
        [sg.Text("Light curve", size=(15, 1)), sg.Combo([NONE_COL], key="-YCOL-", size=(31, 1), readonly=True)],
        [sg.Text("LC error", size=(15, 1)), sg.Combo([NONE_COL], key="-YERRCOL-", size=(31, 1), readonly=True)],
        [sg.Text("Model", size=(15, 1)), sg.Combo([NONE_COL], key="-MODEL_COL-", size=(31, 1), readonly=True)],
        [sg.Text("Residuals", size=(15, 1)), sg.Combo([NONE_COL], key="-RES_COL-", size=(31, 1), readonly=True)],
        [sg.Text("Residual errors", size=(15, 1)), sg.Combo([NONE_COL], key="-RESERR_COL-", size=(31, 1), readonly=True)],
        
        [
            sg.Checkbox(
                "Export rejected points",
                default=False,
                key="-EXPORT_REJECTED_POINTS-",
                tooltip="If disabled, Save curve writes only the points currently kept by manual/sigma cleaning and the latest transit-fit mask.",
            ),
        ],
            
        [
            sg.Text(
                "After loading a table, PhotoCurve Lab tries to select the final detrended columns automatically.",
                size=(54, 2),
            )
        ],
        [
            sg.Button("User manual")
        ],
    ]

    plot_tab = [
        [
            sg.Text("Layout", size=(9, 1)),
            sg.Combo(PLOT_LAYOUTS, default_value="Two panels", key="-PLOT_LAYOUT-", size=(14, 1), readonly=True),
            sg.Text("X mode"),
            sg.Combo(X_MODES, default_value="X - offset", key="-XMODE-", size=(14, 1), readonly=True),
        ],
        [
            sg.Text("X offset/T0", size=(9, 1)),
            sg.Input("auto", key="-XOFFSET-", size=(11, 1)),
            sg.Text("auto = floor(min X)", size=(22, 1)),
        ],
        [sg.Text("Title", size=(9, 1)), sg.Input("", key="-TITLE-", size=(43, 1))],
        [sg.Text("X label", size=(9, 1)), sg.Input("", key="-XLABEL-", size=(43, 1))],
        [sg.Text("Y label", size=(9, 1)), sg.Input("Relative flux", key="-YLABEL-", size=(43, 1))],
        [
            sg.Text("X min/max", size=(9, 1)),
            sg.Input("", key="-XMIN-", size=(8, 1)),
            sg.Input("", key="-XMAX-", size=(8, 1)),
            sg.Text("Y min/max"),
            sg.Input("", key="-YMIN-", size=(8, 1)),
            sg.Input("", key="-YMAX-", size=(8, 1)),
        ],
        [
            sg.Checkbox("Grid", default=False, key="-GRID-"),
            sg.Text("alpha"),
            sg.Input("0.20", key="-GRID_ALPHA-", size=(5, 1)),
            sg.Checkbox("Invert Y", default=False, key="-INVERT_Y-"),
            sg.Checkbox("RMS + N", default=True, key="-SHOW_RMS-"),
        ],
        [
            sg.Checkbox("Legend", default=True, key="-LEGEND-"),
            sg.Text("position"),
            sg.Combo(
                LEGEND_LOCATIONS,
                default_value="Auto",
                key="-LEGEND_LOC-",
                size=(16, 1),
                readonly=True,
            ),
        ],
        [
            sg.Text("Legend", size=(9, 1)),
            sg.Checkbox("LC", default=False, key="-LEG_LC-"),
            sg.Checkbox("Model", default=True, key="-LEG_MODEL-"),
            sg.Checkbox("Res", default=True, key="-LEG_RES-"),
            sg.Checkbox("Zero", default=False, key="-LEG_ZERO-"),
        ],
        [
            sg.Text("Figure", size=(9, 1)),
            sg.Input("8", key="-FIG_W-", size=(5, 1)),
            sg.Text("x"),
            sg.Input("5", key="-FIG_H-", size=(5, 1)),
            sg.Text("DPI"),
            sg.Input("140", key="-DPI-", size=(5, 1)),
            sg.Text("LC offset"),
            sg.Input("0.0", key="-LC_OFFSET-", size=(7, 1)),
        ],
    ]

    style_tab = [
        [
            sg.Text("LC points", size=(13, 1)),
            sg.Combo(COLOURS, default_value="black", key="-LC_COLOUR-", size=(12, 1), readonly=False),
            sg.Text("Error"),
            sg.Combo(COLOURS, default_value="grey", key="-ERR_COLOUR-", size=(12, 1), readonly=False),
        ],
        [
            sg.Text("Best-fit", size=(13, 1)),
            sg.Combo(COLOURS, default_value="red", key="-MODEL_COLOUR-", size=(12, 1), readonly=False),
            sg.Text("Expected"),
            sg.Combo(COLOURS, default_value="tab:cyan", key="-EXPECTED_MODEL_COLOUR-", size=(12, 1), readonly=False),
        ],
        [
            sg.Text("Residuals", size=(13, 1)),
            sg.Combo(COLOURS, default_value="black", key="-RES_COLOUR-", size=(12, 1), readonly=False),
            sg.Text("Zero line"),
            sg.Combo(COLOURS, default_value="grey", key="-ZERO_COLOUR-", size=(12, 1), readonly=False),
        ],
        [
            sg.Text("Markers", size=(13, 1)),
            sg.Combo(MARKERS, default_value="o", key="-MARKER-", size=(8, 1), readonly=True),
            sg.Text("res"),
            sg.Combo(MARKERS, default_value=".", key="-RES_MARKER-", size=(8, 1), readonly=True),
        ],
        [
            sg.Text("Sizes/alpha", size=(13, 1)),
            sg.Input("4", key="-MSIZE-", size=(5, 1)),
            sg.Text("point"),
            sg.Input("0.85", key="-ALPHA-", size=(5, 1)),
            sg.Text("err"),
            sg.Input("0.55", key="-ERR_ALPHA-", size=(5, 1)),
            sg.Text("lw"),
            sg.Input("0.8", key="-LW-", size=(5, 1)),
        ],
    ]

    binning_tab = [
        [
            sg.Checkbox("Enable binning", default=False, key="-BIN_ACTIVE-", tooltip='Enable and set binning points. Only for visualisation purposess'),
            sg.Text("Points/bin"),
            sg.Input("4", key="-BIN_N-", size=(5, 1)),
        ],
        [
            sg.Text("Marker", size=(12, 1)),
            sg.Combo(MARKERS, default_value="s", key="-BIN_MARKER-", size=(10, 1), readonly=True),
            sg.Text("size"),
            sg.Input("6", key="-BIN_MARKER_SIZE-", size=(5, 1)),
            sg.Text("alpha"),
            sg.Input("0.95", key="-BIN_ALPHA-", size=(5, 1)),
        ],
        [
            sg.Text("Colour", size=(12, 1)),
            sg.Combo(COLOURS, default_value="tab:orange", key="-BIN_COLOUR-", size=(12, 1), readonly=False),
            sg.Text("Error"),
            sg.Combo(COLOURS, default_value="tab:orange", key="-BIN_ERR_COLOUR-", size=(12, 1), readonly=False),
        ],
        [
            sg.Checkbox("Show error bars", default=True, key="-BIN_SHOW_ERR-"),
            sg.Checkbox("Show binning legend", default=True, key="-BIN_LEGEND-"),
        ],
    ]

    cleaning_tab = [
        [
            sg.Checkbox("Enable sigma clipping", default=False, key="-CLEAN_ACTIVE-", tooltip='Enable and set the clipping of outliers. Once activated, press the Plot / update button to see the results'),
            sg.Text("Target"),
            sg.Combo(CLEANING_TARGETS, default_value="Residuals", key="-CLEAN_TARGET-", size=(17, 1), readonly=True),
        ],
        [
            sg.Text("Sigma", size=(8, 1)),
            sg.Input("4.0", key="-CLEAN_SIGMA-", size=(6, 1)),
            sg.Text("Iter"),
            sg.Input("3", key="-CLEAN_MAXITER-", size=(4, 1)),
            sg.Text("Center"),
            sg.Combo(CLEANING_CENTRES, default_value="Median", key="-CLEAN_CENTRE-", size=(8, 1), readonly=True),
            sg.Text("Scale"),
            sg.Combo(CLEANING_SCALES, default_value="MAD", key="-CLEAN_SCALE-", size=(8, 1), readonly=True),
        ],
        [
            sg.Checkbox("Show rejected", default=True, key="-CLEAN_SHOW_REJECTED-", tooltip='Showing the rejected points in the plot. Just for visualisation purposes'),
            sg.Checkbox("Rejected legend", default=True, key="-CLEAN_REJ_LEGEND-", tooltip='Showing the rejected point legend in the plot'),
            sg.Checkbox("Apply to stats", default=True, key="-CLEAN_APPLY_STATS-", tooltip='Update the stats tab and analysis with the nes subset of points without the rejected outliers'),
        ],
        [
            sg.Text("Rejected", size=(8, 1)),
            sg.Combo(MARKERS, default_value="x", key="-CLEAN_REJ_MARKER-", size=(7, 1), readonly=True),
            sg.Text("colour"),
            sg.Combo(COLOURS, default_value="tab:red", key="-CLEAN_REJ_COLOUR-", size=(11, 1), readonly=False),
            sg.Text("size"),
            sg.Input("6", key="-CLEAN_REJ_SIZE-", size=(4, 1)),
            sg.Text("alpha"),
            sg.Input("0.9", key="-CLEAN_REJ_ALPHA-", size=(4, 1)),
        ],
        [sg.HorizontalSeparator()],
        [
            sg.Checkbox("Apply manual rejects", default=True, key="-MANUAL_CLEAN_ACTIVE-", enable_events=True, tooltip='Enable (also) the manual rejection of the points'),
            sg.Checkbox("Click-edit plot", default=False, key="-MANUAL_POINT_EDIT-", tooltip='Activate the mouse click rejection, then you can click on a point in the plot to reject or re-activate a point.\nPlot updates in real time'),
            sg.Text("mode"),
            sg.Combo(
                ["Toggle nearest", "Reject nearest", "Restore nearest"],
                default_value="Toggle nearest",
                key="-MANUAL_POINT_MODE-",
                size=(15, 1),
                readonly=True,
            ),
        ],
        [
            sg.Text("Manual points", size=(13, 1)),
            sg.Text("reject 0", key="-MANUAL_REJECT_COUNT-", size=(9, 1)),
            sg.Text("keep 0", key="-MANUAL_KEEP_COUNT-", size=(8, 1)),
            sg.Button("Clear manual points", key="-MANUAL_CLEAR_POINTS-", tooltip='Restore all the manually deleted photometric points'),
            sg.Input("", key="-MANUAL_REJECT_INDICES-", visible=False),
            sg.Input("", key="-MANUAL_KEEP_INDICES-", visible=False),
        ],
        [
            sg.Text(
                "For transits, clip residuals or light curve - model rather than the raw light curve. Use click-edit to reject or restore individual plotted points.",
                size=(54, 3),
            )
        ],
    ]

    detrend_tab = [
        [
            sg.Text(
                "Select one or more decorrelation regressors, then divide the light curve by the fitted baseline.",
                key="-DET_STATUS-",
                size=(54, 2),
                text_color="firebrick",
            )
        ],
        [
            sg.Text("Regressors", size=(13, 1)),
            sg.Button("All", key="-DET_SELECT_ALL-", disabled=True),
            sg.Button("None", key="-DET_SELECT_NONE-", disabled=True),
            sg.Button("Suggested", key="-DET_SELECT_SUGGESTED-", disabled=True),
        ],
        [
            sg.Listbox(
                [],
                key="-DET_REGRESSOR_LIST-",
                size=(28, 8),
                enable_events=True,
                disabled=True,
                no_scrollbar=False,
            ),
            sg.Text(
                "Useful regressors are usually JD_UTC/time, AIRMASS, FWHM/Width, sky background and centroid X/Y.",
                size=(22, 8),
            ),
        ],
        [
            sg.Checkbox("Mask transit", default=True, key="-DET_MASK_TRANSIT-", disabled=True, tooltip='Masking the transit and considering only Out Of Transit points for calculating the detrend.\nI strongly suggest to keep activated, but you can try to deactivate and see the results!'),
            sg.Checkbox("Use clean mask", default=True, key="-DET_USE_CLEANING_MASK-", disabled=True, tooltip='Using the actual data points and neglecting automatically of manually rejected data points in the Cleaning tab\nDeactivate to use all the original data points of your dataset. If you did not reject any point, the result will not change'),
        # ],
        # [
            sg.Text("Poly", size=(5, 1)),
            sg.Combo(["1", "2"], default_value="1", key="-DET_POLY_ORDER-", size=(4, 1), readonly=True, disabled=True),
            sg.Text("sigma"),
            sg.Input("4.0", key="-DET_ROBUST_SIGMA-", size=(5, 1), disabled=True),
            sg.Text("iter"),
            sg.Input("3", key="-DET_ROBUST_ITER-", size=(4, 1), disabled=True),
        ],

        [
            sg.Checkbox("Meridian flip", default=False, key="-DET_FLIP_ACTIVE-", disabled=True, enable_events=True, tooltip='Activate the meridian flip correction, only if during your photometric session your telescope performed a meridian flip and you notice the\n characteristic jump in your light curve. You should insert also the JD_UTC time decimals, that is, something like that: .771\nMeridian flip correction activates with the Run detrending button, like the other detrending methods'),
            sg.Text("time frac"),
            sg.Input("", key="-DET_FLIP_FRAC-", size=(6, 1), disabled=True, enable_events=True),
            # sg.Button("Update line", key="-DET_UPDATE_FLIP_MARKER-", disabled=True),
            sg.Checkbox("Show line", default=True, key="-DET_SHOW_FLIP_MARKER-", disabled=True, enable_events=True, tooltip='After you set the fraction JD_UTC time of the meridian flip, activate this to see a vertical line in the plot corrsponding to the time of the meridian flip.\n If you change the time, you can update the line by deselecting and selecting again this checkboz or by pressing the Plot / update button'),
            sg.Text("Flip mode", size=(9, 1)),
            sg.Combo(
                ["Robust level matching", "Step only", "Step + after-flip slope"],
                default_value="Robust level matching",
                key="-DET_FLIP_MODE-",
                size=(15, 1),
                readonly=True,
                disabled=True,
                tooltip="Robust level matching is recommended. It measures the pre/post-flip level using good out-of-transit points\nand corrects the post-flip part multiplicatively. Step modes are advanced regression-based options.",
            ),
        ],

        [
            sg.Checkbox("Consider fit model", default=False, key="-DET_USE_TRANSIT_MODEL-", disabled=True, tooltip="Use the latest transit fit model when estimating the detrending baseline."),
            sg.Checkbox("Send detrended curve to Data/Transit", default=True, key="-DET_SEND_TO_DATA-", disabled=True, tooltip='Using the result of detrend as the new light curve'),
            sg.Checkbox("Show popup", default=True, key="-DET_SHOW_POPUP-", disabled=True, tooltip='Showing a pop-up info containing the results of detrending once you press the Run detrend button'),
        ],
        [sg.Button("Run detrending", disabled=True, tooltip='Well, this is self-explicative: run the detrend(s) activated, including the meridian flip correction'), sg.Button("Clear detrending", disabled=True, tooltip='Clear all the detrending performed. Also the meridian flip correction will be cleared but the relative checkbox will stay activated!')],
        [
            sg.Multiline(
                "",
                key="-DET_REPORT-",
                size=(54, 8),
                disabled=True,
                autoscroll=False,
                no_scrollbar=False,
            )
        ],
    ]

    stats_tab = [
        [
            sg.Text("Target", size=(9, 1)),
            sg.Combo(STATS_TARGETS, default_value="Both", key="-STATS_TARGET-", size=(14, 1), readonly=True),
            sg.Text("Y type"),
            sg.Combo(Y_DATA_TYPES, default_value="Relative flux", key="-STATS_YTYPE-", size=(14, 1), readonly=True),
        ],
        [sg.Checkbox("Use transformed X axis", default=True, key="-STATS_USE_TRANSFORMED_X-")],
        [sg.Checkbox("Include binned stats", default=True, key="-STATS_INCLUDE_BINNED-")],
        [sg.Checkbox("Show report popup", default=True, key="-STATS_SHOW_POPUP-")],
        [
            sg.Text(
                "Use Compute stats to calculate N, cadence, mean, median, RMS, amplitude and error statistics.",
                size=(54, 2),
            )
        ],
    ]


    comp_tab = [
        [
            sg.Text(
                "Inactive until a table with Source-Sky_T*/C* flux columns is loaded.",
                key="-COMP_STATUS-",
                size=(54, 2),
                text_color="firebrick",
            )
        ],
        [
            sg.Text("Target", size=(9, 1)),
            sg.Combo([], key="-COMP_TARGET-", size=(10, 1), readonly=True, disabled=True, enable_events=True),
            sg.Text("Check"),
            sg.Combo([""], key="-COMP_CHECK-", size=(10, 1), readonly=True, disabled=True, enable_events=True),

            sg.Text("Mode", size=(9, 1)),
            sg.Combo(
                ["Target light curve", "Check star stability"],
                default_value="Target light curve",
                key="-COMP_MODE-",
                size=(22, 1),
                readonly=True,
                disabled=True,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox("Mask expected transit", default=True, key="-COMP_MASK_TRANSIT-", disabled=True, tooltip='Maintain activated if you are running the automatic selection of comparison stars directly on the transiti curve'),
            sg.Text("Poly"),
            sg.Combo(["0", "1", "2"], default_value="1", key="-COMP_POLY_ORDER-", size=(4, 1), readonly=True, disabled=True),
        ],
        [
            sg.Text("Stars", size=(9, 1)),
            sg.Text("min"),
            sg.Input("2", key="-COMP_MIN_STARS-", size=(4, 1), disabled=True),
            sg.Text("max"),
            sg.Input("auto", key="-COMP_MAX_STARS-", size=(6, 1), disabled=True),
            sg.Text("improve %"),
            sg.Input("0.5", key="-COMP_IMPROVE_THRESHOLD-", size=(5, 1), disabled=True),
        ],
        [
            sg.Checkbox("Send optimised curve to Data/Transit", default=True, key="-COMP_SEND_TO_DATA-", disabled=True, tooltip='This will be the new light curve to analyze'),
            sg.Checkbox("Show popup", default=True, key="-COMP_SHOW_POPUP-", disabled=True, tooltip='Showing a pop-up window with the results once you press Run comp optimizer'), sg.Button("Run comp optimizer", disabled=True, tooltip='Run the automatic optimisation for choosing the best set of comparison stars'),
        ],
        [sg.HorizontalSeparator(pad=(0, 8))],
        [sg.Text("Manually adjust the comparison stars")],
        [
            sg.Text("Comparison stars", size=(14, 1)),
            sg.Button("All", key="-COMP_SELECT_ALL-", disabled=True, tooltip='Select all the comparison stars available'),
            sg.Button("None", key="-COMP_SELECT_NONE-", disabled=True, tooltip='Deselect all the comparison stars. Remember then to activate at least one to see a light curve!'),
            sg.Text("click a star to toggle", size=(22, 1)),
        ],
        [
            sg.Text("Diag star", size=(14, 1)),
            sg.Combo([], key="-COMP_DIAG_STAR-", size=(10, 1), readonly=True, disabled=True, enable_events=True, tooltip='Choose a comparison star to inspect. Its light curve is built against the ensemble of the other active comparison stars.'),
            sg.Checkbox("Plot selected comp", default=False, key="-COMP_PLOT_DIAG-", disabled=True, enable_events=True, tooltip='Plot the selected comparison-star diagnostic curve instead of the science target curve.'),
        ],
        [
            sg.Listbox(
                [],
                key="-COMP_STAR_LIST-",
                size=(22, 10),
                enable_events=True,
                disabled=True,
                no_scrollbar=False,
            ),

            sg.Multiline(
                "",
                key="-COMP_REPORT-",
                size=(42, 10),
                disabled=True,
                autoscroll=False,
                no_scrollbar=False,
            ),

        ],
    ]

    transit_tab = [
        [
            sg.Text("Catalogue", size=(9, 1)),
            sg.Input(str(default_catalogue_path()), key="-TR_CATALOG-", size=(36, 1)),
            sg.FileBrowse(button_text="Browse", file_types=(("CSV catalogue", "*.csv"), ("All files", "*.*"))),
            sg.Button("Load catalogue", tooltip='Browse and load here your custom exoplanet catalogue, if you have it'),
        ],
        [
            sg.Text("Source", size=(9, 1)),
            sg.Button("Use NASA", tooltip='Use the NASA extrasolar planets transit database. Warning: older ephemerides'),
            sg.Button("Use ExoClock", tooltip='Use the ExoClock extrasolar planets transit database. Better suited for planets with transit time variations'),
            sg.Text("CSV catalogues can be generated from the tools folder", size=(34, 1)),
        ],
        [
            sg.Text("Planet", size=(9, 1)),
            sg.Combo([], key="-TR_PLANET-", size=(24, 1), readonly=True),
            sg.Text("Filter", size=(9, 1)),
            sg.Combo(TRANSIT_FILTERS, default_value="G", key="-TR_FILTER-", size=(8, 1), readonly=True),
            sg.Text("Exp"),
            sg.Input("300", key="-TR_EXPTIME-", size=(6, 1)),
            sg.Text("s"),
        ],
        [
            sg.Text("Time", size=(9, 1)),
            sg.Combo(TRANSIT_TIME_SYSTEMS, default_value="BJD_TDB", key="-TR_TIME_SYSTEM-", size=(11, 1), readonly=True),
            sg.Text("Stamp"),
            sg.Combo(TRANSIT_TIMESTAMP_REFERENCES, default_value="Mid-exposure", key="-TR_TIMESTAMP_REF-", size=(14, 1), readonly=True),
        ],
        [
            sg.Text("Baseline", size=(9, 1)),
            sg.Combo(TRANSIT_BASELINES, default_value="Linear", key="-TR_BASELINE-", size=(9, 1), readonly=True),
            sg.Text("Model"),
            sg.Combo(TRANSIT_MODEL_ENGINES, default_value="Auto", key="-TR_MODEL_ENGINE-", size=(14, 1), readonly=True),
        ],
        [
            sg.Text("Display", size=(9, 1)),
            sg.Combo(
                TRANSIT_DISPLAY_MODES,
                default_value="Detrended flux",
                key="-TR_DISPLAY_MODE-",
                size=(22, 1),
                readonly=True,
                enable_events=True,
            ),
            sg.Text("shown after Transit diag"),
        ],
        [
            sg.Text("Observatory", size=(10, 1)),
            sg.Text("lat"),
            sg.Input("-27.5", key="-TR_OBS_LAT-", size=(8, 1)),
            sg.Text("lon"),
            sg.Input("-70", key="-TR_OBS_LON-", size=(8, 1)),
            sg.Text("alt"),
            sg.Input("1550", key="-TR_OBS_ALT-", size=(6, 1)),
            sg.Text("m"),
        ],
        [
            sg.Text("Tmid ref", size=(9, 1)),
            sg.Input("", key="-TR_TMID_OVERRIDE-", size=(17, 1)),
            sg.Text("optional BJD_TDB predicted mid-transit"),
        ],
        [
            sg.Text("Fit", size=(9, 1)),
            sg.Checkbox("Tmid", default=True, key="-TR_FIT_TMID-", tooltip='Fitting the mid-transit time'),
            sg.Checkbox("Depth", default=True, key="-TR_FIT_DEPTH-", tooltip='Fitting the depth of the transit'),
            sg.Checkbox("Duration", default=True, key="-TR_FIT_DURATION-", tooltip='Fitting the duration of the transit. If you want to be coherent with HOPS or ExoClock results, this parameter should not be fitted'),
        ],
        [
            sg.Checkbox("Show transit fit on plot", default=True, key="-TR_SET_MODEL_COLUMNS-", enable_events=True, tooltip='Showing the transit fit, the expected model and the residuals in the plot'),
            sg.Checkbox("Show report popup", default=True, key="-TR_SHOW_POPUP-", tooltip='Showing a detailed report of the fit in a pop-up window once you perform the fit by pressing the Run transit model button'),
        ],
        [
            sg.Text("Labels", size=(9, 1)),
            sg.Checkbox("Predicted times", default=False, key="-TR_SHOW_PREDICTED_TIMES-", enable_events=True, tooltip='Showing the predicted times of the transit in the plot'),
            sg.Checkbox("Calculated times", default=False, key="-TR_SHOW_CALCULATED_TIMES-", enable_events=True, tooltip='Showing the calculated times of the transit in the plot.\nIf Predicted times is activated, the O-C value of the transit timing will also be shown in the plot'),
        ],
        [
            sg.Text(
                "Timing labels are optional plot overlays. They are shown only after a transit diagnostic run.",
                size=(54, 2),
            )
        ],
    ]

    control_tabs = sg.TabGroup(
        [
            [
                sg.Tab("Data", data_tab),
                sg.Tab("Plot", plot_tab),
                sg.Tab("Style", style_tab),
                sg.Tab("Binning", binning_tab),
                sg.Tab("Stats", stats_tab),
                sg.Tab("Comp stars", comp_tab),
                sg.Tab("Cleaning", cleaning_tab),
                # sg.Tab("Comp stars", comp_tab),
                sg.Tab("Detrend", detrend_tab),
                # sg.Tab("Stats", stats_tab),
                # sg.Tab("Comp stars", comp_tab),
                sg.Tab("Transit modeling", transit_tab),
            ]
        ],
        expand_x=False,
    )

    buttons_row_1 = [
        sg.Button("Plot / update", button_color=("white", "#2d6cdf"), size = (14,1), tooltip='Plot the lightcurve or update the plot after any change'),
        sg.Button("Compute stats", size = (14,1), tooltip='Compute she statistics of your data, following the parameters in the Stats tab'),
        sg.Button("Run transit model", size = (14,1), tooltip='Run the fit of your transit and comparison with the model, according to the parameters set in the transit modeling tab'),
        sg.Button("Save figure", size = (14,1), tooltip='Save the plot in png format'),
    ]

    buttons_row_2 = [
        
        sg.Button("Save stats", size = (14,1), tooltip='Save the statistics in an ASCII file, according to the stats tab parameters'),
        sg.Button("Save model results", size = (14,1), tooltip='Save the parameters, diagnostics and results of the transit model you have obtained'),
        sg.Button("Save settings", size = (14,1), tooltip='Save the program settings'),
        sg.Button("Load settings", size = (14,1), tooltip='Load your custom parameter settings'),
    ]

    buttons_row_3 = [

        sg.Button("Reset view/data", size = (14,1), tooltip='Reset the plot and all the analysis performed'),
        sg.Button("Save curve", size = (14,1), tooltip='Save the new lightcurve, with the detrend and model, if applied, in csv or ASCII file'),
        sg.Button("Save recipe", size = (14,1), tooltip='Save relevant info to reproduce and share your workflow: apertures, detrend, comparison stars, model settings...'),
        sg.Button("Exit", size = (14,1), button_color= ('black','light blue')),
        # sg.Button("Exit", size = (14,1)),
    ]




    control_column = [
        [sg.Frame("Start here: build a ligh curve or load a light curve file", file_frame, font=("Helvetica", 13, 'bold'))],
        [control_tabs],
        [sg.HorizontalSeparator()],
        buttons_row_1,
        buttons_row_2,
        buttons_row_3,
    ]

    plot_column = [
        [
            sg.Canvas(
                key="-CANVAS-",
                expand_x=True,
                expand_y=True,
                size=(900, 600),
            )
        ],
    ]

    return [
        [sg.Menu([
        ['&File', ['&Build light curve', '&Save figure', 'Save stats', 'Save model results', 'Save settings', 'Load settings', 'Reset view/data', 'Save curve', 'Save recipe',  'E&xit']],
        ['&Data', ['Load table', 'Plot / update']],
        ['&Comp stars', ['Run comp optimizer']],
        ['Detrend', ['Run detrending', 'Clear detrending']],
        ['Transit modeling', ['Use NASA', 'Use ExoClock', 'Load catalogue', 'Run transit model']],
        ['Help', ['User manual']],
        ])],
        
        [
            sg.Column(control_column, vertical_alignment="top", expand_y=False, expand_x=False),
            sg.VSeparator(),
            sg.Column(plot_column, expand_x=True, expand_y=True),
        ],
        [
            sg.Text("Status:"),
            sg.Text("Load an ASCII table to begin.", key="-STATUS-", expand_x=True),
        ],
    ]
