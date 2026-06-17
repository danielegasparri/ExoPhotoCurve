"""Shared constants for PhotoCurve Lab."""

NONE_COL = "-- None --"

COLOURS = [
    "black", "grey", "lightgrey", "red", "darkred", "orange", "gold",
    "green", "limegreen", "blue", "royalblue", "navy", "purple",
    "magenta", "cyan", "tab:blue", "tab:orange", "tab:green",
    "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

MARKERS = ["o", ".", "s", "^", "v", "D", "x", "+", "none"]

PLOT_LAYOUTS = ["Single panel", "Two panels"]
X_MODES = ["Raw X", "X - offset", "Hours from offset"]
LEGEND_LOCATIONS = [
    "Auto",
    "Lower left corner",
    "Lower right corner",
    "Upper left corner",
    "Upper right corner",
]

STATS_TARGETS = ["Light curve", "Residuals", "Both"]
Y_DATA_TYPES = ["Relative flux", "Magnitude", "Generic"]

CLEANING_TARGETS = ["Residuals", "Light curve - model", "Light curve", "Both"]
CLEANING_CENTRES = ["Median", "Mean"]
CLEANING_SCALES = ["MAD", "Std dev"]

TRANSIT_BASELINES = ["Constant", "Linear", "Quadratic"]
TRANSIT_TIME_SYSTEMS = ["BJD_TDB", "JD_UTC", "BJD_UTC", "HJD_UTC", "Other"]
TRANSIT_TIMESTAMP_REFERENCES = ["Exposure start", "Mid-exposure", "Exposure end"]
TRANSIT_MODEL_ENGINES = ["Auto", "Batman physical", "Empirical"]
TRANSIT_DISPLAY_MODES = ["Detrended flux", "Raw flux with baseline"]

TRANSIT_FILTERS = [
    "Clear", "L", "B", "V", "R", "I", "G", "g'", "r'", "i'", "z'",
    "Sloan g", "Sloan r", "Sloan i", "Sloan z", "TESS", "Other",
]

CONFIG_KEYS = [
    "-FILE-", "-DELIM-", "-HEADER-",
    "-XCOL-", "-YCOL-", "-YERRCOL-", "-MODEL_COL-", "-RES_COL-", "-RESERR_COL-",
    "-LC_OFFSET-", "-RES_OFFSET-", "-ZERO_OFFSET-",
    "-LC_COLOUR-", "-MODEL_COLOUR-", "-EXPECTED_MODEL_COLOUR-", "-RES_COLOUR-", "-ZERO_COLOUR-", "-ERR_COLOUR-",
    "-MARKER-", "-RES_MARKER-", "-MSIZE-", "-ALPHA-", "-ERR_ALPHA-", "-LW-",
    "-TITLE-", "-XLABEL-", "-YLABEL-",
    "-XMIN-", "-XMAX-", "-YMIN-", "-YMAX-",
    "-GRID-", "-GRID_ALPHA-", "-FIG_W-", "-FIG_H-", "-DPI-",
    "-LEGEND-", "-LEGEND_LOC-", "-INVERT_Y-",
    "-PLOT_LAYOUT-", "-XMODE-", "-XOFFSET-",
    "-SHOW_RMS-", "-SHOW_RES_OFFSET_TEXT-",
    "-LEG_LC-", "-LEG_MODEL-", "-LEG_RES-", "-LEG_ZERO-",
    "-BIN_ACTIVE-", "-BIN_N-", "-BIN_MARKER-", "-BIN_COLOUR-",
    "-BIN_ERR_COLOUR-", "-BIN_MARKER_SIZE-", "-BIN_ALPHA-",
    "-BIN_SHOW_ERR-", "-BIN_LEGEND-",
    "-STATS_TARGET-", "-STATS_YTYPE-", "-STATS_USE_TRANSFORMED_X-",
    "-STATS_INCLUDE_BINNED-", "-STATS_SHOW_POPUP-",
    "-CLEAN_ACTIVE-", "-CLEAN_TARGET-", "-CLEAN_SIGMA-",
    "-CLEAN_MAXITER-", "-CLEAN_CENTRE-", "-CLEAN_SCALE-",
    "-CLEAN_SHOW_REJECTED-", "-CLEAN_REJ_COLOUR-", "-CLEAN_REJ_MARKER-",
    "-CLEAN_REJ_SIZE-", "-CLEAN_REJ_ALPHA-", "-CLEAN_REJ_LEGEND-",
    "-CLEAN_APPLY_STATS-",
    "-TR_CATALOG-", "-TR_PLANET-", "-TR_FILTER-", "-TR_EXPTIME-",
    "-TR_TIME_SYSTEM-", "-TR_TIMESTAMP_REF-", "-TR_BASELINE-", "-TR_MODEL_ENGINE-", "-TR_DISPLAY_MODE-", "-TR_FIT_TMID-",
    "-TR_FIT_DEPTH-", "-TR_FIT_DURATION-", "-TR_OBS_LAT-", "-TR_OBS_LON-", "-TR_OBS_ALT-",
    "-TR_SHOW_POPUP-", "-TR_SET_MODEL_COLUMNS-",
    "-TR_TMID_OVERRIDE-",
    "-TR_SHOW_PREDICTED_TIMES-", "-TR_SHOW_CALCULATED_TIMES-",
]
