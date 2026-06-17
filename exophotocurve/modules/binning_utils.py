"""On-the-fly light-curve binning utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .numeric_utils import finite_mask


def bin_light_curve(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    n_per_bin: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Bin the light curve by grouping consecutive points in time.

    The data are sorted by the X axis before binning. If valid uncertainties are
    available for all points in a bin, the binned flux is computed as a weighted
    mean and the binned uncertainty is sqrt(1 / sum(weights)). Otherwise, the
    function falls back to an unweighted mean and estimates the uncertainty from
    the standard error of the mean.

    Returns
    -------
    x_bin, y_bin, yerr_bin, n_bin
        Binned X values, binned Y values, binned errors and the number of points
        actually used in each bin. The last bin is kept even if it contains fewer
        than n_per_bin points.
    """
    n_per_bin = max(1, int(n_per_bin))

    base_mask = finite_mask(x, y)
    if not np.any(base_mask):
        return np.array([]), np.array([]), None, np.array([], dtype=int)

    x_valid = x[base_mask]
    y_valid = y[base_mask]
    yerr_valid = None if yerr is None else yerr[base_mask]

    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    if yerr_valid is not None:
        yerr_valid = yerr_valid[order]

    x_bins: List[float] = []
    y_bins: List[float] = []
    yerr_bins: List[float] = []
    n_bins: List[int] = []

    for start in range(0, len(x_valid), n_per_bin):
        stop = min(start + n_per_bin, len(x_valid))
        xb = x_valid[start:stop]
        yb = y_valid[start:stop]
        nb = len(yb)

        if nb == 0:
            continue

        x_bins.append(float(np.nanmean(xb)))
        n_bins.append(nb)

        use_weighted = False
        if yerr_valid is not None:
            eb = np.asarray(yerr_valid[start:stop], dtype=float)
            use_weighted = bool(np.all(np.isfinite(eb)) and np.all(eb > 0))
        else:
            eb = None

        if use_weighted and eb is not None:
            weights = 1.0 / eb**2
            y_mean = float(np.sum(weights * yb) / np.sum(weights))
            y_sigma = float(np.sqrt(1.0 / np.sum(weights)))
        else:
            y_mean = float(np.nanmean(yb))
            if nb > 1:
                y_sigma = float(np.nanstd(yb, ddof=1) / np.sqrt(nb))
            else:
                y_sigma = np.nan

        y_bins.append(y_mean)
        yerr_bins.append(y_sigma)

    yerr_array = np.asarray(yerr_bins, dtype=float)
    if not np.any(np.isfinite(yerr_array)):
        yerr_out: Optional[np.ndarray] = None
    else:
        yerr_out = yerr_array

    return (
        np.asarray(x_bins, dtype=float),
        np.asarray(y_bins, dtype=float),
        yerr_out,
        np.asarray(n_bins, dtype=int),
    )
