# ExoPhotoCurve -- EPC -- v1.0.0

📖 **Author:** Daniele Gasparri  
📅 **Latest Release:** June 2026  

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/badge/license-Non--Commercial-blue)](./LICENSE)

**ExoPhotoCurve is a GUI program designed to create, inspect, correct, analyze and model photometric light curves of known exoplanet transits. It is particularly suited for follow-up observations and fast modeling using the embedded NASA and ExoClock catalogues**

<img width="1841" height="919" alt="Screenshot 2026-06-17 191802" src="https://github.com/user-attachments/assets/2705c839-af44-40ac-9f72-b3684a8bebad" />


## Features

- **Create a light curve** from calibrated and aligned FITS images via aperture differential photometry:
    - Automatic fits sequence recognition
    - Interactive aperture adjustment and positioning
    - Automatic visual feedback to let the user know if a target or comparison star are within the optimal linear range
    - Automatic centroid recentering on all the images of the loaded sequence
    - Saving apertures to be loaded any time you want
    - Photometry file generated compatible with AstroImageJ and automatically passed to the analysis panel
- **Fine-tuning, analyze, and modeling light curves in ASCII or CSV format**:
    - Interactive plot window updated in real time
    - Automatic or manual clipping of outliers
    - Automatic or manual selection of the comparison stars to optimize the signal to noise of the transit
    - Detrending methods: airmass, JD_UTC, FWHM, and meridian flip correction
    - Transit modeling using the physical data of the considered extrasolar planet and comparison with the expected model, using the embedded NASA and ExoClock extrasolar planet databases. 
    - Saving plot, statistics, diagnostics and the full recipe in order to maintain reproducibility of the results.


## System Requirements

- Python 3.10+
- Screen resolution of at least 1600x900 px. Optimal resolution: 1920X1080.
- The following dependencies will be automatically installed: numpy, pandas, matplotlib, scipy, astropy, astropy-iers-data, batman-package
- A 64 bit Standalone (no Python required) Windows installer is also available in the release package

## Installation

You can install **ExoPhotoCurve** using `pip`:

```bash
pip3 install exophotocurve
```
or using the Windows installer, for 64 bit Windows 10+ systems.


## Quick Start

Run ExoPhotoCurve using:

```bash
exophotocurve
```

Once launched, open the "Buld light curve" panel if you want to create your light curve, or just browse and load a light curve table. Accepted light curve format: AstroImageJ, ExoPhotoCurve, HOPS, and in general any table file either in ASCII or CSV format containing at least the time and the relative flux of the target star.

## License

ExoPhotoCurve is licensed under the non-commercial License.
See the LICENSE file for details.

## Contact & Contributions

Found a bug? Want to suggest a feature?  
Drop me an email!  
Contact: Daniele Gasparri – daniele.gasparri@gmail.com  
Take a look also at my website, if you are interested in astronomy: https://www.danielegasparri.com/ 
