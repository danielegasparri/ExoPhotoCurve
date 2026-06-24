# ExoPhotoCurve -- EPC -- v1.0.0

📖 **Author:** Daniele Gasparri  
📅 **Latest Release:** June 2026  

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/badge/license-Non--Commercial-blue)](./LICENSE)

**ExoPhotoCurve is a GUI program designed to create, inspect, correct, analyze and model photometric light curves of known exoplanet transits. It is particularly suited for follow-up observations and fast modeling using the embedded NASA and ExoClock catalogues.**

<img width="1920" height="1032" alt="Screenshot 2026-06-23 231914" src="https://github.com/user-attachments/assets/1351a0c2-9ce4-4725-a6cd-16cd6af13b17" />



## Features

- **Full reduction of raw image sequence** using rigorous scientific approach to preserve and maximize the photometric information:
    - Calibration of raw FITS images with bas, dark and flat frames
    - Automatic alignment handling translation, rotation and meridian flip
    - Monochrome and color data handling. For color data, user selected channel extraction is performed without applying the debayer process
- **Differential aperture photometry** from calibrated and aligned FITS images:
    - Automatic fits sequence recognition
    - Interactive aperture adjustment and positioning
    - Automatic visual feedback to let the user know if a target or comparison star are within the optimal linear range
    - Automatic centroid recentering on all the images of the loaded sequence
    - Saving apertures to be loaded any time you want
    - Photometry file generated compatible with AstroImageJ and automatically passed to the analysis panel
- **Light curve fine-tuning, analysis, and transit modeling**:
    - Interactive plot window updated in real time
    - Automatic or manual clipping of outliers
    - Automatic or manual selection of the comparison stars to optimize the signal-to-noise of the transit
    - Detrending methods: airmass, JD_UTC, FWHM, and meridian flip correction
    - Transit modeling using the physical data of the considered extrasolar planet and comparison with the expected model, using the embedded NASA and ExoClock extrasolar planet databases. 
    - Saving plot, statistics, diagnostics and the full recipe in order to grant reproducibility.


## System Requirements

- Python 3.10+
- Screen resolution of at least 1600x900 px. Optimal resolution: 1920X1080.
- The following dependencies will be automatically installed: numpy, pandas, matplotlib, scipy, astropy, astropy-iers-data, batman-package
- A 64 bit standalone Windows installer (no Python required) is also available in the release package

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

Once launched ExoPhotoCurve, unzip the example reduced sequence of the Kelt-10b transit, open the "Build light curve" panel and start building and analyzing the light curve

## License

ExoPhotoCurve is licensed under the non-commercial License.
See the LICENSE file for details.

## Contact & Contributions

Found a bug? Want to suggest a feature?  
Drop me an email!  
Contact: Daniele Gasparri – daniele.gasparri@gmail.com  
Take a look also at my website, if you are interested in astronomy: https://www.danielegasparri.com/ 
