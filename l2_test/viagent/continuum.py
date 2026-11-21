"""Continuum / spectral-shape diagnostics (Dn4000, breaks, slopes)."""
from __future__ import annotations

import numpy as np


def dn4000(wave, flux, z):
    """Dn4000 index: ratio of mean flux in 4000-4100 to 3850-3950 (rest frame).

    Uses linear interpolation of flux onto rest-frame bins.
    """
    rest = wave / (1.0 + z)
    f = flux
    def mean_flux(lo, hi):
        sel = (rest >= lo) & (rest < hi)
        if sel.sum() < 5:
            return np.nan
        return np.median(f[sel])
    red = mean_flux(4000.0, 4100.0)
    blue = mean_flux(3850.0, 3950.0)
    if blue <= 0 or not np.isfinite(red) or not np.isfinite(blue):
        return np.nan
    return red / blue


def balmer_break(wave, flux, z):
    """Balmer break (B-B): ratio of flux red of 3646 to blue of 3646 (rest)."""
    rest = wave / (1.0 + z)
    f = flux
    def mean_flux(lo, hi):
        sel = (rest >= lo) & (rest < hi)
        if sel.sum() < 5:
            return np.nan
        return np.median(f[sel])
    red = mean_flux(3750.0, 3950.0)
    blue = mean_flux(3500.0, 3600.0)
    if blue <= 0 or not np.isfinite(red) or not np.isfinite(blue):
        return np.nan
    return red / blue


def spectral_slope(wave, flux, z, lo=4500.0, hi=6000.0):
    """Continuum slope over rest-frame window, as f_nu ~ nu^alpha proxy.

    Returns linear fit slope of flux vs wavelength (per 1000 A) over
    the rest-frame window, computed on the continuum.
    """
    rest = wave / (1.0 + z)
    sel = (rest >= lo) & (rest <= hi)
    if sel.sum() < 20:
        return np.nan
    x = rest[sel] / 1000.0
    y = flux[sel]
    # robust slope via median of local differences is overkill; simple fit
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    return float(m)  # per 1000 A rest


def continuum_metrics(wave, flux, z):
    """Bundle all continuum shape metrics at a trial redshift."""
    return {
        "Dn4000": dn4000(wave, flux, z),
        "BalmerBreak": balmer_break(wave, flux, z),
        "slope_4500_6000": spectral_slope(wave, flux, z),
    }
