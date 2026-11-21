"""Spectrum loading and preprocessing for DESI coadd FITS files.

DESI coadd files contain B/R/Z arms as separate HDUs:
  {ARM}_WAVELENGTH, {ARM}_FLUX, {ARM}_IVAR, {ARM}_MASK, {ARM}_RESOLUTION

The FIBERMAP HDU carries target metadata including expert VI results
(VI_Z, VI_SPECTYPE, VI_QUALITY).
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

ARMS = ("B", "R", "Z")
ARMS_COVERAGE = {"B": (3600.0, 5800.0), "R": (5760.0, 7620.0), "Z": (7520.0, 9824.0)}


class Spectrum:
    """A single DESI target spectrum."""

    def __init__(self, filename: str):
        self.filename = filename
        with fits.open(filename) as hdul:
            self.hdul = hdul
            self._load()

    def _load(self):
        hdul = self.hdul
        self.fibermap = hdul["FIBERMAP"].data
        self.targetid = int(self.fibermap["TARGETID"][0])

        self.wave = {}
        self.flux = {}
        self.ivar = {}
        self.mask = {}
        self.resolution = {}
        for arm in ARMS:
            self.wave[arm] = np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=np.float64)
            self.flux[arm] = np.asarray(hdul[f"{arm}_FLUX"].data[0], dtype=np.float64)
            self.ivar[arm] = np.asarray(hdul[f"{arm}_IVAR"].data[0], dtype=np.float64)
            self.mask[arm] = np.asarray(hdul[f"{arm}_MASK"].data[0], dtype=np.int32)
            if f"{arm}_RESOLUTION" in hdul:
                self.resolution[arm] = np.asarray(hdul[f"{arm}_RESOLUTION"].data)
            else:
                self.resolution[arm] = None

        # expert VI results (may be missing)
        fm = self.fibermap
        self.vi_z = float(fm["VI_Z"][0]) if "VI_Z" in fm.dtype.names and not np.isnan(fm["VI_Z"][0]) else None
        self.vi_spectype = (
            str(fm["VI_SPECTYPE"][0]).strip() if "VI_SPECTYPE" in fm.dtype.names else None
        )
        self.vi_quality = (
            float(fm["VI_QUALITY"][0])
            if "VI_QUALITY" in fm.dtype.names and not np.isnan(fm["VI_QUALITY"][0])
            else None
        )

        # per-arm median SNR (signal-to-noise per pixel, from ivar)
        self.snr_arm = {}
        for arm in ARMS:
            iv = self.ivar[arm]
            good = iv > 0
            if good.sum() > 0:
                noise = np.sqrt(1.0 / iv[good])
                # use median flux / median noise as a robust per-pixel SNR proxy
                med_flux = np.median(np.abs(self.flux[arm][good]))
                med_noise = np.median(noise)
                self.snr_arm[arm] = med_flux / med_noise if med_noise > 0 else 0.0
            else:
                self.snr_arm[arm] = 0.0
        self.snr_sum = sum(self.snr_arm.values())

    def arm_range(self, arm: str):
        return self.wave[arm][0], self.wave[arm][-1]

    def close(self):
        self.hdul.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_spectrum(filename: str) -> Spectrum:
    return Spectrum(filename)
