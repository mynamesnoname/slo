"""Feature extraction from a DESI spectrum.

Produces:
- a rebinned, merged spectrum over the full B/R/Z range,
- a Gaussian-smoothed version (sigma in Angstrom),
- a continuum estimate,
- detected emission/absorption features with significance,
- line-window measurements for candidate rest-frame lines at trial redshifts,
- flags for known noisy regions (OH zone, arm junctions, blue edge).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .lines import LINE_TABLE

# Known sky / atmospheric features (observed frame) - from kb/lines.md
SKYLINES = [
    5577.3,  # [O I] green
    5890.0, 5896.0,  # Na D airglow
    6300.3, 6363.8,  # [O I] red pair
    6867.0,  # O2 B band (absorption)
    7605.0,  # O2 A band (absorption)
]

# Wavelength regions to flag
OH_ZONE_STRONG = 7800.0
OH_ZONE_MILD = (7000.0, 7400.0)
JUNCTIONS = [(5760.0, 5800.0), (7520.0, 7620.0)]
BLUE_EDGE = 4000.0


class SpectrumFeatures:
    def __init__(self, spec, sigma_angstrom=15.0):
        self.spec = spec
        self.sigma_angstrom = sigma_angstrom
        self._build_merged()
        self._build_smooth_and_continuum()
        self._detect_features()

    # ------------------------------------------------------------------ #
    # merged spectrum
    # ------------------------------------------------------------------ #
    def _build_merged(self):
        arms = []
        for arm in ("B", "R", "Z"):
            w = self.spec.wave[arm]
            f = self.spec.flux[arm]
            iv = self.spec.ivar[arm]
            m = self.spec.mask[arm]
            # exclude masked pixels and non-positive ivar
            good = (iv > 0) & (m == 0)
            if good.sum() == 0:
                continue
            arms.append((w[good], f[good], iv[good], np.full(good.sum(), arm)))

        if not arms:
            raise ValueError("no usable pixels")

        self.wave = np.concatenate([a[0] for a in arms])
        self.flux = np.concatenate([a[1] for a in arms])
        self.ivar = np.concatenate([a[2] for a in arms])
        self.arm_of = np.concatenate([a[3] for a in arms])

        # sort by wavelength (arms overlap)
        order = np.argsort(self.wave)
        self.wave = self.wave[order]
        self.flux = self.flux[order]
        self.ivar = self.ivar[order]
        self.arm_of = self.arm_of[order]

        self.noise = np.where(self.ivar > 0, 1.0 / np.sqrt(self.ivar), np.inf)

    # ------------------------------------------------------------------ #
    # smoothing + continuum
    # ------------------------------------------------------------------ #
    def _build_smooth_and_continuum(self):
        # convert sigma from Angstrom to pixels (median pixel spacing)
        dw = np.median(np.diff(self.wave))
        sigma_pix = self.sigma_angstrom / dw
        # build Gaussian kernel
        n = max(3, int(round(4 * sigma_pix)))
        x = np.arange(-n, n + 1)
        kern = np.exp(-0.5 * (x / sigma_pix) ** 2)
        kern = kern / kern.sum()

        fl = np.where(np.isfinite(self.flux), self.flux, np.nan)
        self.flux_sm = ndimage.convolve1d(
            np.nan_to_num(fl, nan=np.nanmedian(fl)), kern, mode="nearest"
        )

        # noise of the smoothed flux via error propagation:
        # var_out = conv(var_in, kern^2)
        var = np.where(self.ivar > 0, 1.0 / np.maximum(self.ivar, 1e-30), 1e30)
        var = np.where(np.isfinite(var), var, 1e30)
        var_sm = ndimage.convolve1d(var, kern ** 2, mode="nearest")
        self.noise_sm = np.sqrt(np.maximum(var_sm, 0.0))
        self.noise = np.where(self.ivar > 0, 1.0 / np.sqrt(self.ivar), np.inf)

        # continuum: running median with a wide window (300 A), then smooth
        win = max(9, int(round(300.0 / dw)))
        if win % 2 == 0:
            win += 1
        cont = ndimage.median_filter(self.flux_sm, size=win, mode="nearest")
        self.continuum = ndimage.gaussian_filter1d(cont, sigma=max(sigma_pix, 9.0), mode="nearest")
        self.continuum = np.maximum(self.continuum, 1e-6)

        # normalized residuals
        self.norm = (self.flux_sm - self.continuum) / self.continuum

        # robust local significance via band-pass:
        # subtract a heavily smoothed version (continuum scale) from the
        # line-scale smoothed flux, then normalize by the local scatter of
        # that difference. This measures how much a feature stands out from
        # its own neighbourhood (local contrast), robust to broad continuum
        # shape (breaks, depressions) and correlated noise.
        sigma_wide = max(3 * sigma_pix, int(round(150.0 / dw)))
        flux_wide = ndimage.gaussian_filter1d(self.flux_sm, sigma=sigma_wide, mode="nearest")
        self.flux_wide = flux_wide
        self.diff = self.flux_sm - flux_wide
        # expected noise of the band-passed signal from propagated ivar
        n_wide = max(3, int(round(4 * sigma_wide)))
        xw = np.arange(-n_wide, n_wide + 1)
        kw = np.exp(-0.5 * (xw / sigma_wide) ** 2)
        kw = kw / kw.sum()
        var_wide = ndimage.convolve1d(var, kw ** 2, mode="nearest")
        noise_diff = np.sqrt(np.maximum(var_sm + var_wide, 0.0))
        # local scatter: rolling MAD of diff over +-150 A
        rwin = max(9, int(round(150.0 / dw)))
        if rwin % 2 == 0:
            rwin += 1
        med = ndimage.median_filter(self.diff, size=rwin, mode="nearest")
        mad = ndimage.median_filter(np.abs(self.diff - med), size=rwin, mode="nearest")
        # significance floor: at least the propagated noise of the band-pass
        self.local_rms = np.maximum(1.4826 * mad, 0.7 * noise_diff)
        self.norm_sig = self.diff / self.local_rms

    # ------------------------------------------------------------------ #
    # feature detection (emission peaks & absorption troughs)
    # ------------------------------------------------------------------ #
    def _detect_features(self, snr_threshold=4.0, min_sep_angstrom=8.0):
        sig = self.norm_sig
        # prominence-based peak finding on the smoothed significance
        dw = np.median(np.diff(self.wave))
        min_sep_pix = max(3, int(round(min_sep_angstrom / dw)))

        self.features = []  # dicts: wave, sig, kind ('em'/'abs'), width_pix, arm
        for kind, sign in (("em", 1.0), ("abs", -1.0)):
            s = sign * sig
            peaks, props = signal.find_peaks(
                s,
                height=snr_threshold,
                distance=min_sep_pix,
                prominence=snr_threshold * 0.5,
                width=1,
            )
            for i, p in enumerate(peaks):
                w = self.wave[p]
                # width in Angstrom
                widths = props["widths"][i] * dw
                self.features.append(
                    {
                        "wave": float(w),
                        "sig": float(sign * s[p]),
                        "kind": kind,
                        "width_a": float(widths),
                        "arm": self.arm_of[p],
                    }
                )

        self.features.sort(key=lambda f: -abs(f["sig"]))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def index_at(self, w):
        """Nearest index to observed wavelength w."""
        return int(np.argmin(np.abs(self.wave - w)))

    def flux_window(self, w, half_width=25.0):
        """Flux (smoothed) in window [w-half, w+half]. Returns (wave, flux, continuum)."""
        i = self.index_at(w)
        dw = np.median(np.diff(self.wave))
        n = max(2, int(round(half_width / dw)))
        lo, hi = max(0, i - n), min(len(self.wave), i + n + 1)
        return self.wave[lo:hi], self.flux_sm[lo:hi], self.continuum[lo:hi]

    def measure_line(self, w_obs, rest_name=None, half_width=20.0):
        """Measure line significance at observed wavelength w_obs.

        Returns dict with continuum, peak deviation, significance, and
        whether it looks like emission or absorption. Significance is the
        local-contrast deviation (band-pass filtered) divided by the local
        scatter, i.e. how much the feature stands out from its neighbourhood.
        """
        i = self.index_at(w_obs)
        if abs(self.wave[i] - w_obs) > half_width * 2:
            return None
        cont = self.continuum[i]
        dev = self.diff[i]
        sig = dev / max(self.local_rms[i], 1e-12)
        # propagated-ivar significance (noise of band-pass from ivar only)
        sig_ivar = dev / max(self.noise_sm[i], 1e-12)
        # local extrema check: is this pixel a local max/min in smoothed?
        lo, hi = max(0, i - 3), min(len(self.wave), i + 4)
        local = self.flux_sm[lo:hi]
        is_local_max = self.flux_sm[i] >= local.max()
        is_local_min = self.flux_sm[i] <= local.min()
        return {
            "w_obs": float(w_obs),
            "w_nearest": float(self.wave[i]),
            "cont": float(cont),
            "dev": float(dev),
            "sig": float(sig),
            "sig_ivar": float(sig_ivar),
            "emission": bool(sig > 0 and is_local_max),
            "absorption": bool(sig < 0 and is_local_min),
            "arm": self.arm_of[i],
        }

    def zone_flags(self, w_obs):
        """Return list of contamination flags for an observed wavelength."""
        flags = []
        if w_obs < BLUE_EDGE:
            flags.append("blue-edge-noise")
        for lo, hi in JUNCTIONS:
            if lo <= w_obs <= hi:
                flags.append("arm-junction")
        if w_obs > OH_ZONE_STRONG:
            flags.append("OH-zone")
        elif OH_ZONE_MILD[0] <= w_obs <= OH_ZONE_MILD[1]:
            flags.append("OH-mild")
        for sky in SKYLINES:
            if abs(w_obs - sky) < 4.0:
                flags.append(f"skyline-{sky:.0f}")
        return flags

    # ------------------------------------------------------------------ #
    # candidate line check at a trial redshift
    # ------------------------------------------------------------------ #
    def check_lines_at_z(self, z, line_names=None, snr_report=2.0):
        """For each rest-frame line (optionally restricted), compute the
        observed wavelength and measure significance at that position.

        Returns list of dicts: name, rest, w_obs, measurement, flags,
        in_range. For [O II] the unresolved doublet (3726.0/3729.0) is
        measured at both components and the result stored under 'doublet'.
        """
        if line_names is None:
            table = LINE_TABLE
        else:
            table = [ln for ln in LINE_TABLE if ln["name"] in line_names]

        results = []
        for ln in table:
            w_obs = ln["rest"] * (1.0 + z)
            if w_obs < self.wave[0] + 30 or w_obs > self.wave[-1] - 30:
                results.append(
                    {
                        "name": ln["name"],
                        "rest": ln["rest"],
                        "type": ln["type"],
                        "width_class": ln.get("width_class", ""),
                        "w_obs": float(w_obs),
                        "in_range": False,
                        "meas": None,
                        "flags": self.zone_flags(w_obs),
                        "doublet": None,
                    }
                )
                continue
            meas = self.measure_line(w_obs)
            doublet = None
            if ln["name"] == "[O II]":
                # unresolved doublet 3726.0 / 3729.0
                ma = self.measure_line(3726.0 * (1.0 + z))
                mb = self.measure_line(3729.0 * (1.0 + z))
                if ma and mb:
                    doublet = {
                        "a_sig": ma["sig"],
                        "b_sig": mb["sig"],
                        "a_dev": ma["dev"],
                        "b_dev": mb["dev"],
                    }
            results.append(
                {
                    "name": ln["name"],
                    "rest": ln["rest"],
                    "type": ln["type"],
                    "width_class": ln.get("width_class", ""),
                    "w_obs": float(w_obs),
                    "in_range": True,
                    "meas": meas,
                    "flags": self.zone_flags(w_obs),
                    "doublet": doublet,
                }
            )
        return results

    def feature_summary(self, n_em=12, n_abs=8):
        """Compact text summary of the strongest detected features."""
        em = [f for f in self.features if f["kind"] == "em"][:n_em]
        abs_ = [f for f in self.features if f["kind"] == "abs"][:n_abs]
        lines = []
        lines.append("Strongest emission-like features (observed frame):")
        for f in em:
            ivar_sig = self.norm_sig[self.index_at(f["wave"])]
            lines.append(
                f"  {f['wave']:9.1f} A  sig={f['sig']:+6.1f} (ivar {ivar_sig:+7.1f})  width~{f['width_a']:5.1f} A ({f['arm']})"
            )
        lines.append("Strongest absorption-like features (observed frame):")
        for f in abs_:
            ivar_sig = self.norm_sig[self.index_at(f["wave"])]
            lines.append(
                f"  {f['wave']:9.1f} A  sig={f['sig']:+6.1f} (ivar {ivar_sig:+7.1f})  width~{f['width_a']:5.1f} A ({f['arm']})"
            )
        return "\n".join(lines)
