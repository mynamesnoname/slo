"""Deterministic feature extraction from DESI spectra for visual-inspection-style analysis.

For each input FITS spectrum this module produces a quantitative, auditable feature
report:
  * a combined wavelength/flux/ivar vector (B+R+Z cameras, deduplicated)
  * a robust continuum estimate and Gaussian-smoothed flux
  * per-line integrated S/N measurements over a trial-redshift grid, for a
    rest-frame line catalog (emission and absorption)
  * candidate redshifts (local maxima of a line-weighted score curve) with the
    specific lines that drive each candidate
  * continuum shape diagnostics: D4000 break, blue/red slope, Ly-alpha forest
    indicator, global S/N

Everything here is deterministic; the LLM agent later reasons over this report.
"""

import os
import numpy as np

from astropy.io import fits

# ---------------------------------------------------------------------------
# Rest-frame line catalog
# ---------------------------------------------------------------------------
# name: (rest wavelength Angstrom, expected relative strength, kind)
# kind: 'emi' emission / 'abs' absorption / 'break' continuum break
# Expected strengths are rough priors used only to weight the redshift score.
LINE_CATALOG = {
    # QSO / high-ionization emission
    "LyA":      (1215.67, 4.0, "emi"),
    "SiIV":     (1393.76, 1.2, "emi"),
    "CIV":      (1549.06, 2.6, "emi"),
    "HeII":     (1640.40, 0.9, "emi"),
    "CIII]":    (1908.73, 2.0, "emi"),
    "MgII":     (2796.35, 1.8, "emi"),
    # AGN indicator
    "NeV":      (3425.90, 0.7, "emi"),
    # Galaxy / star-forming emission
    "OII":      (3727.09, 1.6, "emi"),
    "Hdelta":   (4101.73, 0.5, "emi"),
    "Hgamma":   (4340.46, 0.7, "emi"),
    "Hbeta":    (4861.33, 1.0, "emi"),
    "OIII4959": (4958.91, 0.8, "emi"),
    "OIII5007": (5006.84, 1.4, "emi"),
    "NII":      (6583.45, 0.4, "emi"),
    "Halpha":   (6562.80, 1.2, "emi"),
    "SII6716":  (6716.44, 0.3, "emi"),
    "SII6731":  (6730.82, 0.3, "emi"),
    "NeIII":    (3868.71, 0.7, "emi"),
    # Absorption / continuum features
    "CaK":      (3933.66, 1.0, "abs"),
    "CaH":      (3968.47, 0.8, "abs"),
    "MgI":      (5175.30, 0.6, "abs"),
    "NaD":      (5892.50, 0.5, "abs"),
}

# Which lines are usable as "primary" evidence for each class
QSO_LINES = ["LyA", "SiIV", "CIV", "HeII", "CIII]", "MgII", "NeV"]
GAL_LINES = ["OII", "Hdelta", "Hgamma", "Hbeta", "OIII4959", "OIII5007",
             "NII", "Halpha", "CaK", "CaH", "MgI", "NaD"]

D4000_BLUE = (3750.0, 3950.0)
D4000_RED = (4050.0, 4250.0)

# Bright night-sky (OH + atomic) emission lines in the DESI band.
# Spectral features landing within SKY_TOL of these are suspect sky residuals.
SKY_LINES = np.array([
    5577.34, 5890.0, 5896.0, 6300.30, 6363.78, 6863.9,
    6913.0, 6932.0, 6951.0, 6970.0, 6989.0, 7008.0, 7027.0, 7046.0,
    7065.0, 7084.0, 7103.0, 7122.0, 7141.0, 7160.0, 7179.0, 7198.0,
    7217.0, 7236.0, 7255.0, 7274.0, 7293.0, 7312.0, 7331.0, 7350.0,
    7369.0, 7388.0, 7407.0, 7426.0, 7445.0, 7464.0, 7483.0, 7502.0,
    7521.0, 7540.0, 7559.0, 7578.0, 7597.0, 7616.0, 7635.0, 7654.0,
    7673.0, 7692.0, 7711.0, 7730.0, 7749.0, 7768.0, 7787.0, 7806.0,
    7825.0, 7844.0, 7863.0, 7882.0, 7901.0, 7920.0, 7939.0, 7958.0,
    7977.0, 7996.0, 8015.0, 8034.0, 8053.0, 8072.0, 8091.0, 8110.0,
    8129.0, 8148.0, 8167.0, 8186.0, 8205.0, 8224.0, 8243.0, 8262.0,
    8281.0, 8300.0, 8319.0, 8338.0, 8357.0, 8376.0, 8395.0, 8414.0,
    8433.0, 8452.0, 8471.0, 8490.0, 8509.0, 8528.0, 8547.0, 8566.0,
    8585.0, 8604.0, 8623.0, 8642.0, 8661.0, 8680.0, 8699.0, 8718.0,
    8737.0, 8756.0, 8775.0, 8794.0, 8813.0, 8832.0, 8851.0, 8870.0,
    8889.0, 8908.0, 8927.0, 8946.0, 8965.0, 8984.0, 9003.0, 9022.0,
    9041.0, 9060.0, 9079.0, 9098.0, 9117.0, 9136.0, 9155.0, 9174.0,
    9193.0, 9212.0, 9231.0, 9250.0, 9269.0, 9288.0, 9307.0, 9326.0,
    9345.0, 9364.0, 9383.0, 9402.0, 9421.0, 9440.0, 9459.0, 9478.0,
    9497.0, 9516.0, 9535.0, 9554.0, 9573.0, 9592.0, 9611.0, 9630.0,
    9649.0, 9668.0, 9687.0, 9706.0, 9725.0, 9744.0, 9763.0, 9782.0,
    9801.0, 9820.0, 9839.0,
])
SKY_TOL = 2.5


def near_sky(obs_wave):
    """True if the observed wavelength is within SKY_TOL of a bright sky line."""
    i = np.searchsorted(SKY_LINES, obs_wave)
    for j in (i - 1, i):
        if 0 <= j < len(SKY_LINES) and abs(SKY_LINES[j] - obs_wave) <= SKY_TOL:
            return True
    return False


def load_spectrum(path):
    """Load a DESI spectrum FITS, return dict with wavelength, flux, ivar."""
    with fits.open(path) as hdus:
        # Combine B, R, Z cameras
        waves, fluxs, ivars = [], [], []
        for cam in ["B", "R", "Z"]:
            w = hdus[f"{cam}_WAVELENGTH"].data.astype(np.float64)
            f = hdus[f"{cam}_FLUX"].data.astype(np.float64)
            iv = hdus[f"{cam}_IVAR"].data.astype(np.float64)
            # FLUX/IVAR have shape (1, nwave)
            if f.ndim == 2:
                f = f.ravel()
            if iv.ndim == 2:
                iv = iv.ravel()
            waves.append(w)
            fluxs.append(f)
            ivars.append(iv)
        # fibermap / targetid
        fm = hdus["FIBERMAP"].data
        targetid = fm["TARGETID"][0]
    w = np.concatenate(waves)
    f = np.concatenate(fluxs)
    iv = np.concatenate(ivars)
    # sort by wavelength
    order = np.argsort(w)
    w, f, iv = w[order], f[order], iv[order]
    # deduplicate wavelengths, keep the one with higher ivar
    uniq, idx = np.unique(w, return_index=True)
    keep = []
    for u in np.unique(w):
        sel = np.where(w == u)[0]
        keep.append(sel[np.argmax(iv[sel])])
    keep = np.array(keep)
    w, f, iv = w[keep], f[keep], iv[keep]
    return {"wavelength": w, "flux": f, "ivar": iv, "targetid": int(targetid)}


def clean_spectrum(spec, nsig=5.0):
    """Mask ivar<=0 and strong outlier pixels (cosmic/sky residuals)."""
    w, f, iv = spec["wavelength"], spec["flux"], spec["ivar"].copy()
    good = iv > 0
    good_idx = np.where(good)[0]
    rolling = np.full(len(f), np.nan)
    for center in good_idx[::20]:
        lo = max(0, center - 60); hi = min(len(f), center + 61)
        sel = good[lo:hi]
        if sel.sum() > 10:
            rolling[center] = np.median(f[lo:hi][sel])
    mask_ok = ~np.isnan(rolling)
    if mask_ok.sum() > 50:
        rolling = np.interp(np.arange(len(f)), np.where(mask_ok)[0], rolling[mask_ok])
    else:
        rolling = np.nanmedian(f[good])
    resid = f - rolling
    sig = np.full(len(f), np.nan)
    for center in good_idx[::20]:
        lo = max(0, center - 60); hi = min(len(f), center + 61)
        sel = good[lo:hi]
        if sel.sum() > 10:
            sig[center] = 1.4826 * np.median(
                np.abs(resid[lo:hi][sel] - np.median(resid[lo:hi][sel])))
    mask_ok = ~np.isnan(sig)
    if mask_ok.sum() > 50:
        sig = np.interp(np.arange(len(f)), np.where(mask_ok)[0], sig[mask_ok])
    else:
        sig = 1.0
    bad = (np.abs(resid) > nsig * sig) & good
    iv[bad] = 0.0
    spec["ivar"] = iv
    return spec


def continuum(w, f, iv, width_ang=120.0):
    """Robust continuum: local median in a window (iterated), masked by ivar."""
    good = iv > 0
    c = np.full_like(f, np.nan)
    step = max(1, int(width_ang / np.median(np.diff(w))))
    for i in range(0, len(w), step):
        lo = max(0, i - step); hi = min(len(w), i + step)
        sel = good[lo:hi]
        if sel.sum() > 5:
            c[i] = np.median(f[lo:hi][sel])
    ok = ~np.isnan(c)
    if ok.sum() < 20:
        return np.full_like(f, np.nanmedian(f[good]))
    return np.interp(np.arange(len(w)), np.where(ok)[0], c[ok])


def gaussian_smooth(f, sigma_pix=15.0):
    """Gaussian smoothing in pixel space."""
    if sigma_pix <= 0:
        return f.copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(f, sigma_pix, mode="nearest")


def line_sn_at_z(w, f, iv, cont, rest_wave, z, broad=False):
    """Integrated S/N of a line at redshift z.

    Uses a window of +-3 A (narrow) or +-40 A (broad) rest-frame around the
    line center. Returns (sn, n_pixels_used).
    """
    obs = rest_wave * (1.0 + z)
    if broad:
        half = 40.0 * (1.0 + z)
    else:
        half = 3.0 * (1.0 + z)
    sel = (w >= obs - half) & (w <= obs + half) & (iv > 0)
    n = sel.sum()
    if n < 3:
        return 0.0, 0
    d = f[sel] - cont[sel]
    # robust: clip the single worst pixel (e.g. sky residual) at 4 sigma
    if n > 5:
        sig = np.sqrt(1.0 / iv[sel])
        zz = np.abs(d) / np.maximum(sig, 1e-12)
        if zz.max() > 4.0:
            keep = zz < zz.max()
            d = d[keep]
            # update the boolean mask to the kept pixels
            sel_idx = np.where(sel)[0][keep]
            sel = np.zeros_like(sel)
            sel[sel_idx] = True
            if sel.sum() < 3:
                return 0.0, 0
    var = np.sum(1.0 / iv[sel])
    if var <= 0:
        return 0.0, 0
    sn = np.sum(d) / np.sqrt(var)
    return sn, n


def d4000_index(w, f, cont, iv):
    """Dn(4000) style ratio using blue/red windows (continuum-normalized)."""
    def mean_flux(lo, hi):
        sel = (w >= lo) & (w <= hi) & (iv > 0)
        if sel.sum() < 5:
            return np.nan
        return np.mean(f[sel] / np.maximum(cont[sel], 1e-12))
    b = mean_flux(*D4000_BLUE)
    r = mean_flux(*D4000_RED)
    if np.isnan(b) or np.isnan(r) or b <= 0:
        return np.nan
    return r / b


def continuum_slope(w, f, cont, iv, lo=3600.0, hi=9000.0):
    """Power-law slope index of the continuum (f ~ lambda^alpha) in log space."""
    sel = (w >= lo) & (w <= hi) & (iv > 0) & np.isfinite(cont) & (cont > 0)
    if sel.sum() < 50:
        return np.nan
    lw = np.log10(w[sel]); lf = np.log10(cont[sel])
    # robust fit by medians in bins
    bins = np.linspace(lw.min(), lw.max(), 12)
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = (lw >= bins[i]) & (lw < bins[i + 1])
        if m.sum() > 10:
            xs.append(np.median(lw[m])); ys.append(np.median(lf[m]))
    if len(xs) < 4:
        return np.nan
    p = np.polyfit(xs, ys, 1)
    return p[0] / np.log10(np.e)  # d ln f / d ln lambda


def lyap_forest_indicator(w, f, cont, iv, z_qso):
    """Check for Ly-alpha forest absorption blueward of LyA at z_qso."""
    if not (1.7 <= z_qso <= 4.8):
        return 0.0, 0
    lyobs = 1215.67 * (1.0 + z_qso)
    # forest region: between LyB (1025.7) and LyA, in transmission
    lo = 1025.7 * (1.0 + z_qso) + 20.0
    hi = lyobs - 30.0
    if hi <= lo:
        return 0.0, 0
    sel = (w >= lo) & (w <= hi) & (iv > 0)
    if sel.sum() < 30:
        return 0.0, 0
    trans = f[sel] / np.maximum(cont[sel], 1e-12)
    # variance-weighted mean transmission < 0.9 means real absorption
    wts = iv[sel] / np.sum(iv[sel])
    mean_t = np.sum(trans * wts)
    # also fractional variance (forests are patchy)
    frac = np.sqrt(np.sum(wts * (trans - mean_t) ** 2)) / max(mean_t, 1e-6)
    return float(mean_t), float(frac)


def redshift_scan(w, f, iv, cont, zmin=0.0, zmax=4.5, dz=0.002):
    """Score every trial redshift with the weighted line catalog.

    Returns (zgrid, sn_narrow, sn_broad, score) where the S/N arrays have
    shape (nlines, nz). Uses cumulative-sum window sums for speed.
    """
    names = list(LINE_CATALOG.keys())
    zgrid = np.arange(zmin, zmax + dz / 2, dz)
    nz = len(zgrid)
    sn_narrow = np.zeros((len(names), nz))
    sn_broad = np.zeros((len(names), nz))
    good = iv > 0
    d = np.where(good, f - cont, 0.0)
    wgt = np.where(good, 1.0 / np.maximum(iv, 1e-30), 0.0)
    # cumulative sums over the sorted wavelength array
    csum_d = np.concatenate([[0.0], np.cumsum(d)])
    csum_w = np.concatenate([[0.0], np.cumsum(wgt)])
    idx = np.arange(len(w))
    for li, name in enumerate(names):
        rw, _, kind = LINE_CATALOG[name]
        obs = rw * (1.0 + zgrid)
        for zi in range(nz):
            # narrow window +-3 A rest
            lo = np.searchsorted(w, obs[zi] - 3.0 * (1.0 + zgrid[zi]), side="left")
            hi = np.searchsorted(w, obs[zi] + 3.0 * (1.0 + zgrid[zi]), side="right")
            sd = csum_d[hi] - csum_d[lo]
            sw = csum_w[hi] - csum_w[lo]
            if sw > 0 and hi - lo >= 3:
                sn_narrow[li, zi] = sd / np.sqrt(sw)
            # broad window +-40 A rest
            lo = np.searchsorted(w, obs[zi] - 40.0 * (1.0 + zgrid[zi]), side="left")
            hi = np.searchsorted(w, obs[zi] + 40.0 * (1.0 + zgrid[zi]), side="right")
            sd = csum_d[hi] - csum_d[lo]
            sw = csum_w[hi] - csum_w[lo]
            if sw > 0 and hi - lo >= 3:
                sn_broad[li, zi] = sd / np.sqrt(sw)
    return zgrid, sn_narrow, sn_broad


def _line_contrast(w, f, iv, cont, obs, z):
    """Peak |flux - continuum| / continuum in a +-3 A rest window."""
    sel = (w >= obs - 3.0 * (1 + z)) & (w <= obs + 3.0 * (1 + z)) & (iv > 0)
    if sel.sum() < 2:
        return 0.0
    peak = float(np.max(np.abs(f[sel] - cont[sel])))
    cmed = float(np.median(cont[sel]))
    return peak / max(cmed, 1e-12)


def find_candidates(zgrid, sn_narrow, sn_broad, names, ntop=6, min_sep_dz=0.02,
                    w=None, f=None, iv=None, cont=None):
    """Find local maxima of the combined score curve, ranked."""
    score = np.zeros(len(zgrid))
    for li, name in enumerate(names):
        rw, wgt, kind = LINE_CATALOG[name]
        combined = np.maximum(sn_narrow[li], 0.6 * sn_broad[li])
        if kind == "abs":
            # absorption: strong negative S/N is the expected signature
            combined = -combined
        score += wgt * combined
    # local maxima
    peaks = []
    i = 1
    while i < len(zgrid) - 1:
        if score[i] >= score[i - 1] and score[i] > score[i + 1]:
            # refine: find max within +-3 bins (never moving backwards)
            lo = max(0, i - 3); hi = min(len(zgrid), i + 4)
            j = max(i, lo + int(np.argmax(score[lo:hi])))
            peaks.append([zgrid[j], score[j], j])
            i = j + 1
        else:
            i += 1
    # non-maximum suppression by strength: of nearby peaks keep the strongest
    kept = []
    for z, s, j in sorted(peaks, key=lambda p: -p[1]):
        if all(abs(z - k[0]) > min_sep_dz for k in kept):
            kept.append([z, s, j])
    peaks = kept
    peaks.sort(key=lambda p: -p[1])
    out = []
    for z, s, j in peaks[:ntop * 2]:
        # for each line, report S/N at this z (narrow and broad)
        lines = []
        for li, name in enumerate(names):
            if abs(sn_narrow[li, j]) > 2.5 or abs(sn_broad[li, j]) > 2.5:
                rw = LINE_CATALOG[name][0]
                obs = rw * (1 + z)
                contrast = (_line_contrast(w, f, iv, cont, obs, z)
                            if w is not None else 0.0)
                lines.append((name, float(sn_narrow[li, j]), float(sn_broad[li, j]),
                              LINE_CATALOG[name][2], round(contrast, 1)))
        n_secure = len([l for l in lines
                        if abs(l[1]) >= 3.0 or abs(l[2]) >= 3.0])
        out.append({"z": float(z), "score": float(s),
                    "n_secure": n_secure, "lines": lines})
    # re-rank: prefer candidates with more mutually consistent secure lines
    # (a single strong line is often an alias of a different line at another z)
    out.sort(key=lambda c: (-c["n_secure"], -c["score"]))
    return out[:ntop], score


def global_snr(iv, flux, cont):
    """SNR_sum-like: sum of S/N per pixel over the full spectrum."""
    good = iv > 0
    if good.sum() < 100:
        return 0.0
    per_pix = flux[good] * np.sqrt(iv[good])
    # robust: exclude |per_pix| > 10 (lines)
    m = np.abs(per_pix) < 10.0
    if m.sum() < 50:
        m = np.ones_like(per_pix, dtype=bool)
    return float(np.sum(np.abs(per_pix[m])))


def extract_features(path, run_scan=True):
    """Full deterministic feature extraction for one spectrum."""
    spec = load_spectrum(path)
    spec = clean_spectrum(spec)
    w, f, iv = spec["wavelength"], spec["flux"], spec["ivar"]
    cont = continuum(w, f, iv)
    fs = gaussian_smooth(f, 15.0)
    cs = gaussian_smooth(cont, 15.0)
    names = list(LINE_CATALOG.keys())
    feats = {
        "targetid": spec["targetid"],
        "nwave": len(w),
        "wmin": float(w.min()), "wmax": float(w.max()),
        "ivar_fraction": float((iv > 0).mean()),
        "global_snr": global_snr(iv, f, cont),
        "d4000": float(d4000_index(w, f, cont, iv)),
        "slope": float(continuum_slope(w, f, cont, iv)),
        "med_flux": float(np.median(f[iv > 0])),
    }
    if run_scan:
        zgrid, snn, snb = redshift_scan(w, f, iv, cont)
        cands, score = find_candidates(zgrid, snn, snb, names,
                                       w=w, f=f, iv=iv, cont=cont)
        feats["candidates"] = cands
        feats["zgrid"] = zgrid
        feats["sn_narrow"] = snn
        feats["sn_broad"] = snb
        feats["score"] = score
        feats["line_names"] = names
        # best candidate + LyA forest check at that z
        if cands:
            zbest = cands[0]["z"]
            t, fr = lyap_forest_indicator(w, f, cont, iv, zbest)
            feats["lya_forest_at_best"] = (t, fr)
    # feature table at a given z (helper for the report)
    feats["_w"] = w; feats["_f"] = f; feats["_iv"] = iv; feats["_cont"] = cont
    feats["_fs"] = fs; feats["_cs"] = cs
    return feats


def lines_at_z(feats, z, sn_thresh=2.0, include_abs=True):
    """Return list of (name, rest, observed, sn_narrow, sn_broad, kind, expect)
    for all catalog lines at redshift z."""
    if "_w" not in feats:
        return []
    w, f, iv, cont = feats["_w"], feats["_f"], feats["_iv"], feats["_cont"]
    names = list(LINE_CATALOG.keys())
    out = []
    for name in names:
        rw, wgt, kind = LINE_CATALOG[name]
        if kind == "abs" and not include_abs:
            continue
        snn, _ = line_sn_at_z(w, f, iv, cont, rw, z, broad=False)
        snb, _ = line_sn_at_z(w, f, iv, cont, rw, z, broad=True)
        obs = rw * (1 + z)
        if max(abs(snn), abs(snb)) >= sn_thresh:
            # contrast: line peak flux / local continuum (robust to noisy pixels)
            sel = (w >= obs - 3.0 * (1 + z)) & (w <= obs + 3.0 * (1 + z)) & (iv > 0)
            if sel.sum() >= 2:
                peak = float(np.max(np.abs(f[sel] - cont[sel])))
                cmed = float(np.median(cont[sel]))
                contrast = peak / max(cmed, 1e-12)
            else:
                contrast = 0.0
            out.append({
                "name": name, "rest": rw, "obs": obs,
                "sn_narrow": round(float(snn), 1),
                "sn_broad": round(float(snb), 1),
                "kind": kind, "expect": wgt,
                "fwhm_hint": "broad" if abs(snb) > abs(snn) * 1.5 else "narrow",
                "sky": bool(near_sky(obs)),
                "contrast": round(contrast, 1),
            })
    return out


def serialize_report(feats, redrock=None, top_n_cands=5, max_lines_per_cand=6):
    """Human/LLM-readable text report of the deterministic features."""
    L = []
    L.append(f"TARGETID: {feats['targetid']}")
    L.append(f"Wavelength range: {feats['wmin']:.0f}-{feats['wmax']:.0f} A, "
             f"pixels: {feats['nwave']}, usable fraction: {feats['ivar_fraction']:.2f}")
    L.append(f"Global S/N (SNR_sum-like): {feats['global_snr']:.1f}")
    L.append(f"Continuum D4000 break index: {feats['d4000']:.3f}  "
             f"(>1.4 = strong galaxy break, ~1 = featureless/power law)")
    L.append(f"Continuum power-law slope alpha (dlnF/dlnL): {feats['slope']:.2f} "
             f"(negative = blue/rising to blue, positive = red/falling)")
    if redrock is not None:
        L.append(f"REDROCK FIT: z={redrock.get('Z', float('nan')):.4f} "
                 f"spectype={redrock.get('SPECTYPE', '?')} "
                 f"subtype={redrock.get('SUBTYPE', '?')} "
                 f"chi2={redrock.get('CHI2', 0):.0f} "
                 f"deltachi2={redrock.get('DELTACHI2', 0):.0f} "
                 f"zwarn={redrock.get('ZWARN', 0)}")
    L.append("")
    L.append("CANDIDATE REDSHIFTS (from weighted emission/absorption line S/N scan):")
    for i, c in enumerate(feats.get("candidates", [])[:top_n_cands]):
        L.append(f"  #{i + 1} z={c['z']:.4f} score={c['score']:.1f} "
                 f"n_secure={c.get('n_secure', '?')}")
        for name, snn, snb, kind in c["lines"][:max_lines_per_cand]:
            L.append(f"      {name:<8} rest={LINE_CATALOG[name][0]:7.2f} "
                     f"sn_narrow={snn:+.1f} sn_broad={snb:+.1f} ({kind})")
    if "lya_forest_at_best" in feats:
        t, fr = feats["lya_forest_at_best"]
        L.append(f"LyA forest check at best z: mean transmission {t:.2f}, "
                 f"patchiness {fr:.2f} (transmission <0.9 & patchy -> QSO evidence)")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    feats = extract_features(sys.argv[1])
    print(serialize_report(feats))
