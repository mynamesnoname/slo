"""Parse redrock outputs (best-fit FITS + details h5) into per-target results."""
from __future__ import annotations

import os

import h5py
import numpy as np
from astropy.io import fits


def parse_redrock_fits(filename):
    """Parse the rrdesi output FITS: per-target best z / spectype / chi2.

    Returns dict keyed by TARGETID.
    """
    out = {}
    with fits.open(filename) as hdul:
        rs = hdul["REDSHIFTS"].data
        for row in rs:
            out[int(row["TARGETID"])] = {
                "z": float(row["Z"]),
                "zerr": float(row["ZERR"]),
                "zwarn": int(row["ZWARN"]),
                "chi2": float(row["CHI2"]),
                "spectype": str(row["SPECTYPE"]).strip(),
                "subtype": str(row["SUBTYPE"]).strip(),
                "deltachi2": float(row["DELTACHI2"]),
            }
    return out


def parse_redrock_details(filename):
    """Parse the details h5: per-target per-class best z and chi2 from zscan.

    zscan groups are named like 'GALAXY', 'QSO:::HIZ', 'QSO:::LOZ', 'STAR:::G'.
    For each target, for each top-level class, we take the minimum chi2 across
    the class's subgroups and the z at that minimum.

    Returns dict: targetid -> {class: {'z':..., 'chi2':...}, ...}
    """
    out = {}
    with h5py.File(filename, "r") as f:
        targetids = np.asarray(f["targetids"])
        zscan = f["zscan"]
        class_groups = {}
        for grp_name in zscan.keys():
            cls = grp_name.split(":::")[0]
            class_groups.setdefault(cls, []).append(grp_name)

        for i, tid in enumerate(targetids):
            tid = int(tid)
            per_class = {}
            for cls, grps in class_groups.items():
                best_z, best_chi2 = None, np.inf
                for g in grps:
                    zchi2 = np.asarray(zscan[g]["zchi2"])[i]
                    zz = np.asarray(zscan[g]["redshifts"])
                    j = int(np.argmin(zchi2))
                    if zchi2[j] < best_chi2:
                        best_chi2 = float(zchi2[j])
                        best_z = float(zz[j])
                per_class[cls] = {"z": best_z, "chi2": best_chi2}
            out[tid] = per_class
    return out


def parse_redrock_zscan_minima(filename, per_class=("GALAXY", "QSO"), nmin=3,
                               min_sep=0.005):
    """Extract local minima of the zscan chi2 curves for each target.

    Returns dict: targetid -> list of {'z':..., 'chi2':..., 'class':...},
    sorted by chi2, containing up to nmin local minima per class (with
    minimum separation min_sep in z).
    """
    from scipy.signal import argrelextrema

    out = {}
    with h5py.File(filename, "r") as f:
        targetids = np.asarray(f["targetids"])
        zscan = f["zscan"]
        class_groups = {}
        for grp_name in zscan.keys():
            cls = grp_name.split(":::")[0]
            if cls not in per_class:
                continue
            class_groups.setdefault(cls, []).append(grp_name)

        for i, tid in enumerate(targetids):
            tid = int(tid)
            cands = []
            for cls, grps in class_groups.items():
                for g in grps:
                    zchi2 = np.asarray(zscan[g]["zchi2"])[i]
                    zz = np.asarray(zscan[g]["redshifts"])
                    if len(zz) < 3:
                        continue
                    # local minima
                    idx = argrelextrema(zchi2, np.less, order=2)[0]
                    idx = np.concatenate([[int(np.argmin(zchi2))], idx])
                    # dedupe, keep lowest chi2
                    for j in sorted(set(int(k) for k in idx)):
                        cands.append(
                            {"z": float(zz[j]), "chi2": float(zchi2[j]), "class": cls}
                        )
            # sort by chi2, dedupe by z proximity
            cands.sort(key=lambda c: c["chi2"])
            deduped = []
            for c in cands:
                if all(abs(c["z"] - d["z"]) >= min_sep for d in deduped):
                    deduped.append(c)
            out[tid] = deduped[: nmin * len(class_groups)]
    return out


def run_redrock_if_needed(spectra_dir="training_set", results_dir="results", force=False):
    """Run rrdesi on all training spectra if results are not already present."""
    fits_out = os.path.join(results_dir, "redrock_all.fits")
    h5_out = os.path.join(results_dir, "redrock_all.h5")
    if os.path.exists(fits_out) and os.path.exists(h5_out) and not force:
        return fits_out, h5_out
    raise FileNotFoundError(
        f"redrock results missing: {fits_out}. Run rrdesi first."
    )
