"""Evaluation of agent verdicts against expert VI ground truth."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def load_truth(training_dir="training_set"):
    """Load expert VI ground truth for all training spectra.

    Returns DataFrame indexed by (type, index) with VI_SPECTYPE, VI_Z, VI_QUALITY.
    """
    rows = []
    for t in ("BGS", "ELG", "LRG", "QSO"):
        csv = os.path.join(training_dir, t, f"{t}_tr_set.csv")
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            rows.append(
                {
                    "type": t,
                    "index": int(r["index"]),
                    "targetid": int(r["TARGETID"]),
                    "VI_SPECTYPE": str(r["VI_SPECTYPE"]).strip().upper(),
                    "VI_Z": float(r["VI_Z"]),
                    "VI_QUALITY": float(r["VI_QUALITY"]),
                }
            )
    truth = pd.DataFrame(rows)
    truth["file"] = truth.apply(
        lambda r: os.path.join(training_dir, r["type"], f"{r['index']}.fits"), axis=1
    )
    return truth


def classify_match(pred, truth):
    """Binary match: QSO vs Galaxy (STAR/UNKNOWN count as Galaxy mismatch handled by caller)."""
    p = "QSO" if pred == "QSO" else "GALAXY"
    t = "QSO" if truth == "QSO" else "GALAXY"
    return p == t


def z_agreement(pred_z, truth_z, tol=None, velocity_tol_km_s=600.0):
    """Redshift agreement: |dz| < tol OR |dv| < velocity_tol."""
    if pred_z is None or truth_z is None:
        return False, np.inf
    dz = abs(pred_z - truth_z)
    dv = dz / (1.0 + truth_z) * 299792.458  # km/s
    ok = dv < velocity_tol_km_s
    if tol is not None:
        ok = ok or dz < tol
    return bool(ok), dv


def evaluate_verdicts(verdicts, truth, velocity_tol_km_s=600.0):
    """verdicts: dict targetid -> Verdict (or dict). truth: DataFrame.

    Returns (metrics dict, detail DataFrame).
    """
    rows = []
    for _, tr in truth.iterrows():
        v = verdicts.get(tr["targetid"])
        if v is None:
            continue
        if hasattr(v, "to_dict"):
            vd = v.to_dict()
        else:
            vd = v
        z_ok, dv = z_agreement(vd.get("z"), tr["VI_Z"], velocity_tol_km_s=velocity_tol_km_s)
        rows.append(
            {
                "type": tr["type"],
                "index": tr["index"],
                "targetid": tr["targetid"],
                "truth_spectype": tr["VI_SPECTYPE"],
                "pred_spectype": vd.get("spectype"),
                "type_ok": classify_match(vd.get("spectype"), tr["VI_SPECTYPE"]),
                "truth_z": tr["VI_Z"],
                "pred_z": vd.get("z"),
                "dz": abs(vd.get("z", np.nan) - tr["VI_Z"]),
                "dv_km_s": dv,
                "z_ok": z_ok,
                "truth_q": tr["VI_QUALITY"],
                "pred_q": vd.get("quality"),
                "n_secure_lines": vd.get("n_secure_lines"),
                "fatal_issues": vd.get("fatal_issues"),
            }
        )
    det = pd.DataFrame(rows)
    if len(det) == 0:
        return {}, det

    metrics = {
        "n": len(det),
        "type_accuracy": float(det["type_ok"].mean()),
        "z_accuracy_600km_s": float(det["z_ok"].mean()),
        "median_dz": float(det["dz"].median()),
        "mean_dz": float(det["dz"].mean()),
        "median_dv_km_s": float(det["dv_km_s"].median()),
        "quality_mae": float((det["pred_q"] - det["truth_q"]).abs().mean()),
        "quality_exact": float((det["pred_q"].round(1) == det["truth_q"].round(1)).mean()),
    }
    # per-type
    for t in ("BGS", "ELG", "LRG", "QSO"):
        sub = det[det["type"] == t]
        if len(sub) == 0:
            continue
        metrics[f"{t}_type_acc"] = float(sub["type_ok"].mean())
        metrics[f"{t}_z_acc"] = float(sub["z_ok"].mean())
        metrics[f"{t}_median_dz"] = float(sub["dz"].median())
        metrics[f"{t}_quality_mae"] = float((sub["pred_q"] - sub["truth_q"]).abs().mean())
    return metrics, det


def save_verdicts(verdicts, path):
    out = {}
    for tid, v in verdicts.items():
        if hasattr(v, "to_dict"):
            out[tid] = v.to_dict()
        else:
            out[tid] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)


def load_verdicts(path):
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}
