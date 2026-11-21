#!/usr/bin/env python
"""Baseline: evaluate redrock's own z/spectype against expert VI."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viagent.evaluate import classify_match, load_truth, z_agreement
from viagent.redrock_parse import parse_redrock_details, parse_redrock_fits


def main():
    truth = load_truth()
    rr_best = parse_redrock_fits("results/redrock_all.fits")
    rr_class = parse_redrock_details("results/redrock_all.h5")

    rows = []
    for _, tr in truth.iterrows():
        b = rr_best.get(tr["targetid"], {})
        c = rr_class.get(tr["targetid"], {})
        # per-class best: pick min chi2 across GALAXY/QSO
        best_class = min(c, key=lambda k: c[k]["chi2"]) if c else None
        z_ok, dv = z_agreement(b.get("z"), tr["VI_Z"])
        rows.append({
            "type": tr["type"], "index": tr["index"], "targetid": tr["targetid"],
            "truth_spectype": tr["VI_SPECTYPE"],
            "rr_spectype": b.get("spectype"),
            "type_ok": classify_match(b.get("spectype"), tr["VI_SPECTYPE"]),
            "truth_z": tr["VI_Z"], "rr_z": b.get("z"),
            "dz": abs(b.get("z", 0) - tr["VI_Z"]), "dv_km_s": dv, "z_ok": z_ok,
            "rr_best_class": best_class,
            "rr_class_z": c.get(best_class, {}).get("z") if best_class else None,
            "rr_class_chi2": c.get(best_class, {}).get("chi2") if best_class else None,
        })

    import pandas as pd
    det = pd.DataFrame(rows)
    print("=== REDROCK BASELINE vs EXPERT VI ===")
    print(f"n={len(det)}")
    print(f"type_accuracy: {det['type_ok'].mean():.3f}")
    print(f"z_accuracy(<600 km/s): {det['z_ok'].mean():.3f}")
    print(f"median |dz|: {det['dz'].median():.5f}")
    print(f"median |dv|: {det['dv_km_s'].median():.1f} km/s")
    for t in ("BGS", "ELG", "LRG", "QSO"):
        sub = det[det["type"] == t]
        if len(sub):
            print(f"  {t}: n={len(sub)} type_acc={sub['type_ok'].mean():.3f} "
                  f"z_acc={sub['z_ok'].mean():.3f} med_dz={sub['dz'].median():.5f}")
    # z failures
    print("\n-- z failures (>600 km/s) --")
    for _, r in det[~det["z_ok"]].iterrows():
        print(f"  {r['type']}/{r['index']} truth_z={r['truth_z']:.4f} rr_z={r['rr_z']:.4f} dv={r['dv_km_s']:.0f}")
    # type failures
    print("\n-- type failures --")
    for _, r in det[~det["type_ok"]].iterrows():
        print(f"  {r['type']}/{r['index']} truth={r['truth_spectype']} rr={r['rr_spectype']}")
    det.to_csv("results/redrock_baseline.csv", index=False)


if __name__ == "__main__":
    main()
