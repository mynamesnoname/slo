"""Evaluate agent VI results against expert VI ground truth.

Metrics:
  * type accuracy  (QSO vs GALAXY vs expert VI_SPECTYPE)
  * redshift accuracy: |dz|/(1+z) < tol  (tol default 0.003)
  * "VI correct" = type correct AND redshift correct
  * quality: MAE vs expert VI_QUALITY, Spearman rho, fraction |diff|<=1,
    agreement on "secure" (>=3) vs "not secure" (<3)
  * per-type breakdown
"""
import json
import os
import sys

import numpy as np

WORK = "/data2/wbc/slo/l1_test/eva_work"
RESULTS = os.path.join(WORK, "vi_results.json")


def load():
    with open(RESULTS) as fh:
        data = json.load(fh)
    return data


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float) + 1
    rx, ry = rank(x), rank(y)
    return np.corrcoef(rx, ry)[0, 1]


def evaluate(data, dz_tol=0.003, min_quality=0, dz_tol2=0.005):
    results = data["results"]
    gt = data["ground_truth"]
    redrock = data.get("redrock", {})
    rows = []
    for path, res in results.items():
        g = gt[path]
        ttype = g["target_type"]
        z_vi = g["vi_z"]
        type_vi = g["vi_spectype"]
        q_vi = g["vi_quality"]
        z_ag = res.get("vi_z")
        type_ag = res.get("vi_spectype", "?")
        q_ag = res.get("vi_quality", -1)
        if z_ag is None or (isinstance(z_ag, float) and np.isnan(z_ag)):
            z_ag = float("nan")
        z_ok = (not np.isnan(z_ag)) and abs(z_ag - z_vi) / (1 + z_vi) < dz_tol
        z_ok2 = (not np.isnan(z_ag)) and abs(z_ag - z_vi) / (1 + z_vi) < dz_tol2
        type_ok = (type_ag.upper() == type_vi.upper())
        vi_ok = z_ok and type_ok
        vi_ok2 = z_ok2 and type_ok
        rows.append({
            "index": g["index"], "target_type": ttype, "path": path,
            "gt_z": z_vi, "gt_type": type_vi, "gt_q": q_vi,
            "ag_z": z_ag, "ag_type": type_ag, "ag_q": q_ag,
            "z_ok": z_ok, "type_ok": type_ok, "vi_ok": vi_ok,
            "vi_ok2": vi_ok2,
            "dz_frac": abs(z_ag - z_vi) / (1 + z_vi) if not np.isnan(z_ag) else np.nan,
            "dz": abs(z_ag - z_vi) if not np.isnan(z_ag) else np.nan,
            "flags": res.get("flags", []),
            "reasoning": res.get("reasoning", "")[:200],
            "evidence": res.get("evidence", []),
            "fallback": res.get("fallback", False),
            "quality_capped": res.get("quality_capped", False),
        })
    n = len(rows)
    z_ok_frac = np.mean([r["z_ok"] for r in rows])
    t_ok_frac = np.mean([r["type_ok"] for r in rows])
    vi_ok_frac = np.mean([r["vi_ok"] for r in rows])
    vi_ok2_frac = np.mean([r["vi_ok2"] for r in rows])
    # quality-aware: for spectra the expert graded >=3, require z&type correct;
    # for spectra the expert graded <3, require us to be conservative (q<3)
    qa_correct = []
    for r in rows:
        if r["gt_q"] >= 3:
            qa_correct.append(r["vi_ok"])
        else:
            qa_correct.append(r["ag_q"] < 3)
    qa_frac = np.mean(qa_correct)
    n_fallback = sum(1 for r in rows if r.get("fallback"))
    n_capped = sum(1 for r in rows if r.get("quality_capped"))
    dzs = np.array([r["dz_frac"] for r in rows if not np.isnan(r["dz_frac"])])
    med_dz = np.median(dzs) if len(dzs) else np.nan
    q_gt = np.array([r["gt_q"] for r in rows])
    q_ag = np.array([max(0, min(4, r["ag_q"])) for r in rows if r["ag_q"] >= 0])
    q_valid = [r for r in rows if r["ag_q"] >= 0]
    q_mae = np.mean(np.abs(np.array([r["gt_q"] for r in q_valid]) -
                           np.array([r["ag_q"] for r in q_valid])))
    q_rho = spearman(np.array([r["gt_q"] for r in q_valid]),
                     np.array([r["ag_q"] for r in q_valid]))
    q_agree1 = np.mean([abs(r["gt_q"] - r["ag_q"]) <= 1 for r in q_valid])
    # secure agreement
    q_sec_gt = np.array([r["gt_q"] >= 3 for r in q_valid])
    q_sec_ag = np.array([r["ag_q"] >= 3 for r in q_valid])
    q_sec_acc = np.mean(q_sec_gt == q_sec_ag)
    # redrock comparison
    rr_z_ok = []
    rr_t_ok = []
    rr_vi_ok = []
    for r in rows:
        tid = None
        rr = None
        for p, res in results.items():
            if res.get("targetid") == tid:
                break
        # redrock by targetid: need mapping; do in caller
        rr_z_ok.append(np.nan)
    return {
        "n": n,
        "z_ok_frac": z_ok_frac,
        "type_ok_frac": t_ok_frac,
        "vi_ok_frac": vi_ok_frac,
        "vi_ok2_frac": vi_ok2_frac,
        "qa_frac": qa_frac,
        "n_fallback": n_fallback,
        "n_capped": n_capped,
        "med_dz_frac": med_dz,
        "q_mae": q_mae,
        "q_rho": q_rho,
        "q_agree_within1": q_agree1,
        "q_secure_agree": q_sec_acc,
        "rows": rows,
    }


def subset_metrics(rows, gt_q_min=3.0):
    """Metrics on the science-useful subset (expert quality >= min)."""
    sub = [r for r in rows if r["gt_q"] >= gt_q_min]
    if not sub:
        return {}
    return {
        "n": len(sub),
        "z_ok": np.mean([r["z_ok"] for r in sub]),
        "type_ok": np.mean([r["type_ok"] for r in sub]),
        "vi_ok": np.mean([r["vi_ok"] for r in sub]),
        "vi_ok2": np.mean([r["vi_ok2"] for r in sub]),
        "q_within1": np.mean([abs(r["gt_q"] - r["ag_q"]) <= 1 for r in sub]),
        "q_mae": np.mean([abs(r["gt_q"] - r["ag_q"]) for r in sub if r["ag_q"] >= 0]),
    }


def redrock_metrics(data, dz_tol=0.003):
    """Redrock pipeline metrics vs expert VI (for comparison)."""
    results = data["results"]
    gt = data["ground_truth"]
    redrock = data.get("redrock", {})
    n = zok = tok = viok = 0
    for path, res in results.items():
        g = gt[path]
        tid = res.get("targetid")
        rr = redrock.get(str(tid)) if tid is not None else None
        if rr is None:
            continue
        n += 1
        z_ok = abs(rr["Z"] - g["vi_z"]) / (1 + g["vi_z"]) < dz_tol
        t_ok = (rr["SPECTYPE"] == "QSO") == (g["vi_spectype"] == "QSO")
        zok += z_ok; tok += t_ok; viok += (z_ok and t_ok)
    return {"n": n, "z_ok": zok / n, "type_ok": tok / n, "vi_ok": viok / n}


def per_type(rows):
    out = {}
    for t in ["BGS", "ELG", "LRG", "QSO"]:
        sub = [r for r in rows if r["target_type"] == t]
        if not sub:
            continue
        out[t] = {
            "n": len(sub),
            "z_ok": np.mean([r["z_ok"] for r in sub]),
            "type_ok": np.mean([r["type_ok"] for r in sub]),
            "vi_ok": np.mean([r["vi_ok"] for r in sub]),
            "vi_ok2": np.mean([r["vi_ok2"] for r in sub]),
            "q_mae": np.mean([abs(r["gt_q"] - r["ag_q"]) for r in sub if r["ag_q"] >= 0]),
        }
    return out


def main():
    data = load()
    ev = evaluate(data)
    rr = redrock_metrics(data)
    print("=" * 70)
    print(f"n={ev['n']}")
    print(f"redshift accuracy (|dz|/(1+z)<0.003): {ev['z_ok_frac']:.2%}")
    print(f"type accuracy:                          {ev['type_ok_frac']:.2%}")
    print(f"VI correct (z AND type):                {ev['vi_ok_frac']:.2%}")
    print(f"VI correct (dz tol 0.005):              {ev['vi_ok2_frac']:.2%}")
    print(f"quality-aware VI correct:                {ev['qa_frac']:.2%}")
    print(f"fallbacks: {ev['n_fallback']}  quality-capped: {ev['n_capped']}")
    print(f"median |dz|/(1+z):                      {ev['med_dz_frac']:.5f}")
    print(f"quality MAE:                            {ev['q_mae']:.2f}")
    print(f"quality Spearman rho:                   {ev['q_rho']:.2f}")
    print(f"quality within +-1:                     {ev['q_agree_within1']:.2%}")
    print(f"quality secure(>=3) agreement:          {ev['q_secure_agree']:.2%}")
    if rr and rr["n"]:
        print("-" * 70)
        print("redrock pipeline baseline (vs same ground truth):")
        print(f"  z_ok={rr['z_ok']:.2%} type_ok={rr['type_ok']:.2%} "
              f"vi_ok={rr['vi_ok']:.2%} (n={rr['n']})")
    sm = subset_metrics(ev["rows"], 3.0)
    if sm:
        print("-" * 70)
        print(f"science-useful subset (expert q>=3, n={sm['n']}):")
        print(f"  z_ok={sm['z_ok']:.2%} type_ok={sm['type_ok']:.2%} "
              f"vi_ok={sm['vi_ok']:.2%} vi_ok2={sm['vi_ok2']:.2%} "
              f"q_within1={sm['q_within1']:.2%} q_mae={sm['q_mae']:.2f}")
    print("-" * 70)
    print("per-type:")
    for t, m in per_type(ev["rows"]).items():
        print(f"  {t:<4} n={m['n']:>2} z_ok={m['z_ok']:.2%} type_ok={m['type_ok']:.2%} "
              f"vi_ok={m['vi_ok']:.2%} vi_ok2={m['vi_ok2']:.2%} q_mae={m['q_mae']:.2f}")
    print("-" * 70)
    print("worst cases (by |dz|/(1+z)):")
    worst = sorted(ev["rows"], key=lambda r: -(r["dz_frac"] if not np.isnan(r["dz_frac"]) else 99))
    for r in worst[:12]:
        print(f"  {r['target_type']}/{r['index']:>3}: gt z={r['gt_z']:.4f} type={r['gt_type']:<6} "
              f"q={r['gt_q']} | ag z={r['ag_z']:.4f} type={r['ag_type']:<6} q={r['ag_q']} "
              f"z_ok={r['z_ok']} type_ok={r['type_ok']} flags={r['flags']}")
        print(f"      {r['reasoning'][:160]}")


if __name__ == "__main__":
    main()
