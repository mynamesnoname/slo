#!/usr/bin/env python
"""Ensemble evaluation: combine verdicts from multiple runs (majority vote on
type, cluster-consensus on z, median on quality) and evaluate vs expert VI.

This reduces per-run LLM variance and gives the agent's stable opinion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viagent.evaluate import (
    evaluate_verdicts,
    load_truth,
)


def cluster_z(zs, tol=0.005):
    """Cluster z values; return (winning cluster mean, n in cluster, all clusters)."""
    if not zs:
        return None, 0, []
    zs = sorted(zs)
    clusters = []
    cur = [zs[0]]
    for z in zs[1:]:
        if z - cur[-1] <= tol:
            cur.append(z)
        else:
            clusters.append(cur)
            cur = [z]
    clusters.append(cur)
    best = max(clusters, key=len)
    return float(np.mean(best)), len(best), clusters


def ensemble_verdict(per_target_dicts, tol=0.005):
    """Combine per-run verdict dicts into one ensemble verdict dict."""
    out = {}
    for tid, runs in per_target_dicts.items():
        if not runs:
            continue
        # type: majority
        types = [r.get("spectype") for r in runs]
        counts = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        best_type = max(counts, key=counts.get)
        # z: cluster z's of runs agreeing with the majority type; if none, all
        zs_major = [r.get("z") for r in runs if r.get("spectype") == best_type and r.get("z")]
        zs_all = [r.get("z") for r in runs if r.get("z")]
        zs = zs_major if zs_major else zs_all
        z_mean, n_cluster, clusters = cluster_z(zs, tol)
        z = z_mean if z_mean is not None else 0.0
        # quality: median of runs of the majority type (fallback: all)
        qs = [r.get("quality") for r in runs if r.get("spectype") == best_type]
        if not qs:
            qs = [r.get("quality") for r in runs]
        q = float(np.median(qs)) if qs else 0.0
        # evidence: take from the run with most secure lines among majority type
        best_run = max(
            [r for r in runs if r.get("spectype") == best_type] or runs,
            key=lambda r: r.get("n_secure_lines", 0),
        )
        out[tid] = {
            "spectype": best_type,
            "z": z,
            "z_err": best_run.get("z_err"),
            "quality": q,
            "n_secure_lines": best_run.get("n_secure_lines"),
            "n_runs": len(runs),
            "n_type_votes": counts[best_type],
            "z_cluster_n": n_cluster,
            "evidence": best_run.get("evidence", []),
            "reasoning": best_run.get("reasoning", ""),
            "fatal_issues": best_run.get("fatal_issues", []),
            "runs": runs,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="result dirs (default: all results/run* except baselines)")
    ap.add_argument("--out", default="results/ensemble")
    ap.add_argument("--train-dir", default="training_set")
    ap.add_argument("--min-runs", type=int, default=2)
    args = ap.parse_args()

    if args.runs:
        run_dirs = args.runs
    else:
        run_dirs = sorted(
            d for d in os.listdir("results")
            if d.startswith("run") and os.path.exists(f"results/{d}/verdicts.json")
        )
        run_dirs = [f"results/{d}" for d in run_dirs]

    truth = load_truth(args.train_dir)
    per_target = {}
    for rd in run_dirs:
        vp = os.path.join(rd, "verdicts.json")
        if not os.path.exists(vp):
            continue
        with open(vp) as f:
            v = json.load(f)
        for tid, d in v.items():
            per_target.setdefault(str(tid), []).append(d)
        print(f"{rd}: {len(v)} verdicts")

    # keep only targets with enough runs, keyed by int targetid
    ensemble = {}
    for tid, runs in per_target.items():
        if len(runs) >= args.min_runs:
            ensemble[int(tid)] = runs

    verdicts = ensemble_verdict(ensemble)
    print(f"\nensemble over {len(run_dirs)} runs, {len(verdicts)} spectra "
          f"(>= {args.min_runs} runs each)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "verdicts.json"), "w") as f:
        json.dump(verdicts, f, indent=2, ensure_ascii=False, default=str)

    metrics, det = evaluate_verdicts(verdicts, truth)
    print("\n=== ENSEMBLE METRICS ===")
    print(json.dumps(metrics, indent=2))
    det.to_csv(os.path.join(args.out, "detail.csv"), index=False)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
