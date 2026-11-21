#!/usr/bin/env python
"""Run the VI agent over training spectra and evaluate against expert VI.

Usage:
  python run_eval.py --limit 5 --out results/run1
  python run_eval.py --out results/run2 --resume results/run1/verdicts.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viagent.agent import VIAgent
from viagent.evaluate import (
    evaluate_verdicts,
    load_truth,
    save_verdicts,
)
from viagent.kb import KnowledgeBase
from viagent.llm import LLMClient
from viagent.redrock_parse import (
    parse_redrock_details,
    parse_redrock_fits,
    parse_redrock_zscan_minima,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/run1")
    ap.add_argument("--resume", default=None, help="json with already-computed verdicts")
    ap.add_argument("--train-dir", default="training_set")
    ap.add_argument("--redrock-fits", default="results/redrock_all.fits")
    ap.add_argument("--redrock-h5", default="results/redrock_all.h5")
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    truth = load_truth(args.train_dir)
    if args.types:
        truth = truth[truth["type"].isin(args.types)]
    if args.limit:
        truth = truth.sample(n=args.limit, random_state=args.seed)

    rr_best = parse_redrock_fits(args.redrock_fits)
    rr_class = parse_redrock_details(args.redrock_h5)
    rr_zscan = parse_redrock_zscan_minima(args.redrock_h5, nmin=2)

    # resume
    verdicts = {}
    if args.resume and os.path.exists(args.resume):
        from viagent.evaluate import load_verdicts
        verdicts = load_verdicts(args.resume)
        print(f"resumed {len(verdicts)} verdicts from {args.resume}")

    kb = KnowledgeBase()
    llm = LLMClient()
    agent = VIAgent(kb=kb, llm=llm)

    todo = truth[~truth["targetid"].isin(verdicts.keys())]
    print(f"total={len(truth)}  already={len(verdicts)}  todo={len(todo)}", flush=True)

    t0 = time.time()

    def work(row):
        fn = row["file"]
        tid = row["targetid"]
        rrp = rr_class.get(tid)
        rrb = rr_best.get(tid)
        zsc = rr_zscan.get(tid)
        return tid, agent.analyze(fn, redrock_per_class=rrp, redrock_best=rrb,
                                  zscan_candidates=zsc)

    import concurrent.futures

    nworkers = getattr(args, "workers", 6)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as ex:
        futures = {ex.submit(work, tr): tr for _, tr in todo.iterrows()}
        for fut in concurrent.futures.as_completed(futures):
            try:
                tid, v = fut.result()
                verdicts[tid] = v
            except Exception as e:
                tr = futures[fut]
                print(f"ERROR {tr['targetid']} {tr['file']}: {e}", flush=True)
            done += 1
            if done % 5 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] {el:.0f}s elapsed, {len(verdicts)} done", flush=True)
                save_verdicts(verdicts, os.path.join(args.out, "verdicts.json"))

    save_verdicts(verdicts, os.path.join(args.out, "verdicts.json"))

    metrics, det = evaluate_verdicts(verdicts, truth)
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    det.to_csv(os.path.join(args.out, "detail.csv"), index=False)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
