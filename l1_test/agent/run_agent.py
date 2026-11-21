"""Run the LLM VI agent over all training spectra (threaded), save results."""
import csv
import json
import os
import pickle
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from astropy.io import fits  # noqa: E402

from vi_agent import VIAgent  # noqa: E402

BASE = "/data2/wbc/slo/eva_set"
WORK = "/data2/wbc/slo/l1_test/eva_work"
CACHE = os.path.join(WORK, "features_cache.pkl")
RR_FITS = os.path.join(WORK, "redrock_bulk", "rr_all.fits")
OUT = os.path.join(WORK, "vi_results.json")
AUDIT_DIR = os.path.join(WORK, "audit")


def load_ground_truth():
    """Ground truth keyed by full spectrum path (indexes collide across types)."""
    import glob
    gt = {}
    for t in ["BGS", "ELG", "LRG", "QSO"]:
        csv_path = os.path.join(BASE, t, f"{t}_tr_set.csv")
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                idx = int(r["index"])
                path = os.path.join(BASE, t, f"{idx}.fits")
                gt[path] = {
                    "index": idx,
                    "target_type": t,
                    "vi_spectype": r["VI_SPECTYPE"],
                    "vi_z": float(r["VI_Z"]),
                    "vi_quality": float(r["VI_QUALITY"]),
                    "snr_sum": float(r["SNR_sum"]),
                }
    return gt


def load_redrock():
    """Map targetid -> redrock fit dict."""
    tid2fit = {}
    with fits.open(RR_FITS) as hdus:
        d = hdus["REDSHIFTS"].data
        for row in d:
            tid2fit[int(row["TARGETID"])] = {
                "Z": float(row["Z"]),
                "ZERR": float(row["ZERR"]),
                "ZWARN": int(row["ZWARN"]),
                "CHI2": float(row["CHI2"]),
                "SPECTYPE": str(row["SPECTYPE"]),
                "SUBTYPE": str(row["SUBTYPE"]),
                "DELTACHI2": float(row["DELTACHI2"]),
            }
    # map file -> targetid via features cache
    return tid2fit


def map_file_to_targetid(cache):
    mapping = {}
    for path, feats in cache.items():
        mapping[path] = feats["targetid"]
    return mapping


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(CACHE, "rb") as fh:
        cache = pickle.load(fh)
    tid2fit = load_redrock()
    gt = load_ground_truth()
    path2tid = map_file_to_targetid(cache)

    agent = VIAgent()
    results = {}
    errors = []
    lock = threading.Lock()
    sem = threading.Semaphore(8)

    def work(path):
        with sem:
            tid = path2tid[path]
            redrock = tid2fit.get(tid)
            feats = cache[path]
            ttype = os.path.basename(os.path.dirname(path))
            try:
                res = agent.inspect(feats, redrock, target_prior=ttype)
            except Exception as e:  # noqa: BLE001
                res = {"targetid": tid, "vi_z": float("nan"), "vi_spectype": "?",
                       "vi_quality": -1, "evidence": [], "flags": [],
                       "reasoning": f"EXCEPTION: {e}"}
            with lock:
                results[path] = res
                # write audit
                idx = os.path.basename(os.path.dirname(path))
                fname = os.path.basename(path).replace(".fits", "")
                with open(os.path.join(AUDIT_DIR, f"{idx}_{fname}.txt"), "w") as fh:
                    fh.write(res["_audit"]["report"] if "_audit" in res else "")
                    fh.write("\n\n===== LLM CHAIN OF THOUGHT =====\n")
                    fh.write((res["_audit"].get("last_reasoning", "")
                              if "_audit" in res else "") + "\n")
                    fh.write("\n===== LLM OUTPUT =====\n")
                    fh.write(json.dumps(
                        {k: v for k, v in res.items() if not k.startswith("_")},
                        indent=2, ensure_ascii=False))
                print(f"done {path}: z={res['vi_z']:.4f} type={res['vi_spectype']} "
                      f"q={res['vi_quality']} flags={res.get('flags')}", flush=True)

    paths = sorted(cache.keys())
    threads = [threading.Thread(target=work, args=(p,)) for p in paths]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"all done in {time.time() - t0:.0f}s")

    # merge with ground truth for downstream evaluation
    out = {"results": results, "ground_truth": gt, "redrock": tid2fit}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
