"""Extract deterministic features for every training spectrum and cache them."""
import glob
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features  # noqa: E402

BASE = "/data2/wbc/slo/eva_set"
OUT = "/data2/wbc/slo/l1_test/eva_work/features_cache.pkl"


def main():
    files = sorted(glob.glob(os.path.join(BASE, "*", "[0-9]*.fits")))
    cache = {}
    t0 = time.time()
    for i, p in enumerate(files):
        key = p
        t = time.time()
        try:
            feats = extract_features(p)
            cache[key] = feats
            print(f"[{i + 1}/{len(files)}] {p} ({time.time() - t:.1f}s) "
                  f"cands={[c['z'] for c in feats['candidates'][:3]]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(files)}] FAILED {p}: {e}", flush=True)
    with open(OUT, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
