#!/usr/bin/env bash
# Evaluate the l2_test VI agent on the held-out evaluation set (eva_set).
#
# Adaptation notes (no project code modified):
#   input  : --train-dir /data2/wbc/slo/eva_set
#            (load_truth reads {type}_tr_set.csv; eva_set contains symlinks
#             {type}_tr_set.csv -> {type}_eva_set.csv)
#   redrock: fresh rrdesi run over all eva spectra -> $OUT/redrock_all.{fits,h5}
#   output : /data2/wbc/slo/l2_test/eva_results/
#   KB and LLM credentials resolve relative to the project (kb/, .env),
#   independent of cwd.
#
# Final configuration per REPORT.md: 3 independent runs + ensemble
# (majority vote on type, cluster consensus on z, median on quality).

set -euo pipefail

L2=/data2/wbc/slo/l2_test
EVA=/data2/wbc/slo/eva_set
OUT="$L2/eva_results"
N_RUNS=3
WORKERS=8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate langchain
export RR_TEMPLATE_DIR=/data2/wbc/redrock/py/redrock/templates

mkdir -p "$OUT"

# 1. redrock on all eva spectra (~1068 files, takes a while; skipped if products exist)
if [[ ! -s "$OUT/redrock_all.fits" || ! -s "$OUT/redrock_all.h5" ]]; then
    rrdesi -i "$EVA"/*/[0-9]*.fits -o "$OUT/redrock_all.fits" -d "$OUT/redrock_all.h5" \
           --nminima 5 --ncpu 16
fi

# 2. agent runs (resume-safe: re-running this script continues partial runs)
cd "$L2"
RUNS=()
for i in $(seq 1 "$N_RUNS"); do
    rd="$OUT/run$i"
    RUNS+=("$rd")
    python run_eval.py --train-dir "$EVA" \
        --redrock-fits "$OUT/redrock_all.fits" --redrock-h5 "$OUT/redrock_all.h5" \
        --out "$rd" --resume "$rd/verdicts.json" --workers "$WORKERS" \
        2>&1 | tee "$OUT/run$i.log"
done

# 3. ensemble
python run_ensemble.py --runs "${RUNS[@]}" --train-dir "$EVA" --out "$OUT/ensemble" \
    2>&1 | tee "$OUT/ensemble.log"

echo "done. results in $OUT (run*/verdicts.json, ensemble/{verdicts,metrics}.json, ensemble/detail.csv)"
