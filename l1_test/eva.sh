#!/usr/bin/env bash
# Evaluate the VI agent (l1_test/agent) on the held-out evaluation set.
#
# The agent's path variables are configured to:
#   input  : /data2/wbc/slo/eva_set/{BGS,ELG,LRG,QSO}/
#   output : /data2/wbc/slo/l1_test/eva_work/
# Ground truth: run_agent.py reads {type}_tr_set.csv; step 0 below links that
# name to the actual {type}_eva_set.csv so no project code is touched.

set -euo pipefail

L1=/data2/wbc/slo/l1_test
EVA=/data2/wbc/slo/eva_set
AGENT="$L1/agent"
WORK="$L1/eva_work"

# 0. environment + ground-truth CSV links
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate langchain
export RR_TEMPLATE_DIR=/data2/wbc/redrock/py/redrock/templates

mkdir -p "$WORK/redrock_bulk"
for t in BGS ELG LRG QSO; do
    ln -sf "${t}_eva_set.csv" "$EVA/$t/${t}_tr_set.csv"
done

# 1. bulk redrock on all eva spectra (~1068 files, takes a while)
rrdesi -i "$EVA"/*/[0-9]*.fits -o "$WORK/redrock_bulk/rr_all.fits" --ncpu 8

# 2. extract deterministic features (cached in $WORK/features_cache.pkl)
cd "$AGENT"
python extract_all.py

# 3. run the LLM agent (reads API credentials from l1_test/.env)
python run_agent.py

# 4. evaluate against expert VI + export per-spectrum CSV
python evaluate.py
python export_results.py

echo "done. results in $WORK (vi_results.json, vi_results_table.csv, audit/)"
