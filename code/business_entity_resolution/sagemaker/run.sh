#!/usr/bin/env bash
# Full train + predict on a large CPU instance, detached: keeps running if you close the browser.
# Usage (from the repo root):  bash code/business_entity_resolution/sagemaker/run.sh
#   env overrides: SAMPLE=1.0 (share of S1 entities used to fit)  NEG=0.5 (share of easy negatives)  TAG=sm_full
set -euo pipefail
cd "$(dirname "$0")/../../.."
SAMPLE=${SAMPLE:-1.0}; NEG=${NEG:-0.5}; TAG=${TAG:-sm_full}
WORKERS=$(( $(nproc) - 1 ))
mkdir -p "work_$TAG"
nohup .venv/bin/python -W ignore code/business_entity_resolution/src/pipeline.py run \
  --data-dir dataset --work-dir "work_$TAG" --out-dir "output_$TAG" \
  --sample "$SAMPLE" --neg-keep "$NEG" --workers "$WORKERS" > "work_$TAG/run.log" 2>&1 < /dev/null &
echo "started (pid $!). Follow progress with:  tail -f work_$TAG/run.log"
echo "When it prints 'self-check PASS', your submission is output_$TAG/matching_results.tsv"
