#!/usr/bin/env bash
# Build <team>_submission.zip from the repo root. Usage: bash code/business_entity_resolution/make_submission.sh TEAM
set -euo pipefail
TEAM=${1:?team name}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
rm -f "${TEAM}_submission.zip"
zip -r "${TEAM}_submission.zip" output/matching_results.tsv output/candidate_pairs.tsv \
  code/business_entity_resolution/src code/business_entity_resolution/tests \
  code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt \
  Documentation_template.md -x '*/__pycache__/*'
echo "built ${TEAM}_submission.zip"
