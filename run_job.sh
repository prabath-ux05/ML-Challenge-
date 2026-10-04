#!/bin/bash
set -e

echo "=== Starting SageMaker Full Test Inference Job ==="
echo "Working Directory: $(pwd)"

cd /opt/ml/processing/input/code
tar -xzf source-*.tar.gz 2>/dev/null || tar -xzf *.tar.gz 2>/dev/null || true

echo "Installing dependencies..."
pip install --no-cache-dir lightgbm rapidfuzz jellyfish anyascii
pip install -e .

echo "Running Production Pipeline on Complete 1,732,544 S1 Test Queries..."
python3 -m business_entity_resolution.pipeline.entrypoint \
  --s1 /opt/ml/processing/input/source1/test_source1.tsv \
  --s2 /opt/ml/processing/input/source2/test_source2.tsv \
  --s3 /opt/ml/processing/input/source3/test_source3.tsv \
  --models-dir /opt/ml/processing/model \
  --output-dir /opt/ml/processing/output \
  --k 200 \
  --threshold-s2 0.20 \
  --threshold-s3 0.80

echo "Executing Submission Validator on Complete 1,732,544 Output..."
python3 -c "
from business_entity_resolution.pipeline.validator import SubmissionValidator
from business_entity_resolution.utils.io import read_tsv
df_s1 = read_tsv('/opt/ml/processing/input/source1/test_source1.tsv')
expected_s1 = set(df_s1['entity_id'].astype(str))
report = SubmissionValidator.validate_matching_results(
    '/opt/ml/processing/output/matching_results.tsv',
    '/opt/ml/processing/output/candidate_pairs.tsv',
    expected_s1_ids=expected_s1,
    require_deterministic_order=True,
)
print(report.summary())
if not report.is_valid:
    exit(1)
"

echo "=== SageMaker Full Test Inference Completed Successfully ==="
