# Business Entity Resolution — reproducible pipeline

Data → normalisation → multi-pass blocking → 61 pair features → 2-stage gradient-boosted matcher → F0.5-optimal decision → `output/`.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Run end to end
From the `student_resource/` root, which contains `dataset/train`, `dataset/test` and `utils/`:
```bash
python code/business_entity_resolution/src/pipeline.py run \
    --data-dir dataset --model-dir models --out-dir output
```
This does the following:
1. **train**: builds candidates on the training split, reports blocking recall and the oracle ceiling, runs 5-fold GroupKFold (grouped by Source 1 entity) for stage 1 and stage 2, tunes the decision rule on out-of-fold macro-F0.5, and saves `models/model.joblib` and `models/cv_report.json`.
2. **predict**: builds candidates on the test split, scores them with the fold-averaged models, and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It then self-checks that every match is a candidate, that IDs are valid and that there are no duplicates.

You can also run `train` and `predict` separately. Useful flags:
- `--block-cfg '{"k_name":40,"max_per_s1":80}'` to widen blocking.
- `--dump-oof` to write out-of-fold pair scores for error analysis.
- `--dump-scores` to write test pair probabilities.

Validate, then package:
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
bash code/business_entity_resolution/make_submission.sh <team_name>
```

## Smoke test without the real data
```bash
python code/business_entity_resolution/tests/make_synthetic_data.py --out /tmp/synth --n 4000
python code/business_entity_resolution/src/pipeline.py run --data-dir /tmp/synth --model-dir /tmp/m --out-dir /tmp/o
python code/business_entity_resolution/src/evaluate.py --pred /tmp/o/matching_results.tsv \
    --truth /tmp/synth/test/test_ground_truth.tsv --source1 /tmp/synth/test/test_source1.tsv
```

## Layout
| file | role |
|---|---|
| `src/normalize.py` | Unicode→ASCII, abbreviation canonicalisation (names and addresses, including French), legal-suffix stripping, DBA splitting, landmark extraction, postal code and number parsing, city renames, transliteration-tolerant phonetic key |
| `src/blocking.py` | TF-IDF spaces (fitted transductively, without labels) and 7 blocking passes with a per-entity cap |
| `src/features.py` | Pair features (string similarity, TF-IDF cosines, postal and number agreement, name frequency, rank and gap within the candidate set) and stage-2 context features |
| `src/pipeline.py` | Training, CV, decision tuning, prediction, output writing, self-check |
| `src/evaluate.py` | Macro-F0.5 scorer |

Compliance: no external data, APIs, geocoders or pretrained models are used. Every model is a scikit-learn HistGradientBoostingClassifier (BSD-3), trained from scratch on the provided training data only.
