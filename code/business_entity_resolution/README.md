# Business Entity Resolution — reproducible pipeline

The pipeline runs these stages, from raw data to `output/`:
1. Normalisation (parallel, cached)
2. Key-based blocking
3. Pair and group features, computed in chunks
4. Stage-1 gradient-boosted model
5. Stage-2 gradient-boosted model, using context features built from stage-1 scores
6. F0.5-optimal decision rule

It's built for the full challenge size: 2.2M Source 1 records against 10M Source 2/3 records, on a 12-core, 24 GB laptop.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Run end to end
Run from the repository root, which contains `dataset/train`, `dataset/test` and `utils/`:
```bash
python code/business_entity_resolution/src/pipeline.py run \
    --data-dir dataset --work-dir work --out-dir output
```

### Training
1. Normalises all records into `work/norm/*.parquet` using parallel workers. This is cached.
2. Generates candidates with key blocking into `work/train_cand.parquet`. The log reports blocking recall and the oracle macro-F0.5 given those candidates.
3. Computes pair features for every candidate into `work/train_feat_v*/`.
4. Trains in 4 folds (GroupKFold by Source 1 entity). Models are fit on a sample of Source 1 entities: `--sample 0.25`, keeping all positives and `--neg-keep 0.35` of negatives, re-weighted. Each fold model then scores every candidate pair out-of-fold, for stage 1 and stage 2.
5. Tunes the decision rule on held-out entities, then saves `work/model.joblib`, `work/cv_report.json` and `work/train_oof.parquet`.

### Prediction
1. Runs the same blocking and features on the test split.
2. Averages the fold models.
3. Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
4. Self-checks that every match is also a candidate.

You can run `train` and `predict` separately. Cached stages are reused. Delete `work/` to start from scratch, or bump `FEATURE_VERSION` in `pipeline.py` after changing features. Changing `normalize.py` requires deleting `work/norm/`.

Approximate runtimes on an M-series laptop:

| step | time |
|---|---|
| normalisation | ~4 min |
| blocking (train) | ~6 min |
| features (train) | ~15–20 min |
| training | ~40–60 min |
| predict | ~25 min |

## Error analysis
```bash
python code/business_entity_resolution/src/error_analysis.py --work-dir work --n 30
```
Prints false merges and missed matches from the out-of-fold predictions, broken down by country and error type.

## Validate and package
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
bash code/business_entity_resolution/make_submission.sh <team_name>
```

## Smoke test without the real data
```bash
python code/business_entity_resolution/tests/make_synthetic_data.py --out /tmp/synth --n 4000
python code/business_entity_resolution/src/pipeline.py run --data-dir /tmp/synth --work-dir /tmp/w --out-dir /tmp/o --sample 0.5
python code/business_entity_resolution/src/evaluate.py --pred /tmp/o/matching_results.tsv \
    --truth /tmp/synth/test/test_ground_truth.tsv --source1 /tmp/synth/test/test_source1.tsv
```

## Layout
| file | role |
|---|---|
| `src/normalize.py` | Unicode→ASCII, including cleanup of Indic-script transliteration and transliterated state names. Also: abbreviation canonicalisation for names and addresses (including French), legal-suffix and honorific stripping, DBA splitting, web handles and domains, OCR digit fixes, landmark extraction, postal code and number parsing, a phonetic key. |
| `src/prep.py` | Parallel normalisation with a parquet cache |
| `src/blocking.py` | Per-country hash-join blocking on 18 multi-valued keys: exact, phonetic and consonant-skeleton names; rare name and address token combinations; house numbers. Block sizes are capped, and each entity keeps its top 30 candidates by key specificity. |
| `src/features.py` | Vectorised pair features (rapidfuzz `cpdist` string scorers, rarity-weighted token overlap, address and number agreement, name frequency, blocking evidence). Group rank/gap features, and stage-2 context features (competition margins, coherence among an entity's matches). |
| `src/pipeline.py` | Orchestration: CV training, OOF scoring, vectorised F0.5 metric and decision tuning, prediction, output writing |
| `src/error_analysis.py` | False merges and misses from the out-of-fold predictions |
| `src/evaluate.py` | Macro-F0.5 scorer for a `matching_results.tsv` |

## Compliance
No external data, APIs, geocoders or pretrained models are used. The only models are scikit-learn `HistGradientBoostingClassifier` (BSD-3), trained from scratch on the provided training data. TF-style token statistics are computed on each split's own records, without labels.
