# Business Entity Resolution at Scale: Amazon ML Challenge 2026 (team Mesh Minds)

Matching business records across three noisy sources: for every Source 1 entity, find all of its records in Source 2 and Source 3. The data has 2.2M train / 1.7M test reference entities against about 10M records each, in US, Indian and (test-only) French data. The metric is macro F0.5 per entity, singletons included.

**Team:** Prabath D, Syed Amrullah, R Sai Padmini, Vamsi Tarun A (Vel Tech)

## Results (public leaderboard)

| submission | score |
|---|---|
| v2: cross-rank and skeleton blocking, sklearn GBDT | 0.952 |
| v3.1: cluster expansion, LightGBM, expected-F0.5 decisions | 0.967 |
| v4: trained with synthetic decoy groups, France normalisation fix | 0.968 |
| v5: v4 + specific combined blocking keys, house-number agreement context | 0.971 |
| v3.1 + v5 average, France decision ×0.52, French links require v5's agreement | **0.972287** |

## Approach
1. **Normalisation:** transliteration-aware Unicode→ASCII (Indic scripts), abbreviation, legal-form and state/region canonicalisation, house-number parsing, phonetic and consonant-skeleton keys.
2. **Blocking:** per-country polars hash joins on 21 multi-valued keys (rare-token combinations, names × house numbers, address-token pairs × numbers), capped by block size, top 36 per entity.
3. **Cluster expansion:** confident matches act as extra queries to recover records that blocking missed.
4. **Two-stage LightGBM matcher:** about 110 pair features, then context features (competition between entities, coherence with an entity's other matches, house-number agreement, Source 2/3 balance).
5. **Decision:** per-entity expected-F0.5 subset selection with a one-to-one constraint and a "has any match" model for singletons, tuned on held-out entities.

**Key findings** (details in [`Documentation_template.md`](Documentation_template.md)):
- **Train→test shift:** test decoys come in *groups* of 2–3 records at a slightly shifted house number, about 6× more often than in train, while clean copies are unchanged. Training on synthetic decoy groups (`augment_decoys.py`) fixed the behaviour.
- **A label-free test for decoy words:** the share of look-alikes that keep the entity's house number (true copies about 97%, decoys about 5%). It exposed a French normalisation bug and French decoy vocabulary.
- **A held-out simulation engine** (`blend_oof.py`), anchored on one known leaderboard score, predicted v5's leaderboard score exactly and ranked blends without spending submissions.

## Repository layout
```
code/business_entity_resolution/
  src/            pipeline (normalise → block → features → 2-stage LightGBM → decision) + analysis tools
  sagemaker/      scripts for a large CPU instance
  tests/          synthetic-data smoke test
  README.md       detailed run instructions, runtimes, reproduction of the final submission
Documentation_template.md   methodology write-up (challenge template)
```

## Quick start
The challenge data is **not included** (competition terms). Place it in `dataset/train` and `dataset/test`.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt   # macOS: brew install libomp for LightGBM
python code/business_entity_resolution/src/pipeline.py run --data-dir dataset --work-dir work --out-dir output
```
A smoke test on synthetic data, with no challenge data needed:
```bash
python code/business_entity_resolution/tests/make_synthetic_data.py --out /tmp/synth --n 4000
python code/business_entity_resolution/src/pipeline.py run --data-dir /tmp/synth --work-dir /tmp/w --out-dir /tmp/o --sample 0.5
```
See [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) for the full pipeline, runtimes (about 3 h per full run on a 12-core / 24 GB laptop) and the exact steps behind the final submission.

## Compliance
No external data, APIs, geocoders or pretrained models. The models are LightGBM (MIT), trained from scratch on the provided data.
