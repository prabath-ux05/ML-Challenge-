# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Mesh Minds
**Team Members:** Prabath D, Syed Amrullah, R Sai Padmini, Vamsi Tarun A (Vel Tech)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
Our pipeline scales to the full data: 2.2M Source 1 records against 10.3M Source 2/3 records on one laptop. Its stages are:
1. Transliteration-aware normalisation.
2. Hash-join blocking on 18 multi-valued keys.
3. **Cluster expansion**, which recovers matches blocking missed by using an entity's confident matches as extra queries.
4. A two-stage gradient-boosted matcher. Stage 2 adds context features: competition between entities for a record, and coherence among an entity's matches.
5. A decision layer that directly maximises expected per-entity F0.5 under a one-to-one constraint.

The approach is entirely data-driven. It uses no external data, APIs or pretrained models.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:**
  - train: 2,206,821 Source 1 records, 10,320,219 Source 2/3 records, 7,638,365 true links
  - test: 1,732,544 Source 1 records, 9,969,589 Source 2/3 records, including 259k French entities with no French training data
- **Link structure:**
  - Each Source 1 entity has 0–11 matches (mean about 3.5); 5.6% have none.
  - No Source 2/3 record is linked to more than one Source 1 entity, which justifies a one-to-one constraint.
- **Name noise:**
  - legal-suffix variants (Pvt/Private, Ltd/Limited, LLC/L.L.C.)
  - word reordering
  - typos, and OCR confusions like `PRlVATE` and `Fa1con`
  - DBA/"trading as" prefixes
  - web handles and domains (`gulfbrc.com`, `@casillasconcord`)
  - honorifics (`M/s`, `Mr`, `Sri`)
  - names **fully in native Indic scripts**: about 1M train records across Devanagari, Kannada, Tamil, Bengali, Gujarati and Odia
  - pseudonymous trade names with no overlap with the legal name (`Nylakelo`)
- **Address noise:**
  - abbreviations (St/Street, Rd/Road, Rly, Crsg)
  - components reordered, dropped or duplicated
  - landmarks (`Near SBI ATM`, `Opp.`)
  - municipal number formats (`509-511`, `#10433`, `C00139`)
  - transliterated state names (`पश्चिमबंग` → `pshcimbng`)
  - legacy city names (Bombay, Calcutta, Gurgaon)
  - about 3% of target records have **no address at all**
- **Ambiguity:**
  - The name vocabulary is small, so many different entities share a name ("Gold Food", "Green Properties", chain stores).
  - The name alone is often not identifying, and the address is decisive.

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Blocking, a pairwise GBDT classifier, graph-style cluster expansion and a set-level decision.

**Core Innovations:**
1. **Cross-rank rare-token blocking.** Every record emits all combinations of its rarest name and address tokens and house numbers, so a typo that changes which token is "rarest" still links.
2. **Cluster expansion** from confident matches, which lifts candidate recall.
3. **Context and coherence features** in stage 2.
4. **Expected-F0.5 subset selection** per entity, which treats the empty prediction as a first-class option for singletons.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:**
  - Unicode→ASCII transliteration, with Indic clean-up (collapse doubled letters, map transliterated legal words back to pvt/ltd).
  - Canonical abbreviation tables for names and addresses, including French forms (SARL/SAS, Bd, R., Av).
  - US and Indian state codes, including transliterated variants; legacy→current city names.
  - Landmark clauses stripped from the core address; house number and postal code parsing; a phonetic key.
- **Blocking keys used** (all within country; the country label is an open set, so France forms its own blocks):
  - exact core name, phonetic name and **consonant skeleton** (transliteration-robust: `bright consultancy` ≈ `brait knsaltensi` → `brt kns`), with and without the house number
  - name-prefix × address token
  - the 2 rarest name tokens × the 3 rarest address tokens
  - rare tokens × house numbers
  - rare address-token pairs and rare name-token pairs
  - name-only keys reserved for targets with no address

  Each key value is capped by block size, and per entity the 30 candidates with the highest specificity-weighted key evidence are kept (Σ 1/log2(1+|block|)).
- **Cluster expansion.**
  - For each confident stage-1 match (i, u), targets sharing a small block (≤30 records) with u are collected.
  - A target t is kept when it is a near-duplicate of u: name or address token-set similarity ≥ 85.
  - (i, t) is added with sibling-evidence features, up to 10 per entity.
- **Candidate pairs generated:** train 62.2M from blocking + 9.8M from expansion = 72.0M (about 32.6 per Source 1 entity, versus 2.2M × 10.3M ≈ 2.3×10^13 possible pairs, a reduction of more than 99.9999%); test (final submission, union of the v3.1 and v5 candidate sets): **68.25M pairs, 39.4 per Source 1 entity**
- **How we ensured true matches were not lost:**
  - We measured blocking recall and the oracle macro-F0.5 on train at every iteration.
  - Recall went from 90.6% (single keys) to 95.7% (cross-rank keys), then 96.2% (skeleton keys and state transliteration), then 96.3% (zero-padded numbers), then **97.6% with cluster expansion** (oracle F0.5 0.9898).
  - Each miss category (native script, no address, shortened address) was analysed and targeted.

---

## 4. Matching Model

**Features used** (about 118):
- Name features:
  - rapidfuzz ratio, partial ratio, token-sort, token-set, Jaro-Winkler and normalised Levenshtein on the core name
  - phonetic-key ratios; no-space ratios (web handles)
  - best DBA-variant similarity; exact core, full and phonetic equality
  - token Jaccard and containment; **IDF-weighted** overlap
  - max IDF of shared and unshared tokens; acronym match; legal-form conflict
  - native-script flag; name frequency (chain indicator)
- Address features:
  - fuzzy ratios on the core (landmark-free) and full address; token Jaccard and containment
  - IDF-weighted overlap
  - postal code equal, missing or prefix match; house-number equality, containment and number-set Jaccard
  - landmark and empty-address flags
- Other:
  - which blocking keys fired, and their specificity score
  - rank and gap of the pair within its entity's candidates and within the target's competing entities
- Stage-2 context, computed from out-of-fold stage-1 probabilities:
  - rank, max and sum per entity and per target
  - margin to the best competing entity or record
  - **coherence**: similarity to the entity's top-3 confident matches, weighted by probability
  - sibling evidence from expansion

**Model type:**
- **LightGBM** (MIT licence; about 1,000–1,300 trees of 127 leaves per stage-1 fold model), in 2 stages × 4 folds. scikit-learn's `HistGradientBoostingClassifier` (BSD-3) is only a fallback when LightGBM's OpenMP runtime is missing.
- Folds are grouped by Source 1 entity, and every training pair gets an out-of-fold score.
- Models are fit on 30% of entities, keeping every positive, every hard negative (name or address similarity ≥ 80) and 25% of the easy negatives, re-weighted. Test scores average the fold models.
- An entity-level LightGBM "has any match" model, fit out-of-fold on each entity's score profile, feeds the probability of the empty prediction.

**Threshold selection method:**
- Grid search on held-out entities (never used for fitting) with the exact macro-F0.5 metric, including singletons.
- Candidates: a global threshold, and per-entity **expected-F0.5 maximisation**, which picks the prefix of sorted candidates maximising E[F0.5] and predicts empty when P(no match) is higher.
- Both are tried with and without the one-to-one constraint, where a record goes only to its best entity.

---

## 5. Results & Error Analysis

All F0.5 values are out-of-fold macro-F0.5 on held-out Source 1 entities, singletons included.

| run | candidate recall | oracle F0.5 | stage-1 F0.5 | stage-2 F0.5 | public LB |
|---|---|---|---|---|---|
| baseline (single-rank keys, sklearn HGB, 10% sample) | 95.7% | 0.9838 | 0.9604 | 0.9629 | — |
| v2 (cross-rank + skeleton keys, state transliteration, IDF and coherence features, 25% sample) | 96.2% | 0.9861 | 0.9677 | 0.9706 | 0.952 |
| v3.1 (see below) | 97.6% | 0.9898 | 0.9752 | 0.9778 | 0.967 |
| v4: trained with synthetic decoy groups + France fix (augmented train*) | 97.5% | 0.9896 | 0.9702 | 0.9739 | 0.968 |
| v5: v4 + specific blocking keys + number-agreement/source context (augmented train*) | **97.8%** | **0.9908** | 0.9707 | **0.9766** | **0.971** |
| average of v3.1 and v5 (decision re-tuned on held-out entities) | — | — | — | 0.9781 (held-out) | 0.9706 |
| **final: v3.1 + v5 average, France decision ×0.70** | — | — | — | — | **0.971928** |

\* v4/v5 are scored on the decoy-augmented train (1.1M extra hard negatives), a harder set than v3.1's.

v3.1 adds:
- cluster expansion (recall 96.3% → 97.6%)
- the zero-padded number fix and number-distance features
- consonant-skeleton similarity
- French region/department normalisation
- LightGBM on 30% of entities with hard-negative sampling
- the singleton model and segment-specific thresholds

**v4 and v5 (trained for the test decoy pattern):**
- **v4** trains on `dataset_aug`: the original training data plus **1.10M synthetic decoy records in 571k groups** (`src/augment_decoys.py`). Each group copies 1–3 of an entity's own Source 2/3 records to a house number shifted by 1–15, optionally adding or swapping one word from a vocabulary mined from train's own decoys ("Lakeside", "Partners", "Exports", "India"), to 35% of US and 15% of Indian entities (including singletons). v4 also keeps `France` in names. Out-of-fold macro-F0.5 on the *augmented* train: 0.9739 (US 0.9809, India 0.9625, singletons 0.9749), with 1.1M extra hard negatives present.
- **v5** adds three specific blocking keys (rare address-token pair × house number, rare name-token pair × house number, name-token pair × address token). In train, 46% of India's blocking misses shared a key whose block exceeded the caps; the new keys recover about 15k missed matches. v5 also adds stage-2 context features: **house-number agreement** with the entity's confident matches (real copies share the majority number, decoy groups a minority one) and **same-source / other-source context** (80% of entities have copies in both Source 2 and Source 3). It keeps up to 36 candidates per entity and fits on a different 30% entity sample from v4.

**Per slice:**

| slice | baseline | v2 | v3.1 |
|---|---|---|---|
| US | 0.9769 | 0.9803 | **0.9850** |
| India | 0.9415 | 0.9555 | **0.9667** |
| singletons | 0.9527 | 0.9593 | **0.9779** |

**Loss attribution for v2** (F0.5 points lost, out of 2.96 in total):
- blocking misses: 1.28
- model misses: 0.76
- false merges: 0.32
- singletons wrongly matched: 0.23
- mixed: 0.38

This breakdown drove the v3.1 priorities. Expansion targets blocking misses; the singleton model and segment thresholds target singletons.

**Leaderboard vs validation: the train→test shift.**
- v2 scored 0.952 on the public leaderboard against 0.9706 validation, so we measured how test differs from train, using only model-free statistics of the candidate pairs.
- Test has 5.5–5.8 Source 2/3 records per Source 1 entity in every country, against a uniform 4.68 in train. The *clean* copies are unchanged: exact name+address duplicates per entity are 0.581 (US) and 0.240 (India) in both splits, and same-number near-duplicates are also unchanged. The extra records are therefore decoys, not extra true copies.
- **Decoys in test come in groups.** Several records share one slightly shifted house number next to the real business (`… Banco Lakeside Inc | 4889 Bentbrook Dr`, `… Banco Co | 4889 …` beside the real `4882`). In train, 79% of shifted-number look-alikes stood alone, and a *group* of look-alikes at a shifted number meant true copies (US P(true) 0.68 for 2 records, 0.975 for 3+), for example when the Source 1 record itself carried a truncated number. In test, such groups are about 6× more frequent per US entity. So the model learned a signal that test turns against it.
- Re-weighting train entities to the test mix (by near-duplicate, decoy and candidate counts) left v2's validation unchanged (0.967), which confirms that the shift is inside the entity, not in the entity mix.
- **France:** decoys often append a bare word `France` to the real name. Only 7% of such look-alikes share the entity's house number, against 82% of the others. Our normaliser had treated `France` as a noise token and deleted it, which hid the decoy's changed word; v3.2 keeps it.
- The orphan-record hypothesis was tested (by dropping Source 1 entities from train) and rejected.

**Test-time prior correction (`src/recalibrate.py`), evaluated and rejected.**
- Each candidate pair is assigned a segment from its entity's look-alikes (name ≥ 70, address ≥ 85): country × {same house number | shifted number with a group of 1, 2 or 3+ records} × {the entity also has a same-number look-alike}.
- Within every segment the model is well calibrated on train (mean OOF p = true rate to within 0.03), so a label-shift correction applies: each pair's odds are multiplied by odds(π_test)/odds(π_train) for its segment.
- π_test is estimated in two independent ways: by counting (true pairs per entity in each segment as in train, divided by test pairs per entity) and by EM on the model's own test scores (Saerens et al., 2002). For example, US "2 records at a shifted number, entity also has a same-number copy": 0.06 → 0.67 pairs per entity, P(true) 0.968 in train and an estimated 0.087 (count) / 0.057 (EM) in test.
- **Outcome: not used.** On test, the model's *mean* probability in that segment was already 0.107 (0.934 in train): other features give the test decoys away, so the label-shift premise (unchanged P(x|y)) did not hold and the full correction would have erased true matches. We addressed the shift by *training* on synthetic decoy groups instead (v4/v5).
- The corrected probabilities then go through the same tuned expected-F0.5 decision and singleton model.

**Common false positives (wrong merges):**
- near-copies of a business with a slightly shifted house number (`3376` vs `3380`) or one swapped name word (`Jain Services` vs `Jain Infra`)
- native-script names at the same address as a different English-named business
- address-less records with common names assigned to the wrong same-name entity

**Common false negatives (missed matches):**
- native-script names with shortened addresses
- address-less records with typo'd names
- pseudonymous trade names whose address was also altered
- about a third of model misses are records lost to a competing entity under the one-to-one rule, mostly address-less records with common names, which are inherently ambiguous

---

## 6. Conclusion
Our final submission averages two LightGBM pipelines: v3.1, trained on the original data, and v5, trained on data augmented with synthetic decoy groups and using specific combined blocking keys and house-number-agreement context. Public leaderboard: 0.952 (v2) → 0.967 (v3.1) → 0.968 (v4) → 0.971 (v5, v3.1 + v5) → **0.971928** (France decision adjusted).

What made the difference:
- **Measuring the train→test shift instead of guessing.** Model-free statistics showed that clean copies are unchanged in test while decoys come in groups at a shifted house number, about 6× more often per entity. That explained why validation (0.978) overstated the leaderboard, and led to the synthetic decoy groups.
- **A label-free test for decoy words** (how often a name change keeps the real house number) exposed a French normalisation bug (`France` stripped as noise although decoys add it) and showed which French words mark decoys.
- **Recall work driven by miss analysis:** 46% of India's blocking misses shared a key whose block was too large, so we added more specific combined keys (+0.26 points recall, +0.5 India F0.5).
- **A country-specific decision for the unlabelled country.** France inherits US/India thresholds; scaling French pair probabilities by 0.70 before the expected-F0.5 decision (`src/country_adjust.py`, US/India rows unchanged) removed 8,107 low-certainty French links and raised the leaderboard from 0.970584 to 0.971928, i.e. about +0.9 F0.5 points on France.
- **A calibrated simulation engine:** held-out out-of-fold F0.5 anchored to one known leaderboard score predicted v5's leaderboard score exactly (0.971) and let us rank blends without spending submissions.

Limitations and next steps:
- Remaining errors are spread across genuinely ambiguous cases: address-less records with common names (63k train matches never become candidates), Indian-script names without a Latin counterpart, and French chains. France has no labels, and its estimated score is the weakest.
- With more time: group-level (cluster) matching of all Source 2/3 records before assignment, embedding- or character-n-gram nearest-neighbour blocking for the address-less and cross-script cases, and fitting on 100% of entities.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/normalize.py`: text normalisation
- `src/prep.py`: parallel, cached normalisation
- `src/blocking.py`: key blocking
- `src/expand.py`: cluster expansion
- `src/features.py`: pair, group and context features
- `src/pipeline.py`: entry point for training, OOF, decision tuning and prediction
- `src/error_analysis.py`: error analysis
- `src/evaluate.py`: macro-F0.5 scorer

Reproduce the outputs with one command:
```
python code/business_entity_resolution/src/pipeline.py run --data-dir dataset --work-dir work --out-dir output
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. See the README for runtimes and flags.

### B. Additional Results
**Simulation of final-submission options** (held-out out-of-fold macro-F0.5 on 300k entities never used for fitting; "test-like" = with the synthetic decoys, "original" = without them; the simulated leaderboard is anchored on v4's real 0.968):

| option | test-like | original | simulated LB | actual LB |
|---|---|---|---|---|
| v4 | 0.9735 | 0.9754 | 0.968 (anchor) | 0.968 |
| v5 | 0.9766 | 0.9778 | ≈ 0.971 | **0.971** |
| v4 + v5 | 0.9767 | 0.9780 | ≈ 0.971 | — |
| v4 + 2×v5 | 0.9770 | 0.9782 | ≈ 0.9715 | — |
| **v3.1 + v5** | 0.9781 | 0.9794 | ≈ 0.9725 | **0.9706** |
| v3.1 + v4 + v5 | 0.9777 | 0.9789 | ≈ 0.972 | — |

Options containing v3.1 are optimistic in the test-like world, because v3.1 never scored the synthetic decoys on train; the leaderboard confirmed this (0.9706 vs the simulated ≈ 0.9725).

**Reproducing the final submission:**
```
# v3.1: original data, code at commit 2e476ba
git checkout 2e476ba -- code/ && python code/business_entity_resolution/src/pipeline.py run --data-dir dataset --work-dir work --out-dir output_v31
git checkout main -- code/
# v5: synthetic decoy groups + current code
python code/business_entity_resolution/src/augment_decoys.py --data-dir dataset --out-dir dataset_aug
python code/business_entity_resolution/src/pipeline.py run --data-dir dataset_aug --work-dir work_v5 --out-dir output_v5 --folds 4 --block-cfg '{"max_per_s1":36}' --sample-seed 11
# average, decision re-tuned on held-out entities, write output/
python code/business_entity_resolution/src/blend_oof.py eval --works work work_v4 work_v5 --sample-seeds 7 7 11 --spec 'work+work_v5'
python code/business_entity_resolution/src/blend_oof.py apply --works work work_v4 work_v5 --candidate 'work+work_v5' --out-dir output_final
# France decision x0.70 (US/India rows unchanged) -> writes output_fr_0.70/, copied to output/
python code/business_entity_resolution/src/country_adjust.py --works work work_v4 work_v5 --candidate 'work+work_v5' --country france --factors 0.7 --out-prefix output_fr
```
(`work_v4` is only needed as a placeholder in `--works`; its scores are not used by the chosen candidate.)
