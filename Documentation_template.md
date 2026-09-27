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
- **Candidate pairs generated:** train 62.2M from blocking + 9.8M from expansion = 72.0M (about 32.6 per Source 1 entity, versus 2.2M × 10.3M ≈ 2.3×10^13 possible pairs, a reduction of more than 99.9999%); test: TODO
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
| v3.1 (see below) | **97.6%** | **0.9898** | **0.9752** | **0.9778** | **0.967** |

v3.1 adds:
- cluster expansion (recall 96.3% → 97.6%)
- the zero-padded number fix and number-distance features
- consonant-skeleton similarity
- French region/department normalisation
- LightGBM on 30% of entities with hard-negative sampling
- the singleton model and segment-specific thresholds

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

**Test-time prior correction (`src/recalibrate.py`).**
- Each candidate pair is assigned a segment from its entity's look-alikes (name ≥ 70, address ≥ 85): country × {same house number | shifted number with a group of 1, 2 or 3+ records} × {the entity also has a same-number look-alike}.
- Within every segment the model is well calibrated on train (mean OOF p = true rate to within 0.03), so a label-shift correction applies: each pair's odds are multiplied by odds(π_test)/odds(π_train) for its segment.
- π_test is estimated in two independent ways: by counting (true pairs per entity in each segment as in train, divided by test pairs per entity) and by EM on the model's own test scores (Saerens et al., 2002). For example, US "2 records at a shifted number, entity also has a same-number copy": 0.06 → 0.67 pairs per entity, P(true) 0.968 in train and an estimated LB_TODO_RECAL in test.
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
TODO

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
TODO
