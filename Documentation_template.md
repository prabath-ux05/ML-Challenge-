# Methodology — Business Entity Resolution

> Draft. Copy these sections into the official `Documentation_template.md` headings and replace every `TODO` with numbers from `models/cv_report.json` and the training log from the real data.

## 1. Methodology overview
We treat the task as **pairwise classification over a blocked candidate set, followed by a set-level decision that maximises macro-F0.5**:

1. **Normalise** every record, whether Source 1, 2 or 3, into canonical name and address representations.
2. **Block**: union 7 cheap retrieval passes, then cap each Source 1 entity's list to about 60 candidates.
3. **Stage-1 matcher**: a gradient-boosted classifier over 61 pair features.
4. **Stage-2 matcher**: the same model family, with context features added from stage-1 out-of-fold probabilities. These include competition between Source 1 entities for a target, the margin to the runner-up, and coherence among a Source 1 entity's matches.
5. **Decision**: a one-to-one constraint (each Source 2 or 3 record belongs to at most one Source 1 entity), then a threshold or expected-F0.5 subset selection. Whichever does better on out-of-fold macro-F0.5 is used.

Validation is 5-fold **GroupKFold by Source 1 entity**, so no entity's pairs leak across folds. Every decision parameter is tuned on out-of-fold predictions with the exact competition metric, singletons included.

## 2. Candidate generation / blocking
TF-IDF spaces are fitted on all records of a split, using no labels:
- character 2–4-grams of the core name
- character 2–3-grams of the phonetic core name
- word tokens of the core name
- character 2–4-grams of the core address
- word tokens of the address
- word tokens of name plus address

| pass | description | k |
|---|---|---|
| name char TF-IDF | top-k per Source 1 entity, within country | 25 |
| phonetic name TF-IDF | handles transliteration and typos | 15 |
| address char TF-IDF | catches renamed or DBA businesses at the same address | 10 |
| name+address word TF-IDF | | 25 |
| reverse passes | each Source 2/3 record's top Source 1 entities (name, and name+address) | 3 + 3 |
| global name pass | ignores country, so noisy or open-set country labels are safe | 5 |
| postal code + first name token | exact key block | — |

The union is capped at 60 per Source 1 entity using a cheap blended cosine. Reverse-best pairs are always kept. Country is used only to partition the search. An unseen label such as France gets searched within its own partition, and a label with no counterpart in the target sources falls back to searching everything. Nothing is hard-coded to {US, India}.

- Blocking recall ceiling on train: **TODO** (synthetic: 0.9997)
- Oracle macro-F0.5 given the candidates: **TODO**
- Average candidates per Source 1 entity: **TODO** (reduction ratio **TODO**)

## 3. Feature engineering (61 pair + 13 context features)
- **Normalisation**:
  - Unicode→ASCII.
  - Canonical abbreviations: Corp/Corporation, Pvt/Private, Ltd/Limited, St/Street, Rd/Road, Bd/Boulevard, R./Rue, and others.
  - `&`→and; US and Indian state names mapped to their codes; renamed Indian cities (Bombay→Mumbai, and others).
  - Legal forms stripped to make a *core name*: LLC, Inc, Pvt Ltd, SARL, SAS, and others.
  - DBA, "trading as" and bracketed alias splitting.
  - Landmark clauses separated out ("Near SBI ATM", "Opp.").
  - 5- and 6-digit postal codes parsed, including "560 001"; "12B" split into "12 B".
  - A phonetic key for transliteration: sh/s, ee/i, w/v, doubled letters.
- **Name features**:
  - TF-IDF cosines.
  - Levenshtein, ratio, partial ratio, token sort and token set.
  - Jaro-Winkler.
  - Phonetic ratios.
  - Ratio with spaces removed.
  - Best match over DBA variants.
  - Token Jaccard and containment.
  - First-token match.
  - Acronym match.
  - Legal-form conflict.
  - Lengths.
- **Address features**:
  - Cosines and fuzzy ratios on the address without landmarks.
  - Postal code equal, postal 3-digit prefix equal, postal missing.
  - House number equal, number-set Jaccard, disjoint numbers.
  - Landmark present.
  - Containment.
- **Chain and generic-name handling**: how often the core name occurs in each pool. A frequent name like a chain store forces the model to rely on the address.
- **Candidate-set context**: rank and gap to the best cosine within the Source 1 entity's candidates and within the target's competing Source 1 entities.
- **Stage-2 context from stage-1 out-of-fold probabilities**:
  - rank within the Source 1 entity and within the target
  - max and sum
  - count above 0.5
  - margin to the best competitor on both sides
  - coherence: max similarity of a candidate to the Source 1 entity's top-3 confident matches, weighted by probability

The model sees no country identity, so it transfers to France. Country is used only for `same_country` and for the French abbreviation table.

## 4. Model architecture
scikit-learn `HistGradientBoostingClassifier` (BSD-3):
- learning rate 0.05, up to 800 iterations with early stopping
- 48 leaves, min leaf 30, L2 1.0

Five fold models are trained per stage, and test predictions average all five. There are no pretrained or external models, so the parameter count is far below 8B and the license is permissive.

## 5. Decision layer (precision-first)
- **One-to-one**: a Source 2 or 3 record is linked only to the Source 1 entity that scores highest for it. The ground truth has no record linked to two Source 1 entities, and the rule removes duplicate-branch false merges.
- **Threshold mode**: link when p ≥ t, with t tuned on out-of-fold macro-F0.5.
- **Expected-F0.5 mode**: per Source 1 entity, choose the prefix of sorted candidates that maximises expected F0.5. The empty set is also considered, scored at P(no match) = ∏(1−p). This handles singletons explicitly.
- The best mode and parameters on out-of-fold data are saved with the model.

## 6. Results
| setting | macro-F0.5 |
|---|---|
| stage-1 OOF (train) | TODO |
| stage-2 OOF (train) | TODO |
| singletons / matched entities | TODO / TODO |
| per country: US / India | TODO / TODO |
| public leaderboard | TODO |

## 7. Other notes
- **Fair play**: no external data, APIs, geocoding or internet lookups. TF-IDF is fitted on the unlabeled test records only, which is standard transductive preprocessing.
- **Reproducibility**: fixed seeds, pinned `requirements.txt`, and a single command (`pipeline.py run`).
- **Runtime**: about 90 s end to end for 4.6k Source 1 records and 9.4k targets on a laptop CPU. Runtime grows linearly with the number of candidates.
