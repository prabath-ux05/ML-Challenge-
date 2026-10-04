# Business Entity Resolution: Dataset Profiling Report

## Executive Summary

This profiling report details the dataset characteristics across training sources (Source 1, Source 2, Source 3) and training ground truth match pairs, establishing foundational baseline statistics for subsequent phases.

---

## 1. Dataset Dimensions & Identifier Integrity

| Source | Total Rows | Unique Entity IDs | Duplicate IDs | ID Duplication Rate |
| :--- | :---: | :---: | :---: | :---: |
| Source 1 | 2,206,821 | 2,206,821 | 0 | 0.0000% |
| Source 2 | 5,034,616 | 5,034,616 | 0 | 0.0000% |
| Source 3 | 5,285,603 | 5,285,603 | 0 | 0.0000% |

## 2. Missing Value Analysis

| Source | entity_id Missing (%) | business_name Missing (%) | business_address Missing (%) | country Missing (%) |
| :--- | :---: | :---: | :---: | :---: |
| Source 1 | 0.00% | 0.00% | 0.00% | 0.00% |
| Source 2 | 0.00% | 0.00% | 3.36% | 0.00% |
| Source 3 | 0.00% | 0.00% | 3.33% | 0.00% |

## 3. Lexical Attribute Uniqueness

| Source | Unique Names | Name Uniqueness (%) | Unique Addresses | Address Uniqueness (%) | Unique (Name, Addr) Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Source 1 | 1,539,229 | 69.75% | 2,130,606 | 96.55% | 2,206,821 |
| Source 2 | 4,402,008 | 87.43% | 4,337,261 | 86.15% | 4,839,960 |
| Source 3 | 4,651,609 | 88.01% | 4,632,764 | 87.65% | 5,091,046 |

## 4. Geographic Distribution (Top Countries)

- **Source 1**: US: 1,323,633 (60.0%), India: 883,188 (40.0%)
- **Source 2**: US: 3,016,817 (59.9%), India: 2,017,799 (40.1%)
- **Source 3**: US: 3,170,056 (60.0%), India: 2,115,547 (40.0%)

## 5. Training Ground Truth Match Distribution

- **Total S1 Reference Entities**: 2,206,821
- **Total Matched Pairs**: 7,638,365
- **Zero-Match S1 Entities**: 123,247 (5.58%)
- **Single-Match S1 Entities**: 119,157 (5.40%)
- **Multi-Match S1 Entities (>=2)**: 1,964,417 (89.02%)

### Matches per Source 1 Entity Summary Statistics

- **Min / Max**: 0 / 11
- **Mean**: 3.46
- **Median**: 3.0
- **Std Dev**: 1.71
- **Percentiles (P25 / P75 / P95 / P99)**: 2 / 5 / 6 / 8

### Match Frequency Histogram Buckets

| Matches per S1 | Count | Percentage |
| :--- | :---: | :---: |
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6-10 | 252,255 | 11.43% |
| 11-20 | 37 | 0.00% |
| >20 | 0 | 0.00% |

## 6. S2 / S3 Match Source Breakdown

- **Source 2 Matches**: 3,693,619 (48.36% of all matches)
- **Source 3 Matches**: 3,944,746 (51.64% of all matches)
- **Unique Source 2 Entities Matched**: 3,693,619
- **Unique Source 3 Entities Matched**: 3,944,746

### Cross-Source Matching Behavior per S1 Entity

- **Matching Both S2 and S3**: 1,776,047 (80.48%)
- **Matching S2 Only**: 143,029 (6.48%)
- **Matching S3 Only**: 164,498 (7.45%)
- **Matching Neither**: 123,247 (5.58%)

## 7. Exact Lexical Overlap Across Datasets (Pre-Normalization)

| Source Pair | Exact Name Overlap | Overlap % (A) | Overlap % (B) | Exact Addr Overlap | Overlap % (A) | Overlap % (B) | Exact (Name, Addr) Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Source 1 vs Source 2 | 168,382 | 10.94% | 3.83% | 4 | 0.00% | 0.00% | 1 |
| Source 1 vs Source 3 | 176,687 | 11.48% | 3.80% | 170,701 | 8.01% | 3.68% | 17 |
| Source 2 vs Source 3 | 371,391 | 8.44% | 7.98% | 0 | 0.00% | 0.00% | 0 |

## 8. Ground-Truth Match Exactness (Sampled True Pairs)

Evaluated across 25,000 true match pairs:

| Subset | Exact Name Match (%) | Exact Address Match (%) | Exact Both Match (%) |
| :--- | :---: | :---: | :---: |
| **Overall True Pairs** | 4.58% | 2.42% | 0.00% |
| **S1-S2 True Pairs** (12,013) | 4.59% | 0.00% | 0.00% |
| **S1-S3 True Pairs** (12,987) | 4.58% | 4.65% | 0.00% |

> **Key Takeaway**: The low exact match rates on names and addresses between true entity pairs demonstrate the critical necessity for phonetic/token blocking, string similarity metrics, and machine learning entity resolution models in subsequent phases.
