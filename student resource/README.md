# Business Entity Resolution Challenge

A machine learning solution for resolving noisy, multi-source business records across independent data sources without shared identifiers.

## Project Structure

```text
src/business_entity_resolution/
├── data/           # Dataset loading, schemas, and profiling
├── normalization/  # Text, name, and address normalization
├── blocking/       # Candidate pair generation and indexing
├── features/       # Feature extraction for entity pairs
├── matching/       # ML models, training, and decision logic
├── evaluation/     # Metrics, validation splits, and format checks
└── utils/          # Config, logging, and I/O helpers

notebooks/          # Analysis and experimentation notebooks
tests/              # Test suite
```
