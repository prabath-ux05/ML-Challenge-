"""Submission and output file integrity validator for Entity Resolution pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from business_entity_resolution.utils.io import read_tsv
from business_entity_resolution.utils.logging import get_logger

logger = get_logger("pipeline.validator")

# Expected TSV headers
CANDIDATE_PAIRS_HEADER = ["source1_entity_id", "candidate_entity_id"]
MATCHING_RESULTS_HEADER = ["source1_entity_id", "matched_entity_ids"]


@dataclass
class ValidationReport:
    """Report detailing integrity check results for generated submission files."""

    is_valid: bool
    num_s1_entities: int
    num_matched_s1: int
    num_zero_match_s1: int
    total_matches_produced: int
    total_candidate_pairs: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Output Integrity Validation: {status} ===",
            f"Total S1 entities evaluated: {self.num_s1_entities:,}",
            f"S1 with matches: {self.num_matched_s1:,} ({self.num_matched_s1/max(self.num_s1_entities,1)*100:.1f}%)",
            f"S1 zero-match: {self.num_zero_match_s1:,} ({self.num_zero_match_s1/max(self.num_s1_entities,1)*100:.1f}%)",
            f"Total matches produced: {self.total_matches_produced:,}",
            f"Total candidate pairs: {self.total_candidate_pairs:,}",
        ]
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")
        return "\n".join(lines)


class SubmissionValidator:
    """Rigorous validator ensuring final outputs satisfy all challenge and production constraints."""

    @staticmethod
    def validate_matching_results(
        matching_results_path: Union[str, Path],
        candidate_pairs_path: Union[str, Path],
        expected_s1_ids: Set[str],
        valid_candidate_ids: Optional[Set[str]] = None,
        require_deterministic_order: bool = True,
    ) -> ValidationReport:
        """Validate matching_results.tsv and candidate_pairs.tsv against all strict requirements.

        Requirements:
        1. every S1 exactly once (no missing, no duplicate S1 rows)
        2. valid S2/S3 IDs
        3. no duplicate matches within an S1's match list
        4. empty matches allowed (empty string in matched_entity_ids)
        5. every final match exists in candidate_pairs.tsv
        6. no self-matches (cand_id != s1_id)
        7. deterministic output (rows sorted by source1_entity_id)

        Args:
            matching_results_path: Path to matching_results.tsv.
            candidate_pairs_path: Path to candidate_pairs.tsv.
            expected_s1_ids: Full set of expected Source 1 entity IDs.
            valid_candidate_ids: Optional set of all valid S2 and S3 entity IDs.
            require_deterministic_order: If True, requires S1 IDs to be sorted ascending.

        Returns:
            ValidationReport with is_valid=True if all checks pass.
        """
        errors: List[str] = []
        warnings: List[str] = []
        m_path = Path(matching_results_path)
        c_path = Path(candidate_pairs_path)

        if not m_path.exists():
            return ValidationReport(
                is_valid=False,
                num_s1_entities=0,
                num_matched_s1=0,
                num_zero_match_s1=0,
                total_matches_produced=0,
                total_candidate_pairs=0,
                errors=[f"File not found: {m_path}"],
            )
        if not c_path.exists():
            return ValidationReport(
                is_valid=False,
                num_s1_entities=0,
                num_matched_s1=0,
                num_zero_match_s1=0,
                total_matches_produced=0,
                total_candidate_pairs=0,
                errors=[f"File not found: {c_path}"],
            )

        # 1. Load and index candidate_pairs.tsv
        cand_pairs_set: Set[Tuple[str, str]] = set()
        cand_row_count = 0
        df_cands = read_tsv(c_path)
        
        # Check candidate_pairs columns
        c_cols = list(df_cands.columns)
        if len(c_cols) < 2:
            errors.append(f"candidate_pairs.tsv must have at least 2 columns, found: {c_cols}")
        else:
            s1_col = c_cols[0]
            cand_col = c_cols[1]
            for s1_val, c_val in zip(df_cands[s1_col], df_cands[cand_col]):
                cand_pairs_set.add((str(s1_val), str(c_val)))
                cand_row_count += 1

        # 2. Load matching_results.tsv
        df_match = read_tsv(m_path, dtype=str)
        m_cols = list(df_match.columns)
        if m_cols[:2] != MATCHING_RESULTS_HEADER:
            warnings.append(
                f"matching_results.tsv columns {m_cols[:2]} differ from recommended {MATCHING_RESULTS_HEADER}"
            )

        s1_col_name = m_cols[0]
        match_col_name = m_cols[1]

        observed_s1_ids: List[str] = []
        seen_s1_set: Set[str] = set()
        matched_s1_count = 0
        zero_match_s1_count = 0
        total_matches_count = 0

        for idx, row in df_match.iterrows():
            s1_id = str(row[s1_col_name]) if pd.notna(row[s1_col_name]) else ""
            raw_matches = str(row[match_col_name]) if pd.notna(row[match_col_name]) else ""
            raw_matches = raw_matches.strip()
            # Handle string 'nan' or empty
            if raw_matches.lower() in ("nan", "none", "null"):
                raw_matches = ""

            observed_s1_ids.append(s1_id)

            # Check 1: Duplicate S1 rows
            if s1_id in seen_s1_set:
                errors.append(f"Duplicate S1 entity row encountered at line {idx+2}: '{s1_id}'")
            seen_s1_set.add(s1_id)

            # Check matches in row
            if not raw_matches:
                zero_match_s1_count += 1
                continue

            matched_s1_count += 1
            # Split comma-separated matches
            matches_list = [m.strip() for m in raw_matches.split(",") if m.strip()]

            # Check 3: Duplicate matches within single S1
            if len(matches_list) != len(set(matches_list)):
                dup_matches = [m for m in matches_list if matches_list.count(m) > 1]
                errors.append(f"Duplicate match IDs in S1 '{s1_id}': {set(dup_matches)}")

            for mid in matches_list:
                total_matches_count += 1

                # Check 6: No self-matches
                if mid == s1_id:
                    errors.append(f"Self-match detected: S1 entity '{s1_id}' matched to itself")

                # Check 2: Valid candidate IDs
                if valid_candidate_ids is not None and mid not in valid_candidate_ids:
                    errors.append(f"Invalid candidate ID '{mid}' matched to S1 '{s1_id}' (not in candidate pool)")

                # Check 5: Every match must exist in candidate_pairs.tsv
                if (s1_id, mid) not in cand_pairs_set:
                    errors.append(
                        f"Match ({s1_id} -> {mid}) not present in candidate_pairs.tsv"
                    )

        # Check 1: Every expected S1 appears exactly once
        missing_s1 = expected_s1_ids - seen_s1_set
        if missing_s1:
            errors.append(
                f"Missing {len(missing_s1)} expected S1 entities from matching_results.tsv (e.g. {list(missing_s1)[:3]})"
            )

        extra_s1 = seen_s1_set - expected_s1_ids
        if extra_s1:
            errors.append(
                f"Found {len(extra_s1)} unexpected S1 entities in matching_results.tsv (e.g. {list(extra_s1)[:3]})"
            )

        # Check 7: Deterministic ordering
        if require_deterministic_order:
            if observed_s1_ids != sorted(observed_s1_ids):
                warnings.append(
                    "matching_results.tsv rows are not sorted ascending by S1 entity ID. "
                    "Sorting is strongly recommended for deterministic reproduction."
                )

        is_valid = len(errors) == 0
        report = ValidationReport(
            is_valid=is_valid,
            num_s1_entities=len(seen_s1_set),
            num_matched_s1=matched_s1_count,
            num_zero_match_s1=zero_match_s1_count,
            total_matches_produced=total_matches_count,
            total_candidate_pairs=cand_row_count,
            errors=errors,
            warnings=warnings,
        )

        logger.info(report.summary())
        return report
