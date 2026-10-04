"""Production pipeline runner implementing the deterministic frozen V4.1 entity resolution flow."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import psutil

from business_entity_resolution.blocking.candidate_generator import (
    CandidateGenerator,
    STRATEGY_COMBINED_HYBRID,
)
from business_entity_resolution.evaluation.metrics import (
    compute_entity_level_metrics,
    compute_f_beta,
)
from business_entity_resolution.features.pair_features import (
    ALL_FEATURE_NAMES,
    PairFeatureExtractor,
)
from business_entity_resolution.matching.model import EntityMatchClassifier
from business_entity_resolution.pipeline.validator import (
    CANDIDATE_PAIRS_HEADER,
    MATCHING_RESULTS_HEADER,
    SubmissionValidator,
    ValidationReport,
)
from business_entity_resolution.utils.config import S3DataConfig
from business_entity_resolution.utils.io import read_tsv, write_tsv
from business_entity_resolution.utils.logging import get_logger

logger = get_logger("pipeline.runner")


@dataclass
class PipelineConfig:
    """Configuration for production entity resolution pipeline execution."""

    # Input paths (if None, resolved automatically via environment / config)
    s1_path: Optional[Path] = None
    s2_path: Optional[Path] = None
    s3_path: Optional[Path] = None
    ground_truth_path: Optional[Path] = None

    # Output and model directories
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("SM_OUTPUT_DATA_DIR", os.getenv("OUTPUT_DIR", "output"))
        )
    )
    models_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("SM_MODEL_DIR", os.getenv("MODELS_DIR", "models"))
        )
    )

    # Production thresholds (frozen from Phase 8/9 tuning)
    k: int = 200
    threshold_s2: float = 0.20
    threshold_s3: float = 0.80

    # Batch processing and reproducibility
    batch_size: int = 1000
    random_seed: int = 42

    # Qwen semantic reranker flag (strictly False in primary pipeline)
    enable_qwen: bool = False
    qwen_ambiguity_window: Tuple[float, float] = (0.15, 0.40)

    # Optional query/candidate limits for dry-run testing
    limit_s1: Optional[int] = None
    limit_candidates: Optional[int] = None


class ProductionPipelineRunner:
    """Executes the end-to-end entity resolution workflow deterministically."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        np.random.seed(self.config.random_seed)

    def run(self) -> Dict[str, Any]:
        """Execute end-to-end pipeline:
        data -> V4.1 blocking -> 82 features -> LightGBM -> candidate_pairs.tsv -> matching_results.tsv -> validation.
        """
        process = psutil.Process()
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        m_dir = Path(self.config.models_dir)

        logger.info("================================================================")
        logger.info("Starting Frozen V4.1 Entity Resolution Production Pipeline")
        logger.info("Output Directory: %s", out_dir.resolve())
        logger.info("Models Directory: %s", m_dir.resolve())
        logger.info(
            "Candidate Budget: K = %d | Thresholds: S2=%.2f, S3=%.2f | Batch Size: %d",
            self.config.k,
            self.config.threshold_s2,
            self.config.threshold_s3,
            self.config.batch_size,
        )
        logger.info(
            "Qwen Status: %s",
            "ENABLED (EXPERIMENTAL)" if self.config.enable_qwen else "OFF (PRIMARY PRODUCTION)",
        )
        logger.info("Initial Memory: %.1f MB RSS", start_mem)
        logger.info("================================================================")

        # Step 1: Load inputs
        t0 = time.time()
        df_s1, df_s2, df_s3, df_gt = self._load_input_data()
        load_time = time.time() - t0
        logger.info(
            "Loaded data: S1=%d entities, S2=%d entities, S3=%d entities%s (%.2fs)",
            len(df_s1),
            len(df_s2),
            len(df_s3),
            f", GT={len(df_gt)} pairs" if df_gt is not None else "",
            load_time,
        )

        s1_records: List[Tuple[str, str, str, str]] = list(df_s1.itertuples(index=False, name=None))
        s2_records: List[Tuple[str, str, str, str]] = list(df_s2.itertuples(index=False, name=None))
        s3_records: List[Tuple[str, str, str, str]] = list(df_s3.itertuples(index=False, name=None))

        # Sort S1 queries for strict deterministic batch iteration
        s1_records.sort(key=lambda x: str(x[0]))
        all_expected_s1 = {str(r[0]) for r in s1_records}
        valid_candidates = {str(r[0]) for r in s2_records} | {str(r[0]) for r in s3_records}

        s2_dict: Dict[str, Tuple[str, str, str, str]] = {str(r[0]): r for r in s2_records}
        s3_dict: Dict[str, Tuple[str, str, str, str]] = {str(r[0]): r for r in s3_records}

        # Step 2: Fit V4.1 Candidate Generators
        t0 = time.time()
        logger.info("Fitting V4.1 candidate generators on S2 and S3 candidate pools...")
        gen_s2 = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=self.config.k)
        gen_s2.fit(s2_records)
        gen_s3 = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=self.config.k)
        gen_s3.fit(s3_records)
        fit_time = time.time() - t0
        logger.info("Candidate generators fitted in %.2fs", fit_time)

        # Step 3: Load LightGBM matchers & initialize extractors
        logger.info("Loading frozen production LightGBM models...")
        clf_s2 = EntityMatchClassifier.load(m_dir / "lgbm_matcher_s2")
        clf_s3 = EntityMatchClassifier.load(m_dir / "lgbm_matcher_s3")

        extractor_s2 = PairFeatureExtractor(ngram_idf=gen_s2.ngram_idf)
        extractor_s3 = PairFeatureExtractor(ngram_idf=gen_s3.ngram_idf)

        # Step 4: Batched Candidate Generation, Feature Extraction & Scoring
        cand_pairs_path = out_dir / "candidate_pairs.tsv"
        logger.info("Streaming candidate pairs & scoring in batches -> %s", cand_pairs_path)

        matches_per_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        all_generated_cand_pairs: Set[Tuple[str, str]] = set()

        total_s1 = len(s1_records)
        batch_size = max(1, self.config.batch_size)
        total_batches = (total_s1 + batch_size - 1) // batch_size

        total_block_time = 0.0
        total_feat_time = 0.0
        total_score_time = 0.0
        total_pairs_count = 0

        # Open candidate_pairs.tsv for writing
        with open(cand_pairs_path, "w", encoding="utf-8") as f_cand:
            f_cand.write("\t".join(CANDIDATE_PAIRS_HEADER) + "\n")

            for batch_idx in range(total_batches):
                b_start = batch_idx * batch_size
                b_end = min(total_s1, b_start + batch_size)
                batch_queries = s1_records[b_start:b_end]

                # 4a. Blocking & Candidate Generation
                t_b0 = time.time()
                batch_s2_pair_items = []
                batch_s3_pair_items = []
                batch_cand_lines = []

                for q in batch_queries:
                    s1_id, s1_name, s1_addr, s1_country = q
                    s1_id = str(s1_id)

                    # S2 candidates
                    cands_meta_s2 = gen_s2.generate_candidates_with_metadata(
                        s1_name, s1_addr, s1_country, k=self.config.k
                    )
                    for cm in cands_meta_s2:
                        cid = str(cm.get("candidate_id") or cm.get("entity_id"))
                        batch_cand_lines.append(f"{s1_id}\t{cid}\n")
                        all_generated_cand_pairs.add((s1_id, cid))
                        if cid in s2_dict:
                            batch_s2_pair_items.append((q, s2_dict[cid], "S2", cm))

                    # S3 candidates
                    cands_meta_s3 = gen_s3.generate_candidates_with_metadata(
                        s1_name, s1_addr, s1_country, k=self.config.k
                    )
                    for cm in cands_meta_s3:
                        cid = str(cm.get("candidate_id") or cm.get("entity_id"))
                        batch_cand_lines.append(f"{s1_id}\t{cid}\n")
                        all_generated_cand_pairs.add((s1_id, cid))
                        if cid in s3_dict:
                            batch_s3_pair_items.append((q, s3_dict[cid], "S3", cm))

                # Flush candidate pairs to disk
                f_cand.writelines(batch_cand_lines)
                total_pairs_count += len(batch_cand_lines)
                total_block_time += (time.time() - t_b0)

                # 4b. Feature Extraction
                t_f0 = time.time()
                df_feats_s2 = (
                    extractor_s2.extract_batch_dataframe(batch_s2_pair_items, include_identifiers=True)
                    if batch_s2_pair_items
                    else pd.DataFrame()
                )
                df_feats_s3 = (
                    extractor_s3.extract_batch_dataframe(batch_s3_pair_items, include_identifiers=True)
                    if batch_s3_pair_items
                    else pd.DataFrame()
                )
                total_feat_time += (time.time() - t_f0)

                # 4c. LightGBM Scoring & Threshold Decisions
                t_s0 = time.time()
                if not df_feats_s2.empty:
                    probas_s2 = clf_s2.predict_proba(df_feats_s2[ALL_FEATURE_NAMES])
                    for s1_val, c_val, prob in zip(
                        df_feats_s2["source1_id"], df_feats_s2["candidate_id"], probas_s2
                    ):
                        s1_str = str(s1_val)
                        c_str = str(c_val)
                        prob_val = float(prob)
                        if prob_val >= self.config.threshold_s2 and c_str != s1_str:
                            matches_per_s1[s1_str].append((c_str, prob_val))

                if not df_feats_s3.empty:
                    probas_s3 = clf_s3.predict_proba(df_feats_s3[ALL_FEATURE_NAMES])
                    for s1_val, c_val, prob in zip(
                        df_feats_s3["source1_id"], df_feats_s3["candidate_id"], probas_s3
                    ):
                        s1_str = str(s1_val)
                        c_str = str(c_val)
                        prob_val = float(prob)
                        if prob_val >= self.config.threshold_s3 and c_str != s1_str:
                            matches_per_s1[s1_str].append((c_str, prob_val))
                total_score_time += (time.time() - t_s0)

                if (batch_idx + 1) % max(1, total_batches // 5) == 0 or (batch_idx + 1) == total_batches:
                    curr_mem = process.memory_info().rss / (1024 * 1024)
                    logger.info(
                        "Processed batch %d/%d (%d S1 queries, %d candidate pairs, memory: %.1f MB RSS)",
                        batch_idx + 1,
                        total_batches,
                        b_end,
                        total_pairs_count,
                        curr_mem,
                    )

        logger.info(
            "Saved %d candidate pairs to %s (block: %.2fs, feat: %.2fs, score: %.2fs)",
            total_pairs_count,
            cand_pairs_path,
            total_block_time,
            total_feat_time,
            total_score_time,
        )

        # Step 5: Format & Write matching_results.tsv
        t0 = time.time()
        matching_results_path = out_dir / "matching_results.tsv"
        logger.info("Formatting final matches -> %s", matching_results_path)

        matching_results_rows: List[Dict[str, str]] = []
        for s1_id in sorted(all_expected_s1):
            cand_list = matches_per_s1.get(s1_id, [])
            # Deduplicate preserving confidence descending order
            seen_cands: Set[str] = set()
            sorted_unique_cands: List[str] = []
            for cid, _ in sorted(cand_list, key=lambda x: x[1], reverse=True):
                if cid not in seen_cands:
                    seen_cands.add(cid)
                    sorted_unique_cands.append(cid)

            matching_results_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(sorted_unique_cands),
            })

        df_matching_results = pd.DataFrame(matching_results_rows)
        write_tsv(df_matching_results, matching_results_path)
        decision_time = time.time() - t0
        logger.info("Saved %d rows to %s in %.2fs", len(df_matching_results), matching_results_path, decision_time)

        # Step 6: Validate output integrity
        logger.info("Executing rigorous submission integrity validation...")
        val_report = SubmissionValidator.validate_matching_results(
            matching_results_path=matching_results_path,
            candidate_pairs_path=cand_pairs_path,
            expected_s1_ids=all_expected_s1,
            valid_candidate_ids=valid_candidates,
            require_deterministic_order=True,
        )

        # Step 7: Labeled Validation Metrics (if ground truth available)
        eval_metrics: Optional[Dict[str, Any]] = None
        if df_gt is not None:
            eval_metrics = self._evaluate_with_ground_truth(
                df_gt=df_gt,
                matching_results_rows=matching_results_rows,
                all_generated_cand_pairs=all_generated_cand_pairs,
                s1_records=s1_records,
                s2_dict=s2_dict,
                s3_dict=s3_dict,
            )

        total_runtime = time.time() - start_time
        peak_mem = process.memory_info().rss / (1024 * 1024)

        summary: Dict[str, Any] = {
            "is_valid": val_report.is_valid,
            "total_s1_entities": val_report.num_s1_entities,
            "matched_s1_count": val_report.num_matched_s1,
            "zero_match_s1_count": val_report.num_zero_match_s1,
            "total_matches_produced": val_report.total_matches_produced,
            "total_candidate_pairs": val_report.total_candidate_pairs,
            "matching_results_path": str(matching_results_path),
            "candidate_pairs_path": str(cand_pairs_path),
            "runtime_seconds": round(total_runtime, 2),
            "peak_memory_mb": round(peak_mem, 1),
            "timings": {
                "load_time": round(load_time, 2),
                "fit_time": round(fit_time, 2),
                "block_time": round(total_block_time, 2),
                "feat_time": round(total_feat_time, 2),
                "score_time": round(total_score_time, 2),
                "decision_time": round(decision_time, 2),
            },
        }

        if eval_metrics:
            summary["labeled_validation"] = eval_metrics

        manifest_path = out_dir / "pipeline_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Pipeline completed successfully in %.2fs. Peak memory: %.1f MB. Manifest: %s", total_runtime, peak_mem, manifest_path)
        return summary

    def _evaluate_with_ground_truth(
        self,
        df_gt: pd.DataFrame,
        matching_results_rows: List[Dict[str, str]],
        all_generated_cand_pairs: Set[Tuple[str, str]],
        s1_records: List[Tuple[str, str, str, str]],
        s2_dict: Dict[str, Any],
        s3_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute comprehensive labeled validation metrics against ground truth."""
        logger.info("Evaluating predictions against Ground Truth...")
        s1_query_ids = {str(r[0]) for r in s1_records}

        gt_col_s1 = df_gt.columns[0]
        gt_col_match = df_gt.columns[1]

        gt_s2_map: Dict[str, Set[str]] = defaultdict(set)
        gt_s3_map: Dict[str, Set[str]] = defaultdict(set)
        gt_overall_map: Dict[str, Set[str]] = defaultdict(set)

        for s1_val, m_val in zip(df_gt[gt_col_s1], df_gt[gt_col_match]):
            s1_id = str(s1_val)
            if s1_id not in s1_query_ids:
                continue
            m_str = str(m_val).strip() if pd.notna(m_val) else ""
            if not m_str or m_str.lower() in ("nan", "none", "null"):
                continue
            for mid in m_str.split(","):
                mid = mid.strip()
                if not mid:
                    continue
                gt_overall_map[s1_id].add(mid)
                if mid in s2_dict:
                    gt_s2_map[s1_id].add(mid)
                elif mid in s3_dict:
                    gt_s3_map[s1_id].add(mid)
                elif mid.startswith("S2") or mid.startswith("s2"):
                    gt_s2_map[s1_id].add(mid)
                elif mid.startswith("S3") or mid.startswith("s3"):
                    gt_s3_map[s1_id].add(mid)

        # 1. Candidate Recall
        total_gt_s2 = sum(len(cands) for cands in gt_s2_map.values())
        total_gt_s3 = sum(len(cands) for cands in gt_s3_map.values())
        total_gt_all = sum(len(cands) for cands in gt_overall_map.values())

        found_s2 = sum(1 for s1, cands in gt_s2_map.items() for c in cands if (s1, c) in all_generated_cand_pairs)
        found_s3 = sum(1 for s1, cands in gt_s3_map.items() for c in cands if (s1, c) in all_generated_cand_pairs)
        found_all = sum(1 for s1, cands in gt_overall_map.items() for c in cands if (s1, c) in all_generated_cand_pairs)

        cand_recall_s2 = (found_s2 / total_gt_s2) if total_gt_s2 > 0 else 1.0
        cand_recall_s3 = (found_s3 / total_gt_s3) if total_gt_s3 > 0 else 1.0
        cand_recall_all = (found_all / total_gt_all) if total_gt_all > 0 else 1.0

        # 2. Predicted Match Maps
        pred_map: Dict[str, Set[str]] = {}
        for row in matching_results_rows:
            s1_id = row["source1_entity_id"]
            m_str = row["matched_entity_ids"]
            cands = {c.strip() for c in m_str.split(",") if c.strip()} if m_str else set()
            pred_map[s1_id] = cands

        # 3. Overall Entity Level Metrics
        precisions = []
        recalls = []
        f0_5_scores = []
        zero_match_correct = 0
        zero_match_total = 0
        single_match_f0_5 = []
        multi_match_f0_5 = []

        tp_total = 0
        fp_total = 0
        fn_total = 0

        for s1_id in s1_query_ids:
            true_set = gt_overall_map.get(s1_id, set())
            pred_set = pred_map.get(s1_id, set())

            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            tp_total += tp
            fp_total += fp
            fn_total += fn

            # Precision & Recall
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if not pred_set and not true_set else 0.0)
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if not true_set else 0.0)
            precisions.append(prec)
            recalls.append(rec)

            f0_5 = compute_f_beta(prec, rec, beta=0.5)
            f0_5_scores.append(f0_5)

            # Subgroup breakdowns
            num_true = len(true_set)
            if num_true == 0:
                zero_match_total += 1
                if len(pred_set) == 0:
                    zero_match_correct += 1
            elif num_true == 1:
                single_match_f0_5.append(f0_5)
            else:
                multi_match_f0_5.append(f0_5)

        macro_prec = float(np.mean(precisions)) if precisions else 0.0
        macro_rec = float(np.mean(recalls)) if recalls else 0.0
        macro_f0_5 = float(np.mean(f0_5_scores)) if f0_5_scores else 0.0

        pairwise_prec = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
        pairwise_rec = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
        pairwise_f0_5 = compute_f_beta(pairwise_prec, pairwise_rec, beta=0.5)

        zero_match_acc = (zero_match_correct / zero_match_total) if zero_match_total > 0 else 1.0
        single_f0_5 = float(np.mean(single_match_f0_5)) if single_match_f0_5 else 0.0
        multi_f0_5 = float(np.mean(multi_match_f0_5)) if multi_match_f0_5 else 0.0

        res = {
            "candidate_recall_s2": round(cand_recall_s2 * 100, 2),
            "candidate_recall_s3": round(cand_recall_s3 * 100, 2),
            "candidate_recall_overall": round(cand_recall_all * 100, 2),
            "entity_macro_f0_5": round(macro_f0_5, 4),
            "entity_macro_precision": round(macro_prec, 4),
            "entity_macro_recall": round(macro_rec, 4),
            "pairwise_f0_5": round(pairwise_f0_5, 4),
            "pairwise_precision": round(pairwise_prec, 4),
            "pairwise_recall": round(pairwise_rec, 4),
            "subgroups": {
                "zero_match_accuracy": round(zero_match_acc * 100, 2),
                "zero_match_count": zero_match_total,
                "single_match_f0_5": round(single_f0_5, 4),
                "single_match_count": len(single_match_f0_5),
                "multi_match_f0_5": round(multi_f0_5, 4),
                "multi_match_count": len(multi_match_f0_5),
            },
        }

        logger.info(
            "Labeled Validation Metrics: Cand Recall=%.2f%% | Macro F0.5=%.4f (P=%.4f, R=%.4f) | ZeroAcc=%.2f%%, SingleF0.5=%.4f, MultiF0.5=%.4f",
            res["candidate_recall_overall"],
            res["entity_macro_f0_5"],
            res["entity_macro_precision"],
            res["entity_macro_recall"],
            res["subgroups"]["zero_match_accuracy"],
            res["subgroups"]["single_match_f0_5"],
            res["subgroups"]["multi_match_f0_5"],
        )
        return res

    def _load_input_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Locate and load source1, source2, source3, and optional ground truth DataFrames."""
        cfg = self.config
        data_cfg = S3DataConfig()

        def resolve_file(explicit_path: Optional[Path], dataset_name: str) -> Optional[Path]:
            if explicit_path is not None:
                p = Path(explicit_path)
                if p.exists():
                    return p

            # 1. SageMaker channel environments
            sm_channel = os.getenv(f"SM_CHANNEL_{dataset_name.upper()}")
            if sm_channel:
                p = Path(sm_channel)
                if p.is_file():
                    return p
                if p.is_dir():
                    tsv_files = sorted(list(p.glob("**/*.tsv")))
                    if tsv_files:
                        return tsv_files[0]

            # 2. General SageMaker input directory
            sm_input = Path(os.getenv("SM_INPUT_DIR", "/opt/ml/processing/input"))
            if sm_input.exists():
                for fname in [
                    f"{dataset_name}.tsv",
                    f"test_{dataset_name}.tsv",
                    f"train_{dataset_name}.tsv",
                ]:
                    cand = sm_input / fname
                    if cand.exists():
                        return cand
                    cand_subdir = sm_input / dataset_name / fname
                    if cand_subdir.exists():
                        return cand_subdir

            # 3. Local data cache (test or train)
            test_local = data_cfg.get_local_path("test", dataset_name)
            if test_local.exists():
                return test_local
            train_local = data_cfg.get_local_path("train", dataset_name)
            if train_local.exists():
                return train_local

            return None

        p1 = resolve_file(cfg.s1_path, "source1")
        p2 = resolve_file(cfg.s2_path, "source2")
        p3 = resolve_file(cfg.s3_path, "source3")
        pgt = resolve_file(cfg.ground_truth_path, "ground_truth")

        if p1 is None or not p1.exists():
            raise FileNotFoundError("Cannot locate input file for source1. Provide --s1 or place in data/raw/")
        if p2 is None or not p2.exists():
            raise FileNotFoundError("Cannot locate input file for source2. Provide --s2 or place in data/raw/")
        if p3 is None or not p3.exists():
            raise FileNotFoundError("Cannot locate input file for source3. Provide --s3 or place in data/raw/")

        logger.info("Reading Source 1: %s", p1)
        df_s1 = read_tsv(p1, nrows=cfg.limit_s1)
        logger.info("Reading Source 2: %s", p2)
        df_s2 = read_tsv(p2, nrows=cfg.limit_candidates)
        logger.info("Reading Source 3: %s", p3)
        df_s3 = read_tsv(p3, nrows=cfg.limit_candidates)

        df_gt: Optional[pd.DataFrame] = None
        if cfg.ground_truth_path is not None:
            pgt = resolve_file(cfg.ground_truth_path, "ground_truth")
            if pgt is not None and pgt.exists():
                logger.info("Reading Ground Truth: %s", pgt)
                df_gt = read_tsv(pgt)
        elif "train" in str(p1).lower():
            pgt = resolve_file(None, "ground_truth")
            if pgt is not None and pgt.exists():
                logger.info("Reading Ground Truth: %s", pgt)
                df_gt = read_tsv(pgt)

        return df_s1, df_s2, df_s3, df_gt
