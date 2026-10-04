"""Production CLI entrypoint for local execution and AWS SageMaker processing."""

import argparse
import os
from pathlib import Path
import sys

from business_entity_resolution.pipeline.runner import PipelineConfig, ProductionPipelineRunner
from business_entity_resolution.utils.logging import get_logger

logger = get_logger("pipeline.entrypoint")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run production entity resolution matching pipeline (V4.1 Blocker + LightGBM)."
    )
    # Inputs
    parser.add_argument("--s1", type=str, default=None, help="Path to Source 1 TSV")
    parser.add_argument("--s2", type=str, default=None, help="Path to Source 2 TSV")
    parser.add_argument("--s3", type=str, default=None, help="Path to Source 3 TSV")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional path to Ground Truth TSV")

    # Directories
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("SM_OUTPUT_DATA_DIR", os.getenv("OUTPUT_DIR", "output")),
        help="Directory where candidate_pairs.tsv and matching_results.tsv are saved",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.getenv("SM_MODEL_DIR", os.getenv("MODELS_DIR", "models")),
        help="Directory containing trained LightGBM model files",
    )

    # Parameters
    parser.add_argument("--k", type=int, default=200, help="Candidate budget cap per target source")
    parser.add_argument("--threshold-s2", type=float, default=0.20, help="Decision threshold for S2")
    parser.add_argument("--threshold-s3", type=float, default=0.80, help="Decision threshold for S3")
    parser.add_argument("--limit-s1", type=int, default=None, help="Limit number of S1 entities (for testing)")
    parser.add_argument("--limit-candidates", type=int, default=None, help="Limit number of S2/S3 candidates (for testing)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    # Experimental Qwen flag
    parser.add_argument(
        "--enable-qwen",
        action="store_true",
        default=False,
        help="Enable optional experimental Qwen semantic reranker (strictly OFF by default)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Entity Resolution Production Runner...")

    config = PipelineConfig(
        s1_path=Path(args.s1) if args.s1 else None,
        s2_path=Path(args.s2) if args.s2 else None,
        s3_path=Path(args.s3) if args.s3 else None,
        ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        k=args.k,
        threshold_s2=args.threshold_s2,
        threshold_s3=args.threshold_s3,
        limit_s1=args.limit_s1,
        limit_candidates=args.limit_candidates,
        random_seed=args.random_seed,
        enable_qwen=args.enable_qwen,
    )

    runner = ProductionPipelineRunner(config)
    results = runner.run()

    if not results.get("is_valid", False):
        logger.error("Submission validation failed! Please check pipeline errors.")
        sys.exit(1)

    logger.info("Production pipeline run completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
