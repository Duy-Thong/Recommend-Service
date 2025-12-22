"""
Run Evaluation of the Recommendation System.

Evaluates the recommendation system against ground truth using:
- MRR (Mean Reciprocal Rank)
- NDCG@5, NDCG@10 (Normalized Discounted Cumulative Gain)
- Hit Rate@5, Hit Rate@10

Prerequisites:
1. Ground truth must be generated first (run generate_ground_truth.py)
2. Recommendations must be calculated (run run_full_pipeline.py)

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --ground-truth ./custom_ground_truth.csv
    python scripts/run_evaluation.py --compare
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import recommend_service
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection
from recommend_service.services.evaluation import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_connection() -> bool:
    """Test database connection"""
    db = DatabaseConnection()
    if db.test_connection():
        logger.info("Database connection successful")
        return True
    else:
        logger.error("Database connection failed")
        return False


def main():
    """Main function to run evaluation"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run evaluation of the recommendation system"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="./evaluation_data/ground_truth.csv",
        help="Path to ground truth CSV file (default: ./evaluation_data/ground_truth.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_data/evaluation_results.json",
        help="Path to save evaluation results JSON (default: ./evaluation_data/evaluation_results.json)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison with expected results from section_54_danh_gia.tex"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Use relaxed matching based on title embedding similarity"
    )
    parser.add_argument(
        "--similar-jobs",
        action="store_true",
        help="Use similar_jobs table for relaxed matching (recommended)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for relaxed matching (default: 0.8)"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RECOMMENDATION SYSTEM EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"Ground truth: {args.ground_truth}")
    logger.info(f"Output: {args.output}")
    if args.similar_jobs:
        logger.info(f"Mode: SIMILAR JOBS TABLE (threshold={args.threshold})")
    elif args.relaxed:
        logger.info(f"Mode: RELAXED EMBEDDING MATCHING (threshold={args.threshold})")
    else:
        logger.info("Mode: EXACT ID MATCHING")
    logger.info("=" * 70)

    # Check if ground truth file exists
    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        logger.error("")
        logger.error("Please generate ground truth first:")
        logger.error("  python scripts/generate_ground_truth.py")
        sys.exit(1)

    # Test connection
    if not test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    try:
        # Create evaluator
        evaluator = Evaluator(
            ground_truth_path=args.ground_truth,
            output_path=args.output
        )

        # Run evaluation
        if args.similar_jobs:
            logger.info(f"Running evaluation with SIMILAR JOBS TABLE (threshold={args.threshold})...")
            result = evaluator.evaluate_with_similar_jobs(similarity_threshold=args.threshold)
            evaluator.save_results(result)
        elif args.relaxed:
            logger.info(f"Running RELAXED evaluation with threshold={args.threshold}...")
            result = evaluator.evaluate_relaxed(similarity_threshold=args.threshold)
            evaluator.save_results(result)
        else:
            result = evaluator.run()

        # Show comparison if requested
        if args.compare:
            evaluator.print_comparison_table(result)

        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATION COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {args.output}")
        logger.info("")
        logger.info("Summary:")
        logger.info("-" * 70)
        print(result)
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
