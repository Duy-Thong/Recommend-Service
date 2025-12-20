"""
Run Full Comparison Evaluation.

Runs all comparison experiments as described in section_54_danh_gia.tex:
- Bảng 2: So sánh với phương pháp cơ sở (Random, TF-IDF, SimCSE, Cascade)
- Bảng 3: Ảnh hưởng của các vòng lọc (1-layer, 2-layer, 3-layer)

Prerequisites:
1. Ground truth must be generated (run generate_ground_truth.py)
2. Recommendations must be calculated (run run_full_pipeline.py)

Usage:
    python scripts/run_comparison.py
    python scripts/run_comparison.py --baseline-only
    python scripts/run_comparison.py --ablation-only
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import recommend_service
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection
from recommend_service.services.evaluation import ComparisonEvaluator

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
    """Main function to run comparison evaluation"""
    parser = argparse.ArgumentParser(
        description="Run comparison evaluation of recommendation methods"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="./evaluation_data/ground_truth.csv",
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_data/comparison_results.json",
        help="Path to save comparison results"
    )
    parser.add_argument(
        "--cv-limit",
        type=int,
        default=2000,
        help="Maximum number of CVs to evaluate (default: 2000)"
    )
    parser.add_argument(
        "--job-limit",
        type=int,
        default=5000,
        help="Maximum number of jobs to use (default: 5000)"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline comparison (Bảng 2)"
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Only run ablation study (Bảng 3)"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("COMPARISON EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"Ground truth: {args.ground_truth}")
    logger.info(f"Output: {args.output}")
    logger.info(f"CV limit: {args.cv_limit}")
    logger.info(f"Job limit: {args.job_limit}")
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
        evaluator = ComparisonEvaluator(
            ground_truth_path=args.ground_truth,
            output_path=args.output,
            cv_limit=args.cv_limit,
            job_limit=args.job_limit
        )

        # Determine what to run
        run_baseline = not args.ablation_only
        run_ablation = not args.baseline_only

        # Run evaluation
        result = evaluator.run(
            run_baseline=run_baseline,
            run_ablation=run_ablation
        )

        # Print tables
        if result.baseline_comparison:
            evaluator.print_baseline_table(result.baseline_comparison)
            evaluator.print_expected_comparison(result.baseline_comparison)

        if result.ablation_study:
            evaluator.print_ablation_table(result.ablation_study)

        logger.info("")
        logger.info("=" * 70)
        logger.info("COMPARISON EVALUATION COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {args.output}")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Comparison evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
