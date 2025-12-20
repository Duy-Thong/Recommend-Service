"""
Generate Ground Truth for Evaluation.

Uses sentence-transformers/paraphrase-multilingual-mpnet-base-v2 to embed titles
and FAISS to find the most similar job for each CV.

Output: CSV file with columns (cv_id, job_id, cv_title, job_title, similarity)

Usage:
    python scripts/generate_ground_truth.py
    python scripts/generate_ground_truth.py --cv-limit 2000 --job-limit 5000
    python scripts/generate_ground_truth.py --output ./evaluation_data/custom_ground_truth.csv
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import recommend_service
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection
from recommend_service.services.evaluation import GroundTruthGenerator

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
    """Main function to generate ground truth"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate ground truth CV-Job pairs for evaluation"
    )
    parser.add_argument(
        "--cv-limit",
        type=int,
        default=2000,
        help="Maximum number of CVs to process (default: 2000)"
    )
    parser.add_argument(
        "--job-limit",
        type=int,
        default=5000,
        help="Maximum number of jobs to process (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_data/ground_truth.csv",
        help="Output path for ground truth CSV (default: ./evaluation_data/ground_truth.csv)"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="./faiss_data/title_jobs.faiss",
        help="Path to save FAISS index (default: ./faiss_data/title_jobs.faiss)"
    )
    parser.add_argument(
        "--no-save-index",
        action="store_true",
        help="Don't save FAISS index to disk"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("GROUND TRUTH GENERATION")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"CV limit: {args.cv_limit}")
    logger.info(f"Job limit: {args.job_limit}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Index path: {args.index_path}")
    logger.info("=" * 70)

    # Test connection first
    if not test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    try:
        # Create generator
        generator = GroundTruthGenerator(
            cv_limit=args.cv_limit,
            job_limit=args.job_limit,
            index_path=args.index_path,
            output_path=args.output
        )

        # Run generation
        output_path = generator.run(save_index=not args.no_save_index)

        if output_path:
            logger.info("")
            logger.info("=" * 70)
            logger.info("SUCCESS!")
            logger.info("=" * 70)
            logger.info(f"Ground truth saved to: {output_path}")
            if not args.no_save_index:
                logger.info(f"FAISS index saved to: {args.index_path}")
            logger.info("")
            logger.info("Next step: Run the recommendation pipeline, then evaluate:")
            logger.info("  1. python scripts/run_full_pipeline.py")
            logger.info("  2. python scripts/run_evaluation.py")
            logger.info("=" * 70)
        else:
            logger.error("No ground truth pairs generated. Check database content.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to generate ground truth: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
