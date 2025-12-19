"""
Script to run the full recommendation pipeline.

The order doesn't matter much because:
- Both services automatically generate embeddings and save to DB (with hash caching)
- Shared FAISS index is built/loaded as needed

However, for optimal performance:
- Recommended: Run CV-Job recommendations first (embeds both jobs & CVs)
- Then run Similar Jobs (reuses job embeddings, only needs to load index or calculate similarities)

Usage:
    # Run both services (recommended order: CV-Job first, then Similar Jobs)
    python scripts/run_full_pipeline.py

    # Run in reverse order (Similar Jobs first)
    python scripts/run_full_pipeline.py --reverse
"""
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import recommend_service
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection
from recommend_service.services.similar_jobs_recommendation import SimilarJobsRecommendationService
from recommend_service.services.recommendation import RecommendationService

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
    """Main function to run full recommendation pipeline"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run full recommendation pipeline")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run in reverse order (Similar Jobs first, then CV-Job recommendations)"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FULL RECOMMENDATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Top K jobs: {settings.top_k_jobs}")
    if args.reverse:
        logger.info("Order: Similar Jobs → CV-Job Recommendations (reversed)")
    else:
        logger.info("Order: CV-Job Recommendations → Similar Jobs (recommended)")
    logger.info("=" * 70)

    # Test connection first
    if not test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    total_stats = {
        "similar_jobs": {},
        "cv_recommendations": {}
    }

    if args.reverse:
        # Reverse order: Similar Jobs first
        step1_name = "CALCULATE SIMILAR JOBS"
        step2_name = "GENERATE CV-JOB RECOMMENDATIONS"

        # Step 1: Calculate Similar Jobs
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"STEP 1: {step1_name}")
        logger.info("=" * 70)

        try:
            similar_jobs_service = SimilarJobsRecommendationService()
            similar_jobs_stats = similar_jobs_service.run()
            total_stats["similar_jobs"] = similar_jobs_stats

            logger.info("=" * 70)
            logger.info("Step 1 completed successfully!")
            logger.info(f"Similar Jobs Statistics: {similar_jobs_stats}")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            logger.error("Aborting pipeline due to Step 1 failure")
            sys.exit(1)

        # Step 2: Generate CV-Job Recommendations
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"STEP 2: {step2_name}")
        logger.info("=" * 70)

        try:
            recommendation_service = RecommendationService(use_faiss=True)
            cv_rec_stats = recommendation_service.run()
            total_stats["cv_recommendations"] = cv_rec_stats

            logger.info("=" * 70)
            logger.info("Step 2 completed successfully!")
            logger.info(f"CV Recommendation Statistics: {cv_rec_stats}")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Recommended order: CV-Job Recommendations first
        step1_name = "GENERATE CV-JOB RECOMMENDATIONS"
        step2_name = "CALCULATE SIMILAR JOBS"

        # Step 1: Generate CV-Job Recommendations (embeds jobs & CVs, builds index)
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"STEP 1: {step1_name}")
        logger.info("=" * 70)

        try:
            recommendation_service = RecommendationService(use_faiss=True)
            cv_rec_stats = recommendation_service.run()
            total_stats["cv_recommendations"] = cv_rec_stats

            logger.info("=" * 70)
            logger.info("Step 1 completed successfully!")
            logger.info(f"CV Recommendation Statistics: {cv_rec_stats}")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            logger.error("Aborting pipeline due to Step 1 failure")
            sys.exit(1)

        # Step 2: Calculate Similar Jobs (reuses job embeddings and index)
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"STEP 2: {step2_name}")
        logger.info("=" * 70)

        try:
            similar_jobs_service = SimilarJobsRecommendationService()
            similar_jobs_stats = similar_jobs_service.run()
            total_stats["similar_jobs"] = similar_jobs_stats

            logger.info("=" * 70)
            logger.info("Step 2 completed successfully!")
            logger.info(f"Similar Jobs Statistics: {similar_jobs_stats}")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            sys.exit(1)

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Summary:")
    logger.info("-" * 70)
    logger.info("Step 1 - Similar Jobs:")
    logger.info(f"  - Jobs processed: {total_stats['similar_jobs'].get('jobs_processed', 0)}")
    logger.info(f"  - Jobs embedded: {total_stats['similar_jobs'].get('jobs_embedded', 0)}")
    logger.info(f"  - Similar jobs created: {total_stats['similar_jobs'].get('similar_jobs_created', 0)}")
    logger.info("")
    logger.info("Step 2 - CV-Job Recommendations:")
    logger.info(f"  - CVs processed: {total_stats['cv_recommendations'].get('cvs_processed', 0)}")
    logger.info(f"  - CVs embedded: {total_stats['cv_recommendations'].get('cvs_embedded', 0)}")
    logger.info(f"  - Jobs processed: {total_stats['cv_recommendations'].get('jobs_processed', 0)}")
    logger.info(f"  - Jobs embedded: {total_stats['cv_recommendations'].get('jobs_embedded', 0)}")
    logger.info(f"  - Recommendations created: {total_stats['cv_recommendations'].get('recommendations_created', 0)}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Shared FAISS index location: ./faiss_data/shared_jobs.faiss")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
