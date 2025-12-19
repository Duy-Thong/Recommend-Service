"""
Script to run the full recommendation pipeline:
1. Calculate similar jobs (builds shared FAISS index)
2. Generate CV-Job recommendations (uses shared FAISS index)

Usage:
    python scripts/run_full_pipeline.py
"""
import sys
import logging
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
    logger.info("=" * 70)
    logger.info("FULL RECOMMENDATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Top K jobs: {settings.top_k_jobs}")
    logger.info("=" * 70)

    # Test connection first
    if not test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    total_stats = {
        "similar_jobs": {},
        "cv_recommendations": {}
    }

    # Step 1: Calculate Similar Jobs (builds shared FAISS index)
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: CALCULATE SIMILAR JOBS")
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

    # Step 2: Generate CV-Job Recommendations (uses shared FAISS index)
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: GENERATE CV-JOB RECOMMENDATIONS")
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
