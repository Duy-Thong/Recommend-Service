"""
Script to calculate and store similar jobs based on title embeddings.

Usage:
    python scripts/calculate_similar_jobs.py
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path to import recommend_service
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection
from recommend_service.services.similar_jobs_recommendation import SimilarJobsRecommendationService

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
    """Main function to calculate similar jobs"""
    logger.info("=" * 50)
    logger.info("Calculate Similar Jobs Script")
    logger.info("=" * 50)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info("=" * 50)

    # Test connection first
    if not test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    # Run similar jobs calculation
    service = SimilarJobsRecommendationService()

    try:
        stats = service.run()
        logger.info("=" * 50)
        logger.info("Similar jobs calculation completed successfully!")
        logger.info(f"Statistics: {stats}")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"Failed to calculate similar jobs: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
