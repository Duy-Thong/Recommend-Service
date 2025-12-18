import argparse
import logging
import sys

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection
from recommend_service.scheduler import RecommendationScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("recommend_service.log")
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
    parser = argparse.ArgumentParser(description="Job Recommendation Service")
    parser.add_argument(
        "--mode",
        choices=["schedule", "once", "test"],
        default="schedule",
        help="Run mode: 'schedule' for continuous scheduling, 'once' for single run, 'test' for connection test"
    )
    parser.add_argument(
        "--no-immediate",
        action="store_true",
        help="Don't run immediately when starting scheduler"
    )

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Job Recommendation Service")
    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Top K jobs: {settings.top_k_jobs}")
    logger.info(f"Schedule interval: {settings.schedule_interval_hours} hours")
    logger.info("=" * 50)

    # Test connection first
    if not test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    scheduler = RecommendationScheduler()

    if args.mode == "test":
        logger.info("Connection test passed. Exiting.")
        sys.exit(0)
    elif args.mode == "once":
        stats = scheduler.run_once()
        logger.info(f"Completed. Stats: {stats}")
    else:  # schedule
        run_immediately = not args.no_immediate
        scheduler.start(run_immediately=run_immediately)


if __name__ == "__main__":
    main()
