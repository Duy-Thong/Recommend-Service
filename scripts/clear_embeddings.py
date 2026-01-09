"""
Script to clear all embeddings from database.
This is needed when changing the embedding method (e.g., from mean pooling to CLS token).
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Clear all embeddings from jobs and cvs tables"""
    logger.info("=" * 70)
    logger.info("CLEARING ALL EMBEDDINGS")
    logger.info("=" * 70)

    db = DatabaseConnection()

    try:
        # Clear job embeddings
        logger.info("Clearing job embeddings...")
        with db.get_cursor() as cursor:
            cursor.execute('''
                UPDATE jobs
                SET "titleEmbedding" = NULL,
                    "skillsEmbedding" = NULL,
                    "requirementEmbedding" = NULL,
                    "contentHash" = NULL
                WHERE "titleEmbedding" IS NOT NULL
                   OR "skillsEmbedding" IS NOT NULL
                   OR "requirementEmbedding" IS NOT NULL
            ''')
            jobs_cleared = cursor.rowcount
        logger.info(f"Jobs cleared: {jobs_cleared}")

        # Clear CV embeddings
        logger.info("Clearing CV embeddings...")
        with db.get_cursor() as cursor:
            cursor.execute('''
                UPDATE cvs
                SET "titleEmbedding" = NULL,
                    "skillsEmbedding" = NULL,
                    "experienceEmbedding" = NULL,
                    "contentHash" = NULL
                WHERE "titleEmbedding" IS NOT NULL
                   OR "skillsEmbedding" IS NOT NULL
                   OR "experienceEmbedding" IS NOT NULL
            ''')
            cvs_cleared = cursor.rowcount
        logger.info(f"CVs cleared: {cvs_cleared}")

        logger.info("=" * 70)
        logger.info("EMBEDDINGS CLEARED SUCCESSFULLY")
        logger.info(f"Total jobs cleared: {jobs_cleared}")
        logger.info(f"Total CVs cleared: {cvs_cleared}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to clear embeddings: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
