"""
Script to query and display similar jobs for a given job.

Usage:
    python scripts/query_similar_jobs.py <job_id>
    python scripts/query_similar_jobs.py --all  # Show similar jobs for all jobs
"""
import sys
import io
import logging
from pathlib import Path

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import recommend_service
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database import DatabaseConnection, SimilarJobRepository, JobRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def query_similar_jobs(job_id: str, limit: int = 10):
    """Query and display similar jobs for a given job"""
    db = DatabaseConnection()
    similar_job_repo = SimilarJobRepository(db)
    job_repo = JobRepository(db)

    # Get job details
    query = """
        SELECT id, title, location, "experienceLevel", type
        FROM jobs
        WHERE id = %s
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (job_id,))
        job = cursor.fetchone()

    if not job:
        logger.error(f"Job with ID {job_id} not found")
        return

    print("\n" + "=" * 80)
    print(f"TARGET JOB: {job['title']}")
    print(f"Job ID: {job['id']}")
    print(f"Location: {job.get('location', 'N/A')}")
    print(f"Experience Level: {job.get('experienceLevel', 'N/A')}")
    print(f"Type: {job.get('type', 'N/A')}")
    print("=" * 80)

    # Get similar jobs
    similar_jobs = similar_job_repo.get_similar_jobs(job_id, limit)

    if not similar_jobs:
        print("\nNo similar jobs found.")
        return

    print(f"\nTOP {len(similar_jobs)} SIMILAR JOBS:")
    print("-" * 80)

    for idx, similar_job in enumerate(similar_jobs, 1):
        print(f"\n{idx}. {similar_job['title']}")
        print(f"   Similarity: {similar_job['similarity']:.4f}")
        print(f"   Job ID: {similar_job['similarJobId']}")
        print(f"   Location: {similar_job.get('location', 'N/A')}")
        print(f"   Experience Level: {similar_job.get('experienceLevel', 'N/A')}")
        print(f"   Type: {similar_job.get('type', 'N/A')}")

    print("\n" + "=" * 80 + "\n")


def query_all_jobs_with_similar():
    """Query and display all jobs that have similar jobs"""
    db = DatabaseConnection()

    query = """
        SELECT DISTINCT j.id, j.title, COUNT(sj."similarJobId") as similar_count
        FROM jobs j
        JOIN similar_jobs sj ON j.id = sj."jobId"
        WHERE j.status = 'ACTIVE'
        GROUP BY j.id, j.title
        ORDER BY similar_count DESC
    """

    with db.get_cursor() as cursor:
        cursor.execute(query)
        jobs = cursor.fetchall()

    if not jobs:
        print("\nNo jobs with similar jobs found.")
        return

    print("\n" + "=" * 80)
    print("ALL JOBS WITH SIMILAR JOBS:")
    print("=" * 80)

    for idx, job in enumerate(jobs, 1):
        print(f"\n{idx}. {job['title']}")
        print(f"   Job ID: {job['id']}")
        print(f"   Similar Jobs Count: {job['similar_count']}")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(jobs)} jobs have similar jobs calculated")
    print("=" * 80 + "\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/query_similar_jobs.py <job_id>")
        print("  python scripts/query_similar_jobs.py --all")
        sys.exit(1)

    # Test connection
    db = DatabaseConnection()
    if not db.test_connection():
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)

    if sys.argv[1] == "--all":
        query_all_jobs_with_similar()
    else:
        job_id = sys.argv[1]
        query_similar_jobs(job_id)


if __name__ == "__main__":
    main()
