"""Quick script to check record counts in database"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database.connection import DatabaseConnection

db = DatabaseConnection()

tables = [
    "companies",
    "users",
    "jobs",
    "cvs",
    "cv_skills",
    "work_experiences",
    "job_requirements",
    "job_benefits",
    "salaries"
]

with db.get_cursor() as cursor:
    print("=== Record Counts ===")
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) as cnt FROM {table}')
        count = cursor.fetchone()['cnt']
        print(f"{table}: {count}")

    print("\n=== Embedding Counts ===")
    # Jobs with embeddings
    cursor.execute('SELECT COUNT(*) as cnt FROM jobs WHERE "titleEmbedding" IS NOT NULL')
    jobs_with_emb = cursor.fetchone()['cnt']
    cursor.execute('SELECT COUNT(*) as cnt FROM jobs')
    total_jobs = cursor.fetchone()['cnt']
    print(f"Jobs with embeddings: {jobs_with_emb}/{total_jobs}")

    # CVs with embeddings
    cursor.execute('SELECT COUNT(*) as cnt FROM cvs WHERE "titleEmbedding" IS NOT NULL')
    cvs_with_emb = cursor.fetchone()['cnt']
    cursor.execute('SELECT COUNT(*) as cnt FROM cvs')
    total_cvs = cursor.fetchone()['cnt']
    print(f"CVs with embeddings: {cvs_with_emb}/{total_cvs}")

    print("\n=== Recommendation Counts ===")
    # CV-Job recommendations
    cursor.execute('SELECT COUNT(*) as cnt FROM recommend_jobs_for_cv')
    rec_count = cursor.fetchone()['cnt']
    cursor.execute('SELECT COUNT(DISTINCT "cvId") as cnt FROM recommend_jobs_for_cv')
    cvs_with_rec = cursor.fetchone()['cnt']
    print(f"Total CV-Job recommendations: {rec_count}")
    print(f"CVs with recommendations: {cvs_with_rec}/{total_cvs}")

    # Similar jobs
    cursor.execute('SELECT COUNT(*) as cnt FROM similar_jobs')
    similar_count = cursor.fetchone()['cnt']
    cursor.execute('SELECT COUNT(DISTINCT "jobId") as cnt FROM similar_jobs')
    jobs_with_similar = cursor.fetchone()['cnt']
    print(f"Total similar job pairs: {similar_count}")
    print(f"Jobs with similar jobs: {jobs_with_similar}/{total_jobs}")
