import sys
sys.path.insert(0, ".")

from recommend_service.database import DatabaseConnection

db = DatabaseConnection()

with db.get_cursor() as cursor:
    # Jobs embedding progress
    cursor.execute('''
        SELECT
            COUNT(*) as total_jobs,
            COUNT("titleEmbedding") as jobs_with_embedding
        FROM jobs
        WHERE status = 'ACTIVE'
    ''')
    job_stats = cursor.fetchone()

    # CVs embedding progress
    cursor.execute('''
        SELECT
            COUNT(*) as total_cvs,
            COUNT("titleEmbedding") as cvs_with_embedding
        FROM cvs
        WHERE "isMain" = true
    ''')
    cv_stats = cursor.fetchone()

print("=== EMBEDDING PROGRESS ===")
job_pct = 100 * job_stats['jobs_with_embedding'] / job_stats['total_jobs'] if job_stats['total_jobs'] > 0 else 0
cv_pct = 100 * cv_stats['cvs_with_embedding'] / cv_stats['total_cvs'] if cv_stats['total_cvs'] > 0 else 0
print(f"Jobs: {job_stats['jobs_with_embedding']}/{job_stats['total_jobs']} ({job_pct:.1f}%)")
print(f"CVs:  {cv_stats['cvs_with_embedding']}/{cv_stats['total_cvs']} ({cv_pct:.1f}%)")
