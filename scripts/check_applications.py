"""Check applications table for ground truth data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database import DatabaseConnection

db = DatabaseConnection()

# Get applications table structure
with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'applications'
        ORDER BY ordinal_position
    """)
    columns = cursor.fetchall()

print("Applications table structure:")
for col in columns:
    print(f"  - {col['column_name']}: {col['data_type']}")

# Count applications
with db.get_cursor() as cursor:
    cursor.execute("SELECT COUNT(*) as cnt FROM applications")
    count = cursor.fetchone()["cnt"]
    print(f"\nTotal applications: {count}")

# Check if applications link CVs to Jobs
with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT COUNT(DISTINCT "cvId") as cvs, COUNT(DISTINCT "jobId") as jobs
        FROM applications
        WHERE "cvId" IS NOT NULL AND "jobId" IS NOT NULL
    """)
    result = cursor.fetchone()
    print(f"Unique CVs with applications: {result['cvs']}")
    print(f"Unique Jobs with applications: {result['jobs']}")

# Sample applications
print("\nSample applications (first 5):")
with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT a."cvId", a."jobId", a.status, a."createdAt",
               c.title as cv_title, j.title as job_title
        FROM applications a
        LEFT JOIN cvs c ON a."cvId" = c.id
        LEFT JOIN jobs j ON a."jobId" = j.id
        WHERE a."cvId" IS NOT NULL AND a."jobId" IS NOT NULL
        LIMIT 5
    """)
    apps = cursor.fetchall()
    for app in apps:
        print(f"  CV: {app['cv_title'][:40] if app['cv_title'] else 'N/A'}")
        print(f"  Job: {app['job_title'][:40] if app['job_title'] else 'N/A'}")
        print(f"  Status: {app['status']}")
        print("  ---")

# Check application statuses
print("\nApplication statuses:")
with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT status, COUNT(*) as cnt
        FROM applications
        GROUP BY status
        ORDER BY cnt DESC
    """)
    statuses = cursor.fetchall()
    for s in statuses:
        print(f"  - {s['status']}: {s['cnt']}")
