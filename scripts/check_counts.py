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
