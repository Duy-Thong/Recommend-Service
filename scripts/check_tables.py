"""Check database tables for application/interaction data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database import DatabaseConnection

db = DatabaseConnection()

# Get all tables
with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [r["table_name"] for r in cursor.fetchall()]

print("Tables in database:")
for t in tables:
    print(f"  - {t}")

# Check for application-related tables
application_keywords = ['application', 'apply', 'candidate', 'hire', 'interview', 'save', 'bookmark', 'favorite', 'click', 'view']
print("\nTables that might contain user interaction data:")
for t in tables:
    for kw in application_keywords:
        if kw in t.lower():
            print(f"  - {t}")
            break

# Check job_applications if exists
if 'job_applications' in tables:
    with db.get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as cnt FROM job_applications")
        count = cursor.fetchone()["cnt"]
        print(f"\njob_applications: {count} records")
