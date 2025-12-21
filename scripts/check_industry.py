import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
from recommend_service.database import DatabaseConnection

db = DatabaseConnection()
with db.get_cursor() as cursor:
    # Check distinct industries
    cursor.execute('''
        SELECT industry, COUNT(*) as cnt
        FROM jobs
        WHERE industry IS NOT NULL AND industry != ''
        GROUP BY industry
        ORDER BY cnt DESC
        LIMIT 30
    ''')
    industries = cursor.fetchall()
    print('Industries in jobs table:')
    for ind in industries:
        print(f'  - {ind["industry"]}: {ind["cnt"]} jobs')

    # Count jobs without industry
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE industry IS NULL OR industry = ''")
    null_cnt = cursor.fetchone()[0]
    print(f'\nJobs without industry: {null_cnt}')

    # Total jobs
    cursor.execute("SELECT COUNT(*) FROM jobs")
    total = cursor.fetchone()[0]
    print(f'Total jobs: {total}')
