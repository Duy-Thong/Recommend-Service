# -*- coding: utf-8 -*-
"""
Test Word2Vec job recommendations.
"""
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, '.')

from recommend_service.services.evaluation.baseline_methods import Word2VecRecommender
from recommend_service.database import DatabaseConnection

def main():
    # Load model
    model_path = "models/word2vec/baomoi_400d.model.bin"
    print("Loading Word2Vec model...")
    recommender = Word2VecRecommender(model_path=model_path)
    print("Model loaded!")

    # Get a sample CV from database
    db = DatabaseConnection()
    with db.get_cursor() as cursor:
        # Get CV
        cursor.execute('''
            SELECT c.id, c.title, c.summary, c."currentPosition"
            FROM cvs c
            WHERE c.title IS NOT NULL
            LIMIT 1
        ''')
        cv_row = cursor.fetchone()
        cv_id = cv_row['id']

        # Get skills for this CV
        cursor.execute('''
            SELECT "skillName" FROM cv_skills WHERE "cvId" = %s
        ''', (cv_id,))
        skills = [s['skillName'] for s in cursor.fetchall()]

        # Get work experiences
        cursor.execute('''
            SELECT title, company, description
            FROM work_experiences WHERE "cvId" = %s
        ''', (cv_id,))
        experiences = cursor.fetchall()

        # Build CV dict for recommender
        cv = {
            'id': cv_id,
            'title': cv_row['title'],
            'summary': cv_row['summary'] or '',
            'currentPosition': cv_row['currentPosition'] or '',
            'skills': ', '.join(skills),
            'experience': ' '.join([
                f"{exp['title']} {exp['description'] or ''}"
                for exp in experiences
            ])
        }

    print(f"\n{'='*60}")
    print("TEST CV")
    print('='*60)
    print(f"ID: {cv['id']}")
    print(f"Title: {cv['title']}")
    print(f"Current Position: {cv['currentPosition']}")
    print(f"Skills: {cv['skills'][:100]}..." if len(cv['skills']) > 100 else f"Skills: {cv['skills']}")

    # Get jobs
    with db.get_cursor() as cursor:
        cursor.execute('''
            SELECT j.id, j.title, j.description
            FROM jobs j
            WHERE j.status = 'ACTIVE' AND j.title IS NOT NULL
            LIMIT 100
        ''')
        job_rows = cursor.fetchall()

        jobs = []
        for job_row in job_rows:
            job_id = job_row['id']

            # Get job requirements
            cursor.execute('''
                SELECT title, description FROM job_requirements WHERE "jobId" = %s
            ''', (job_id,))
            requirements = [f"{r['title']} {r['description'] or ''}" for r in cursor.fetchall()]

            jobs.append({
                'id': job_id,
                'title': job_row['title'],
                'description': job_row['description'] or '',
                'requirements': ' '.join(requirements)
            })

    print(f"\nLoaded {len(jobs)} jobs for testing")

    # Fit the recommender with jobs (list of (job_id, job_title))
    print("\nFitting Word2Vec recommender...")
    job_tuples = [(j['id'], j['title']) for j in jobs]
    recommender.fit(job_tuples)

    # Build a mapping from job_id to job dict
    job_dict = {j['id']: j for j in jobs}

    # Get recommendations
    print(f"\n{'='*60}")
    print("WORD2VEC RECOMMENDATIONS")
    print('='*60)

    # Use CV title for recommendation
    cv_text = f"{cv['title']} {cv['currentPosition']} {cv['skills']}"
    recommendations = recommender.recommend(cv_text, top_k=10)

    for i, (job_id, score) in enumerate(recommendations, 1):
        job = job_dict[job_id]
        title = job['title'][:55] if len(job['title']) > 55 else job['title']
        print(f"{i:2}. [{score:.4f}] {title}")

if __name__ == "__main__":
    main()
