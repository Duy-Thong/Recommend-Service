"""
Generate Recommendations for Ablation Study (1, 2, 3 layer cascade).

Generates recommendations for different cascade configurations:
- cascade_1layer.csv: Title only (FAISS)
- cascade_2layer.csv: Title + Experience
- cascade_3layer.csv: Title + Experience + Skills (full cascade - same as cascade.csv)

Output format: cv_id, job_id, similarity, rank

Usage:
    python scripts/generate_ablation_recommendations.py
    python scripts/generate_ablation_recommendations.py --cv-limit 1000
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.config import settings
from recommend_service.database import DatabaseConnection, CVRepository, JobRepository
from recommend_service.models import CVData, JobData
from recommend_service.services.embedding import EmbeddingService
from recommend_service.services.similarity import SimilarityService
from recommend_service.services.evaluation.baseline_methods import CascadeRecommender

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study recommendations")
    parser.add_argument("--cv-limit", type=int, default=5000, help="Max CVs to process")
    parser.add_argument("--job-limit", type=int, default=5000, help="Max jobs to use")
    parser.add_argument("--top-k", type=int, default=30, help="Number of recommendations per CV")
    parser.add_argument("--output-dir", type=str, default="./evaluation_data/recommendations",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ABLATION STUDY RECOMMENDATIONS GENERATION")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"CV limit: {args.cv_limit}")
    logger.info(f"Job limit: {args.job_limit}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 70)

    # Load data
    db = DatabaseConnection()
    cv_repo = CVRepository(db)
    job_repo = JobRepository(db)

    logger.info("Loading CVs...")
    raw_cvs = cv_repo.get_main_cvs()[:args.cv_limit]

    # Batch load all CV skills and experiences
    logger.info("Loading CV skills and experiences in batch...")
    all_cv_skills = cv_repo.get_all_cv_skills()
    all_cv_experiences = cv_repo.get_all_cv_experiences()

    cvs = []
    for raw_cv in tqdm(raw_cvs, desc="Processing CVs"):
        cv_id = raw_cv["id"]
        title = raw_cv.get("title", "")
        if not title:
            continue

        skills = all_cv_skills.get(cv_id, [])
        experiences = all_cv_experiences.get(cv_id, [])
        cv = CVData.from_db_row(raw_cv, skills, experiences)
        cvs.append(cv)

    logger.info(f"Loaded {len(cvs)} CVs")

    logger.info("Loading Jobs...")
    raw_jobs = job_repo.get_active_jobs()[:args.job_limit]

    # Batch load all job skills and requirements
    logger.info("Loading job skills and requirements in batch...")
    all_job_skills = job_repo.get_all_job_skills()
    all_job_requirements = job_repo.get_all_job_requirements()

    jobs = []
    for raw_job in tqdm(raw_jobs, desc="Processing Jobs"):
        job_id = raw_job["id"]
        skills = all_job_skills.get(job_id, [])
        requirements = all_job_requirements.get(job_id, [])
        job = JobData.from_db_row(raw_job, skills, requirements)
        jobs.append(job)

    logger.info(f"Loaded {len(jobs)} jobs")

    # Initialize similarity service
    logger.info("Building similarity service...")
    similarity_service = SimilarityService()
    similarity_service.build_index(jobs)

    # Generate for each layer configuration
    for num_layers in [1, 2, 3]:
        logger.info("\n" + "=" * 70)
        logger.info(f"Generating {num_layers}-layer cascade recommendations...")
        logger.info("=" * 70)

        if num_layers == 3:
            # For 3-layer (full cascade), export directly from database table
            logger.info("Exporting 3-layer recommendations from database (recommend_jobs_for_cv table)...")
            with db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT
                        "cvId" as cv_id,
                        "jobId" as job_id,
                        similarity,
                        ROW_NUMBER() OVER (PARTITION BY "cvId" ORDER BY similarity DESC) as rank
                    FROM recommend_jobs_for_cv
                    ORDER BY "cvId", rank
                """)
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]

            df = pd.DataFrame(data)
            output_path = output_dir / f"cascade_{num_layers}layer.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(data)} recommendations from database to {output_path}")
        else:
            # For 1-layer and 2-layer, run the algorithm
            recommender = CascadeRecommender(num_layers=num_layers)
            recommender.fit(jobs, similarity_service)

            data = []
            for cv in tqdm(cvs, desc=f"{num_layers}-layer"):
                try:
                    recommendations = recommender.recommend(cv, top_k=args.top_k)
                    for rank, (job_id, similarity) in enumerate(recommendations, 1):
                        data.append({
                            'cv_id': cv.id,
                            'job_id': job_id,
                            'similarity': similarity,
                            'rank': rank
                        })
                except Exception as e:
                    logger.warning(f"Failed to get recommendations for CV {cv.id}: {e}")
                    continue

            df = pd.DataFrame(data)
            output_path = output_dir / f"cascade_{num_layers}layer.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(data)} recommendations to {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY GENERATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"All recommendations saved to {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
