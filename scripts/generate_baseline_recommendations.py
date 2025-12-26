"""
Generate Recommendations for All Baseline Methods and Ablation Study.

Generates recommendations for all methods and saves to CSV files:
- random.csv: Random recommendations
- tfidf.csv: TF-IDF + Cosine recommendations
- simcse.csv: SimCSE (title only) recommendations
- cascade_1layer.csv: Cascade 1 layer (title only - ablation)
- cascade_2layer.csv: Cascade 2 layers (title + experience - ablation)
- cascade_3layer.csv: Cascade 3 layers (title + exp + skills - ablation)
- cascade.csv: Cascade (from DB - recommend_jobs_for_cv)

Output format: cv_id, job_id, similarity, rank

Usage:
    python scripts/generate_baseline_recommendations.py
    python scripts/generate_baseline_recommendations.py --cv-limit 1000 --job-limit 5000
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
from recommend_service.services.evaluation.baseline_methods import (
    RandomRecommender,
    TFIDFRecommender,
    JaccardRecommender,
    Word2VecRecommender,
    CascadeRecommender
)

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
    parser = argparse.ArgumentParser(description="Generate baseline recommendations")
    parser.add_argument("--cv-limit", type=int, default=10000, help="Max CVs to scan (will filter to ground truth)")
    parser.add_argument("--top-k", type=int, default=30, help="Number of recommendations per CV")
    parser.add_argument("--output-dir", type=str, default="./evaluation_data/recommendations",
                        help="Output directory for CSV files")
    parser.add_argument("--word2vec-model", type=str, default=None,
                        help="Path to Word2Vec model file (.bin or .bin.gz)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BASELINE RECOMMENDATIONS GENERATION")
    logger.info("=" * 70)
    logger.info(f"Database: {settings.database_url_clean}")
    logger.info(f"CV limit (for scanning): {args.cv_limit}")
    logger.info(f"Jobs: ALL ACTIVE jobs in DB")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 70)

    # Load ground truth to get CV IDs
    logger.info("Loading ground truth...")
    ground_truth_path = Path(__file__).parent.parent / "evaluation_data" / "ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    gt_cv_ids = set(ground_truth_df['cv_id'].unique())
    logger.info(f"Found {len(gt_cv_ids)} unique CVs in ground truth")

    # Load data
    db = DatabaseConnection()
    cv_repo = CVRepository(db)
    job_repo = JobRepository(db)

    logger.info("Loading CVs from ground truth...")
    raw_cvs = cv_repo.get_main_cvs()[:args.cv_limit]

    # Batch load all CV skills and experiences
    logger.info("Loading CV skills and experiences in batch...")
    all_cv_skills = cv_repo.get_all_cv_skills()
    all_cv_experiences = cv_repo.get_all_cv_experiences()

    cvs = []
    cv_titles = []
    for raw_cv in tqdm(raw_cvs, desc="Processing CVs"):
        cv_id = raw_cv["id"]
        # Only include CVs from ground truth
        if cv_id not in gt_cv_ids:
            continue

        title = raw_cv.get("title", "")
        if not title:
            continue

        skills = all_cv_skills.get(cv_id, [])
        experiences = all_cv_experiences.get(cv_id, [])
        cv = CVData.from_db_row(raw_cv, skills, experiences)
        cvs.append(cv)
        cv_titles.append((cv_id, title))

    logger.info(f"Loaded {len(cvs)} CVs (from ground truth)")

    logger.info("Loading ALL active Jobs...")
    raw_jobs = job_repo.get_active_jobs()  # Load ALL active jobs

    # Batch load all job skills and requirements
    logger.info("Loading job skills and requirements in batch...")
    all_job_skills = job_repo.get_all_job_skills()
    all_job_requirements = job_repo.get_all_job_requirements()

    jobs = []
    job_titles = []
    for raw_job in tqdm(raw_jobs, desc="Processing Jobs"):
        job_id = raw_job["id"]
        title = raw_job.get("title", "")

        skills = all_job_skills.get(job_id, [])
        requirements = all_job_requirements.get(job_id, [])
        job = JobData.from_db_row(raw_job, skills, requirements)
        jobs.append(job)
        job_titles.append((job_id, title))

    logger.info(f"Loaded {len(jobs)} jobs (ALL ACTIVE)")

    cv_ids = [cv_id for cv_id, _ in cv_titles]

    # 1. Random
    logger.info("\n" + "=" * 70)
    logger.info("Generating Random recommendations...")
    logger.info("=" * 70)
    random_recommender = RandomRecommender(seed=42)
    random_recommender.fit([job_id for job_id, _ in job_titles])

    random_data = []
    for cv_id in tqdm(cv_ids, desc="Random"):
        job_ids = random_recommender.recommend(cv_id, top_k=args.top_k)
        for rank, job_id in enumerate(job_ids, 1):
            random_data.append({
                'cv_id': cv_id,
                'job_id': job_id,
                'similarity': 0.0,  # Random has no similarity score
                'rank': rank
            })

    random_df = pd.DataFrame(random_data)
    random_path = output_dir / "random.csv"
    random_df.to_csv(random_path, index=False)
    logger.info(f"Saved {len(random_data)} recommendations to {random_path}")

    # 2. TF-IDF
    logger.info("\n" + "=" * 70)
    logger.info("Generating TF-IDF recommendations...")
    logger.info("=" * 70)
    tfidf_recommender = TFIDFRecommender()
    tfidf_recommender.fit(job_titles)

    tfidf_data = []
    for cv_id, cv_title in tqdm(cv_titles, desc="TF-IDF"):
        recommendations = tfidf_recommender.recommend(cv_title, top_k=args.top_k)
        for rank, (job_id, similarity) in enumerate(recommendations, 1):
            tfidf_data.append({
                'cv_id': cv_id,
                'job_id': job_id,
                'similarity': similarity,
                'rank': rank
            })

    tfidf_df = pd.DataFrame(tfidf_data)
    tfidf_path = output_dir / "tfidf.csv"
    tfidf_df.to_csv(tfidf_path, index=False)
    logger.info(f"Saved {len(tfidf_data)} recommendations to {tfidf_path}")

    # 3. Jaccard Similarity
    logger.info("\n" + "=" * 70)
    logger.info("Generating Jaccard Similarity recommendations...")
    logger.info("=" * 70)
    jaccard_recommender = JaccardRecommender()
    jaccard_recommender.fit(job_titles)

    jaccard_data = []
    for cv_id, cv_title in tqdm(cv_titles, desc="Jaccard"):
        recommendations = jaccard_recommender.recommend(cv_title, top_k=args.top_k)
        for rank, (job_id, similarity) in enumerate(recommendations, 1):
            jaccard_data.append({
                'cv_id': cv_id,
                'job_id': job_id,
                'similarity': similarity,
                'rank': rank
            })

    jaccard_df = pd.DataFrame(jaccard_data)
    jaccard_path = output_dir / "jaccard.csv"
    jaccard_df.to_csv(jaccard_path, index=False)
    logger.info(f"Saved {len(jaccard_data)} recommendations to {jaccard_path}")

    # 4. Word2Vec (if model provided)
    if args.word2vec_model:
        logger.info("\n" + "=" * 70)
        logger.info("Generating Word2Vec recommendations...")
        logger.info("=" * 70)

        try:
            word2vec_recommender = Word2VecRecommender(model_path=args.word2vec_model)
            word2vec_recommender.fit(job_titles)

            word2vec_data = []
            for cv_id, cv_title in tqdm(cv_titles, desc="Word2Vec"):
                recommendations = word2vec_recommender.recommend(cv_title, top_k=args.top_k)
                for rank, (job_id, similarity) in enumerate(recommendations, 1):
                    word2vec_data.append({
                        'cv_id': cv_id,
                        'job_id': job_id,
                        'similarity': similarity,
                        'rank': rank
                    })

            word2vec_df = pd.DataFrame(word2vec_data)
            word2vec_path = output_dir / "word2vec.csv"
            word2vec_df.to_csv(word2vec_path, index=False)
            logger.info(f"Saved {len(word2vec_data)} recommendations to {word2vec_path}")
        except Exception as e:
            logger.error(f"Failed to generate Word2Vec recommendations: {e}")
            logger.warning("Skipping Word2Vec baseline")
    else:
        logger.info("\n" + "=" * 70)
        logger.info("Skipping Word2Vec (no model path provided)")
        logger.info("Use --word2vec-model to specify model path")
        logger.info("=" * 70)

    # 5-6. Cascade Ablation Study (1, 2 layers)
    logger.info("\n" + "=" * 70)
    logger.info("Initializing similarity service for cascade...")
    logger.info("=" * 70)
    embedding_service = EmbeddingService()
    similarity_service = SimilarityService()
    similarity_service.build_index(jobs)

    # Only generate 1 and 2 layers (3 layers = full cascade from DB)
    for num_layers in [1, 2]:
        logger.info("\n" + "=" * 70)
        logger.info(f"Generating Cascade {num_layers}-layer recommendations (ablation)...")
        logger.info("=" * 70)

        cascade_recommender = CascadeRecommender(num_layers=num_layers)
        cascade_recommender.fit(jobs, similarity_service)

        cascade_layer_data = []
        for cv in tqdm(cvs, desc=f"Cascade {num_layers}-layer"):
            try:
                recommendations = cascade_recommender.recommend(cv, top_k=args.top_k)
                for rank, (job_id, similarity) in enumerate(recommendations, 1):
                    cascade_layer_data.append({
                        'cv_id': cv.id,
                        'job_id': job_id,
                        'similarity': similarity,
                        'rank': rank
                    })
            except Exception as e:
                logger.warning(f"Failed to get recommendations for CV {cv.id}: {e}")
                continue

        cascade_layer_df = pd.DataFrame(cascade_layer_data)
        cascade_layer_path = output_dir / f"cascade_{num_layers}layer.csv"
        cascade_layer_df.to_csv(cascade_layer_path, index=False)
        logger.info(f"Saved {len(cascade_layer_data)} recommendations to {cascade_layer_path}")

    # 3. Cascade 3-layer (from DB - full cascade)
    logger.info("\n" + "=" * 70)
    logger.info("Exporting Cascade recommendations from DB...")
    logger.info("=" * 70)
    cascade_data = []
    with db.get_cursor() as cursor:
        for cv_id in tqdm(cv_ids, desc="Cascade"):
            cursor.execute('''
                SELECT "jobId", similarity
                FROM recommend_jobs_for_cv
                WHERE "cvId" = %s
                ORDER BY similarity DESC
                LIMIT %s
            ''', (cv_id, args.top_k))
            rows = cursor.fetchall()
            for rank, row in enumerate(rows, 1):
                cascade_data.append({
                    'cv_id': cv_id,
                    'job_id': row['jobId'],
                    'similarity': row['similarity'],
                    'rank': rank
                })

    cascade_df = pd.DataFrame(cascade_data)
    cascade_path = output_dir / "cascade.csv"
    cascade_df.to_csv(cascade_path, index=False)
    logger.info(f"Saved {len(cascade_data)} recommendations to {cascade_path}")

    logger.info("\n" + "=" * 70)
    logger.info("GENERATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"All recommendations saved to {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
