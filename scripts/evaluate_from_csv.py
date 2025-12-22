"""
Evaluate Baseline Methods from CSV with Relaxed Similarity.

Reads recommendations from CSV files and evaluates using relaxed similarity matching.

Usage:
    python scripts/evaluate_from_csv.py --threshold 0.7
    python scripts/evaluate_from_csv.py --threshold 0.7 --methods random tfidf simcse cascade
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database import DatabaseConnection
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def load_job_embeddings(db: DatabaseConnection, job_ids: list) -> dict:
    """Load titleEmbedding for jobs from database."""
    embeddings = {}
    query = """
        SELECT id, "titleEmbedding"
        FROM jobs
        WHERE id = ANY(%s) AND "titleEmbedding" IS NOT NULL
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (job_ids,))
        rows = cursor.fetchall()

    for row in rows:
        emb = row["titleEmbedding"]
        if isinstance(emb, str):
            import json as json_module
            emb = json_module.loads(emb)
        if emb:
            embeddings[str(row["id"])] = np.array(emb)

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def evaluate_method_relaxed(
    ground_truth: dict,
    recommendations: dict,
    gt_embeddings: dict,
    pred_embeddings: dict,
    threshold: float = 0.7
) -> dict:
    """
    Evaluate with relaxed similarity matching.

    Args:
        ground_truth: Dict of cv_id -> gt_job_id
        recommendations: Dict of cv_id -> list of pred_job_ids
        gt_embeddings: Dict of job_id -> embedding
        pred_embeddings: Dict of job_id -> embedding
        threshold: Similarity threshold for matching

    Returns:
        Dict with MRR, NDCG@5, NDCG@10, Hit Rate@5, Hit Rate@10, Hit Rate@30
    """
    reciprocal_ranks = []
    hits_at_5 = 0
    hits_at_10 = 0
    hits_at_30 = 0
    dcg_at_5 = []
    dcg_at_10 = []
    dcg_at_30 = []

    for cv_id, gt_job_id in ground_truth.items():
        if cv_id not in recommendations:
            continue

        pred_job_ids = recommendations[cv_id]
        if not pred_job_ids:
            continue

        # Get ground truth embedding
        gt_emb = gt_embeddings.get(gt_job_id)
        if gt_emb is None:
            continue

        # Find first hit position
        hit_rank = None
        for rank, pred_job_id in enumerate(pred_job_ids, 1):
            pred_emb = pred_embeddings.get(pred_job_id)
            if pred_emb is not None:
                sim = cosine_similarity(gt_emb, pred_emb)
                if sim >= threshold:
                    hit_rank = rank
                    break

        # Compute metrics
        if hit_rank:
            reciprocal_ranks.append(1.0 / hit_rank)
            if hit_rank <= 5:
                hits_at_5 += 1
            if hit_rank <= 10:
                hits_at_10 += 1
            if hit_rank <= 30:
                hits_at_30 += 1

            # DCG (relevance = 1 for hit, 0 otherwise)
            if hit_rank <= 5:
                dcg_at_5.append(1.0 / np.log2(hit_rank + 1))
            else:
                dcg_at_5.append(0.0)

            if hit_rank <= 10:
                dcg_at_10.append(1.0 / np.log2(hit_rank + 1))
            else:
                dcg_at_10.append(0.0)

            if hit_rank <= 30:
                dcg_at_30.append(1.0 / np.log2(hit_rank + 1))
            else:
                dcg_at_30.append(0.0)
        else:
            reciprocal_ranks.append(0.0)
            dcg_at_5.append(0.0)
            dcg_at_10.append(0.0)
            dcg_at_30.append(0.0)

    num_queries = len(reciprocal_ranks)
    idcg = 1.0  # Ideal DCG for single relevant item at rank 1

    return {
        "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_5": np.mean(dcg_at_5) / idcg if dcg_at_5 else 0.0,
        "ndcg_at_10": np.mean(dcg_at_10) / idcg if dcg_at_10 else 0.0,
        "ndcg_at_30": np.mean(dcg_at_30) / idcg if dcg_at_30 else 0.0,
        "hit_rate_at_5": hits_at_5 / num_queries if num_queries > 0 else 0.0,
        "hit_rate_at_10": hits_at_10 / num_queries if num_queries > 0 else 0.0,
        "hit_rate_at_30": hits_at_30 / num_queries if num_queries > 0 else 0.0,
        "num_queries": num_queries,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "hits_at_30": hits_at_30
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendations from CSV")
    parser.add_argument("--ground-truth", type=str, default="./evaluation_data/ground_truth.csv",
                        help="Path to ground truth CSV")
    parser.add_argument("--rec-dir", type=str, default="./evaluation_data/recommendations",
                        help="Directory containing recommendation CSV files")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold for relaxed matching")
    parser.add_argument("--methods", nargs="+", default=["random", "tfidf", "simcse", "cascade"],
                        help="Methods to evaluate")
    parser.add_argument("--output", type=str, default="./evaluation_data/comparison_results_relaxed.json",
                        help="Output path for results")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EVALUATION FROM CSV WITH RELAXED SIMILARITY")
    logger.info("=" * 70)
    logger.info(f"Ground truth: {args.ground_truth}")
    logger.info(f"Recommendations dir: {args.rec_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Methods: {args.methods}")
    logger.info("=" * 70)

    # Load ground truth
    logger.info("Loading ground truth...")
    gt_df = pd.read_csv(args.ground_truth)
    ground_truth = dict(zip(gt_df['cv_id'].astype(str), gt_df['job_id'].astype(str)))
    logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

    # Load job embeddings
    db = DatabaseConnection()
    rec_dir = Path(args.rec_dir)

    # Collect all unique job IDs
    all_job_ids = set(ground_truth.values())
    for method in args.methods:
        csv_path = rec_dir / f"{method}.csv"
        if csv_path.exists():
            rec_df = pd.read_csv(csv_path)
            all_job_ids.update(rec_df['job_id'].astype(str).unique())

    logger.info(f"Loading embeddings for {len(all_job_ids)} unique jobs...")
    job_embeddings = load_job_embeddings(db, list(all_job_ids))
    logger.info(f"Loaded {len(job_embeddings)} job embeddings")

    # Evaluate each method
    results = []
    method_names = {
        "random": "Ngẫu nhiên",
        "tfidf": "TF-IDF + Cosine",
        "simcse": "SimCSE (chỉ tiêu đề)",
        "cascade": "Cascade (3 vòng lọc)"
    }

    for method in args.methods:
        csv_path = rec_dir / f"{method}.csv"
        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}, skipping...")
            continue

        logger.info(f"\nEvaluating {method}...")
        rec_df = pd.read_csv(csv_path)

        # Build recommendations dict
        recommendations = {}
        for cv_id, group in rec_df.groupby('cv_id'):
            recommendations[str(cv_id)] = group['job_id'].astype(str).tolist()

        # Evaluate
        metrics = evaluate_method_relaxed(
            ground_truth,
            recommendations,
            job_embeddings,
            job_embeddings,
            threshold=args.threshold
        )

        result = {
            "method": method_names.get(method, method),
            **metrics
        }
        results.append(result)

        logger.info(f"  MRR: {metrics['mrr']:.4f}")
        logger.info(f"  NDCG@5: {metrics['ndcg_at_5']:.4f}")
        logger.info(f"  NDCG@10: {metrics['ndcg_at_10']:.4f}")
        logger.info(f"  Hit Rate@5: {metrics['hit_rate_at_5']:.4f} ({metrics['hits_at_5']}/{metrics['num_queries']})")
        logger.info(f"  Hit Rate@10: {metrics['hit_rate_at_10']:.4f} ({metrics['hits_at_10']}/{metrics['num_queries']})")
        logger.info(f"  Hit Rate@30: {metrics['hit_rate_at_30']:.4f} ({metrics['hits_at_30']}/{metrics['num_queries']})")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth_path": args.ground_truth,
        "threshold": args.threshold,
        "results": results
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
