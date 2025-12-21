"""
Evaluator Service for Recommendation System.

Orchestrates the evaluation process:
1. Load ground truth from CSV
2. Get predictions from recommendation system
3. Compute evaluation metrics
4. Save results
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recommend_service.database import DatabaseConnection, RecommendationRepository
from .metrics import EvaluationMetrics, EvaluationResult

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Main evaluator service that orchestrates the evaluation process.

    Loads ground truth, retrieves predictions from the recommendation system,
    and computes evaluation metrics.
    """

    def __init__(
        self,
        ground_truth_path: str = "./evaluation_data/ground_truth.csv",
        output_path: str = "./evaluation_data/evaluation_results.json"
    ):
        """
        Initialize the evaluator.

        Args:
            ground_truth_path: Path to ground truth CSV file
            output_path: Path to save evaluation results JSON
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.output_path = Path(output_path)

        self.db = DatabaseConnection()
        self.rec_repo = RecommendationRepository(self.db)

    def load_ground_truth(self) -> Dict[str, str]:
        """
        Load ground truth from CSV file.

        Returns:
            Dict mapping cv_id -> job_id
        """
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {self.ground_truth_path}\n"
                "Run 'python scripts/generate_ground_truth.py' first."
            )

        df = pd.read_csv(self.ground_truth_path)

        ground_truth = {}
        for _, row in df.iterrows():
            cv_id = str(row["cv_id"])
            job_id = str(row["job_id"])
            ground_truth[cv_id] = job_id

        logger.info(f"Loaded {len(ground_truth)} ground truth pairs from {self.ground_truth_path}")
        return ground_truth

    def load_ground_truth_with_embeddings(self) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Load ground truth with job embeddings for relaxed matching.

        Returns:
            Tuple of (ground_truth dict, job_embeddings dict)
        """
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {self.ground_truth_path}\n"
                "Run 'python scripts/generate_ground_truth.py' first."
            )

        df = pd.read_csv(self.ground_truth_path)

        ground_truth = {}
        job_ids = []
        for _, row in df.iterrows():
            cv_id = str(row["cv_id"])
            job_id = str(row["job_id"])
            ground_truth[cv_id] = job_id
            job_ids.append(job_id)

        # Load embeddings for ground truth jobs
        job_embeddings = {}
        unique_job_ids = list(set(job_ids))

        query = """
            SELECT id, "titleEmbedding"
            FROM jobs
            WHERE id = ANY(%s) AND "titleEmbedding" IS NOT NULL
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (unique_job_ids,))
            rows = cursor.fetchall()

        for row in rows:
            emb = row["titleEmbedding"]
            if isinstance(emb, str):
                import json as json_module
                emb = json_module.loads(emb)
            if emb:
                job_embeddings[str(row["id"])] = np.array(emb)

        logger.info(f"Loaded {len(ground_truth)} ground truth pairs, {len(job_embeddings)} job embeddings")
        return ground_truth, job_embeddings

    def get_predictions(
        self,
        cv_ids: List[str],
        top_k: int = 30
    ) -> Dict[str, List[str]]:
        """
        Get recommendation predictions for given CVs.

        Uses RecommendationRepository to get recommendations from database.

        Args:
            cv_ids: List of CV IDs to get predictions for
            top_k: Number of top recommendations to retrieve

        Returns:
            Dict mapping cv_id -> list of job_ids (ranked by similarity)
        """
        predictions = {}

        for cv_id in cv_ids:
            try:
                recommendations = self.rec_repo.get_recommendations_for_cv(cv_id, limit=top_k)

                # Extract job IDs from recommendations
                job_ids = [str(rec["jobId"]) for rec in recommendations]
                predictions[cv_id] = job_ids
            except Exception as e:
                logger.warning(f"Failed to get predictions for CV {cv_id}: {e}")
                predictions[cv_id] = []

        # Count CVs with predictions
        cvs_with_predictions = sum(1 for jobs in predictions.values() if jobs)
        logger.info(f"Got predictions for {cvs_with_predictions}/{len(cv_ids)} CVs")

        return predictions

    def get_predictions_with_embeddings(
        self,
        cv_ids: List[str],
        top_k: int = 30
    ) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
        """
        Get predictions with job embeddings for relaxed matching.

        Returns:
            Tuple of (predictions dict, job_embeddings dict)
        """
        predictions = {}
        all_job_ids = set()

        for cv_id in cv_ids:
            try:
                recommendations = self.rec_repo.get_recommendations_for_cv(cv_id, limit=top_k)
                job_ids = [str(rec["jobId"]) for rec in recommendations]
                predictions[cv_id] = job_ids
                all_job_ids.update(job_ids)
            except Exception as e:
                logger.warning(f"Failed to get predictions for CV {cv_id}: {e}")
                predictions[cv_id] = []

        # Load embeddings for all predicted jobs
        job_embeddings = {}
        if all_job_ids:
            query = """
                SELECT id, "titleEmbedding"
                FROM jobs
                WHERE id = ANY(%s) AND "titleEmbedding" IS NOT NULL
            """
            with self.db.get_cursor() as cursor:
                cursor.execute(query, (list(all_job_ids),))
                rows = cursor.fetchall()

            for row in rows:
                emb = row["titleEmbedding"]
                if isinstance(emb, str):
                    import json as json_module
                    emb = json_module.loads(emb)
                if emb:
                    job_embeddings[str(row["id"])] = np.array(emb)

        cvs_with_predictions = sum(1 for jobs in predictions.values() if jobs)
        logger.info(f"Got predictions for {cvs_with_predictions}/{len(cv_ids)} CVs, {len(job_embeddings)} job embeddings")

        return predictions, job_embeddings

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def evaluate(self) -> EvaluationResult:
        """
        Run full evaluation.

        Returns:
            EvaluationResult with all metrics
        """
        # Load ground truth
        logger.info("Loading ground truth...")
        ground_truth = self.load_ground_truth()
        logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

        # Get predictions
        logger.info("Getting predictions from recommendation system...")
        cv_ids = list(ground_truth.keys())
        predictions = self.get_predictions(cv_ids, top_k=30)

        # Filter out CVs without predictions for cleaner evaluation
        valid_ground_truth = {
            cv_id: job_id
            for cv_id, job_id in ground_truth.items()
            if cv_id in predictions and predictions[cv_id]
        }

        if len(valid_ground_truth) < len(ground_truth):
            logger.warning(
                f"Only {len(valid_ground_truth)}/{len(ground_truth)} CVs have predictions. "
                "Make sure to run the recommendation pipeline first."
            )

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        result = EvaluationMetrics.evaluate(valid_ground_truth, predictions)

        return result

    def evaluate_relaxed(self, similarity_threshold: float = 0.85) -> EvaluationResult:
        """
        Run evaluation with relaxed matching based on title embedding similarity.

        Instead of exact ID match, a prediction is considered a "hit" if any
        recommended job has title embedding similarity >= threshold with ground truth job.

        Args:
            similarity_threshold: Minimum cosine similarity to consider a match (default: 0.85)

        Returns:
            EvaluationResult with all metrics
        """
        # Load ground truth with embeddings
        logger.info("Loading ground truth with embeddings...")
        ground_truth, gt_job_embeddings = self.load_ground_truth_with_embeddings()
        logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

        # Get predictions with embeddings
        logger.info("Getting predictions with embeddings...")
        cv_ids = list(ground_truth.keys())
        predictions, pred_job_embeddings = self.get_predictions_with_embeddings(cv_ids, top_k=30)

        # Filter out CVs without predictions
        valid_cv_ids = [
            cv_id for cv_id in ground_truth.keys()
            if cv_id in predictions and predictions[cv_id]
        ]

        if len(valid_cv_ids) < len(ground_truth):
            logger.warning(
                f"Only {len(valid_cv_ids)}/{len(ground_truth)} CVs have predictions."
            )

        # For relaxed matching, we need to find the best matching predicted job
        # for each ground truth job based on embedding similarity
        logger.info(f"Computing relaxed matching with threshold={similarity_threshold}...")

        # Create "virtual" predictions where we replace job IDs with the ground truth job ID
        # if similarity >= threshold (to reuse existing metrics)
        relaxed_predictions = {}

        for cv_id in valid_cv_ids:
            gt_job_id = ground_truth[cv_id]
            pred_job_ids = predictions[cv_id]

            # Get ground truth job embedding
            gt_emb = gt_job_embeddings.get(gt_job_id)
            if gt_emb is None:
                relaxed_predictions[cv_id] = pred_job_ids
                continue

            # Find first matching position in predictions (only count one hit)
            relaxed_list = []
            first_hit_found = False
            for pred_job_id in pred_job_ids:
                if first_hit_found:
                    # Already found a hit, keep remaining predictions as-is
                    relaxed_list.append(pred_job_id)
                    continue

                pred_emb = pred_job_embeddings.get(pred_job_id)
                if pred_emb is not None:
                    sim = self.cosine_similarity(gt_emb, pred_emb)
                    if sim >= similarity_threshold:
                        # This is a "hit" - use ground truth ID to mark as match
                        relaxed_list.append(gt_job_id)
                        first_hit_found = True
                    else:
                        relaxed_list.append(pred_job_id)
                else:
                    relaxed_list.append(pred_job_id)

            relaxed_predictions[cv_id] = relaxed_list

        # Build valid ground truth
        valid_ground_truth = {cv_id: ground_truth[cv_id] for cv_id in valid_cv_ids}

        # Compute metrics using relaxed predictions
        logger.info("Computing evaluation metrics with relaxed matching...")
        result = EvaluationMetrics.evaluate(valid_ground_truth, relaxed_predictions)

        return result

    def get_similar_jobs_map(
        self,
        job_ids: List[str],
        threshold: float = 0.8
    ) -> Dict[str, set]:
        """
        Get similar jobs for a list of job IDs from the similar_jobs table.

        Args:
            job_ids: List of job IDs to find similar jobs for
            threshold: Minimum similarity threshold (default: 0.8)

        Returns:
            Dict mapping job_id -> set of similar job IDs
        """
        if not job_ids:
            return {}

        query = """
            SELECT "jobId", "similarJobId", similarity
            FROM similar_jobs
            WHERE "jobId" = ANY(%s) AND similarity >= %s
        """
        similar_map = {job_id: set() for job_id in job_ids}

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (job_ids, threshold))
            rows = cursor.fetchall()

        for row in rows:
            job_id = str(row["jobId"])
            similar_job_id = str(row["similarJobId"])
            if job_id in similar_map:
                similar_map[job_id].add(similar_job_id)

        # Also add reverse relationships
        query_reverse = """
            SELECT "jobId", "similarJobId", similarity
            FROM similar_jobs
            WHERE "similarJobId" = ANY(%s) AND similarity >= %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query_reverse, (job_ids, threshold))
            rows = cursor.fetchall()

        for row in rows:
            similar_job_id = str(row["similarJobId"])
            job_id = str(row["jobId"])
            if similar_job_id in similar_map:
                similar_map[similar_job_id].add(job_id)

        return similar_map

    def check_job_similarity(
        self,
        pred_job_id: str,
        gt_job_id: str,
        threshold: float = 0.8
    ) -> bool:
        """
        Check if two jobs are similar based on similar_jobs table.

        Args:
            pred_job_id: Predicted job ID
            gt_job_id: Ground truth job ID
            threshold: Minimum similarity threshold

        Returns:
            True if jobs are similar (in either direction)
        """
        query = """
            SELECT 1 FROM similar_jobs
            WHERE (("jobId" = %s AND "similarJobId" = %s)
                   OR ("jobId" = %s AND "similarJobId" = %s))
                  AND similarity >= %s
            LIMIT 1
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (pred_job_id, gt_job_id, gt_job_id, pred_job_id, threshold))
            return cursor.fetchone() is not None

    def evaluate_with_similar_jobs(self, similarity_threshold: float = 0.8) -> EvaluationResult:
        """
        Run evaluation using the similar_jobs table for relaxed matching.

        A prediction is considered a "hit" if:
        1. The predicted job ID exactly matches the ground truth job ID, OR
        2. The predicted job is similar to the ground truth job (in similar_jobs table)

        Args:
            similarity_threshold: Minimum similarity in similar_jobs table (default: 0.8)

        Returns:
            EvaluationResult with all metrics
        """
        # Load ground truth
        logger.info("Loading ground truth...")
        ground_truth = self.load_ground_truth()
        logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

        # Get predictions
        logger.info("Getting predictions from recommendation system...")
        cv_ids = list(ground_truth.keys())
        predictions = self.get_predictions(cv_ids, top_k=30)

        # Filter out CVs without predictions
        valid_cv_ids = [
            cv_id for cv_id in ground_truth.keys()
            if cv_id in predictions and predictions[cv_id]
        ]

        if len(valid_cv_ids) < len(ground_truth):
            logger.warning(
                f"Only {len(valid_cv_ids)}/{len(ground_truth)} CVs have predictions."
            )

        # Collect all unique job IDs (both ground truth and predictions)
        gt_job_ids = set(ground_truth[cv_id] for cv_id in valid_cv_ids)
        all_pred_job_ids = set()
        for cv_id in valid_cv_ids:
            all_pred_job_ids.update(predictions[cv_id])

        all_job_ids = list(gt_job_ids | all_pred_job_ids)
        logger.info(f"Loading similar jobs for {len(all_job_ids)} unique jobs (threshold={similarity_threshold})...")

        # Build a similarity lookup: (job_a, job_b) -> True if similar
        similarity_pairs = set()
        query = """
            SELECT "jobId", "similarJobId"
            FROM similar_jobs
            WHERE ("jobId" = ANY(%s) OR "similarJobId" = ANY(%s))
                  AND similarity >= %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (all_job_ids, all_job_ids, similarity_threshold))
            rows = cursor.fetchall()

        for row in rows:
            job_a = str(row["jobId"])
            job_b = str(row["similarJobId"])
            similarity_pairs.add((job_a, job_b))
            similarity_pairs.add((job_b, job_a))  # Add reverse

        logger.info(f"Loaded {len(rows)} similar job pairs")

        # Create relaxed predictions
        logger.info("Computing relaxed matching with similar_jobs table...")
        relaxed_predictions = {}
        hits_found = 0

        for cv_id in valid_cv_ids:
            gt_job_id = ground_truth[cv_id]
            pred_job_ids = predictions[cv_id]

            # Find first matching position
            relaxed_list = []
            first_hit_found = False

            for pred_job_id in pred_job_ids:
                if first_hit_found:
                    relaxed_list.append(pred_job_id)
                    continue

                # Check if this prediction matches ground truth or is similar
                is_similar = (pred_job_id, gt_job_id) in similarity_pairs
                if pred_job_id == gt_job_id or is_similar:
                    # This is a "hit"
                    relaxed_list.append(gt_job_id)
                    first_hit_found = True
                    hits_found += 1
                else:
                    relaxed_list.append(pred_job_id)

            relaxed_predictions[cv_id] = relaxed_list

        logger.info(f"Found {hits_found} hits in relaxed matching")

        # Build valid ground truth
        valid_ground_truth = {cv_id: ground_truth[cv_id] for cv_id in valid_cv_ids}

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        result = EvaluationMetrics.evaluate(valid_ground_truth, relaxed_predictions)

        return result

    def save_results(self, result: EvaluationResult) -> str:
        """
        Save evaluation results to JSON file.

        Args:
            result: EvaluationResult to save

        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = {
            "timestamp": datetime.now().isoformat(),
            "ground_truth_path": str(self.ground_truth_path),
            "results": result.to_dict()
        }

        # Save to JSON
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation results to {self.output_path}")
        return str(self.output_path)

    def run(self) -> EvaluationResult:
        """
        Run evaluation and save results.

        Returns:
            EvaluationResult with all metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Evaluation")
        logger.info("=" * 60)

        # Run evaluation
        result = self.evaluate()

        # Log results
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"MRR:           {result.mrr:.4f}")
        logger.info(f"NDCG@5:        {result.ndcg_at_5:.4f}")
        logger.info(f"NDCG@10:       {result.ndcg_at_10:.4f}")
        logger.info(f"NDCG@30:       {result.ndcg_at_30:.4f}")
        logger.info(f"Hit Rate@5:    {result.hit_rate_at_5:.4f} ({result.num_hits_at_5}/{result.num_queries})")
        logger.info(f"Hit Rate@10:   {result.hit_rate_at_10:.4f} ({result.num_hits_at_10}/{result.num_queries})")
        logger.info(f"Hit Rate@30:   {result.hit_rate_at_30:.4f} ({result.num_hits_at_30}/{result.num_queries})")
        logger.info("=" * 60)

        # Save results
        self.save_results(result)

        return result

    def print_comparison_table(self, result: EvaluationResult) -> None:
        """
        Print comparison with expected results from section_54_danh_gia.tex.

        Args:
            result: Current evaluation result
        """
        # Expected results from the report
        expected = {
            "mrr": 0.847,
            "ndcg_at_5": 0.782,
            "ndcg_at_10": 0.814,
            "ndcg_at_30": 0.850,
            "hit_rate_at_5": 0.891,
            "hit_rate_at_10": 0.934,
            "hit_rate_at_30": 0.970
        }

        print("\n" + "=" * 70)
        print("COMPARISON WITH EXPECTED RESULTS (from section_54_danh_gia.tex)")
        print("=" * 70)
        print(f"{'Metric':<15} {'Current':<12} {'Expected':<12} {'Difference':<12}")
        print("-" * 70)

        metrics = [
            ("MRR", result.mrr, expected["mrr"]),
            ("NDCG@5", result.ndcg_at_5, expected["ndcg_at_5"]),
            ("NDCG@10", result.ndcg_at_10, expected["ndcg_at_10"]),
            ("NDCG@30", result.ndcg_at_30, expected["ndcg_at_30"]),
            ("Hit Rate@5", result.hit_rate_at_5, expected["hit_rate_at_5"]),
            ("Hit Rate@10", result.hit_rate_at_10, expected["hit_rate_at_10"]),
            ("Hit Rate@30", result.hit_rate_at_30, expected["hit_rate_at_30"]),
        ]

        for name, current, exp in metrics:
            diff = current - exp
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            print(f"{name:<15} {current:<12.4f} {exp:<12.4f} {diff_str:<12}")

        print("=" * 70 + "\n")
