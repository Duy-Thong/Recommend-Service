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
from typing import Dict, List, Optional

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
        logger.info(f"Hit Rate@5:    {result.hit_rate_at_5:.4f} ({result.num_hits_at_5}/{result.num_queries})")
        logger.info(f"Hit Rate@10:   {result.hit_rate_at_10:.4f} ({result.num_hits_at_10}/{result.num_queries})")
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
            "hit_rate_at_5": 0.891,
            "hit_rate_at_10": 0.934
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
            ("Hit Rate@5", result.hit_rate_at_5, expected["hit_rate_at_5"]),
            ("Hit Rate@10", result.hit_rate_at_10, expected["hit_rate_at_10"]),
        ]

        for name, current, exp in metrics:
            diff = current - exp
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            print(f"{name:<15} {current:<12.4f} {exp:<12.4f} {diff_str:<12}")

        print("=" * 70 + "\n")
