"""
Comparison Evaluator for Baseline Methods and Ablation Study.

Implements the comparison tables from section_54_danh_gia.tex:
- Bảng 2: So sánh với phương pháp cơ sở (Random, TF-IDF, SimCSE title-only, Cascade)
- Bảng 3: Ảnh hưởng của các vòng lọc (1-layer, 2-layer, 3-layer)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from recommend_service.database import DatabaseConnection, CVRepository, JobRepository
from recommend_service.models import CVData, JobData
from recommend_service.services.embedding import EmbeddingService
from recommend_service.services.similarity import SimilarityService

from .baseline_methods import (
    RandomRecommender,
    TFIDFRecommender,
    TitleOnlyRecommender,
    CascadeRecommender
)
from .metrics import EvaluationMetrics, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Result for a single method evaluation."""
    method_name: str
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    avg_time_ms: float  # Average time per query in milliseconds
    num_queries: int


@dataclass
class ComparisonResult:
    """Results for all method comparisons."""
    baseline_comparison: List[MethodResult] = field(default_factory=list)  # Bảng 2
    ablation_study: List[MethodResult] = field(default_factory=list)  # Bảng 3

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "baseline_comparison": [
                {
                    "method": r.method_name,
                    "mrr": r.mrr,
                    "ndcg_at_5": r.ndcg_at_5,
                    "ndcg_at_10": r.ndcg_at_10,
                    "hit_rate_at_5": r.hit_rate_at_5,
                    "hit_rate_at_10": r.hit_rate_at_10,
                    "avg_time_ms": r.avg_time_ms,
                    "num_queries": r.num_queries
                }
                for r in self.baseline_comparison
            ],
            "ablation_study": [
                {
                    "config": r.method_name,
                    "ndcg_at_10": r.ndcg_at_10,
                    "hit_rate_at_10": r.hit_rate_at_10,
                    "avg_time_ms": r.avg_time_ms
                }
                for r in self.ablation_study
            ]
        }


class ComparisonEvaluator:
    """
    Evaluator for comparing multiple recommendation methods.

    Runs all baseline methods and ablation configurations, then computes
    evaluation metrics for each.
    """

    def __init__(
        self,
        ground_truth_path: str = "./evaluation_data/ground_truth.csv",
        output_path: str = "./evaluation_data/comparison_results.json",
        cv_limit: int = 2000,
        job_limit: int = 5000
    ):
        """
        Initialize comparison evaluator.

        Args:
            ground_truth_path: Path to ground truth CSV file
            output_path: Path to save comparison results
            cv_limit: Maximum number of CVs to evaluate
            job_limit: Maximum number of jobs to use
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.output_path = Path(output_path)
        self.cv_limit = cv_limit
        self.job_limit = job_limit

        self.db = DatabaseConnection()
        self.cv_repo = CVRepository(self.db)
        self.job_repo = JobRepository(self.db)

        # Data containers
        self.cvs: List[CVData] = []
        self.jobs: List[JobData] = []
        self.cv_titles: List[Tuple[str, str]] = []  # (cv_id, title)
        self.job_titles: List[Tuple[str, str]] = []  # (job_id, title)
        self.ground_truth: Dict[str, str] = {}

    def load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth from CSV file."""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {self.ground_truth_path}\n"
                "Run 'python scripts/generate_ground_truth.py' first."
            )

        df = pd.read_csv(self.ground_truth_path)
        ground_truth = {str(row["cv_id"]): str(row["job_id"]) for _, row in df.iterrows()}

        logger.info(f"Loaded {len(ground_truth)} ground truth pairs")
        return ground_truth

    def load_data(self) -> None:
        """Load CVs and Jobs from database."""
        logger.info("Loading data from database...")

        # Load CVs
        raw_cvs = self.cv_repo.get_main_cvs()[:self.cv_limit]
        self.cvs = []
        self.cv_titles = []

        for raw_cv in raw_cvs:
            cv_id = raw_cv["id"]
            title = raw_cv.get("title", "")

            skills = self.cv_repo.get_cv_skills(cv_id)
            experiences = self.cv_repo.get_cv_experiences(cv_id)

            cv = CVData.from_db_row(raw_cv, skills, experiences)
            self.cvs.append(cv)
            self.cv_titles.append((cv_id, title))

        logger.info(f"Loaded {len(self.cvs)} CVs")

        # Load Jobs
        raw_jobs = self.job_repo.get_active_jobs()[:self.job_limit]
        self.jobs = []
        self.job_titles = []

        for raw_job in raw_jobs:
            job_id = raw_job["id"]
            title = raw_job.get("title", "")

            skills = self.job_repo.get_job_skills(job_id)
            requirements = self.job_repo.get_job_requirements(job_id)

            job = JobData.from_db_row(raw_job, skills, requirements)
            self.jobs.append(job)
            self.job_titles.append((job_id, title))

        logger.info(f"Loaded {len(self.jobs)} jobs")

    def _evaluate_method(
        self,
        method_name: str,
        predictions: Dict[str, List[str]],
        total_time_ms: float
    ) -> MethodResult:
        """
        Evaluate a method's predictions.

        Args:
            method_name: Name of the method
            predictions: Dict mapping cv_id -> list of job_ids
            total_time_ms: Total time taken in milliseconds

        Returns:
            MethodResult with all metrics
        """
        result = EvaluationMetrics.evaluate(self.ground_truth, predictions)

        return MethodResult(
            method_name=method_name,
            mrr=result.mrr,
            ndcg_at_5=result.ndcg_at_5,
            ndcg_at_10=result.ndcg_at_10,
            hit_rate_at_5=result.hit_rate_at_5,
            hit_rate_at_10=result.hit_rate_at_10,
            avg_time_ms=total_time_ms / len(predictions) if predictions else 0,
            num_queries=result.num_queries
        )

    def evaluate_random(self) -> MethodResult:
        """Evaluate random baseline."""
        logger.info("Evaluating Random baseline...")

        recommender = RandomRecommender(seed=42)
        recommender.fit([job_id for job_id, _ in self.job_titles])

        cv_ids = list(self.ground_truth.keys())

        start_time = time.time()
        predictions = recommender.recommend_batch(cv_ids, top_k=10)
        total_time_ms = (time.time() - start_time) * 1000

        return self._evaluate_method("Ngẫu nhiên", predictions, total_time_ms)

    def evaluate_tfidf(self) -> MethodResult:
        """Evaluate TF-IDF baseline."""
        logger.info("Evaluating TF-IDF + Cosine baseline...")

        recommender = TFIDFRecommender()
        recommender.fit(self.job_titles)

        # Get CV titles for ground truth CVs
        cv_id_to_title = {cv_id: title for cv_id, title in self.cv_titles}
        cvs_to_evaluate = [
            (cv_id, cv_id_to_title.get(cv_id, ""))
            for cv_id in self.ground_truth.keys()
            if cv_id in cv_id_to_title
        ]

        start_time = time.time()
        predictions = recommender.recommend_batch(cvs_to_evaluate, top_k=10)
        total_time_ms = (time.time() - start_time) * 1000

        return self._evaluate_method("TF-IDF + Cosin", predictions, total_time_ms)

    def evaluate_title_only(self, embedding_service: EmbeddingService) -> MethodResult:
        """
        Evaluate title-only embedding baseline.

        Args:
            embedding_service: EmbeddingService instance
        """
        logger.info("Evaluating SimCSE (title only) baseline...")

        recommender = TitleOnlyRecommender()
        recommender.fit(self.job_titles, embedding_service)

        # Get CV titles for ground truth CVs
        cv_id_to_title = {cv_id: title for cv_id, title in self.cv_titles}
        cvs_to_evaluate = [
            (cv_id, cv_id_to_title.get(cv_id, ""))
            for cv_id in self.ground_truth.keys()
            if cv_id in cv_id_to_title
        ]

        start_time = time.time()
        predictions = recommender.recommend_batch(cvs_to_evaluate, top_k=10)
        total_time_ms = (time.time() - start_time) * 1000

        return self._evaluate_method("SimCSE (chỉ tiêu đề)", predictions, total_time_ms)

    def evaluate_cascade(
        self,
        num_layers: int,
        similarity_service: SimilarityService
    ) -> MethodResult:
        """
        Evaluate cascade filtering with specified number of layers.

        Args:
            num_layers: Number of filtering layers (1, 2, or 3)
            similarity_service: SimilarityService instance
        """
        layer_names = {
            1: "Vòng 1 (tiêu đề)",
            2: "Vòng 1 + 2 (+ kinh nghiệm)",
            3: "Vòng 1 + 2 + 3 (+ kỹ năng)"
        }
        logger.info(f"Evaluating {layer_names[num_layers]}...")

        recommender = CascadeRecommender(num_layers=num_layers)
        recommender.fit(self.jobs, similarity_service)

        # Filter CVs to those in ground truth
        cv_id_set = set(self.ground_truth.keys())
        cvs_to_evaluate = [cv for cv in self.cvs if cv.id in cv_id_set]

        start_time = time.time()
        predictions = recommender.recommend_batch(cvs_to_evaluate, top_k=10)
        total_time_ms = (time.time() - start_time) * 1000

        return self._evaluate_method(layer_names[num_layers], predictions, total_time_ms)

    def run_baseline_comparison(self) -> List[MethodResult]:
        """
        Run baseline comparison (Bảng 2).

        Compares: Random, TF-IDF, SimCSE title-only, Cascade (proposed)
        """
        logger.info("=" * 60)
        logger.info("Running Baseline Comparison (Bảng 2)")
        logger.info("=" * 60)

        results = []

        # 1. Random
        results.append(self.evaluate_random())

        # 2. TF-IDF
        results.append(self.evaluate_tfidf())

        # 3. SimCSE (title only) - uses PhoBERT
        embedding_service = EmbeddingService()
        embedding_service.initialize()
        results.append(self.evaluate_title_only(embedding_service))

        # 4. Cascade Filtering (proposed) - full 3 layers
        similarity_service = SimilarityService()
        jobs_with_embeddings = [j for j in self.jobs if j.title_embedding]
        if jobs_with_embeddings:
            similarity_service.build_index(jobs_with_embeddings)
            results.append(self.evaluate_cascade(3, similarity_service))

        return results

    def run_ablation_study(self) -> List[MethodResult]:
        """
        Run ablation study (Bảng 3).

        Compares: 1-layer, 2-layer, 3-layer cascade filtering
        """
        logger.info("=" * 60)
        logger.info("Running Ablation Study (Bảng 3)")
        logger.info("=" * 60)

        results = []

        # Build similarity service
        similarity_service = SimilarityService()
        jobs_with_embeddings = [j for j in self.jobs if j.title_embedding]

        if not jobs_with_embeddings:
            logger.warning("No jobs with embeddings found. Run recommendation pipeline first.")
            return results

        similarity_service.build_index(jobs_with_embeddings)

        # 1-layer (title only)
        results.append(self.evaluate_cascade(1, similarity_service))

        # 2-layer (title + experience)
        results.append(self.evaluate_cascade(2, similarity_service))

        # 3-layer (title + experience + skills)
        results.append(self.evaluate_cascade(3, similarity_service))

        return results

    def run(
        self,
        run_baseline: bool = True,
        run_ablation: bool = True
    ) -> ComparisonResult:
        """
        Run full comparison evaluation.

        Args:
            run_baseline: Whether to run baseline comparison
            run_ablation: Whether to run ablation study

        Returns:
            ComparisonResult with all results
        """
        logger.info("=" * 70)
        logger.info("COMPARISON EVALUATION")
        logger.info("=" * 70)

        # Load data
        self.ground_truth = self.load_ground_truth()
        self.load_data()

        result = ComparisonResult()

        if run_baseline:
            result.baseline_comparison = self.run_baseline_comparison()

        if run_ablation:
            result.ablation_study = self.run_ablation_study()

        # Save results
        self.save_results(result)

        return result

    def save_results(self, result: ComparisonResult) -> str:
        """Save comparison results to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now().isoformat(),
            "ground_truth_path": str(self.ground_truth_path),
            "cv_count": len(self.cvs),
            "job_count": len(self.jobs),
            "results": result.to_dict()
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved comparison results to {self.output_path}")
        return str(self.output_path)

    def print_baseline_table(self, results: List[MethodResult]) -> None:
        """Print baseline comparison table (Bảng 2 format)."""
        print("\n" + "=" * 75)
        print("Bảng 2: So sánh với phương pháp cơ sở")
        print("=" * 75)
        print(f"{'Phương pháp':<25} {'MRR':<10} {'NDCG@10':<12} {'Hit Rate@10':<12}")
        print("-" * 75)

        for r in results:
            print(f"{r.method_name:<25} {r.mrr:<10.3f} {r.ndcg_at_10:<12.3f} {r.hit_rate_at_10:<12.3f}")

        print("=" * 75)

    def print_ablation_table(self, results: List[MethodResult]) -> None:
        """Print ablation study table (Bảng 3 format)."""
        print("\n" + "=" * 75)
        print("Bảng 3: Ảnh hưởng của các vòng lọc")
        print("=" * 75)
        print(f"{'Cấu hình':<30} {'NDCG@10':<12} {'Hit Rate@10':<15} {'Thời gian (ms)':<15}")
        print("-" * 75)

        for r in results:
            print(f"{r.method_name:<30} {r.ndcg_at_10:<12.3f} {r.hit_rate_at_10:<15.3f} {r.avg_time_ms:<15.1f}")

        print("=" * 75)

    def print_expected_comparison(self, results: List[MethodResult]) -> None:
        """Print comparison with expected results from the report."""
        expected_baseline = {
            "Ngẫu nhiên": {"mrr": 0.089, "ndcg_at_10": 0.112, "hit_rate_at_10": 0.203},
            "TF-IDF + Cosin": {"mrr": 0.524, "ndcg_at_10": 0.487, "hit_rate_at_10": 0.612},
            "SimCSE (chỉ tiêu đề)": {"mrr": 0.723, "ndcg_at_10": 0.695, "hit_rate_at_10": 0.847},
            "Vòng 1 + 2 + 3 (+ kỹ năng)": {"mrr": 0.847, "ndcg_at_10": 0.814, "hit_rate_at_10": 0.934}
        }

        print("\n" + "=" * 90)
        print("So sánh với kết quả kỳ vọng (từ section_54_danh_gia.tex)")
        print("=" * 90)
        print(f"{'Phương pháp':<30} {'Metric':<12} {'Actual':<10} {'Expected':<10} {'Diff':<10}")
        print("-" * 90)

        for r in results:
            if r.method_name in expected_baseline:
                exp = expected_baseline[r.method_name]
                for metric, actual, expected in [
                    ("MRR", r.mrr, exp["mrr"]),
                    ("NDCG@10", r.ndcg_at_10, exp["ndcg_at_10"]),
                    ("Hit Rate@10", r.hit_rate_at_10, exp["hit_rate_at_10"])
                ]:
                    diff = actual - expected
                    diff_str = f"{diff:+.3f}"
                    print(f"{r.method_name:<30} {metric:<12} {actual:<10.3f} {expected:<10.3f} {diff_str:<10}")
                print("-" * 90)

        print("=" * 90)
