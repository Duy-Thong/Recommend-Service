"""
Evaluation Metrics for Recommendation System.

Implements ranking metrics as described in section_54_danh_gia.tex:
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    num_queries: int
    num_hits_at_5: int
    num_hits_at_10: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mrr": self.mrr,
            "ndcg_at_5": self.ndcg_at_5,
            "ndcg_at_10": self.ndcg_at_10,
            "hit_rate_at_5": self.hit_rate_at_5,
            "hit_rate_at_10": self.hit_rate_at_10,
            "num_queries": self.num_queries,
            "num_hits_at_5": self.num_hits_at_5,
            "num_hits_at_10": self.num_hits_at_10
        }

    def __str__(self) -> str:
        """Format results as string."""
        return (
            f"Evaluation Results:\n"
            f"  MRR:           {self.mrr:.4f}\n"
            f"  NDCG@5:        {self.ndcg_at_5:.4f}\n"
            f"  NDCG@10:       {self.ndcg_at_10:.4f}\n"
            f"  Hit Rate@5:    {self.hit_rate_at_5:.4f}\n"
            f"  Hit Rate@10:   {self.hit_rate_at_10:.4f}\n"
            f"  Queries:       {self.num_queries}\n"
            f"  Hits@5:        {self.num_hits_at_5}\n"
            f"  Hits@10:       {self.num_hits_at_10}"
        )


class EvaluationMetrics:
    """
    Calculator for ranking evaluation metrics.

    Metrics follow the formulas from section_54_danh_gia.tex:
    - MRR: Mean Reciprocal Rank
    - NDCG@K: Normalized Discounted Cumulative Gain at K
    - Hit Rate@K: Hit Ratio at K
    """

    @staticmethod
    def reciprocal_rank(relevant_item: str, ranked_list: List[str]) -> float:
        """
        Calculate reciprocal rank for a single query.

        Args:
            relevant_item: The ground truth job_id
            ranked_list: List of recommended job_ids in order (best first)

        Returns:
            1/rank if found, 0.0 if not found
        """
        try:
            rank = ranked_list.index(relevant_item) + 1  # 1-indexed
            return 1.0 / rank
        except ValueError:
            return 0.0

    @staticmethod
    def mrr(ground_truth: Dict[str, str], predictions: Dict[str, List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = (1/|Q|) * sum(1/rank_i)

        where rank_i is the position of the first relevant item for query i.

        Args:
            ground_truth: Dict mapping cv_id -> relevant job_id
            predictions: Dict mapping cv_id -> list of recommended job_ids (ranked)

        Returns:
            MRR value between 0 and 1
        """
        if not ground_truth:
            return 0.0

        reciprocal_ranks = []

        for cv_id, relevant_job in ground_truth.items():
            if cv_id not in predictions:
                reciprocal_ranks.append(0.0)
                continue

            ranked_list = predictions[cv_id]
            rr = EvaluationMetrics.reciprocal_rank(relevant_job, ranked_list)
            reciprocal_ranks.append(rr)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = sum_{i=1}^{K} (rel_i / log2(i+1))

        Args:
            relevances: List of relevance scores (0 or 1 for binary relevance)
            k: Number of top results to consider

        Returns:
            DCG@K value
        """
        relevances = relevances[:k]
        if not relevances:
            return 0.0

        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0, log2(1) = 0

        return dcg

    @staticmethod
    def ndcg_at_k(
        relevant_item: str,
        ranked_list: List[str],
        k: int
    ) -> float:
        """
        Calculate Normalized DCG at K for a single query.

        For binary relevance (0 or 1):
        - rel_i = 1 if ranked_list[i] == relevant_item, else 0
        - IDCG@K = 1/log2(2) = 1.0 (ideal: relevant item at position 1)

        NDCG@K = DCG@K / IDCG@K

        Args:
            relevant_item: The ground truth job_id
            ranked_list: List of recommended job_ids in order (best first)
            k: Number of top results to consider

        Returns:
            NDCG@K value between 0 and 1
        """
        # Create binary relevance scores
        relevances = [1.0 if item == relevant_item else 0.0 for item in ranked_list[:k]]

        dcg = EvaluationMetrics.dcg_at_k(relevances, k)

        # IDCG for single relevant item: item at position 1
        idcg = 1.0 / np.log2(2)  # = 1.0

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def ndcg(
        ground_truth: Dict[str, str],
        predictions: Dict[str, List[str]],
        k: int
    ) -> float:
        """
        Calculate Mean NDCG@K across all queries.

        Args:
            ground_truth: Dict mapping cv_id -> relevant job_id
            predictions: Dict mapping cv_id -> list of recommended job_ids
            k: Number of top results to consider

        Returns:
            Mean NDCG@K value between 0 and 1
        """
        if not ground_truth:
            return 0.0

        ndcg_scores = []

        for cv_id, relevant_job in ground_truth.items():
            if cv_id not in predictions:
                ndcg_scores.append(0.0)
                continue

            ranked_list = predictions[cv_id]
            ndcg_score = EvaluationMetrics.ndcg_at_k(relevant_job, ranked_list, k)
            ndcg_scores.append(ndcg_score)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    @staticmethod
    def hit_rate_at_k(
        ground_truth: Dict[str, str],
        predictions: Dict[str, List[str]],
        k: int
    ) -> Tuple[float, int]:
        """
        Calculate Hit Rate at K.

        Hit Rate@K = (1/|Q|) * sum(indicator[relevant item in top K])

        Args:
            ground_truth: Dict mapping cv_id -> relevant job_id
            predictions: Dict mapping cv_id -> list of recommended job_ids
            k: Number of top results to consider

        Returns:
            Tuple of (hit_rate, num_hits)
        """
        if not ground_truth:
            return 0.0, 0

        hits = 0

        for cv_id, relevant_job in ground_truth.items():
            if cv_id in predictions:
                top_k = predictions[cv_id][:k]
                if relevant_job in top_k:
                    hits += 1

        hit_rate = hits / len(ground_truth)
        return hit_rate, hits

    @classmethod
    def evaluate(
        cls,
        ground_truth: Dict[str, str],
        predictions: Dict[str, List[str]]
    ) -> EvaluationResult:
        """
        Run all evaluation metrics.

        Args:
            ground_truth: Dict mapping cv_id -> relevant job_id
            predictions: Dict mapping cv_id -> list of recommended job_ids

        Returns:
            EvaluationResult with all metrics
        """
        if not ground_truth:
            logger.warning("Empty ground truth provided")
            return EvaluationResult(
                mrr=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                hit_rate_at_5=0.0,
                hit_rate_at_10=0.0,
                num_queries=0,
                num_hits_at_5=0,
                num_hits_at_10=0
            )

        # Calculate MRR
        mrr = cls.mrr(ground_truth, predictions)

        # Calculate NDCG@5 and NDCG@10
        ndcg_at_5 = cls.ndcg(ground_truth, predictions, k=5)
        ndcg_at_10 = cls.ndcg(ground_truth, predictions, k=10)

        # Calculate Hit Rate@5 and Hit Rate@10
        hit_rate_at_5, num_hits_5 = cls.hit_rate_at_k(ground_truth, predictions, k=5)
        hit_rate_at_10, num_hits_10 = cls.hit_rate_at_k(ground_truth, predictions, k=10)

        result = EvaluationResult(
            mrr=mrr,
            ndcg_at_5=ndcg_at_5,
            ndcg_at_10=ndcg_at_10,
            hit_rate_at_5=hit_rate_at_5,
            hit_rate_at_10=hit_rate_at_10,
            num_queries=len(ground_truth),
            num_hits_at_5=num_hits_5,
            num_hits_at_10=num_hits_10
        )

        return result
