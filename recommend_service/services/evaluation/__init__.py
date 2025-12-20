"""
Evaluation module for Job Recommendation System.

This module provides:
- Ground truth generation using title embeddings
- Evaluation metrics (MRR, NDCG@K, Hit Rate@K)
- Baseline methods comparison (Random, TF-IDF, SimCSE title-only)
- Ablation study (1-layer, 2-layer, 3-layer cascade filtering)
"""

from .title_embedding_service import TitleEmbeddingService
from .ground_truth_generator import GroundTruthGenerator, GroundTruthPair
from .metrics import EvaluationMetrics, EvaluationResult
from .evaluator import Evaluator
from .baseline_methods import (
    RandomRecommender,
    TFIDFRecommender,
    TitleOnlyRecommender,
    CascadeRecommender
)
from .comparison_evaluator import ComparisonEvaluator, ComparisonResult, MethodResult

__all__ = [
    # Core evaluation
    "TitleEmbeddingService",
    "GroundTruthGenerator",
    "GroundTruthPair",
    "EvaluationMetrics",
    "EvaluationResult",
    "Evaluator",
    # Baseline methods
    "RandomRecommender",
    "TFIDFRecommender",
    "TitleOnlyRecommender",
    "CascadeRecommender",
    # Comparison evaluation
    "ComparisonEvaluator",
    "ComparisonResult",
    "MethodResult",
]
