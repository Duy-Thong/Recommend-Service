import logging
import numpy as np
from typing import List, Optional

from recommend_service.models import CVData, JobData

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Service for calculating similarity between CV and Job embeddings.

    Currently uses only title embedding for similarity.
    You can extend this to use weighted combination of multiple embeddings.
    """

    def __init__(self):
        # Weights for different embedding types (for future use)
        self.title_weight = 1.0
        self.skills_weight = 0.0  # Set to 0 for now, can be adjusted later
        self.experience_weight = 0.0  # Set to 0 for now

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0

        try:
            a = np.array(vec1)
            b = np.array(vec2)

            if a.shape != b.shape:
                return 0.0

            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_similarity(self, cv: CVData, job: JobData) -> float:
        """
        Calculate overall similarity between a CV and a Job.

        Currently uses only title embedding.
        Modify this method to change similarity calculation logic.

        Args:
            cv: CV data with embeddings
            job: Job data with embeddings

        Returns:
            Similarity score between 0 and 1
        """
        # ============================================
        # MODIFY THIS METHOD TO CHANGE SIMILARITY LOGIC
        # ============================================

        # Currently: Only use title embedding
        title_sim = self.cosine_similarity(
            cv.title_embedding or [],
            job.title_embedding or []
        )

        # For future: Weighted combination
        # skills_sim = self.cosine_similarity(
        #     cv.skills_embedding or [],
        #     job.skills_embedding or []
        # )
        #
        # exp_sim = self.cosine_similarity(
        #     cv.experience_embedding or [],
        #     job.requirement_embedding or []
        # )
        #
        # total_weight = self.title_weight + self.skills_weight + self.experience_weight
        # if total_weight == 0:
        #     return 0.0
        #
        # similarity = (
        #     self.title_weight * title_sim +
        #     self.skills_weight * skills_sim +
        #     self.experience_weight * exp_sim
        # ) / total_weight

        similarity = title_sim

        return max(0.0, min(1.0, similarity))

    def find_top_k_jobs(
        self,
        cv: CVData,
        jobs: List[JobData],
        top_k: int = 20
    ) -> List[dict]:
        """
        Find top K most similar jobs for a CV.

        Args:
            cv: CV data with embeddings
            jobs: List of job data with embeddings
            top_k: Number of top jobs to return

        Returns:
            List of dicts with job_id and similarity score
        """
        similarities = []

        for job in jobs:
            sim = self.calculate_similarity(cv, job)
            similarities.append({
                "job_id": job.id,
                "similarity": sim,
                "job_title": job.title
            })

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top K
        return similarities[:top_k]

    def batch_calculate_similarities(
        self,
        cvs: List[CVData],
        jobs: List[JobData],
        top_k: int = 20
    ) -> dict:
        """
        Calculate top K job recommendations for multiple CVs.

        Args:
            cvs: List of CV data
            jobs: List of job data
            top_k: Number of top jobs per CV

        Returns:
            Dict mapping cv_id to list of recommendations
        """
        results = {}

        for cv in cvs:
            if not cv.title_embedding:
                logger.warning(f"CV {cv.id} has no title embedding, skipping")
                continue

            top_jobs = self.find_top_k_jobs(cv, jobs, top_k)
            results[cv.id] = top_jobs

        return results
