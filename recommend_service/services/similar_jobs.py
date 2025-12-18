import logging
import numpy as np
from typing import List, Optional

from recommend_service.models import JobData

logger = logging.getLogger(__name__)


class SimilarJobsService:
    """
    Service for finding similar jobs based on title embeddings.
    """

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

    def calculate_job_similarity(self, job1: JobData, job2: JobData) -> float:
        """
        Calculate similarity between two jobs based on title embeddings.

        Args:
            job1: First job data with title embedding
            job2: Second job data with title embedding

        Returns:
            Similarity score between 0 and 1
        """
        if not job1.title_embedding or not job2.title_embedding:
            return 0.0

        similarity = self.cosine_similarity(
            job1.title_embedding,
            job2.title_embedding
        )

        return max(0.0, min(1.0, similarity))

    def find_similar_jobs(
        self,
        target_job: JobData,
        all_jobs: List[JobData],
        top_k: int = 10
    ) -> List[dict]:
        """
        Find top K most similar jobs for a given job.

        Args:
            target_job: The job to find similar jobs for
            all_jobs: List of all available jobs
            top_k: Number of similar jobs to return (default: 10)

        Returns:
            List of dicts with job_id, similarity score, and job_title
        """
        similarities = []

        for job in all_jobs:
            # Skip the target job itself
            if job.id == target_job.id:
                continue

            sim = self.calculate_job_similarity(target_job, job)

            # Only include jobs with non-zero similarity
            if sim > 0:
                similarities.append({
                    "job_id": job.id,
                    "similar_job_id": job.id,
                    "similarity": sim,
                    "job_title": job.title
                })

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top K
        return similarities[:top_k]

    def batch_calculate_similar_jobs(
        self,
        jobs: List[JobData],
        top_k: int = 10
    ) -> dict:
        """
        Calculate top K similar jobs for multiple jobs.

        Args:
            jobs: List of job data
            top_k: Number of similar jobs per job

        Returns:
            Dict mapping job_id to list of similar jobs
        """
        results = {}

        for target_job in jobs:
            if not target_job.title_embedding:
                logger.warning(f"Job {target_job.id} has no title embedding, skipping")
                continue

            similar_jobs = self.find_similar_jobs(target_job, jobs, top_k)
            results[target_job.id] = similar_jobs

            logger.info(f"Found {len(similar_jobs)} similar jobs for job {target_job.id}")

        return results
