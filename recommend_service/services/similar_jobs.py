import logging
import numpy as np
import faiss
import pickle
from typing import List, Optional, Dict
from pathlib import Path

from recommend_service.models import JobData

logger = logging.getLogger(__name__)


class SimilarJobsService:
    """
    Service for finding similar jobs based on title embeddings using FAISS for fast similarity search.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize the service with an optional FAISS index path.

        Args:
            index_path: Path to save/load FAISS index
            index_type: Type of FAISS index ("Flat" or "IVFFlat")
            nlist: Number of clusters for IVF index (k-means)
            nprobe: Number of clusters to search in IVF index
        """
        self.index: Optional[faiss.Index] = None
        self.job_ids: List[str] = []
        self.job_titles: List[str] = []
        self.index_path = index_path
        self.dimension: Optional[int] = None
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

    def build_index(self, jobs: List[JobData]) -> None:
        """
        Build FAISS index from job embeddings.

        Args:
            jobs: List of jobs with title embeddings
        """
        valid_jobs = [job for job in jobs if job.title_embedding]

        if not valid_jobs:
            logger.warning("No valid jobs with embeddings to build index")
            return

        # Get dimension from first embedding
        self.dimension = len(valid_jobs[0].title_embedding)

        # Prepare embeddings and metadata
        embeddings = []
        self.job_ids = []
        self.job_titles = []

        for job in valid_jobs:
            embedding = np.array(job.title_embedding, dtype=np.float32)
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)
            self.job_ids.append(job.id)
            self.job_titles.append(job.title)

        embeddings_matrix = np.array(embeddings, dtype=np.float32)

        # Create appropriate FAISS index based on type
        if self.index_type == "IVFFlat":
            # Use IVF (Inverted File) with K-means clustering
            # Adjust nlist based on dataset size
            n_samples = len(embeddings)
            nlist = min(self.nlist, max(1, n_samples // 10))  # At least 10 samples per cluster

            logger.info(f"Building IVFFlat index with {nlist} clusters (k-means)")

            # Create quantizer (uses Flat index for centroids)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index (k-means clustering)
            logger.info(f"Training index with {n_samples} samples...")
            self.index.train(embeddings_matrix)

            # Set number of clusters to probe during search
            self.index.nprobe = min(self.nprobe, nlist)

            # Add vectors to index
            self.index.add(embeddings_matrix)

            logger.info(f"Built IVFFlat index: {nlist} clusters, nprobe={self.index.nprobe}")
        else:
            # Use Flat index (brute force)
            logger.info("Building Flat index (brute force)")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings_matrix)

        logger.info(f"Built FAISS {self.index_type} index with {len(self.job_ids)} jobs, dimension {self.dimension}")

    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            path: Path to save index (uses self.index_path if not provided)
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        save_path = path or self.index_path
        if not save_path:
            logger.warning("No path specified to save index")
            return

        index_file = Path(save_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_file))
        logger.info(f"Saved FAISS index to {save_path}")

        # Save metadata (job_ids, job_titles, etc.)
        metadata_file = index_file.with_suffix('.metadata.pkl')
        metadata = {
            'job_ids': self.job_ids,
            'job_titles': self.job_titles,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_file}")

    def load_index(self, path: Optional[str] = None) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            path: Path to load index from (uses self.index_path if not provided)
        """
        load_path = path or self.index_path
        if not load_path:
            logger.warning("No path specified to load index")
            return

        index_file = Path(load_path)
        if not index_file.exists():
            logger.warning(f"Index file not found: {load_path}")
            return

        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        logger.info(f"Loaded FAISS index from {load_path}")

        # Load metadata
        metadata_file = index_file.with_suffix('.metadata.pkl')
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            self.job_ids = metadata.get('job_ids', [])
            self.job_titles = metadata.get('job_titles', [])
            self.dimension = metadata.get('dimension')
            self.index_type = metadata.get('index_type', 'Flat')
            self.nlist = metadata.get('nlist', 100)
            self.nprobe = metadata.get('nprobe', 10)

            # Set nprobe for IVF index
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe

            logger.info(f"Loaded metadata: {len(self.job_ids)} jobs, type={self.index_type}")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")

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
        all_jobs: List[JobData] = None,
        top_k: int = 10
    ) -> List[dict]:
        """
        Find top K most similar jobs for a given job using FAISS index.

        Args:
            target_job: The job to find similar jobs for
            all_jobs: List of all available jobs (only used for fallback if index not built)
            top_k: Number of similar jobs to return (default: 10)

        Returns:
            List of dicts with job_id, similarity score, and job_title
        """
        # Use FAISS if index is available
        if self.index is not None and self.job_ids:
            return self._find_similar_jobs_faiss(target_job, top_k)

        # Fallback to original method
        if all_jobs is None:
            logger.warning("No FAISS index and no jobs provided, cannot find similar jobs")
            return []

        return self._find_similar_jobs_legacy(target_job, all_jobs, top_k)

    def _find_similar_jobs_faiss(self, target_job: JobData, top_k: int) -> List[dict]:
        """
        Find similar jobs using FAISS index (fast method).

        Args:
            target_job: The job to find similar jobs for
            top_k: Number of similar jobs to return

        Returns:
            List of dicts with job_id, similarity score, and job_title
        """
        if not target_job.title_embedding:
            return []

        # Normalize query embedding
        query_embedding = np.array(target_job.title_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search in FAISS index (top_k + 1 to account for the job itself)
        k = min(top_k + 1, len(self.job_ids))
        distances, indices = self.index.search(query_embedding, k)

        # Build results
        similarities = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            job_id = self.job_ids[idx]

            # Skip the target job itself
            if job_id == target_job.id:
                continue

            similarities.append({
                "job_id": target_job.id,
                "similar_job_id": job_id,
                "similarity": float(dist),  # Already cosine similarity due to normalization
                "job_title": self.job_titles[idx]
            })

            if len(similarities) >= top_k:
                break

        return similarities

    def _find_similar_jobs_legacy(
        self,
        target_job: JobData,
        all_jobs: List[JobData],
        top_k: int
    ) -> List[dict]:
        """
        Legacy method: Find similar jobs without FAISS (slower, for fallback).

        Args:
            target_job: The job to find similar jobs for
            all_jobs: List of all available jobs
            top_k: Number of similar jobs to return

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
                    "job_id": target_job.id,
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
        top_k: int = 10,
        use_faiss: bool = True
    ) -> Dict[str, List[dict]]:
        """
        Calculate top K similar jobs for multiple jobs.
        Uses FAISS for vectorized batch processing (much faster).

        Args:
            jobs: List of job data
            top_k: Number of similar jobs per job
            use_faiss: Whether to use FAISS for batch processing (default: True)

        Returns:
            Dict mapping job_id to list of similar jobs
        """
        # Build index if using FAISS and not already built
        if use_faiss and self.index is None:
            logger.info("Building FAISS index for batch calculation")
            self.build_index(jobs)

        results = {}

        if use_faiss and self.index is not None:
            # Use FAISS batch search for better performance
            results = self._batch_calculate_faiss(jobs, top_k)
        else:
            # Fallback to legacy method
            for target_job in jobs:
                if not target_job.title_embedding:
                    logger.warning(f"Job {target_job.id} has no title embedding, skipping")
                    continue

                similar_jobs = self.find_similar_jobs(target_job, jobs, top_k)
                results[target_job.id] = similar_jobs

                if len(results) % 100 == 0:
                    logger.info(f"Processed {len(results)} jobs")

        logger.info(f"Completed batch calculation for {len(results)} jobs")
        return results

    def _batch_calculate_faiss(
        self,
        jobs: List[JobData],
        top_k: int
    ) -> Dict[str, List[dict]]:
        """
        Batch calculate similar jobs using FAISS vectorized operations.

        Args:
            jobs: List of job data
            top_k: Number of similar jobs per job

        Returns:
            Dict mapping job_id to list of similar jobs
        """
        # Prepare query embeddings
        valid_jobs = [job for job in jobs if job.title_embedding]
        query_embeddings = []
        query_job_ids = []

        for job in valid_jobs:
            embedding = np.array(job.title_embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            query_embeddings.append(embedding)
            query_job_ids.append(job.id)

        if not query_embeddings:
            logger.warning("No valid embeddings for batch calculation")
            return {}

        # Batch search using FAISS
        query_matrix = np.array(query_embeddings, dtype=np.float32)
        k = min(top_k + 1, len(self.job_ids))  # +1 to account for self-match
        distances, indices = self.index.search(query_matrix, k)

        # Build results
        results = {}
        for i, job_id in enumerate(query_job_ids):
            similarities = []

            for dist, idx in zip(distances[i], indices[i]):
                if idx == -1:
                    continue

                similar_job_id = self.job_ids[idx]

                # Skip the job itself
                if similar_job_id == job_id:
                    continue

                similarities.append({
                    "job_id": job_id,
                    "similar_job_id": similar_job_id,
                    "similarity": float(dist),
                    "job_title": self.job_titles[idx]
                })

                if len(similarities) >= top_k:
                    break

            results[job_id] = similarities

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(query_job_ids)} jobs in batch")

        return results
