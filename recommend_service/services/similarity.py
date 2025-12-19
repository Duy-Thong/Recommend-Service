import logging
import numpy as np
import faiss
import pickle
from typing import List, Optional, Dict
from pathlib import Path

from recommend_service.models import CVData, JobData

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Service for calculating similarity between CV and Job embeddings using FAISS for fast search.

    Currently uses only title embedding for similarity.
    You can extend this to use weighted combination of multiple embeddings.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize the service with optional FAISS index.

        Args:
            index_path: Path to save/load FAISS index for jobs
            index_type: Type of FAISS index ("Flat" or "IVFFlat")
            nlist: Number of clusters for IVF index (k-means)
            nprobe: Number of clusters to search in IVF index
        """
        # Weights for different embedding types (for future use)
        self.title_weight = 1.0
        self.skills_weight = 0.0  # Set to 0 for now, can be adjusted later
        self.experience_weight = 0.0  # Set to 0 for now

        # FAISS index for jobs
        self.index: Optional[faiss.Index] = None
        self.job_ids: List[str] = []
        self.job_titles: List[str] = []
        self.index_path = index_path
        self.dimension: Optional[int] = None
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

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
            n_samples = len(embeddings)
            nlist = min(self.nlist, max(1, n_samples // 10))

            logger.info(f"Building IVFFlat index for jobs with {nlist} clusters (k-means)")

            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index
            logger.info(f"Training index with {n_samples} job samples...")
            self.index.train(embeddings_matrix)

            # Set number of clusters to probe
            self.index.nprobe = min(self.nprobe, nlist)

            # Add vectors to index
            self.index.add(embeddings_matrix)

            logger.info(f"Built IVFFlat index: {nlist} clusters, nprobe={self.index.nprobe}")
        else:
            # Use Flat index (brute force)
            logger.info("Building Flat index for jobs (brute force)")
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

        # Save metadata
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

    def find_top_k_jobs(
        self,
        cv: CVData,
        jobs: List[JobData] = None,
        top_k: int = 20
    ) -> List[dict]:
        """
        Find top K most similar jobs for a CV using FAISS index.

        Args:
            cv: CV data with embeddings
            jobs: List of job data (only used for fallback if index not built)
            top_k: Number of top jobs to return

        Returns:
            List of dicts with job_id and similarity score
        """
        # Use FAISS if index is available
        if self.index is not None and self.job_ids:
            return self._find_top_k_jobs_faiss(cv, top_k)

        # Fallback to original method
        if jobs is None:
            logger.warning("No FAISS index and no jobs provided, cannot find jobs")
            return []

        return self._find_top_k_jobs_legacy(cv, jobs, top_k)

    def find_top_k_jobs_cascade(
        self,
        cv: CVData,
        jobs_dict: Dict[str, JobData],
        k1: int = 1000,
        k2: int = 100,
        k3: int = 10
    ) -> List[dict]:
        """
        Find top K jobs using 3-round cascade filtering:
        Round 1 (FAISS): Title similarity -> Top K1 jobs (default 1000)
        Round 2 (Loop):  Experience-Requirements similarity -> Top K2 jobs (default 100)
        Round 3 (Loop):  Skills similarity -> Top K3 jobs (default 10)

        Args:
            cv: CV data with embeddings
            jobs_dict: Dictionary mapping job_id to JobData (for fast lookup)
            k1: Number of jobs to select in round 1 (title filtering)
            k2: Number of jobs to select in round 2 (experience filtering)
            k3: Number of jobs to select in round 3 (skills filtering)

        Returns:
            List of dicts with job_id and similarity score
        """
        if self.index is None or not self.job_ids:
            logger.error("FAISS index not built. Cannot perform cascade filtering.")
            return []

        if not cv.title_embedding:
            logger.warning(f"CV has no title embedding, cannot perform cascade filtering")
            return []

        # ============================================
        # Round 1: FAISS search by Title
        # ============================================
        logger.debug(f"Round 1: Searching top {k1} jobs by title similarity")

        query_vec = np.array(cv.title_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k1_actual = min(k1, len(self.job_ids))
        distances, indices = self.index.search(query_vec, k1_actual)

        top_k1_ids = [self.job_ids[idx] for idx in indices[0] if idx != -1]
        logger.debug(f"Round 1 complete: Found {len(top_k1_ids)} jobs")

        if not top_k1_ids:
            logger.warning("Round 1 returned no jobs")
            return []

        # ============================================
        # Round 2: Experience-Requirements filtering
        # ============================================
        logger.debug(f"Round 2: Filtering {len(top_k1_ids)} -> {k2} jobs by experience-requirements")

        exp_scores = []
        for job_id in top_k1_ids:
            job = jobs_dict.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found in jobs_dict")
                continue

            # Calculate experience-requirements similarity
            if cv.experience_embedding and job.requirement_embedding:
                sim = self.cosine_similarity(cv.experience_embedding, job.requirement_embedding)
            elif not cv.experience_embedding:
                # CV has no experience (e.g., fresher) -> lower score but not eliminate
                sim = 0.3
            elif not job.requirement_embedding:
                # Job has no specific requirements -> neutral score
                sim = 0.5
            else:
                sim = 0.0

            exp_scores.append((job_id, sim))

        # Sort by experience-requirements similarity
        exp_scores.sort(key=lambda x: x[1], reverse=True)
        top_k2_ids = [job_id for job_id, _ in exp_scores[:k2]]
        logger.debug(f"Round 2 complete: Selected {len(top_k2_ids)} jobs")

        if not top_k2_ids:
            logger.warning("Round 2 returned no jobs")
            return []

        # ============================================
        # Round 3: Skills-Skills filtering
        # ============================================
        logger.debug(f"Round 3: Filtering {len(top_k2_ids)} -> {k3} jobs by skills")

        skill_scores = []
        for job_id in top_k2_ids:
            job = jobs_dict.get(job_id)
            if not job:
                continue

            # Calculate skills similarity
            if cv.skills_embedding and job.skills_embedding:
                sim = self.cosine_similarity(cv.skills_embedding, job.skills_embedding)
            elif not cv.skills_embedding:
                # CV has no skills listed -> lower score
                sim = 0.3
            elif not job.skills_embedding:
                # Job has no skills requirement -> neutral score
                sim = 0.5
            else:
                sim = 0.0

            skill_scores.append({
                "job_id": job_id,
                "similarity": sim
            })

        # Sort by skills similarity
        skill_scores.sort(key=lambda x: x["similarity"], reverse=True)
        top_k3 = skill_scores[:k3]
        logger.debug(f"Round 3 complete: Final {len(top_k3)} jobs selected")

        return top_k3

    def _find_top_k_jobs_faiss(self, cv: CVData, top_k: int) -> List[dict]:
        """
        Find top K jobs using FAISS index (fast method).

        Args:
            cv: CV data with embeddings
            top_k: Number of top jobs to return

        Returns:
            List of dicts with job_id and similarity score
        """
        if not cv.title_embedding:
            return []

        # Normalize query embedding
        query_embedding = np.array(cv.title_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search in FAISS index
        k = min(top_k, len(self.job_ids))
        distances, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            results.append({
                "job_id": self.job_ids[idx],
                "similarity": float(dist)  # Already cosine similarity due to normalization
            })

        return results

    def _find_top_k_jobs_legacy(
        self,
        cv: CVData,
        jobs: List[JobData],
        top_k: int
    ) -> List[dict]:
        """
        Legacy method: Find top K jobs without FAISS (slower, for fallback).

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
                "similarity": sim
            })

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top K
        return similarities[:top_k]

    def batch_calculate_similarities(
        self,
        cvs: List[CVData],
        jobs: List[JobData] = None,
        top_k: int = 20,
        use_faiss: bool = True
    ) -> Dict[str, List[dict]]:
        """
        Calculate top K job recommendations for multiple CVs.
        Uses FAISS for vectorized batch processing (much faster).

        Args:
            cvs: List of CV data
            jobs: List of job data (only used if index not built)
            top_k: Number of top jobs per CV
            use_faiss: Whether to use FAISS for batch processing (default: True)

        Returns:
            Dict mapping cv_id to list of recommendations
        """
        # Build index if using FAISS and not already built
        if use_faiss and self.index is None and jobs:
            logger.info("Building FAISS index for batch calculation")
            self.build_index(jobs)

        results = {}

        if use_faiss and self.index is not None:
            # Use FAISS batch search for better performance
            results = self._batch_calculate_faiss(cvs, top_k)
        else:
            # Fallback to legacy method
            if jobs is None:
                logger.warning("No FAISS index and no jobs provided")
                return {}

            for cv in cvs:
                if not cv.title_embedding:
                    logger.warning(f"CV {cv.id} has no title embedding, skipping")
                    continue

                top_jobs = self.find_top_k_jobs(cv, jobs, top_k)
                results[cv.id] = top_jobs

                if len(results) % 100 == 0:
                    logger.info(f"Processed {len(results)} CVs")

        logger.info(f"Completed batch calculation for {len(results)} CVs")
        return results

    def _batch_calculate_faiss(
        self,
        cvs: List[CVData],
        top_k: int
    ) -> Dict[str, List[dict]]:
        """
        Batch calculate job recommendations using FAISS vectorized operations.

        Args:
            cvs: List of CV data
            top_k: Number of top jobs per CV

        Returns:
            Dict mapping cv_id to list of recommendations
        """
        # Prepare query embeddings from CVs
        valid_cvs = [cv for cv in cvs if cv.title_embedding]
        query_embeddings = []
        query_cv_ids = []

        for cv in valid_cvs:
            embedding = np.array(cv.title_embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            query_embeddings.append(embedding)
            query_cv_ids.append(cv.id)

        if not query_embeddings:
            logger.warning("No valid CV embeddings for batch calculation")
            return {}

        # Batch search using FAISS
        query_matrix = np.array(query_embeddings, dtype=np.float32)
        k = min(top_k, len(self.job_ids))
        distances, indices = self.index.search(query_matrix, k)

        # Build results
        results = {}
        for i, cv_id in enumerate(query_cv_ids):
            recommendations = []

            for dist, idx in zip(distances[i], indices[i]):
                if idx == -1:
                    continue

                recommendations.append({
                    "job_id": self.job_ids[idx],
                    "similarity": float(dist)
                })

            results[cv_id] = recommendations

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(query_cv_ids)} CVs in batch")

        return results
