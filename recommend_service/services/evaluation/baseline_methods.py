"""
Baseline Methods for Comparison.

Implements baseline recommendation methods for comparison:
1. Random: Random job recommendations
2. TF-IDF + Cosine: TF-IDF vectorization with cosine similarity
3. Title-only Embedding: Only use title embeddings (no cascade)

These baselines are compared against the proposed cascade filtering method
as described in section_54_danh_gia.tex (Bảng 2: So sánh với phương pháp cơ sở).
"""

import logging
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RandomRecommender:
    """
    Random recommendation baseline.

    Simply returns random jobs for each CV.
    Expected results from report: MRR=0.089, NDCG@10=0.112, Hit Rate@10=0.203
    """

    def __init__(self, seed: int = 42):
        """
        Initialize random recommender.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.job_ids: List[str] = []

    def fit(self, job_ids: List[str]) -> None:
        """
        Store job IDs for random sampling.

        Args:
            job_ids: List of all job IDs
        """
        self.job_ids = job_ids
        logger.info(f"RandomRecommender: Loaded {len(job_ids)} jobs")

    def recommend(self, cv_id: str, top_k: int = 10) -> List[str]:
        """
        Get random job recommendations.

        Args:
            cv_id: CV ID (not used, just for interface consistency)
            top_k: Number of recommendations

        Returns:
            List of randomly selected job IDs
        """
        random.seed(hash(cv_id) + self.seed)  # Deterministic per CV
        k = min(top_k, len(self.job_ids))
        return random.sample(self.job_ids, k)

    def recommend_batch(
        self,
        cv_ids: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get random recommendations for multiple CVs.

        Args:
            cv_ids: List of CV IDs
            top_k: Number of recommendations per CV

        Returns:
            Dict mapping cv_id -> list of job_ids
        """
        return {cv_id: self.recommend(cv_id, top_k) for cv_id in cv_ids}


class TFIDFRecommender:
    """
    TF-IDF + Cosine Similarity baseline.

    Uses TF-IDF vectorization of titles and cosine similarity for matching.
    Expected results from report: MRR=0.524, NDCG@10=0.487, Hit Rate@10=0.612
    """

    def __init__(self):
        """Initialize TF-IDF recommender."""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.job_ids: List[str] = []
        self.job_vectors = None

    def fit(self, jobs: List[Tuple[str, str]]) -> None:
        """
        Fit TF-IDF vectorizer on job titles.

        Args:
            jobs: List of (job_id, job_title) tuples
        """
        self.job_ids = [j[0] for j in jobs]
        job_titles = [j[1] for j in jobs]

        # Handle empty titles
        job_titles = [t if t and t.strip() else " " for t in job_titles]

        logger.info(f"TFIDFRecommender: Fitting on {len(jobs)} jobs...")
        self.job_vectors = self.vectorizer.fit_transform(job_titles)
        logger.info(f"TFIDFRecommender: Vocabulary size = {len(self.vectorizer.vocabulary_)}")

    def recommend(self, cv_title: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get job recommendations for a CV title.

        Args:
            cv_title: CV title text
            top_k: Number of recommendations

        Returns:
            List of (job_id, similarity_score) tuples
        """
        if not cv_title or not cv_title.strip():
            return []

        # Transform CV title
        cv_vector = self.vectorizer.transform([cv_title])

        # Compute cosine similarity
        similarities = cosine_similarity(cv_vector, self.job_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.job_ids[i], float(similarities[i])) for i in top_indices]

    def recommend_batch(
        self,
        cvs: List[Tuple[str, str]],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get recommendations for multiple CVs.

        Args:
            cvs: List of (cv_id, cv_title) tuples
            top_k: Number of recommendations per CV

        Returns:
            Dict mapping cv_id -> list of job_ids
        """
        result = {}
        for cv_id, cv_title in cvs:
            recommendations = self.recommend(cv_title, top_k)
            result[cv_id] = [job_id for job_id, _ in recommendations]
        return result


class TitleOnlyRecommender:
    """
    Title-only Embedding baseline (SimCSE without cascade filtering).

    Uses embedding similarity on titles only, without experience/skills filtering.
    This corresponds to "Vòng 1 (tiêu đề)" in Bảng 3 of the report.
    Expected results from report: MRR=0.723, NDCG@10=0.695, Hit Rate@10=0.847
    """

    def __init__(self):
        """Initialize title-only recommender."""
        self.job_ids: List[str] = []
        self.job_embeddings = None
        self.embedding_service = None
        self.index = None

    def fit(
        self,
        jobs: List[Tuple[str, str]],
        embedding_service=None
    ) -> None:
        """
        Build FAISS index from job title embeddings.

        Args:
            jobs: List of (job_id, job_title) tuples
            embedding_service: EmbeddingService instance (uses PhoBERT)
        """
        import faiss

        self.job_ids = [j[0] for j in jobs]
        job_titles = [j[1] for j in jobs]

        if embedding_service is None:
            from recommend_service.services.embedding import EmbeddingService
            embedding_service = EmbeddingService()

        self.embedding_service = embedding_service

        logger.info(f"TitleOnlyRecommender: Generating embeddings for {len(jobs)} jobs...")

        # Generate embeddings batch
        embeddings = []
        for title in job_titles:
            emb = self.embedding_service.get_embedding(title)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0.0] * 768)  # PhoBERT dimension

        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        logger.info(f"TitleOnlyRecommender: Built FAISS index with {len(self.job_ids)} jobs")

    def recommend(self, cv_title: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get job recommendations for a CV title.

        Args:
            cv_title: CV title text
            top_k: Number of recommendations

        Returns:
            List of (job_id, similarity_score) tuples
        """
        if not cv_title or not cv_title.strip():
            return []

        # Get CV title embedding
        cv_embedding = self.embedding_service.get_embedding(cv_title)
        if not cv_embedding:
            return []

        cv_embedding = np.array(cv_embedding, dtype=np.float32).reshape(1, -1)

        # Normalize
        norm = np.linalg.norm(cv_embedding)
        if norm > 0:
            cv_embedding = cv_embedding / norm

        # Search
        k = min(top_k, len(self.job_ids))
        distances, indices = self.index.search(cv_embedding, k)

        return [
            (self.job_ids[idx], float(distances[0][i]))
            for i, idx in enumerate(indices[0])
            if idx != -1
        ]

    def recommend_batch(
        self,
        cvs: List[Tuple[str, str]],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get recommendations for multiple CVs.

        Args:
            cvs: List of (cv_id, cv_title) tuples
            top_k: Number of recommendations per CV

        Returns:
            Dict mapping cv_id -> list of job_ids
        """
        result = {}
        for cv_id, cv_title in cvs:
            recommendations = self.recommend(cv_title, top_k)
            result[cv_id] = [job_id for job_id, _ in recommendations]
        return result


class CascadeRecommender:
    """
    Cascade Filtering Recommender with configurable layers.

    Supports different configurations for ablation study:
    - 1 layer: Title only (FAISS search)
    - 2 layers: Title + Experience
    - 3 layers: Title + Experience + Skills (full cascade)

    This corresponds to Bảng 3 "Ảnh hưởng của các vòng lọc" in the report.
    """

    def __init__(self, num_layers: int = 3):
        """
        Initialize cascade recommender.

        Args:
            num_layers: Number of filtering layers (1, 2, or 3)
        """
        if num_layers not in [1, 2, 3]:
            raise ValueError("num_layers must be 1, 2, or 3")

        self.num_layers = num_layers
        self.similarity_service = None
        self.jobs_dict = {}

    def fit(self, jobs, similarity_service=None) -> None:
        """
        Initialize with jobs and similarity service.

        Args:
            jobs: List of JobData objects
            similarity_service: SimilarityService instance
        """
        if similarity_service is None:
            from recommend_service.services.similarity import SimilarityService
            similarity_service = SimilarityService()
            similarity_service.build_index(jobs)

        self.similarity_service = similarity_service
        self.jobs_dict = {job.id: job for job in jobs}

        logger.info(f"CascadeRecommender: Initialized with {len(jobs)} jobs, {self.num_layers} layers")

    def recommend(self, cv, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get job recommendations for a CV using cascade filtering.

        Args:
            cv: CVData object
            top_k: Number of recommendations

        Returns:
            List of (job_id, similarity_score) tuples
        """
        if self.num_layers == 1:
            # Title only (FAISS)
            results = self.similarity_service._find_top_k_jobs_faiss(cv, top_k)
        elif self.num_layers == 2:
            # Title + Experience
            results = self.similarity_service.find_top_k_jobs_cascade(
                cv, self.jobs_dict,
                k1=1000, k2=top_k, k3=top_k  # Skip layer 3 by setting k3=k2
            )
        else:
            # Full cascade (Title + Experience + Skills)
            results = self.similarity_service.find_top_k_jobs_cascade(
                cv, self.jobs_dict,
                k1=1000, k2=100, k3=top_k
            )

        return [(r["job_id"], r["similarity"]) for r in results]

    def recommend_batch(
        self,
        cvs,
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get recommendations for multiple CVs.

        Args:
            cvs: List of CVData objects
            top_k: Number of recommendations per CV

        Returns:
            Dict mapping cv_id -> list of job_ids
        """
        result = {}
        for cv in cvs:
            recommendations = self.recommend(cv, top_k)
            result[cv.id] = [job_id for job_id, _ in recommendations]
        return result
