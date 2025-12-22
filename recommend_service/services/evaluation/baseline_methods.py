"""
Baseline Methods for Comparison.

Implements baseline recommendation methods for comparison:
1. Random: Random job recommendations
2. TF-IDF + Cosine: TF-IDF vectorization with cosine similarity
3. Jaccard Similarity: Set-based word overlap
4. Word2Vec Average: Average word embeddings
5. Title-only Embedding: Only use title embeddings (no cascade)

These baselines are compared against the proposed cascade filtering method
as described in section_54_danh_gia.tex (Bảng 2: So sánh với phương pháp cơ sở).
"""

import logging
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, cache_dir: str = "./cache/baselines"):
        """Initialize title-only recommender.

        Args:
            cache_dir: Directory to cache embeddings and index
        """
        self.job_ids: List[str] = []
        self.job_embeddings = None
        self.embedding_service = None
        self.index = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, num_jobs: int) -> Path:
        """Get cache file path based on number of jobs."""
        return self.cache_dir / f"title_only_{num_jobs}_jobs.pkl"

    def fit(
        self,
        jobs: List[Tuple[str, str]],
        embedding_service=None,
        use_cache: bool = True
    ) -> None:
        """
        Build FAISS index from job title embeddings.

        Args:
            jobs: List of (job_id, job_title) tuples
            embedding_service: EmbeddingService instance (uses PhoBERT)
            use_cache: Whether to use cached embeddings if available
        """
        import faiss

        self.job_ids = [j[0] for j in jobs]
        job_titles = [j[1] for j in jobs]

        # Try to load from cache
        cache_path = self._get_cache_path(len(jobs))
        if use_cache and cache_path.exists():
            logger.info(f"TitleOnlyRecommender: Loading from cache {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.job_ids = cache_data['job_ids']
                    embeddings = cache_data['embeddings']

                    # Rebuild FAISS index
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(embeddings)

                    logger.info(f"TitleOnlyRecommender: Loaded {len(self.job_ids)} jobs from cache")

                    # Initialize embedding service for CV embeddings
                    if embedding_service is None:
                        from recommend_service.services.embedding import EmbeddingService
                        embedding_service = EmbeddingService()
                    self.embedding_service = embedding_service
                    return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating...")

        # Cache miss or disabled - generate embeddings
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

        # Save to cache
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'job_ids': self.job_ids,
                        'embeddings': embeddings
                    }, f)
                logger.info(f"TitleOnlyRecommender: Saved embeddings to cache {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

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
            # Title + Experience (weighted average)
            results = self.similarity_service.find_top_k_jobs_cascade(
                cv, self.jobs_dict,
                k1=1000, k2=top_k, k3=top_k,
                num_layers=2  # Stop after layer 2
            )
        else:
            # Full cascade (Title + Experience + Skills weighted average)
            results = self.similarity_service.find_top_k_jobs_cascade(
                cv, self.jobs_dict,
                k1=1000, k2=100, k3=top_k,
                num_layers=3  # Use all 3 layers
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


class JaccardRecommender:
    """
    Jaccard Similarity baseline.

    Uses set-based word overlap:
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Very fast, no model needed, simple lexical matching.
    """

    def __init__(self):
        """Initialize Jaccard recommender."""
        self.job_ids: List[str] = []
        self.job_word_sets: List[set] = []

    def _tokenize(self, text: str) -> set:
        """
        Simple tokenization: lowercase, split by non-alphanumeric.

        Args:
            text: Input text

        Returns:
            Set of words
        """
        if not text:
            return set()
        # Lowercase and split by non-word characters
        words = re.findall(r'\w+', text.lower())
        return set(words)

    def fit(self, jobs: List[Tuple[str, str]]) -> None:
        """
        Build word sets for all jobs.

        Args:
            jobs: List of (job_id, job_title) tuples
        """
        self.job_ids = [j[0] for j in jobs]
        self.job_word_sets = [self._tokenize(j[1]) for j in jobs]
        logger.info(f"JaccardRecommender: Built word sets for {len(self.job_ids)} jobs")

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Args:
            set_a: First word set
            set_b: Second word set

        Returns:
            Jaccard similarity score [0, 1]
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

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

        cv_words = self._tokenize(cv_title)
        if not cv_words:
            return []

        # Calculate Jaccard similarity with all jobs
        similarities = []
        for job_id, job_words in zip(self.job_ids, self.job_word_sets):
            sim = self._jaccard_similarity(cv_words, job_words)
            similarities.append((job_id, sim))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class Word2VecRecommender:
    """
    Word2Vec Average Embedding baseline.

    Uses pre-trained Vietnamese Word2Vec model:
    - Tokenize title into words
    - Get embedding for each word
    - Average all word embeddings
    - Compute cosine similarity

    Model: https://github.com/sonvx/word2vecVN
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Word2Vec recommender.

        Args:
            model_path: Path to pre-trained Word2Vec model (.bin or .bin.gz)
        """
        self.model_path = model_path
        self.model = None
        self.job_ids: List[str] = []
        self.job_embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        """Load pre-trained Word2Vec model using gensim."""
        if self.model is not None:
            return

        try:
            from gensim.models import KeyedVectors

            if self.model_path is None:
                raise ValueError("model_path is required for Word2VecRecommender")

            logger.info(f"Loading Word2Vec model from {self.model_path}")

            # Try loading as binary format
            if self.model_path.endswith('.bin') or self.model_path.endswith('.bin.gz'):
                self.model = KeyedVectors.load_word2vec_format(
                    self.model_path,
                    binary=True
                )
            else:
                self.model = KeyedVectors.load(self.model_path)

            logger.info(f"Word2Vec model loaded: {len(self.model)} words, {self.model.vector_size}D")
        except Exception as e:
            logger.error(f"Failed to load Word2Vec model: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, split by non-alphanumeric.

        Args:
            text: Input text

        Returns:
            List of words
        """
        if not text:
            return []
        words = re.findall(r'\w+', text.lower())
        return words

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get average Word2Vec embedding for text.

        Args:
            text: Input text

        Returns:
            Average embedding vector or None
        """
        words = self._tokenize(text)
        if not words:
            return None

        vectors = []
        for word in words:
            if word in self.model:
                vectors.append(self.model[word])

        if not vectors:
            return None

        # Average all word vectors
        return np.mean(vectors, axis=0)

    def fit(self, jobs: List[Tuple[str, str]]) -> None:
        """
        Build embeddings for all jobs.

        Args:
            jobs: List of (job_id, job_title) tuples
        """
        self._load_model()

        self.job_ids = [j[0] for j in jobs]
        embeddings = []

        for job_id, job_title in jobs:
            emb = self._get_text_embedding(job_title)
            if emb is not None:
                embeddings.append(emb)
            else:
                # If no embedding, use zero vector
                embeddings.append(np.zeros(self.model.vector_size))

        self.job_embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize embeddings
        norms = np.linalg.norm(self.job_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.job_embeddings = self.job_embeddings / norms

        logger.info(f"Word2VecRecommender: Built embeddings for {len(self.job_ids)} jobs")

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

        cv_emb = self._get_text_embedding(cv_title)
        if cv_emb is None:
            return []

        # Normalize CV embedding
        cv_emb = cv_emb.reshape(1, -1)
        norm = np.linalg.norm(cv_emb)
        if norm > 0:
            cv_emb = cv_emb / norm

        # Compute cosine similarity with all jobs
        similarities = np.dot(self.job_embeddings, cv_emb.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (self.job_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]
