"""
Ground Truth Generator for Evaluation.

Generates ground truth CV-Job pairs using combined text embedding similarity.
- CV: title + summary (consistent with recommendation system)
- Job: title + description (consistent with recommendation system)

Uses sentence-transformers/paraphrase-multilingual-mpnet-base-v2 for embeddings
and FAISS for efficient similarity search.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from recommend_service.database import DatabaseConnection
from .title_embedding_service import TitleEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthPair:
    """A ground truth CV-Job matching pair."""
    cv_id: str
    job_id: str
    cv_title: str
    job_title: str
    similarity: float


@dataclass
class CVWithText:
    """CV with combined text for embedding."""
    cv_id: str
    title: str
    combined_text: str  # title + summary


@dataclass
class JobWithText:
    """Job with combined text for embedding."""
    job_id: str
    title: str
    combined_text: str  # title + description


class GroundTruthGenerator:
    """
    Generates ground truth CV-Job pairs using combined text embedding similarity.

    Process:
    1. Load CVs with summary and Jobs with description from database
    2. Combine title + summary (CV) and title + description (Job)
    3. Embed combined texts using paraphrase-multilingual-mpnet-base-v2
    4. Build FAISS index from job embeddings
    5. For each CV, find the most similar job (top-1)
    6. Save pairs to CSV
    """

    def __init__(
        self,
        cv_limit: int = 5000,
        job_limit: int = 5000,
        index_path: str = "./faiss_data/title_jobs.faiss",
        output_path: str = "./evaluation_data/ground_truth.csv"
    ):
        """
        Initialize the ground truth generator.

        Args:
            cv_limit: Maximum number of CVs to process
            job_limit: Maximum number of jobs to process
            index_path: Path to save/load FAISS index
            output_path: Path to save ground truth CSV
        """
        self.cv_limit = cv_limit
        self.job_limit = job_limit
        self.index_path = Path(index_path)
        self.output_path = Path(output_path)

        self.db = DatabaseConnection()
        self.embedding_service = TitleEmbeddingService()

        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.job_ids: List[str] = []
        self.job_titles: List[str] = []
        self.dimension: Optional[int] = None

    def load_cvs(self) -> List[Tuple[str, str]]:
        """
        Load CVs from database (legacy method for backward compatibility).

        Returns:
            List of (cv_id, cv_title) tuples
        """
        query = """
            SELECT id, title
            FROM cvs
            WHERE "isMain" = true AND title IS NOT NULL AND title != ''
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (self.cv_limit,))
            rows = cursor.fetchall()

        cvs = [(row["id"], row["title"]) for row in rows if row["title"]]
        logger.info(f"Loaded {len(cvs)} CVs from database")
        return cvs

    def load_cvs_with_summary(self) -> List[CVWithText]:
        """
        Load CVs with summary for combined embedding (consistent with recommendation system).

        Returns:
            List of CVWithText objects with combined text (title + summary)
        """
        cv_query = """
            SELECT id, title, summary
            FROM cvs
            WHERE "isMain" = true AND title IS NOT NULL AND title != ''
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(cv_query, (self.cv_limit,))
            cv_rows = cursor.fetchall()

        if not cv_rows:
            return []

        # Build CVWithText objects
        cvs = []
        for row in cv_rows:
            cv_id = row["id"]
            title = row["title"]
            if not title:
                continue

            # Combine title + summary (consistent with recommendation system)
            summary = row.get("summary") or ""
            combined_text = f"{title} {summary}".strip()

            cvs.append(CVWithText(
                cv_id=cv_id,
                title=title,
                combined_text=combined_text
            ))

        logger.info(f"Loaded {len(cvs)} CVs with summary data")
        return cvs

    def load_jobs(self) -> List[Tuple[str, str]]:
        """
        Load jobs from database (legacy method for backward compatibility).

        Returns:
            List of (job_id, job_title) tuples
        """
        # Note: For evaluation purposes, we include all jobs regardless of expiry date
        # This ensures we have enough data diversity for proper evaluation
        query = """
            SELECT id, title
            FROM jobs
            WHERE status = 'ACTIVE'
                AND title IS NOT NULL AND title != ''
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (self.job_limit,))
            rows = cursor.fetchall()

        jobs = [(row["id"], row["title"]) for row in rows if row["title"]]
        logger.info(f"Loaded {len(jobs)} jobs from database")
        return jobs

    def load_jobs_with_description(self) -> List[JobWithText]:
        """
        Load jobs with description for combined embedding (consistent with recommendation system).

        Returns:
            List of JobWithText objects with combined text (title + description)
        """
        job_query = """
            SELECT id, title, description
            FROM jobs
            WHERE status = 'ACTIVE'
                AND title IS NOT NULL AND title != ''
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(job_query, (self.job_limit,))
            job_rows = cursor.fetchall()

        if not job_rows:
            return []

        # Build JobWithText objects
        jobs = []
        for row in job_rows:
            job_id = row["id"]
            title = row["title"]
            if not title:
                continue

            # Combine title + description (consistent with recommendation system)
            description = row.get("description") or ""
            combined_text = f"{title} {description}".strip()

            jobs.append(JobWithText(
                job_id=job_id,
                title=title,
                combined_text=combined_text
            ))

        logger.info(f"Loaded {len(jobs)} jobs with description data")
        return jobs

    def load_jobs_with_db_embeddings(self) -> List[Tuple[str, str, np.ndarray]]:
        """
        Load jobs with their titleEmbedding from database.
        Uses the same embeddings that the recommendation system uses.

        Returns:
            List of (job_id, job_title, embedding) tuples
        """
        query = """
            SELECT id, title, "titleEmbedding"
            FROM jobs
            WHERE status = 'ACTIVE'
                AND title IS NOT NULL AND title != ''
                AND "titleEmbedding" IS NOT NULL
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (self.job_limit,))
            rows = cursor.fetchall()

        import json as json_module
        jobs = []
        for row in rows:
            if row["title"] and row["titleEmbedding"]:
                emb = row["titleEmbedding"]
                if isinstance(emb, str):
                    emb = json_module.loads(emb)
                jobs.append((row["id"], row["title"], np.array(emb, dtype=np.float32)))

        logger.info(f"Loaded {len(jobs)} jobs with embeddings from database")
        return jobs

    def load_cvs_with_db_embeddings(self) -> List[Tuple[str, str, np.ndarray]]:
        """
        Load CVs with their titleEmbedding from database.
        Uses the same embeddings that the recommendation system uses.

        Returns:
            List of (cv_id, cv_title, embedding) tuples
        """
        query = """
            SELECT id, title, "titleEmbedding"
            FROM cvs
            WHERE "isMain" = true
                AND title IS NOT NULL AND title != ''
                AND "titleEmbedding" IS NOT NULL
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (self.cv_limit,))
            rows = cursor.fetchall()

        import json as json_module
        cvs = []
        for row in rows:
            if row["title"] and row["titleEmbedding"]:
                emb = row["titleEmbedding"]
                if isinstance(emb, str):
                    emb = json_module.loads(emb)
                cvs.append((row["id"], row["title"], np.array(emb, dtype=np.float32)))

        logger.info(f"Loaded {len(cvs)} CVs with embeddings from database")
        return cvs

    def build_job_title_index(self, jobs: List[Tuple[str, str]]) -> None:
        """
        Build FAISS index from job titles (legacy method).

        Args:
            jobs: List of (job_id, job_title) tuples
        """
        # Store metadata
        self.job_ids = [j[0] for j in jobs]
        self.job_titles = [j[1] for j in jobs]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(jobs)} job titles...")
        embeddings = self.embedding_service.get_embeddings_batch(
            self.job_titles,
            batch_size=32,
            show_progress=True
        )

        self.dimension = embeddings.shape[1]

        # Build FAISS index (FlatIP for cosine similarity with normalized vectors)
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(f"Built FAISS index with {len(self.job_ids)} jobs, dimension={self.dimension}")

    def build_job_combined_index(self, jobs: List[JobWithText]) -> None:
        """
        Build FAISS index from job combined texts (title + requirements).

        Args:
            jobs: List of JobWithText objects
        """
        # Store metadata
        self.job_ids = [j.job_id for j in jobs]
        self.job_titles = [j.title for j in jobs]
        combined_texts = [j.combined_text for j in jobs]

        # Generate embeddings from combined text
        logger.info(f"Generating embeddings for {len(jobs)} jobs (title + requirements)...")
        embeddings = self.embedding_service.get_embeddings_batch(
            combined_texts,
            batch_size=32,
            show_progress=True
        )

        self.dimension = embeddings.shape[1]

        # Build FAISS index (FlatIP for cosine similarity with normalized vectors)
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(f"Built FAISS index with {len(self.job_ids)} jobs, dimension={self.dimension}")

    def find_best_matching_job(self, cv_title: str) -> Tuple[str, str, float]:
        """
        Find the most similar job for a CV based on title (legacy method).

        Args:
            cv_title: The CV title to match

        Returns:
            Tuple of (job_id, job_title, similarity_score)
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_job_title_index first.")

        # Get CV title embedding
        cv_embedding = self.embedding_service.get_embedding(cv_title)
        cv_embedding = cv_embedding.reshape(1, -1)

        # Search for top-1 match
        distances, indices = self.index.search(cv_embedding, 1)

        if indices[0][0] == -1:
            return "", "", 0.0

        idx = indices[0][0]
        similarity = float(distances[0][0])

        return self.job_ids[idx], self.job_titles[idx], similarity

    def find_best_matching_job_combined(self, cv_combined_text: str) -> Tuple[str, str, float]:
        """
        Find the most similar job for a CV based on combined text (title + experience).

        Args:
            cv_combined_text: The CV combined text to match

        Returns:
            Tuple of (job_id, job_title, similarity_score)
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_job_combined_index first.")

        # Get CV combined text embedding
        cv_embedding = self.embedding_service.get_embedding(cv_combined_text)
        cv_embedding = cv_embedding.reshape(1, -1)

        # Search for top-1 match
        distances, indices = self.index.search(cv_embedding, 1)

        if indices[0][0] == -1:
            return "", "", 0.0

        idx = indices[0][0]
        similarity = float(distances[0][0])

        return self.job_ids[idx], self.job_titles[idx], similarity

    def generate(self, use_combined_text: bool = True) -> List[GroundTruthPair]:
        """
        Generate ground truth pairs.

        Args:
            use_combined_text: If True, use combined text (title + experience/requirements).
                             If False, use title only (legacy behavior).

        Returns:
            List of GroundTruthPair objects
        """
        if use_combined_text:
            return self._generate_with_combined_text()
        else:
            return self._generate_with_title_only()

    def _generate_with_title_only(self) -> List[GroundTruthPair]:
        """Generate ground truth using title only (legacy method)."""
        # Load data
        logger.info("Loading CVs and Jobs from database...")
        cvs = self.load_cvs()
        jobs = self.load_jobs()

        if not cvs:
            logger.warning("No CVs found in database")
            return []

        if not jobs:
            logger.warning("No jobs found in database")
            return []

        # Build FAISS index for jobs
        self.build_job_title_index(jobs)

        # Find best matching job for each CV
        logger.info("Finding best matching jobs for each CV...")
        pairs = []

        for cv_id, cv_title in tqdm(cvs, desc="Matching CVs"):
            if not cv_title or not cv_title.strip():
                continue

            job_id, job_title, similarity = self.find_best_matching_job(cv_title)

            if job_id:
                pairs.append(GroundTruthPair(
                    cv_id=cv_id,
                    job_id=job_id,
                    cv_title=cv_title,
                    job_title=job_title,
                    similarity=similarity
                ))

        logger.info(f"Generated {len(pairs)} ground truth pairs")
        return pairs

    def _generate_with_combined_text(self) -> List[GroundTruthPair]:
        """Generate ground truth using combined text (title + summary/description)."""
        # Load data with summary/description (consistent with recommendation system)
        logger.info("Loading CVs with summary and Jobs with description...")
        cvs = self.load_cvs_with_summary()
        jobs = self.load_jobs_with_description()

        if not cvs:
            logger.warning("No CVs found in database")
            return []

        if not jobs:
            logger.warning("No jobs found in database")
            return []

        # Build FAISS index for jobs using combined text
        self.build_job_combined_index(jobs)

        # Find best matching job for each CV
        logger.info("Finding best matching jobs for each CV (using title + summary/description)...")
        pairs = []

        for cv in tqdm(cvs, desc="Matching CVs"):
            if not cv.combined_text or not cv.combined_text.strip():
                continue

            job_id, job_title, similarity = self.find_best_matching_job_combined(cv.combined_text)

            if job_id:
                pairs.append(GroundTruthPair(
                    cv_id=cv.cv_id,
                    job_id=job_id,
                    cv_title=cv.title,
                    job_title=job_title,
                    similarity=similarity
                ))

        logger.info(f"Generated {len(pairs)} ground truth pairs (title + summary/description)")
        return pairs

    def save_to_csv(self, pairs: List[GroundTruthPair]) -> str:
        """
        Save ground truth pairs to CSV file.

        Args:
            pairs: List of GroundTruthPair objects

        Returns:
            Path to saved CSV file
        """
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        data = [
            {
                "cv_id": p.cv_id,
                "job_id": p.job_id,
                "cv_title": p.cv_title,
                "job_title": p.job_title,
                "similarity": p.similarity
            }
            for p in pairs
        ]

        df = pd.DataFrame(data)
        df.to_csv(self.output_path, index=False, encoding="utf-8")

        logger.info(f"Saved {len(pairs)} ground truth pairs to {self.output_path}")
        return str(self.output_path)

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata
        metadata_path = self.index_path.with_suffix(".meta")
        metadata = {
            "job_ids": self.job_ids,
            "job_titles": self.job_titles,
            "dimension": self.dimension
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved FAISS index to {self.index_path}")

    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.index_path.exists():
            logger.info(f"No existing index found at {self.index_path}")
            return False

        metadata_path = self.index_path.with_suffix(".meta")
        if not metadata_path.exists():
            logger.info(f"No metadata file found at {metadata_path}")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self.job_ids = metadata["job_ids"]
            self.job_titles = metadata["job_titles"]
            self.dimension = metadata["dimension"]

            logger.info(f"Loaded FAISS index with {len(self.job_ids)} jobs")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def run(self, save_index: bool = True, use_combined_text: bool = True) -> str:
        """
        Run the full ground truth generation pipeline.

        Args:
            save_index: Whether to save the FAISS index for reuse
            use_combined_text: If True, use combined text (title + experience/requirements).
                             If False, use title only (legacy behavior).

        Returns:
            Path to the saved ground truth CSV file
        """
        logger.info("=" * 50)
        logger.info("Starting Ground Truth Generation")
        logger.info("=" * 50)
        logger.info(f"CV limit: {self.cv_limit}")
        logger.info(f"Job limit: {self.job_limit}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Mode: {'COMBINED TEXT (title + experience/requirements)' if use_combined_text else 'TITLE ONLY'}")
        logger.info("=" * 50)

        # Generate pairs
        pairs = self.generate(use_combined_text=use_combined_text)

        if not pairs:
            logger.warning("No ground truth pairs generated")
            return ""

        # Save to CSV
        output_path = self.save_to_csv(pairs)

        # Optionally save index
        if save_index:
            self.save_index()

        logger.info("=" * 50)
        logger.info("Ground Truth Generation Complete")
        logger.info(f"Total pairs: {len(pairs)}")
        logger.info(f"Output file: {output_path}")
        logger.info("=" * 50)

        return output_path
