"""
Ground Truth Generator for Evaluation.

Generates ground truth CV-Job pairs using title embedding similarity.
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


class GroundTruthGenerator:
    """
    Generates ground truth CV-Job pairs using title embedding similarity.

    Process:
    1. Load CVs and Jobs from database
    2. Embed all job titles using paraphrase-multilingual-mpnet-base-v2
    3. Build FAISS index from job title embeddings
    4. For each CV, find the most similar job (top-1) based on title
    5. Save pairs to CSV
    """

    def __init__(
        self,
        cv_limit: int = 2000,
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
        Load CVs from database.

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

    def load_jobs(self) -> List[Tuple[str, str]]:
        """
        Load jobs from database.

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

    def build_job_title_index(self, jobs: List[Tuple[str, str]]) -> None:
        """
        Build FAISS index from job titles.

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

    def find_best_matching_job(self, cv_title: str) -> Tuple[str, str, float]:
        """
        Find the most similar job for a CV based on title.

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

    def generate(self) -> List[GroundTruthPair]:
        """
        Generate ground truth pairs.

        Returns:
            List of GroundTruthPair objects
        """
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

    def run(self, save_index: bool = True) -> str:
        """
        Run the full ground truth generation pipeline.

        Args:
            save_index: Whether to save the FAISS index for reuse

        Returns:
            Path to the saved ground truth CSV file
        """
        logger.info("=" * 50)
        logger.info("Starting Ground Truth Generation")
        logger.info("=" * 50)
        logger.info(f"CV limit: {self.cv_limit}")
        logger.info(f"Job limit: {self.job_limit}")
        logger.info(f"Output path: {self.output_path}")
        logger.info("=" * 50)

        # Generate pairs
        pairs = self.generate()

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
