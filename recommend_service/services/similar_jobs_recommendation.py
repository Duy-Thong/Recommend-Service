import logging
from typing import List

from recommend_service.config import settings
from recommend_service.database import (
    DatabaseConnection,
    JobRepository,
    SimilarJobRepository
)
from recommend_service.models import JobData
from recommend_service.services.embedding import EmbeddingService
from recommend_service.services.similar_jobs import SimilarJobsService

logger = logging.getLogger(__name__)


class SimilarJobsRecommendationService:
    """
    Service to calculate and store similar jobs based on title embeddings.
    """

    def __init__(self, shared_index_path: str = "./faiss_data/shared_jobs.faiss"):
        self.db = DatabaseConnection()
        self.job_repo = JobRepository(self.db)
        self.similar_job_repo = SimilarJobRepository(self.db)
        self.embedding_service = EmbeddingService()
        # Use shared FAISS index
        self.similar_jobs_service = SimilarJobsService(
            index_path=shared_index_path,
            index_type="IVFFlat",
            nlist=100,
            nprobe=10
        )
        self.top_k = 10  # Get top 10 similar jobs

    def run(self) -> dict:
        """
        Run the similar jobs calculation pipeline:
        1. Load all active jobs from DB
        2. Generate/update embeddings if needed
        3. Calculate similarities between jobs
        4. Save similar jobs to database

        Returns:
            Summary statistics
        """
        logger.info("=" * 50)
        logger.info("Starting similar jobs calculation pipeline")
        logger.info("=" * 50)

        stats = {
            "jobs_processed": 0,
            "jobs_embedded": 0,
            "similar_jobs_created": 0
        }

        # Step 1: Load and process Jobs
        logger.info("Step 1: Loading and processing Jobs")
        jobs = self._load_and_embed_jobs()
        stats["jobs_processed"] = len(jobs)
        stats["jobs_embedded"] = len([j for j in jobs if j.title_embedding])
        logger.info(f"Processed {len(jobs)} jobs")

        if not jobs or len(jobs) < 2:
            logger.warning("Not enough jobs to calculate similarities (need at least 2)")
            return stats

        # Step 2: Calculate similar jobs
        logger.info("Step 2: Calculating similar jobs")
        similar_jobs_count = self._generate_similar_jobs(jobs)
        stats["similar_jobs_created"] = similar_jobs_count

        logger.info("=" * 50)
        logger.info(f"Pipeline completed. Stats: {stats}")
        logger.info("=" * 50)

        return stats

    def _load_and_embed_jobs(self) -> List[JobData]:
        """Load jobs from DB and generate embeddings if needed"""
        jobs_data = []
        raw_jobs = self.job_repo.get_all_jobs()  # Get ALL jobs, not just active ones

        for raw_job in raw_jobs:
            job_id = raw_job["id"]

            # Get related data
            skills = self.job_repo.get_job_skills(job_id)
            requirements = self.job_repo.get_job_requirements(job_id)

            # Create JobData object
            job = JobData.from_db_row(raw_job, skills, requirements)

            # Check if embedding needs update
            current_hash = JobRepository.compute_content_hash(raw_job, skills, requirements)
            needs_update = job.content_hash != current_hash or not job.title_embedding

            if needs_update:
                logger.info(f"Generating embeddings for job: {job_id}")
                self._generate_job_embeddings(job, skills, requirements, current_hash)

            jobs_data.append(job)

        return jobs_data

    def _generate_job_embeddings(
        self,
        job: JobData,
        skills: List[dict],
        requirements: List[dict],
        content_hash: str
    ) -> None:
        """Generate and save embeddings for a job"""
        # Generate title embedding
        title_embedding = self.embedding_service.get_embedding(job.title)

        # Generate skills embedding
        skills_text = " ".join([(s.get("skillName") or "") for s in skills])
        skills_embedding = self.embedding_service.get_embedding(skills_text) if skills_text.strip() else None

        # Generate requirements embedding
        req_text = " ".join([f"{r.get('title') or ''} {r.get('description') or ''}" for r in requirements])
        requirement_embedding = self.embedding_service.get_embedding(req_text) if req_text.strip() else None

        # Update in DB
        self.job_repo.update_job_embeddings(
            job.id,
            title_embedding,
            skills_embedding,
            requirement_embedding,
            content_hash
        )

        # Update in memory
        job.title_embedding = title_embedding
        job.skills_embedding = skills_embedding
        job.requirement_embedding = requirement_embedding
        job.content_hash = content_hash

    def _generate_similar_jobs(self, jobs: List[JobData]) -> int:
        """Generate and save similar jobs for all jobs"""
        total_similar_jobs = 0

        # Filter jobs with embeddings
        jobs_with_embeddings = [j for j in jobs if j.title_embedding]

        if not jobs_with_embeddings or len(jobs_with_embeddings) < 2:
            logger.warning("Not enough jobs with embeddings found (need at least 2)")
            return 0

        logger.info(f"Calculating similar jobs for {len(jobs_with_embeddings)} jobs")

        # Build shared FAISS index
        logger.info("Building shared FAISS index from jobs")
        self.similar_jobs_service.build_index(jobs_with_embeddings)

        # Save the shared index for reuse by RecommendationService
        try:
            self.similar_jobs_service.save_index()
            logger.info("Saved shared FAISS index successfully")
        except Exception as e:
            logger.warning(f"Failed to save shared FAISS index: {e}")

        # Calculate similar jobs for all jobs
        all_similar_jobs = self.similar_jobs_service.batch_calculate_similar_jobs(
            jobs_with_embeddings,
            self.top_k,
            use_faiss=True
        )

        # Save to database
        if all_similar_jobs:
            self.similar_job_repo.batch_upsert_similar_jobs(all_similar_jobs)
            total_similar_jobs = sum(len(similar_jobs) for similar_jobs in all_similar_jobs.values())

        return total_similar_jobs
