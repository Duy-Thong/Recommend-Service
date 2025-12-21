import logging
from typing import List, Tuple

from recommend_service.config import settings
from recommend_service.database import (
    DatabaseConnection,
    CVRepository,
    JobRepository,
    RecommendationRepository
)
from recommend_service.models import CVData, JobData
from recommend_service.services.embedding import EmbeddingService
from recommend_service.services.similarity import SimilarityService

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(
        self,
        use_faiss: bool = True,
        shared_index_path: str = "./faiss_data/shared_jobs.faiss"
    ):
        """
        Initialize recommendation service.

        Args:
            use_faiss: Whether to use FAISS for fast similarity search (default: True)
            shared_index_path: Path to shared FAISS index (default: ./faiss_data/shared_jobs.faiss)
                              This index is shared with SimilarJobsRecommendationService
        """
        self.db = DatabaseConnection()
        self.cv_repo = CVRepository(self.db)
        self.job_repo = JobRepository(self.db)
        self.rec_repo = RecommendationRepository(self.db)
        self.embedding_service = EmbeddingService()

        # Initialize SimilarityService with shared FAISS index
        self.use_faiss = use_faiss
        self.similarity_service = SimilarityService(
            index_path=shared_index_path,
            index_type="IVFFlat",
            nlist=100,
            nprobe=10
        )

        # Cascade filtering settings from config
        self.use_cascade = settings.use_cascade_filtering
        self.cascade_k1 = settings.cascade_k1
        self.cascade_k2 = settings.cascade_k2
        self.cascade_k3 = settings.cascade_k3

        self.top_k = settings.top_k_jobs
        self.batch_size = settings.batch_size

        logger.info(f"Cascade filtering: {'ENABLED' if self.use_cascade else 'DISABLED'}")
        if self.use_cascade:
            logger.info(f"Cascade K values: K1={self.cascade_k1}, K2={self.cascade_k2}, K3={self.cascade_k3}")

    def run(self) -> dict:
        """
        Run the full recommendation pipeline:
        1. Load CVs and Jobs from DB
        2. Generate/update embeddings
        3. Calculate similarities
        4. Save recommendations

        Returns:
            Summary statistics
        """
        logger.info("=" * 50)
        logger.info("Starting recommendation pipeline")
        logger.info("=" * 50)

        stats = {
            "cvs_processed": 0,
            "jobs_processed": 0,
            "cvs_embedded": 0,
            "jobs_embedded": 0,
            "recommendations_created": 0
        }

        # Step 1: Load and process Jobs
        logger.info("Step 1: Loading and processing Jobs")
        jobs = self._load_and_embed_jobs()
        stats["jobs_processed"] = len(jobs)
        stats["jobs_embedded"] = len([j for j in jobs if j.title_embedding])
        logger.info(f"Processed {len(jobs)} jobs")

        if not jobs:
            logger.warning("No active jobs found, skipping recommendation")
            return stats

        # Step 2: Load and process CVs
        logger.info("Step 2: Loading and processing CVs")
        cvs = self._load_and_embed_cvs()
        stats["cvs_processed"] = len(cvs)
        stats["cvs_embedded"] = len([c for c in cvs if c.title_embedding])
        logger.info(f"Processed {len(cvs)} CVs")

        if not cvs:
            logger.warning("No main CVs found, skipping recommendation")
            return stats

        # Step 3: Calculate similarities and save recommendations
        logger.info("Step 3: Calculating similarities and saving recommendations")
        recommendations_count = self._generate_recommendations(cvs, jobs)
        stats["recommendations_created"] = recommendations_count

        logger.info("=" * 50)
        logger.info(f"Pipeline completed. Stats: {stats}")
        logger.info("=" * 50)

        return stats

    def _load_and_embed_jobs(self) -> List[JobData]:
        """Load jobs from DB and generate embeddings if needed"""
        jobs_data = []
        raw_jobs = self.job_repo.get_active_jobs()

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
        # Generate title embedding (title + description)
        title_text = f"{job.title} {job.description or ''}"
        title_embedding = self.embedding_service.get_embedding(title_text)

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

    def _load_and_embed_cvs(self) -> List[CVData]:
        """Load CVs from DB and generate embeddings if needed"""
        cvs_data = []
        raw_cvs = self.cv_repo.get_main_cvs()

        for raw_cv in raw_cvs:
            cv_id = raw_cv["id"]

            # Get related data
            skills = self.cv_repo.get_cv_skills(cv_id)
            experiences = self.cv_repo.get_cv_experiences(cv_id)

            # Create CVData object
            cv = CVData.from_db_row(raw_cv, skills, experiences)

            # Check if embedding needs update
            current_hash = CVRepository.compute_content_hash(raw_cv, skills, experiences)
            needs_update = cv.content_hash != current_hash or not cv.title_embedding

            if needs_update:
                logger.info(f"Generating embeddings for CV: {cv_id}")
                self._generate_cv_embeddings(cv, skills, experiences, current_hash)

            cvs_data.append(cv)

        return cvs_data

    def _generate_cv_embeddings(
        self,
        cv: CVData,
        skills: List[dict],
        experiences: List[dict],
        content_hash: str
    ) -> None:
        """Generate and save embeddings for a CV"""
        # Generate title embedding (title + summary)
        title_text = f"{cv.title} {cv.summary or ''}"
        title_embedding = self.embedding_service.get_embedding(title_text)

        # Generate skills embedding
        skills_text = " ".join([(s.get("skillName") or "") for s in skills])
        skills_embedding = self.embedding_service.get_embedding(skills_text) if skills_text.strip() else None

        # Generate experience embedding
        exp_text = " ".join([f"{e.get('title') or ''} {e.get('description') or ''}" for e in experiences])
        experience_embedding = self.embedding_service.get_embedding(exp_text) if exp_text.strip() else None

        # Update in DB
        self.cv_repo.update_cv_embeddings(
            cv.id,
            title_embedding,
            skills_embedding,
            experience_embedding,
            content_hash
        )

        # Update in memory
        cv.title_embedding = title_embedding
        cv.skills_embedding = skills_embedding
        cv.experience_embedding = experience_embedding
        cv.content_hash = content_hash

    def _generate_recommendations(self, cvs: List[CVData], jobs: List[JobData]) -> int:
        """Generate and save recommendations for all CVs using shared FAISS index"""
        total_recommendations = 0

        # Filter jobs with embeddings
        jobs_with_embeddings = [j for j in jobs if j.title_embedding]

        if not jobs_with_embeddings:
            logger.warning("No jobs with embeddings found")
            return 0

        if self.use_faiss:
            # Try to load existing shared FAISS index first
            try:
                logger.info("Attempting to load shared FAISS index")
                self.similarity_service.load_index()

                # Verify loaded index has data
                if self.similarity_service.index is not None and self.similarity_service.job_ids:
                    logger.info(f"Successfully loaded shared FAISS index with {len(self.similarity_service.job_ids)} jobs")
                else:
                    raise ValueError("Loaded index is empty")

            except Exception as e:
                logger.warning(f"Failed to load shared FAISS index: {e}")
                logger.info("Building new FAISS index from current jobs")
                self.similarity_service.build_index(jobs_with_embeddings)

                # Save the index for future use
                try:
                    self.similarity_service.save_index()
                    logger.info("FAISS index saved successfully")
                except Exception as save_error:
                    logger.warning(f"Failed to save FAISS index: {save_error}")

        # Build jobs_dict for cascade filtering
        jobs_dict = {job.id: job for job in jobs_with_embeddings}

        # Process CVs
        for cv in cvs:
            if not cv.title_embedding:
                logger.warning(f"CV {cv.id} has no embedding, skipping")
                continue

            # Choose filtering method
            if self.use_cascade:
                # Use cascade filtering (3 rounds: title -> experience -> skills)
                logger.debug(f"Using cascade filtering for CV {cv.id}")
                top_jobs = self.similarity_service.find_top_k_jobs_cascade(
                    cv,
                    jobs_dict,
                    k1=self.cascade_k1,
                    k2=self.cascade_k2,
                    k3=self.cascade_k3
                )
            else:
                # Use original method (title only)
                logger.debug(f"Using original filtering for CV {cv.id}")
                top_jobs = self.similarity_service.find_top_k_jobs(
                    cv,
                    jobs_with_embeddings,
                    self.top_k
                )

            # Save recommendations
            if top_jobs:
                self.rec_repo.upsert_recommendations(cv.id, top_jobs)
                total_recommendations += len(top_jobs)
                logger.info(f"Saved {len(top_jobs)} recommendations for CV: {cv.id}")

        return total_recommendations
