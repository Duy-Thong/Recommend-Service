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
        """Load jobs from DB and generate embeddings if needed (batch optimized)"""
        jobs_data = []
        jobs_to_embed = []  # (job, skills, requirements, content_hash)

        # Batch load all data in 3 queries instead of N*2 queries
        logger.info("Loading all jobs data...")
        raw_jobs = self.job_repo.get_active_jobs()
        logger.info(f"Loaded {len(raw_jobs)} jobs")

        logger.info("Loading all job skills...")
        all_skills = self.job_repo.get_all_job_skills()
        logger.info(f"Loaded skills for {len(all_skills)} jobs")

        logger.info("Loading all job requirements...")
        all_requirements = self.job_repo.get_all_job_requirements()
        logger.info(f"Loaded requirements for {len(all_requirements)} jobs")

        # Phase 1: Process all jobs and identify which need embedding
        logger.info("Processing jobs...")
        for raw_job in raw_jobs:
            job_id = raw_job["id"]

            # Get related data from pre-loaded dicts
            skills = all_skills.get(job_id, [])
            requirements = all_requirements.get(job_id, [])

            # Create JobData object
            job = JobData.from_db_row(raw_job, skills, requirements)

            # Check if embedding needs update
            current_hash = JobRepository.compute_content_hash(raw_job, skills, requirements)
            needs_update = job.content_hash != current_hash or not job.title_embedding

            if needs_update:
                jobs_to_embed.append((job, skills, requirements, current_hash))

            jobs_data.append(job)

        # Phase 2: Batch embed jobs that need updating
        if jobs_to_embed:
            logger.info(f"Batch embedding {len(jobs_to_embed)} jobs...")
            self._batch_embed_jobs(jobs_to_embed)

        return jobs_data

    def _batch_embed_jobs(self, jobs_to_embed: list) -> None:
        """Batch embed multiple jobs at once for better performance"""
        batch_size = self.batch_size
        total_batches = (len(jobs_to_embed) + batch_size - 1) // batch_size

        for i in range(0, len(jobs_to_embed), batch_size):
            batch = jobs_to_embed[i:i + batch_size]
            logger.info(f"Processing job batch {i // batch_size + 1}/{total_batches}")

            # Prepare texts for batch embedding
            title_texts = []
            skills_texts = []
            req_texts = []

            for job, skills, requirements, _ in batch:
                # Title text
                title_texts.append(f"{job.title} {job.description or ''}")

                # Skills text
                skills_text = " ".join([(s.get("skillName") or "") for s in skills])
                skills_texts.append(skills_text if skills_text.strip() else "")

                # Requirements text
                req_text = " ".join([f"{r.get('title') or ''} {r.get('description') or ''}" for r in requirements])
                req_texts.append(req_text if req_text.strip() else "")

            # Batch embed all texts
            title_embeddings = self.embedding_service.get_embeddings_batch(title_texts)
            skills_embeddings = self.embedding_service.get_embeddings_batch(skills_texts)
            req_embeddings = self.embedding_service.get_embeddings_batch(req_texts)

            # Prepare batch updates for DB
            job_updates = []
            for idx, (job, _, _, content_hash) in enumerate(batch):
                title_emb = title_embeddings[idx] if title_embeddings[idx] else None
                skills_emb = skills_embeddings[idx] if skills_embeddings[idx] else None
                req_emb = req_embeddings[idx] if req_embeddings[idx] else None

                # Add to batch update list
                job_updates.append((job.id, title_emb, skills_emb, req_emb, content_hash))

                # Update in memory
                job.title_embedding = title_emb
                job.skills_embedding = skills_emb
                job.requirement_embedding = req_emb
                job.content_hash = content_hash

            # Batch save to DB
            self.job_repo.batch_update_job_embeddings(job_updates)

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
        """Load CVs from DB and generate embeddings if needed (batch optimized)"""
        cvs_data = []
        cvs_to_embed = []  # (cv, skills, experiences, content_hash)

        # Batch load all data in 3 queries instead of N*2 queries
        logger.info("Loading all CVs data...")
        raw_cvs = self.cv_repo.get_main_cvs()
        logger.info(f"Loaded {len(raw_cvs)} CVs")

        logger.info("Loading all CV skills...")
        all_skills = self.cv_repo.get_all_cv_skills()
        logger.info(f"Loaded skills for {len(all_skills)} CVs")

        logger.info("Loading all CV experiences...")
        all_experiences = self.cv_repo.get_all_cv_experiences()
        logger.info(f"Loaded experiences for {len(all_experiences)} CVs")

        # Phase 1: Process all CVs and identify which need embedding
        logger.info("Processing CVs...")
        for raw_cv in raw_cvs:
            cv_id = raw_cv["id"]

            # Get related data from pre-loaded dicts
            skills = all_skills.get(cv_id, [])
            experiences = all_experiences.get(cv_id, [])

            # Create CVData object
            cv = CVData.from_db_row(raw_cv, skills, experiences)

            # Check if embedding needs update
            current_hash = CVRepository.compute_content_hash(raw_cv, skills, experiences)
            needs_update = cv.content_hash != current_hash or not cv.title_embedding

            if needs_update:
                cvs_to_embed.append((cv, skills, experiences, current_hash))

            cvs_data.append(cv)

        # Phase 2: Batch embed CVs that need updating
        if cvs_to_embed:
            logger.info(f"Batch embedding {len(cvs_to_embed)} CVs...")
            self._batch_embed_cvs(cvs_to_embed)

        return cvs_data

    def _batch_embed_cvs(self, cvs_to_embed: list) -> None:
        """Batch embed multiple CVs at once for better performance"""
        batch_size = self.batch_size
        total_batches = (len(cvs_to_embed) + batch_size - 1) // batch_size

        for i in range(0, len(cvs_to_embed), batch_size):
            batch = cvs_to_embed[i:i + batch_size]
            logger.info(f"Processing CV batch {i // batch_size + 1}/{total_batches}")

            # Prepare texts for batch embedding
            title_texts = []
            skills_texts = []
            exp_texts = []

            for cv, skills, experiences, _ in batch:
                # Title text
                title_texts.append(f"{cv.title} {cv.summary or ''}")

                # Skills text
                skills_text = " ".join([(s.get("skillName") or "") for s in skills])
                skills_texts.append(skills_text if skills_text.strip() else "")

                # Experience text
                exp_text = " ".join([f"{e.get('title') or ''} {e.get('description') or ''}" for e in experiences])
                exp_texts.append(exp_text if exp_text.strip() else "")

            # Batch embed all texts
            title_embeddings = self.embedding_service.get_embeddings_batch(title_texts)
            skills_embeddings = self.embedding_service.get_embeddings_batch(skills_texts)
            exp_embeddings = self.embedding_service.get_embeddings_batch(exp_texts)

            # Prepare batch updates for DB
            cv_updates = []
            for idx, (cv, _, _, content_hash) in enumerate(batch):
                title_emb = title_embeddings[idx] if title_embeddings[idx] else None
                skills_emb = skills_embeddings[idx] if skills_embeddings[idx] else None
                exp_emb = exp_embeddings[idx] if exp_embeddings[idx] else None

                # Add to batch update list
                cv_updates.append((cv.id, title_emb, skills_emb, exp_emb, content_hash))

                # Update in memory
                cv.title_embedding = title_emb
                cv.skills_embedding = skills_emb
                cv.experience_embedding = exp_emb
                cv.content_hash = content_hash

            # Batch save to DB
            self.cv_repo.batch_update_cv_embeddings(cv_updates)

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
            # Always rebuild FAISS index to ensure it matches current jobs in DB
            logger.info("Building FAISS index from current jobs")
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
