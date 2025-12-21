import json
import hashlib
import logging
from typing import List, Optional
from datetime import datetime

from .connection import DatabaseConnection

logger = logging.getLogger(__name__)


class CVRepository:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def get_main_cvs(self) -> List[dict]:
        """Get all main CVs for recommendation"""
        query = """
            SELECT
                c.id,
                c.title,
                c.summary,
                c."titleEmbedding",
                c."skillsEmbedding",
                c."experienceEmbedding",
                c."contentHash"
            FROM cvs c
            WHERE c."isMain" = true
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def get_cv_skills(self, cv_id: str) -> List[dict]:
        """Get skills for a CV"""
        query = """
            SELECT "skillName"
            FROM cv_skills
            WHERE "cvId" = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (cv_id,))
            return cursor.fetchall()

    def get_cv_experiences(self, cv_id: str) -> List[dict]:
        """Get work experiences for a CV"""
        query = """
            SELECT title, description
            FROM work_experiences
            WHERE "cvId" = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (cv_id,))
            return cursor.fetchall()

    def get_all_cv_skills(self) -> dict:
        """Get all skills for all CVs in one query, returns dict {cv_id: [skills]}"""
        query = """
            SELECT "cvId", "skillName"
            FROM cv_skills
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        result = {}
        for row in rows:
            cv_id = row["cvId"]
            if cv_id not in result:
                result[cv_id] = []
            result[cv_id].append({"skillName": row["skillName"]})
        return result

    def get_all_cv_experiences(self) -> dict:
        """Get all experiences for all CVs in one query, returns dict {cv_id: [experiences]}"""
        query = """
            SELECT "cvId", title, description
            FROM work_experiences
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        result = {}
        for row in rows:
            cv_id = row["cvId"]
            if cv_id not in result:
                result[cv_id] = []
            result[cv_id].append({"title": row["title"], "description": row["description"]})
        return result

    def update_cv_embeddings(
        self,
        cv_id: str,
        title_embedding: List[float],
        skills_embedding: Optional[List[float]],
        experience_embedding: Optional[List[float]],
        content_hash: str,
    ) -> None:
        """Update embeddings for a CV"""
        query = """
            UPDATE cvs
            SET
                "titleEmbedding" = %s,
                "skillsEmbedding" = %s,
                "experienceEmbedding" = %s,
                "contentHash" = %s,
                "updatedAt" = %s
            WHERE id = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                query,
                (
                    json.dumps(title_embedding),
                    json.dumps(skills_embedding) if skills_embedding else None,
                    json.dumps(experience_embedding) if experience_embedding else None,
                    content_hash,
                    datetime.utcnow(),
                    cv_id,
                ),
            )

    def batch_update_cv_embeddings(self, cv_updates: List[tuple]) -> None:
        """Batch update embeddings for multiple CVs

        Args:
            cv_updates: List of (cv_id, title_embedding, skills_embedding, experience_embedding, content_hash)
        """
        if not cv_updates:
            return

        query = """
            UPDATE cvs
            SET
                "titleEmbedding" = %s,
                "skillsEmbedding" = %s,
                "experienceEmbedding" = %s,
                "contentHash" = %s,
                "updatedAt" = %s
            WHERE id = %s
        """
        now = datetime.utcnow()
        with self.db.get_cursor() as cursor:
            params = [
                (
                    json.dumps(title_emb) if title_emb else None,
                    json.dumps(skills_emb) if skills_emb else None,
                    json.dumps(exp_emb) if exp_emb else None,
                    content_hash,
                    now,
                    cv_id,
                )
                for cv_id, title_emb, skills_emb, exp_emb, content_hash in cv_updates
            ]
            cursor.executemany(query, params)
            logger.info(f"Batch updated embeddings for {len(cv_updates)} CVs")

    @staticmethod
    def compute_content_hash(
        cv: dict, skills: List[dict], experiences: List[dict]
    ) -> str:
        """Compute hash of CV content to detect changes (only fields used for embedding)"""
        content = {
            "title": cv.get("title") or "",
            "summary": cv.get("summary") or "",
            "skills": [(s.get("skillName") or "") for s in skills],
            "experiences": [
                (e.get("title") or "") + (e.get("description") or "")
                for e in experiences
            ],
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()


class JobRepository:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def get_active_jobs(self) -> List[dict]:
        """Get all active jobs for recommendation"""
        query = """
            SELECT
                j.id,
                j.title,
                j.description,
                j."titleEmbedding",
                j."skillsEmbedding",
                j."requirementEmbedding",
                j."contentHash"
            FROM jobs j
            WHERE j.status = 'ACTIVE'
                AND (j."expiresAt" IS NULL OR j."expiresAt" > NOW())
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def get_all_jobs(self) -> List[dict]:
        """Get all jobs (regardless of status) for similar jobs calculation"""
        query = """
            SELECT
                j.id,
                j.title,
                j.description,
                j."titleEmbedding",
                j."skillsEmbedding",
                j."requirementEmbedding",
                j."contentHash"
            FROM jobs j
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def get_job_skills(self, job_id: str) -> List[dict]:
        """Get skills for a job"""
        query = """
            SELECT "skillName"
            FROM job_skills
            WHERE "jobId" = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (job_id,))
            return cursor.fetchall()

    def get_job_requirements(self, job_id: str) -> List[dict]:
        """Get requirements for a job"""
        query = """
            SELECT title, description
            FROM job_requirements
            WHERE "jobId" = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (job_id,))
            return cursor.fetchall()

    def get_all_job_skills(self) -> dict:
        """Get all skills for all jobs in one query, returns dict {job_id: [skills]}"""
        query = """
            SELECT "jobId", "skillName"
            FROM job_skills
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        result = {}
        for row in rows:
            job_id = row["jobId"]
            if job_id not in result:
                result[job_id] = []
            result[job_id].append({"skillName": row["skillName"]})
        return result

    def get_all_job_requirements(self) -> dict:
        """Get all requirements for all jobs in one query, returns dict {job_id: [requirements]}"""
        query = """
            SELECT "jobId", title, description
            FROM job_requirements
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        result = {}
        for row in rows:
            job_id = row["jobId"]
            if job_id not in result:
                result[job_id] = []
            result[job_id].append({"title": row["title"], "description": row["description"]})
        return result

    def update_job_embeddings(
        self,
        job_id: str,
        title_embedding: List[float],
        skills_embedding: Optional[List[float]],
        requirement_embedding: Optional[List[float]],
        content_hash: str,
    ) -> None:
        """Update embeddings for a job"""
        query = """
            UPDATE jobs
            SET
                "titleEmbedding" = %s,
                "skillsEmbedding" = %s,
                "requirementEmbedding" = %s,
                "contentHash" = %s,
                "updatedAt" = %s
            WHERE id = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                query,
                (
                    json.dumps(title_embedding),
                    json.dumps(skills_embedding) if skills_embedding else None,
                    (
                        json.dumps(requirement_embedding)
                        if requirement_embedding
                        else None
                    ),
                    content_hash,
                    datetime.utcnow(),
                    job_id,
                ),
            )

    def batch_update_job_embeddings(self, job_updates: List[tuple]) -> None:
        """Batch update embeddings for multiple jobs

        Args:
            job_updates: List of (job_id, title_embedding, skills_embedding, requirement_embedding, content_hash)
        """
        if not job_updates:
            return

        query = """
            UPDATE jobs
            SET
                "titleEmbedding" = %s,
                "skillsEmbedding" = %s,
                "requirementEmbedding" = %s,
                "contentHash" = %s,
                "updatedAt" = %s
            WHERE id = %s
        """
        now = datetime.utcnow()
        with self.db.get_cursor() as cursor:
            params = [
                (
                    json.dumps(title_emb) if title_emb else None,
                    json.dumps(skills_emb) if skills_emb else None,
                    json.dumps(req_emb) if req_emb else None,
                    content_hash,
                    now,
                    job_id,
                )
                for job_id, title_emb, skills_emb, req_emb, content_hash in job_updates
            ]
            cursor.executemany(query, params)
            logger.info(f"Batch updated embeddings for {len(job_updates)} jobs")

    @staticmethod
    def compute_content_hash(
        job: dict, skills: List[dict], requirements: List[dict]
    ) -> str:
        """Compute hash of job content to detect changes (only fields used for embedding)"""
        content = {
            "title": job.get("title") or "",
            "description": job.get("description") or "",
            "skills": [(s.get("skillName") or "") for s in skills],
            "requirements": [
                (r.get("title") or "") + (r.get("description") or "")
                for r in requirements
            ],
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()


class RecommendationRepository:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def upsert_recommendations(self, cv_id: str, recommendations: List[dict]) -> None:
        """Upsert job recommendations for a CV"""
        # First, delete old recommendations for this CV
        delete_query = """
            DELETE FROM recommend_jobs_for_cv
            WHERE "cvId" = %s
        """

        # Then insert new recommendations
        insert_query = """
            INSERT INTO recommend_jobs_for_cv ("id", "cvId", "jobId", "similarity", "createdAt", "updatedAt")
            VALUES (gen_random_uuid(), %s, %s, %s, NOW(), NOW())
            ON CONFLICT ("cvId", "jobId")
            DO UPDATE SET similarity = EXCLUDED.similarity, "updatedAt" = NOW()
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(delete_query, (cv_id,))

            for rec in recommendations:
                cursor.execute(insert_query, (cv_id, rec["job_id"], rec["similarity"]))

            logger.info(
                f"Upserted {len(recommendations)} recommendations for CV: {cv_id}"
            )

    def get_recommendations_for_cv(self, cv_id: str, limit: int = 20) -> List[dict]:
        """Get job recommendations for a CV"""
        query = """
            SELECT r."jobId", r.similarity, j.title, j.description
            FROM recommend_jobs_for_cv r
            JOIN jobs j ON r."jobId" = j.id
            WHERE r."cvId" = %s
            ORDER BY r.similarity DESC
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (cv_id, limit))
            return cursor.fetchall()


class SimilarJobRepository:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def upsert_similar_jobs(self, job_id: str, similar_jobs: List[dict]) -> None:
        """Upsert similar jobs for a given job"""
        # First, delete old similar jobs for this job
        delete_query = """
            DELETE FROM similar_jobs
            WHERE "jobId" = %s
        """

        # Then insert new similar jobs
        insert_query = """
            INSERT INTO similar_jobs ("id", "jobId", "similarJobId", "similarity", "createdAt", "updatedAt")
            VALUES (gen_random_uuid(), %s, %s, %s, NOW(), NOW())
            ON CONFLICT ("jobId", "similarJobId")
            DO UPDATE SET similarity = EXCLUDED.similarity, "updatedAt" = NOW()
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(delete_query, (job_id,))

            for similar_job in similar_jobs:
                cursor.execute(
                    insert_query,
                    (job_id, similar_job["similar_job_id"], similar_job["similarity"])
                )

            logger.info(
                f"Upserted {len(similar_jobs)} similar jobs for Job: {job_id}"
            )

    def get_similar_jobs(self, job_id: str, limit: int = 10) -> List[dict]:
        """Get similar jobs for a given job"""
        query = """
            SELECT
                sj."similarJobId",
                sj.similarity,
                j.title,
                j.description,
                j.location,
                j."experienceLevel",
                j.type
            FROM similar_jobs sj
            JOIN jobs j ON sj."similarJobId" = j.id
            WHERE sj."jobId" = %s
                AND j.status = 'ACTIVE'
                AND (j."expiresAt" IS NULL OR j."expiresAt" > NOW())
            ORDER BY sj.similarity DESC
            LIMIT %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (job_id, limit))
            return cursor.fetchall()

    def batch_upsert_similar_jobs(self, all_similar_jobs: dict) -> None:
        """Batch upsert similar jobs for multiple jobs

        Args:
            all_similar_jobs: Dict mapping job_id to list of similar jobs
        """
        total_inserted = 0
        for job_id, similar_jobs in all_similar_jobs.items():
            if similar_jobs:
                self.upsert_similar_jobs(job_id, similar_jobs)
                total_inserted += len(similar_jobs)

        logger.info(f"Batch upserted {total_inserted} similar job relationships for {len(all_similar_jobs)} jobs")
