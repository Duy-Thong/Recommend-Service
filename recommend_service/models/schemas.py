from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CVData:
    id: str
    title: str
    current_position: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experiences: List[str] = field(default_factory=list)
    title_embedding: Optional[List[float]] = None
    skills_embedding: Optional[List[float]] = None
    experience_embedding: Optional[List[float]] = None
    content_hash: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: dict, skills: List[dict], experiences: List[dict]) -> "CVData":
        return cls(
            id=row["id"],
            title=row.get("title", ""),
            current_position=row.get("currentPosition"),
            skills=[s["skillName"] for s in skills],
            experiences=[f"{e['title']} - {e.get('description', '')}" for e in experiences],
            title_embedding=row.get("titleEmbedding"),
            skills_embedding=row.get("skillsEmbedding"),
            experience_embedding=row.get("experienceEmbedding"),
            content_hash=row.get("contentHash")
        )


@dataclass
class JobData:
    id: str
    title: str
    skills: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    title_embedding: Optional[List[float]] = None
    skills_embedding: Optional[List[float]] = None
    requirement_embedding: Optional[List[float]] = None
    content_hash: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: dict, skills: List[dict], requirements: List[dict]) -> "JobData":
        return cls(
            id=row["id"],
            title=row.get("title", ""),
            skills=[s["skillName"] for s in skills],
            requirements=[f"{r['title']} - {r.get('description', '')}" for r in requirements],
            title_embedding=row.get("titleEmbedding"),
            skills_embedding=row.get("skillsEmbedding"),
            requirement_embedding=row.get("requirementEmbedding"),
            content_hash=row.get("contentHash")
        )


@dataclass
class RecommendationResult:
    cv_id: str
    job_id: str
    similarity: float
    job_title: Optional[str] = None
