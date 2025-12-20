import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:1234@localhost:5432/jobsconnect?schema=public"
    )

    # Embedding model
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    )

    # Recommendation settings
    top_k_jobs: int = int(os.getenv("TOP_K_JOBS", "20"))

    # Cascade filtering settings
    use_cascade_filtering: bool = os.getenv("USE_CASCADE_FILTERING", "true").lower() == "true"
    cascade_k1: int = int(os.getenv("CASCADE_K1", "1000"))  # Round 1: Title filtering
    cascade_k2: int = int(os.getenv("CASCADE_K2", "300"))   # Round 2: Experience filtering
    cascade_k3: int = int(os.getenv("CASCADE_K3", "30"))    # Round 3: Skills filtering

    # Scheduler settings (in hours)
    schedule_interval_hours: int = int(os.getenv("SCHEDULE_INTERVAL_HOURS", "12"))

    # Batch size for processing
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))

    @property
    def database_url_clean(self) -> str:
        """Remove schema parameter for psycopg2 compatibility"""
        url = self.database_url
        if "?schema=" in url:
            url = url.split("?schema=")[0]
        return url


settings = Settings()
