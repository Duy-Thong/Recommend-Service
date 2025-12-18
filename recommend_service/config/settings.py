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
        "vovanphuc/phobert-base-v2"
    )

    # Recommendation settings
    top_k_jobs: int = int(os.getenv("TOP_K_JOBS", "20"))

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
