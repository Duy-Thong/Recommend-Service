import logging
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

from recommend_service.config import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self):
        self.connection_string = settings.database_url_clean

    @contextmanager
    def get_connection(self) -> Generator:
        """Get a database connection with automatic cleanup"""
        conn = None
        try:
            conn = psycopg.connect(self.connection_string)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True) -> Generator:
        """Get a database cursor with automatic cleanup"""
        with self.get_connection() as conn:
            row_factory = dict_row if dict_cursor else None
            cursor = conn.cursor(row_factory=row_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database operation error: {e}")
                raise
            finally:
                cursor.close()

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
