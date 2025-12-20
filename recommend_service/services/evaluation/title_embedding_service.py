"""
Title Embedding Service using sentence-transformers.

Uses paraphrase-multilingual-mpnet-base-v2 model for multilingual title embeddings.
This is separate from the main EmbeddingService which uses PhoBERT.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TitleEmbeddingService:
    """
    Embedding service using sentence-transformers for title similarity.

    This service is optimized for title-based matching using a multilingual model
    that supports Vietnamese and other languages.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the title embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to paraphrase-multilingual-mpnet-base-v2.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self._initialized = False
        self.dimension: Optional[int] = None

    def initialize(self) -> None:
        """Load the SentenceTransformer model."""
        if self._initialized:
            return

        logger.info(f"Loading sentence-transformer model: {self.model_name}")

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self._initialized = True

            # Get embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.dimension = test_embedding.shape[1]

            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector as numpy array
        """
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                show_progress_bar=False
            )
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            2D numpy array of shape (len(texts), dimension) with normalized embeddings
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        # Replace empty texts with placeholder
        processed_texts = [t if t and t.strip() else " " for t in texts]

        try:
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                show_progress_bar=show_progress
            )

            # Set zero vectors for originally empty texts
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    embeddings[i] = np.zeros(self.dimension, dtype=np.float32)

            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self._initialized:
            self.initialize()
        return self.dimension
