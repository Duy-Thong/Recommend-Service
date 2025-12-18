import logging
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from recommend_service.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embedding_model
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    def initialize(self) -> None:
        """Load the PhoBERT model and tokenizer"""
        if self._initialized:
            return

        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            return []

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            return embedding.cpu().tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not self._initialized:
            self.initialize()

        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[] for _ in texts]

        try:
            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)

            result = embeddings.cpu().tolist()

            # Map back to original positions (handle empty texts)
            final_result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    final_result.append(result[valid_idx])
                    valid_idx += 1
                else:
                    final_result.append([])

            return final_result
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [[] for _ in texts]

    def combine_texts_for_embedding(self, texts: List[str]) -> str:
        """Combine multiple texts into one for embedding"""
        valid_texts = [t for t in texts if t and t.strip()]
        return " ".join(valid_texts)
