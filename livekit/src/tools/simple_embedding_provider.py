"""
Simple embedding provider implementations.
Uses sentence-transformers for high-quality embeddings with fallback to hash-based embeddings.
"""

import logging
from typing import List
import hashlib

logger = logging.getLogger(__name__)


class SimpleEmbeddingProvider:
    """Simple embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding provider.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback hash-based embeddings.")
            self.model = None
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        if self.model is not None:
            try:
                embedding = self.model.encode(text, show_progress_bar=False)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                return self._hash_based_embedding(text)
        else:
            return self._hash_based_embedding(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                return [self._hash_based_embedding(text) for text in texts]
        else:
            return [self._hash_based_embedding(text) for text in texts]
    
    @staticmethod
    def _hash_based_embedding(text: str, dim: int = 384) -> List[float]:
        """Create a simple hash-based embedding as fallback.
        
        Args:
            text: Text to embed
            dim: Dimension of the embedding vector
            
        Returns:
            Simple embedding vector
        """
        # Use hash to create pseudo-random but consistent embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to embedding values
        embedding = []
        for i in range(dim):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to -1 to 1 range
            val = (byte_val / 128.0) - 1.0
            embedding.append(val)
        
        return embedding
