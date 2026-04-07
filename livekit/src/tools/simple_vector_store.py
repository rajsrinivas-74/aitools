"""
Simple in-memory vector store implementation for demonstration.
Uses basic cosine similarity for searching.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """Simple in-memory vector store with cosine similarity search."""
    
    def __init__(self, save_path: str = "vector_index.json"):
        """Initialize the simple vector store.
        
        Args:
            save_path: Path to save/load the index
        """
        self.save_path = save_path
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> {content, vector, metadata}
        self.load()
    
    def add_document(self, doc_id: str, content: str, vector: List[float]) -> None:
        """Add a document to the store.
        
        Args:
            doc_id: Document identifier
            content: Document content
            vector: Document embedding vector
        """
        self.documents[doc_id] = {
            "content": content,
            "vector": vector,
            "doc_id": doc_id
        }
        logger.info(f"Added document {doc_id}")
    
    def search(self, query_vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results with content, score, and id
        """
        if not self.documents:
            return []
        
        # Calculate cosine similarity for all documents
        scores = []
        for doc_id, doc_data in self.documents.items():
            similarity = self._cosine_similarity(query_vector, doc_data["vector"])
            scores.append({
                "doc_id": doc_id,
                "score": similarity,
                "content": doc_data["content"]
            })
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        results = []
        for item in scores[:top_k]:
            results.append({
                "id": item["doc_id"],
                "content": item["content"],
                "score": item["score"]
            })
        
        return results
    
    def save(self) -> None:
        """Save the index to disk."""
        try:
            # Only save doc_id, content, and vector (not metadata as it's large)
            data = {
                doc_id: {
                    "content": doc["content"],
                    "vector": doc["vector"]
                }
                for doc_id, doc in self.documents.items()
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Vector index saved to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load(self) -> None:
        """Load the index from disk."""
        try:
            if Path(self.save_path).exists():
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                
                for doc_id, doc_data in data.items():
                    self.documents[doc_id] = {
                        "content": doc_data["content"],
                        "vector": doc_data["vector"],
                        "doc_id": doc_id
                    }
                
                logger.info(f"Loaded {len(self.documents)} documents from {self.save_path}")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        if len(vec1) != len(vec2):
            # Pad shorter vector with zeros
            max_len = max(len(vec1), len(vec2))
            vec1 = list(vec1) + [0] * (max_len - len(vec1))
            vec2 = list(vec2) + [0] * (max_len - len(vec2))
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
