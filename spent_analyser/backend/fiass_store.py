"""
FAISS Vector Store for transaction embeddings and semantic search.
"""

import logging
import json
import pickle
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from backend.models import TransactionChunk, ProcessedTransaction
from config.settings import settings

logger = logging.getLogger(__name__)


class FAISSStoreError(Exception):
    """Custom exception for FAISS operations."""
    pass


class FAISSStore:
    """FAISS vector store for transaction embeddings and similarity search."""
    
    def __init__(self):
        """Initialize FAISS store."""
        if faiss is None:
            raise FAISSStoreError("FAISS not installed")
        
        self.logger = logging.getLogger(__name__)
        self.index: Optional[Any] = None
        self.metadata: List[Dict] = []
        self.chunks: List[TransactionChunk] = []
        self.index_path = settings.FAISS_INDEX_DIR / f"{settings.FAISS_INDEX_NAME}.index"
        self.metadata_path = settings.FAISS_INDEX_DIR / f"{settings.FAISS_INDEX_NAME}_metadata.json"
        
        self._load_index_if_exists()
    
    def add_embeddings(
        self,
        chunks: List[TransactionChunk],
        embeddings: List[List[float]]
    ) -> None:
        """Add embeddings to FAISS index."""
        self.logger.debug(f"Adding {len(chunks)} embeddings to FAISS index")
        if not chunks or not embeddings:
            self.logger.warning("No chunks or embeddings provided")
            return
        
        if len(chunks) != len(embeddings):
            raise FAISSStoreError("Chunks and embeddings count mismatch")
        
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            if self.index is None:
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
            
            self.index.add(embeddings_array)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                self.chunks.append(chunk)
                
                metadata_entry = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "timestamp": datetime.now().isoformat(),
                    "transaction_count": len(chunk.transactions),
                }
                self.metadata.append(metadata_entry)
            
            self.logger.info(f"Added {len(chunks)} chunks to FAISS index")
            
        except Exception as e:
            self.logger.error(f"Error adding embeddings: {str(e)}")
            raise FAISSStoreError(f"Failed to add embeddings: {str(e)}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar transactions."""
        if self.index is None or len(self.metadata) == 0:
            return []
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_array, min(k, len(self.metadata)))
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result["distance"] = float(distance)
                    result["similarity_score"] = float(1 / (1 + distance))
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching index: {str(e)}")
            return []
    
    def save_index(self) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            return
        
        try:
            faiss.write_index(self.index, str(self.index_path))
            
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            self.logger.info(f"Saved FAISS index to {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
    
    def _load_index_if_exists(self) -> None:
        """Load existing FAISS index if available."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                self.logger.info(f"Loaded FAISS index with {len(self.metadata)} entries")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing index: {str(e)}")
    
    def clear(self) -> None:
        """Clear all data."""
        self.index = None
        self.metadata = []
        self.chunks = []