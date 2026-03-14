"""Shared utilities and base classes for RAG systems.

This module provides:
- Common utility functions (text chunking, environment loading, logging)
- Embedding model abstractions
- Base classes and interfaces (ContextBlock, BaseRetriever)
- LLM response generation utilities

Each concern is clearly separated for better cohesion.
"""

import os
import logging
from typing import List, Callable, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv, dotenv_values
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

try:
    import openai
except Exception:
    openai = None


logger = logging.getLogger(__name__)


# ============================================================================
# Error Types (Structured Error Handling)
# ============================================================================

class RAGError(Exception):
    """Base exception for RAG system errors"""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/API responses"""
        return {
            "error": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


class ValidationError(RAGError):
    """Validation error for invalid inputs"""
    pass


class ConfigurationError(RAGError):
    """Configuration error"""
    pass


class RetrieverError(RAGError):
    """Retriever operation error"""
    pass


class LLMError(RAGError):
    """LLM operation error"""
    pass


# ============================================================================
# Input Validation
# ============================================================================

def validate_query(query: str, min_length: int = 1, max_length: int = 10000) -> str:
    """Validate and normalize user query.
    
    Args:
        query: User query string
        min_length: Minimum query length
        max_length: Maximum query length
        
    Returns:
        Validated query string
        
    Raises:
        ValidationError: If query is invalid
    """
    if not isinstance(query, str):
        raise ValidationError(
            f"Query must be string, got {type(query).__name__}",
            error_code="INVALID_QUERY_TYPE"
        )
    
    query = query.strip()
    
    if len(query) < min_length:
        raise ValidationError(
            f"Query too short (min {min_length} chars)",
            error_code="QUERY_TOO_SHORT",
            context={"length": len(query), "min_length": min_length}
        )
    
    if len(query) > max_length:
        raise ValidationError(
            f"Query too long (max {max_length} chars)",
            error_code="QUERY_TOO_LONG",
            context={"length": len(query), "max_length": max_length}
        )
    
    return query


# ============================================================================
# Base Classes for Legacy Compatibility
# ============================================================================

class RAGIndexer(ABC):
    """Abstract base class for RAG indexers (legacy - for backward compatibility).
    
    VectorSearchIndexer and KnowledgeGraphIndexer inherit from this.
    Provides common initialization patterns for LLM and environment loading.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo", env_file: str = None):
        """Initialize RAG indexer.
        
        Args:
            llm_model: LLM model to use for answering
            env_file: Path to .env file
        """
        self.env_vars = load_env_file(env_file)
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self._validate_dependencies()
    
    @abstractmethod
    def _validate_dependencies(self):
        """Validate required dependencies. Implemented by subclasses."""
        pass
    
    def index_document(self, path: str, chunk_size: int = 500, overlap: int = 100):
        """Index a document.
        
        Args:
            path: Path to document file
            chunk_size: Size of chunks
            overlap: Overlap between chunks
        """
        raise NotImplementedError("Subclasses must implement index_document()")
    
    def query_index(self, question: str, k: int = 4):
        """Query the index.
        
        Args:
            question: Question to ask
            k: Number of results to return
            
        Returns:
            List of results
        """
        raise NotImplementedError("Subclasses must implement query_index()")


# ============================================================================
# Text Processing Utilities
# ============================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Chunk text into overlapping pieces.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between consecutive chunks in characters
        
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == length:
            break
        start = end - overlap
    return [c for c in chunks if c]


# ============================================================================
# Environment Loading
# ============================================================================

def load_env_file(path: str = None) -> dict:
    """Load environment variables from .env file.
    
    Args:
        path: Path to .env file (if None, searches standard locations)
        
    Returns:
        Dictionary of loaded environment variables
    """
    if not DOTENV_AVAILABLE:
        return {}
    candidates = [] if path else [os.path.expanduser("~/.env"), ".env"]
    if path:
        candidates = [path]
    for p in candidates:
        if p and os.path.exists(p):
            vals = dotenv_values(p)
            load_dotenv(p)
            return {k: v for k, v in vals.items() if k is not None}
    return {}


# ============================================================================
# Logging Utilities
# ============================================================================

def _short_repr(obj, maxlen: int = 200) -> str:
    """Create a short representation of an object for logging.
    
    Args:
        obj: Object to represent
        maxlen: Maximum length of string representation
        
    Returns:
        Short string representation
    """
    try:
        if isinstance(obj, str):
            return obj if len(obj) <= maxlen else obj[:maxlen] + "..."
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}(len={len(obj)})"
        return repr(obj)
    except Exception:
        return "<unrepresentable>"


def log_calls(func: Callable) -> Callable:
    """Decorator to log function entry, exit, and exceptions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with logging
    """
    def wrapper(*args, **kwargs):
        try:
            logger.debug("ENTER %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))
            res = func(*args, **kwargs)
            logger.debug("EXIT %s -> %s", func.__name__, _short_repr(res))
            return res
        except Exception as e:
            logger.exception("EXCEPTION in %s: %s", func.__name__, e)
            raise
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ============================================================================
# Embedding Models
# ============================================================================

class EmbeddingModel(ABC):
    """Abstract base class for embedding models.
    
    Provides interface for text embedding implementations.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model wrapper.
    
    Uses OpenAI's embedding API to generate text embeddings.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """Initialize OpenAI embedding model.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai is None:
            raise RuntimeError("openai package is required (pip install openai)")

    @log_calls
    def embed(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = openai.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except AttributeError:
            # Older OpenAI API
            response = openai.Embedding.create(input=text, model=self.model)
            return response['data'][0]['embedding']

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            response = openai.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except AttributeError:
            # Older OpenAI API
            response = openai.Embedding.create(input=texts, model=self.model)
            return [item['embedding'] for item in response['data']]


# ============================================================================
# Data Structures (Context/Results)
# ============================================================================

@dataclass
class ContextBlock:
    """Represents a single block of retrieved context with metadata.
    
    Used across all retrievers to provide structured context with
    source attribution and confidence scores.
    """
    content: str
    source: str  # e.g., "vector_search", "graph_search", "web_search"
    score: float  # Relevance or confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.
        
        Returns:
            Dictionary representation of the context block
        """
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }


# ============================================================================
# Base Retriever Interface
# ============================================================================

class BaseRetriever(ABC):
    """Abstract base class for all retriever implementations.
    
    Defines the interface that all concrete retrievers (vector, graph, web)
    must implement. Enables polymorphism and decoupling from implementations.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        pass
    
    @abstractmethod
    def get_context_blocks(self, query: str, top_k: int = 5) -> List[ContextBlock]:
        """Retrieve context blocks with metadata.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of ContextBlock objects with source and score information
        """
        pass

    @abstractmethod
    def generate_response(self, query: str) -> str:
        """Generate a response for the query using this retriever's sources.
        
        Combines retrieval and LLM generation in a single step.
        
        Args:
            query: User query
            
        Returns:
            Generated response string
        """
        pass


# ============================================================================
# LLM Response Generation
# ============================================================================

def generate_response_from_contexts(question: str, context_blocks: List, 
                                    llm_model: str = "gpt-3.5-turbo",
                                    include_source_attribution: bool = True) -> str:
    """Generate LLM response from context blocks.
    
    This utility is used by multiple sources (including multi-source aggregation
    in the orchestrator) to synthesize responses from retrieved context.
    
    Args:
        question: The user's question
        context_blocks: List of context strings or ContextBlock objects
        llm_model: LLM model to use
        include_source_attribution: Whether to include source info in formatted context
        
    Returns:
        LLM-generated response string
        
    Raises:
        RuntimeError: If openai package is not installed
    """
    if openai is None:
        raise RuntimeError("openai package is required (pip install openai)")
    
    # Extract content from various formats
    context_texts = []
    sources_used = set()
    
    for block in context_blocks:
        if isinstance(block, ContextBlock):
            content = block.content
            source = block.source
            score = block.score
            if include_source_attribution:
                formatted = f"[{source.upper()} - confidence: {score:.2f}]\n{content}"
            else:
                formatted = content
            context_texts.append(formatted)
            sources_used.add(source)
        elif isinstance(block, dict):
            content = block.get("content", "")
            source = block.get("source", "unknown")
            score = block.get("score", 0.0)
            if include_source_attribution and source != "unknown":
                formatted = f"[{source.upper()} - confidence: {score:.2f}]\n{content}"
            else:
                formatted = content
            context_texts.append(formatted)
            sources_used.add(source)
        else:
            # Handle plain strings
            context_texts.append(str(block))
    
    # Generate response
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer is not contained in the context, say you do not know the answer. "
        "Be concise and direct."
    )
    
    context_text = "\n\n---\n\n".join(context_texts) if context_texts else "No context available."
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        resp = openai.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except AttributeError:
        # Older OpenAI client API
        resp = openai.ChatCompletion.create(
            model=llm_model,
            messages=messages,
            temperature=0.0
        )
        return resp['choices'][0]['message']['content'].strip()
