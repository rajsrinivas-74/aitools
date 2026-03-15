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
import inspect
from functools import wraps

try:
    from dotenv import load_dotenv, dotenv_values
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

try:
    import openai
except Exception:
    openai = None

from app_config import get_config


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
    
    def __init__(self, llm_model: str = None, env_file: str = None):
        """Initialize RAG indexer.
        
        Args:
            llm_model: LLM model to use for answering. If None, uses config default.
            env_file: Path to .env file
        """
        self.env_vars = load_env_file(env_file)
        
        # Use provided model or get from config
        if llm_model is None:
            config = get_config()
            llm_model = config.get_default_llm_model()
        
        self.llm_model = llm_model
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
    
    Supports both synchronous and asynchronous functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with logging
    """
    # Handle async functions
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                logger.debug("ENTER %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))
                res = await func(*args, **kwargs)
                logger.debug("EXIT %s -> %s", func.__name__, _short_repr(res))
                return res
            except Exception as e:
                logger.exception("EXCEPTION in %s: %s", func.__name__, e)
                raise
        return async_wrapper
    else:
        # Handle sync functions
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                logger.debug("ENTER %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))
                res = func(*args, **kwargs)
                logger.debug("EXIT %s -> %s", func.__name__, _short_repr(res))
                return res
            except Exception as e:
                logger.exception("EXCEPTION in %s: %s", func.__name__, e)
                raise
        return sync_wrapper


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
                                    llm_model: str = None,
                                    include_source_attribution: bool = True) -> str:
    """Generate LLM response from context blocks.
    
    This utility is used by multiple sources (including multi-source aggregation
    in the orchestrator) to synthesize responses from retrieved context.
    
    Args:
        question: The user's question
        context_blocks: List of context strings or ContextBlock objects
        llm_model: LLM model to use. If None, uses config default.
        include_source_attribution: Whether to include source info in formatted context
        
    Returns:
        LLM-generated response string
        
    Raises:
        RuntimeError: If openai package is not installed
    """
    if openai is None:
        raise RuntimeError("openai package is required (pip install openai)")
    
    # Use default model from config if not provided
    if llm_model is None:
        config = get_config()
        llm_model = config.get_default_llm_model()
    
    # Extract content from various formats
    context_texts = []
    sources_used = set()
    
    # DEBUG: Log context block types
    logger.warning("Processing %d context blocks", len(context_blocks))
    for i, block in enumerate(context_blocks):
        block_type = type(block).__name__
        logger.warning("Block %d: type=%s, len=%s", i, block_type, len(str(block)) if block else 0)
        if i == 0:
            block_str = str(block)
            logger.warning("Block 0 first 200 chars: %r", block_str[:200])
    
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
    system_prompt = """
You are an AI assistant answering questions using retrieved knowledge.

You will be given:
1. A user question
2. Context retrieved from external sources

Guidelines:
- Answer the question using ONLY the information in the provided context.
- Do not use prior knowledge or make assumptions.
- If the answer cannot be found in the context, respond with:
  "I do not know based on the provided context."
- If multiple context snippets are provided, combine relevant information.
- If the context contains conflicting information, acknowledge the uncertainty.
- Keep the response concise, factual, and directly related to the user's question.
"""

    
    # Sanitize context text to handle special characters properly
    context_text = "\n\n---\n\n".join(context_texts) if context_texts else "No context available."
    
     # Remove null bytes and normalize line endings for ASCII-safe encoding
    context_text = context_text.replace('\x00', '')  # Remove null bytes
    context_text = context_text.replace('\r\n', '\n')  # Normalize CRLF to LF
    context_text = context_text.replace('\r', '\n')  # Normalize CR to LF
    
    # Remove potentially problematic Unicode characters but keep spaces and common punctuation
    # Keep ASCII printable chars (32-126) and common whitespace (9=tab, 10=lf)
    cleaned_chars = []
    for char in context_text:
        code = ord(char)
        # Keep: spaces (32), printable ASCII (33-126), tab (9), newline (10)
        if code >= 32 and code <= 126:  # Printable ASCII
            cleaned_chars.append(char)
        elif code in (9, 10):  # Tab and newline
            cleaned_chars.append(char)
        elif code > 127:  # Try to keep valid UTF-8 characters
            try:
                char.encode('utf-8')
                cleaned_chars.append(char)
            except:
                # Skip invalid UTF-8
                pass
        # Skip all other control characters
    context_text = ''.join(cleaned_chars).strip()
    
    # Truncate if too long (prevent token overflow)
    max_context_length = 7000  # Leave room for question and response
    if len(context_text) > max_context_length:
        context_text = context_text[:max_context_length] + "\n[... context truncated ...]"
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    logger.debug("Prepared user prompt length: %d chars", len(user_prompt))
    
    # Additional safety: ensure UTF-8 encoding
    try:
        context_text_utf8 = context_text.encode('utf-8', errors='replace').decode('utf-8')
        user_prompt_utf8 = user_prompt.encode('utf-8', errors='replace').decode('utf-8')
        system_prompt_utf8 = system_prompt.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as enc_err:
        logger.warning("Could not ensure UTF-8 encoding: %s", enc_err)
        user_prompt_utf8 = user_prompt
        system_prompt_utf8 = system_prompt
    
    # Create message list with UTF-8 safe content
    messages = [
        {"role": "system", "content": system_prompt_utf8},
        {"role": "user", "content": user_prompt_utf8},
    ]
    
    # Log the exact JSON that will be sent
    import json
    json_payload = json.dumps(messages)
    logger.warning("JSON body length: %d, first 300 chars: %r", len(json_payload), json_payload[:300])
    
    # Debug: Check content before sending
    logger.warning("System prompt length: %d", len(system_prompt))
    logger.warning("User prompt length: %d", len(user_prompt))
    logger.warning("User prompt first 200 chars: %r", user_prompt[:200])
    logger.warning("User prompt last 200 chars: %r", user_prompt[-200:])
    
    # Verify JSON is valid
    import json
    try:
        json_str = json.dumps(messages)
        logger.warning("JSON payload size: %d bytes", len(json_str))
    except Exception as json_err:
        logger.error("JSON serialization error: %s", json_err)
        # Try to extract which character is causing the issue
        for i, char in enumerate(user_prompt):
            try:
                json.dumps({"test": user_prompt[:i+1]})
            except:
                logger.error("JSON error at position %d, char code: %d, char: %r", i, ord(char), char)
                break
        raise
    
    try:
        resp = openai.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as openai_err:
        # Check if it's a BadRequest/JSON error - try with NO context, just the question
        error_str = str(openai_err)
        if ("400" in error_str or "BadRequest" in error_str or "JSON" in error_str) and "could not parse" in error_str:
            logger.warning("JSON encoding error detected, trying without context")
            # Try with just question, no context
            simple_prompt = f"Answer this question: {question}"
            simple_messages = [
                {"role": "user", "content": simple_prompt},
            ]
            try:
                logger.warning("Retrying with question-only prompt")
                resp = openai.chat.completions.create(
                    model=llm_model,
                    messages=simple_messages,
                    temperature=0.0
                )
                answer = resp.choices[0].message.content.strip()
                return f"Answer (without source context): {answer}\n\nNote: Context synthesis failed due to JSON encoding issues."
            except Exception as retry_err:
                logger.error("Question-only retry also failed: %s", retry_err)
                # Return a summary of what we found instead of trying OpenAI again
                logger.warning("Giving up on OpenAI, returning summary of retrieved context")
                summary = f"Retrieved {len(context_blocks)} documents related to: {question}\n\nContext summary:\n"
                for i, block in enumerate(context_blocks[:3]):  # Just summarize first 3
                    block_str = str(block)[:300]
                    summary += f"\n{i+1}. {block_str}..."
                return summary
