"""Shared utilities for RAG systems.

Common functions and base classes for vector search and graph search RAG implementations.
"""

import os
import logging
from typing import List, Callable
from abc import ABC, abstractmethod

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
# Utility Functions
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
            logger.info("ENTER %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))
            res = func(*args, **kwargs)
            logger.info("EXIT %s -> %s", func.__name__, _short_repr(res))
            return res
        except Exception as e:
            logger.exception("EXCEPTION in %s: %s", func.__name__, e)
            raise
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


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
# Base Classes
# ============================================================================

class RAGIndexer(ABC):
    """Abstract base class for RAG systems."""

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
        """Validate required dependencies. Must be implemented by subclasses."""
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

    def answer_with_llm(self, question: str, contexts: List[str]) -> str:
        """Generate an answer using LLM.
        
        Args:
            question: The question
            contexts: Context chunks
            
        Returns:
            Generated answer
        """
        if openai is None:
            raise RuntimeError("openai package is required (pip install openai)")
        
        system = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer is not contained in the context, say you do not know the answer. Be concise."
        )
        context_text = "\n\n---\n\n".join(contexts)
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer using only the context above."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        
        try:
            resp = openai.chat.completions.create(
                model=self.llm_model, 
                messages=messages, 
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except AttributeError:
            # Older OpenAI client API
            resp = openai.ChatCompletion.create(
                model=self.llm_model, 
                messages=messages, 
                temperature=0.0
            )
            return resp['choices'][0]['message']['content'].strip()

    def close(self):
        """Clean up resources. Override in subclasses if needed."""
        pass


# ============================================================================
# LLM Response Generation
# ============================================================================

def generate_response(question: str, contexts: List[str], llm_model: str = "gpt-3.5-turbo",
                     system_prompt: str = None, include_sources: bool = False) -> str:
    """Generate an LLM response from a question and context blocks.
    
    This is a common utility function used by all RAG modules (vector search, graph search, web search).
    
    Args:
        question: The user's question
        contexts: List of context strings to use for answering
        llm_model: LLM model to use (default: gpt-3.5-turbo)
        system_prompt: Optional custom system prompt (uses default if None)
        include_sources: If True and contexts have source info, include it in response
        
    Returns:
        LLM-generated response
        
    Raises:
        RuntimeError: If openai package is not installed
        Exception: If LLM call fails
    """
    if openai is None:
        raise RuntimeError("openai package is required (pip install openai)")
    
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer is not contained in the context, say you do not know the answer. Be concise and direct."
        )
    
    # Format context text
    if not contexts:
        context_text = "No context available."
    elif include_sources:
        # Format with source attribution if available
        context_text = "\n\n---\n\n".join(contexts)
    else:
        # Simple concatenation
        context_text = "\n\n---\n\n".join(contexts)
    
    # Build prompt
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


def generate_response_from_contexts(question: str, context_blocks: List[dict], 
                                      llm_model: str = "gpt-3.5-turbo",
                                      include_source_attribution: bool = True) -> dict:
    """Generate LLM response from structured context blocks.
    
    This function is designed to work with context blocks that have metadata like source and score.
    
    Args:
        question: The user's question
        context_blocks: List of context block dictionaries with keys:
            - content (str): The context text
            - source (str, optional): Where the context came from (e.g., "vector_search")
            - score (float, optional): Relevance score
            - metadata (dict, optional): Additional metadata
        llm_model: LLM model to use
        include_source_attribution: Whether to include source info in formatted context
        
    Returns:
        Dictionary with:
        - response: The LLM-generated response
        - sources_used: List of unique sources in the context
        - context_count: Number of context blocks used
    """
    if openai is None:
        raise RuntimeError("openai package is required (pip install openai)")
    
    # Extract content and track sources
    context_texts = []
    sources_used = set()
    
    for block in context_blocks:
        if isinstance(block, dict):
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
        "You are a helpful assistant. Use the provided context from multiple sources to answer the user's question. "
        "Synthesize information across all sources when relevant. "
        "If the answer is not contained in the context, say you do not know the answer."
    )
    
    context_text = "\n\n---\n\n".join(context_texts) if context_texts else "No context available."
    user_prompt = f"Context from multiple sources:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    
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
        response_text = resp.choices[0].message.content.strip()
    except AttributeError:
        # Older OpenAI client API
        resp = openai.ChatCompletion.create(
            model=llm_model,
            messages=messages,
            temperature=0.0
        )
        response_text = resp['choices'][0]['message']['content'].strip()
    
    return {
        "response": response_text,
        "sources_used": list(sources_used),
        "context_count": len(context_texts),
        "context_blocks": context_blocks
    }


def close(self):
        """Clean up resources. Override in subclasses if needed."""
        pass


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

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
    """OpenAI embedding model wrapper."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """Initialize OpenAI embedding model.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key
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
            # Older API
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
            # Older API
            response = openai.Embedding.create(input=texts, model=self.model)
            return [item['embedding'] for item in response['data']]
