"""
Vector Search Tool for LiveKit Agent - Standalone Vector Search Implementation
Provides vector-based document retrieval and search capabilities.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    content: str
    score: float
    source: str
    id: str = ""


@dataclass
class ContextBlock:
    """Represents a context block for answer generation."""
    content: str
    score: float
    source: str


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def add_document(self, doc_id: str, content: str, vector: List[float]) -> None:
        """Add a document to the vector store."""
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Persist the vector store."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding generation."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class VectorSearchAgentTool:
    """Standalone Vector Search Tool for LiveKit agents.
    
    This class provides vector-based semantic search capabilities with support for:
    - Document indexing and retrieval
    - Async search operations
    - Context-based answer generation
    - Pluggable vector storage and embedding backends
    """
    
    def __init__(
        self, 
        vector_store: Optional[VectorStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_generator: Optional[callable] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the vector search tool.
        
        Args:
            vector_store: Custom vector storage backend (injectable)
            embedding_provider: Custom embedding provider (injectable)
            llm_generator: Custom LLM response generator (injectable)
            config: Configuration dictionary with optional keys:
                - index_path: Path to save/load index
                - embedding_model: Model name for embeddings
                - vector_dim: Vector dimension size
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_generator = llm_generator
        self.config = config or {}
        self.documents: Dict[str, str] = {}  # Store original documents
        self.document_metadata: Dict[str, Dict] = {}  # Store metadata
        
        logger.info("VectorSearchAgentTool initialized as standalone service")
    
    def set_vector_store(self, vector_store: VectorStore) -> None:
        """Set or update the vector storage backend."""
        self.vector_store = vector_store
        logger.info("Vector store backend updated")
    
    def set_embedding_provider(self, embedding_provider: EmbeddingProvider) -> None:
        """Set or update the embedding provider."""
        self.embedding_provider = embedding_provider
        logger.info("Embedding provider updated")
    
    def set_llm_generator(self, generator: callable) -> None:
        """Set or update the LLM response generator."""
        self.llm_generator = generator
        logger.info("LLM generator updated")
    
    def add_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a document to the vector search index.
        
        Args:
            doc_id: Unique document identifier
            content: Document content text
            metadata: Optional metadata (source, timestamp, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.embedding_provider:
            logger.error("Embedding provider not configured")
            return False
        
        if not self.vector_store:
            logger.error("Vector store not configured")
            return False
        
        try:
            # Generate embedding for the document
            embedding = self.embedding_provider.embed_text(content)
            
            # Store in vector store
            self.vector_store.add_document(doc_id, content, embedding)
            
            # Store document and metadata
            self.documents[doc_id] = content
            self.document_metadata[doc_id] = metadata or {"source": "unknown"}
            
            logger.info(f"Document added: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Add multiple documents to the index.
        
        Args:
            documents: List of dicts with 'id', 'content', and optional 'metadata'
            
        Returns:
            Dictionary mapping doc_id to success status
        """
        if not self.embedding_provider:
            logger.error("Embedding provider not configured")
            return {doc["id"]: False for doc in documents}
        
        results = {}
        try:
            # Extract content for batch embedding
            contents = [doc["content"] for doc in documents]
            embeddings = self.embedding_provider.embed_batch(contents)
            
            # Add each document
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc["id"]
                content = doc["content"]
                metadata = doc.get("metadata", {})
                
                try:
                    self.vector_store.add_document(doc_id, content, embedding)
                    self.documents[doc_id] = content
                    self.document_metadata[doc_id] = metadata
                    results[doc_id] = True
                except Exception as e:
                    logger.error(f"Failed to add document {doc_id}: {e}")
                    results[doc_id] = False
                    
        except Exception as e:
            logger.error(f"Batch document addition failed: {e}")
            for doc in documents:
                results[doc["id"]] = False
        
        return results
    
    async def search_knowledge_base(self, query: str, top_k: int = 4) -> str:
        """Search the knowledge base for relevant information.
        
        Args:
            query: Search query/question
            top_k: Number of top results to return (default: 4)
            
        Returns:
            JSON string with search results
        """
        if not self.vector_store or not self.embedding_provider:
            return json.dumps({
                "error": "Vector search not fully configured",
                "results": []
            })
        
        try:
            loop = asyncio.get_event_loop()
            
            # Generate query embedding
            query_embedding = await loop.run_in_executor(
                None,
                self.embedding_provider.embed_text,
                query
            )
            
            # Search vector store
            raw_results = await loop.run_in_executor(
                None,
                self.vector_store.search,
                query_embedding,
                top_k
            )
            
            # Format results
            formatted_results = []
            for result in raw_results:
                doc_id = result.get("id", "")
                metadata = self.document_metadata.get(doc_id, {})
                
                formatted_results.append({
                    "content": result.get("content", ""),
                    "score": float(result.get("score", 0.0)),
                    "source": metadata.get("source", "vector_search"),
                    "id": doc_id,
                    "metadata": metadata
                })
            
            return json.dumps({
                "query": query,
                "count": len(formatted_results),
                "results": formatted_results
            })
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return json.dumps({
                "error": str(e),
                "query": query,
                "results": []
            })
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Index documents into the vector store.
        
        Args:
            documents: List of documents with structure:
                {
                    "id": "doc_id",
                    "content": "document text",
                    "metadata": {"source": "...", ...}
                }
            
        Returns:
            JSON string with indexing results
        """
        if not self.vector_store or not self.embedding_provider:
            return json.dumps({
                "error": "Vector search not fully configured",
                "indexed": 0
            })
        
        try:
            loop = asyncio.get_event_loop()
            
            # Add documents in batch
            results = await loop.run_in_executor(
                None,
                self.add_documents_batch,
                documents
            )
            
            indexed_count = sum(1 for v in results.values() if v)
            failed_count = len(results) - indexed_count
            
            # Persist the vector store
            await loop.run_in_executor(None, self.vector_store.save)
            
            return json.dumps({
                "indexed": indexed_count,
                "failed": failed_count,
                "total": len(documents),
                "message": f"Indexed {indexed_count} documents successfully"
            })
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return json.dumps({
                "error": str(e),
                "indexed": 0
            })
    
    async def get_answer(self, query: str, top_k: int = 5) -> str:
        """Search knowledge base and generate an answer.
        
        Args:
            query: User question
            top_k: Number of context blocks to retrieve
            
        Returns:
            JSON string with answer and source information
        """
        if not self.vector_store or not self.embedding_provider:
            return json.dumps({
                "error": "Vector search not fully configured",
                "answer": ""
            })
        
        try:
            loop = asyncio.get_event_loop()
            
            # Generate query embedding
            query_embedding = await loop.run_in_executor(
                None,
                self.embedding_provider.embed_text,
                query
            )
            
            # Retrieve context blocks
            raw_results = await loop.run_in_executor(
                None,
                self.vector_store.search,
                query_embedding,
                top_k
            )
            
            # Build context blocks
            context_blocks = []
            context_text = ""
            
            for result in raw_results:
                doc_id = result.get("id", "")
                metadata = self.document_metadata.get(doc_id, {})
                content = result.get("content", "")
                
                context_blocks.append({
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "score": float(result.get("score", 0.0)),
                    "source": metadata.get("source", "unknown"),
                    "id": doc_id
                })
                context_text += f"\n{content}"
            
            # Generate answer if LLM generator is available
            answer = ""
            if self.llm_generator:
                answer = await loop.run_in_executor(
                    None,
                    self.llm_generator,
                    query,
                    context_text
                )
            else:
                answer = "LLM generator not configured. Please provide context above."
            
            return json.dumps({
                "query": query,
                "answer": answer,
                "sources": context_blocks,
                "source_count": len(context_blocks)
            })
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return json.dumps({
                "error": str(e),
                "query": query,
                "answer": ""
            })
    
    def _read_document_file(self, file_path: str) -> Optional[str]:
        """Read document content from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            File content as string, or None if reading fails
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Handle PDF files
            if file_ext == '.pdf':
                return self._extract_pdf_text(file_path)
            
            # Text file formats
            if file_ext in ['.txt', '.md', '.markdown', '.rst', '.json', '.xml', '.yaml', '.yml', '.csv']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Try reading as text by default
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # If UTF-8 fails, try with latin-1 encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def _extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content, or None if extraction fails
        """
        try:
            # Try using PyPDF2
            try:
                from PyPDF2 import PdfReader
                
                text_content = []
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    num_pages = len(pdf_reader.pages)
                    
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            text_content.append(f"[Page {page_num + 1}]\n{text}")
                
                if text_content:
                    logger.info(f"Extracted {num_pages} pages from {pdf_path}")
                    return "\n\n".join(text_content)
                else:
                    logger.warning(f"No text extracted from PDF: {pdf_path}")
                    return None
                    
            except ImportError:
                # Fallback: Try pdfplumber
                try:
                    import pdfplumber
                    
                    text_content = []
                    with pdfplumber.open(pdf_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                text_content.append(f"[Page {page_num + 1}]\n{text}")
                    
                    if text_content:
                        logger.info(f"Extracted {len(pdf.pages)} pages from {pdf_path}")
                        return "\n\n".join(text_content)
                    else:
                        logger.warning(f"No text extracted from PDF: {pdf_path}")
                        return None
                        
                except ImportError:
                    logger.error(f"PDF libraries not available. Install: pip install PyPDF2 pdfplumber")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to extract PDF text from {pdf_path}: {e}")
            return None
    
    def _get_supported_files(self, folder_path: str) -> List[Path]:
        """Get all supported document files from a folder.
        
        Args:
            folder_path: Path to the documents folder
            
        Returns:
            List of Path objects for supported files
        """
        supported_extensions = {
            '.txt', '.md', '.markdown', '.rst', '.json', '.xml', 
            '.yaml', '.yml', '.csv', '.log', '.doc', '.docx', '.pdf'
        }
        
        folder = Path(folder_path)
        supported_files = []
        
        if not folder.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return supported_files
        
        if not folder.is_dir():
            logger.warning(f"Path is not a directory: {folder_path}")
            return supported_files
        
        # Recursively find all supported files
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported documents in {folder_path}")
        return supported_files
    
    async def load_documents_from_folder(self, folder_path: str = "documents") -> str:
        """Load and index all documents from a folder.
        
        Args:
            folder_path: Path to the documents folder (default: "documents")
            
        Returns:
            JSON string with loading results
        """
        if not self.vector_store or not self.embedding_provider:
            return json.dumps({
                "error": "Vector search not fully configured",
                "loaded": 0
            })
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get all supported files
            supported_files = await loop.run_in_executor(
                None,
                self._get_supported_files,
                folder_path
            )
            
            if not supported_files:
                return json.dumps({
                    "error": f"No supported documents found in {folder_path}",
                    "loaded": 0,
                    "folder": folder_path
                })
            
            # Read and prepare documents
            documents_to_load = []
            
            for file_path in supported_files:
                content = await loop.run_in_executor(
                    None,
                    self._read_document_file,
                    str(file_path)
                )
                
                if content:
                    documents_to_load.append({
                        "id": str(file_path),
                        "content": content,
                        "metadata": {
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": file_path.suffix.lower()
                        }
                    })
            
            if not documents_to_load:
                return json.dumps({
                    "error": "No documents could be read from the folder",
                    "loaded": 0,
                    "folder": folder_path
                })
            
            # Add documents in batch
            results = await loop.run_in_executor(
                None,
                self.add_documents_batch,
                documents_to_load
            )
            
            loaded_count = sum(1 for v in results.values() if v)
            failed_count = len(results) - loaded_count
            
            # Persist the vector store
            await loop.run_in_executor(None, self.vector_store.save)
            
            logger.info(f"Loaded {loaded_count} documents from {folder_path}")
            
            return json.dumps({
                "loaded": loaded_count,
                "failed": failed_count,
                "total": len(documents_to_load),
                "folder": folder_path,
                "message": f"Loaded {loaded_count} documents from {folder_path}"
            })
            
        except Exception as e:
            logger.error(f"Failed to load documents from folder {folder_path}: {e}")
            return json.dumps({
                "error": str(e),
                "loaded": 0,
                "folder": folder_path
            })
    
    def save_index(self) -> bool:
        """Persist the vector index to disk."""
        if not self.vector_store:
            logger.error("Vector store not configured")
            return False
        
        try:
            self.vector_store.save()
            logger.info("Vector index saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load the vector index from disk."""
        if not self.vector_store:
            logger.error("Vector store not configured")
            return False
        
        try:
            self.vector_store.load()
            logger.info("Vector index loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector search tool."""
        return {
            "total_documents": len(self.documents),
            "vector_store_configured": self.vector_store is not None,
            "embedding_provider_configured": self.embedding_provider is not None,
            "llm_generator_configured": self.llm_generator is not None,
            "config": self.config
        }


# Factory functions for creating configured instances

def create_vector_search_tool(
    vector_store: Optional[VectorStore] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    llm_generator: Optional[callable] = None,
    config: Optional[Dict[str, Any]] = None
) -> VectorSearchAgentTool:
    """Create a configured Vector Search tool instance.
    
    Args:
        vector_store: Custom vector storage backend
        embedding_provider: Custom embedding provider
        llm_generator: Custom LLM response generator
        config: Configuration dictionary
        
    Returns:
        Configured VectorSearchAgentTool instance
    """
    return VectorSearchAgentTool(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        llm_generator=llm_generator,
        config=config
    )


def create_basic_vector_search_tool(
    config: Optional[Dict[str, Any]] = None
) -> VectorSearchAgentTool:
    """Create a basic Vector Search tool with no backends configured.
    
    Backends can be added later using set_vector_store(), 
    set_embedding_provider(), and set_llm_generator() methods.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Unconfigured VectorSearchAgentTool instance ready for setup
    """
    return VectorSearchAgentTool(config=config)


# Default singleton instance
_default_vector_search_tool: Optional[VectorSearchAgentTool] = None


def get_default_vector_search_tool() -> VectorSearchAgentTool:
    """Get or create the default Vector Search tool instance.
    
    Returns a singleton instance with default configuration.
    Use this when you want a quick, zero-configuration tool instance.
    
    Note: You still need to configure backends using:
        - set_vector_store()
        - set_embedding_provider()
        - set_llm_generator()
    
    Returns:
        Default VectorSearchAgentTool instance
    """
    global _default_vector_search_tool
    if _default_vector_search_tool is None:
        default_config = {
            "index_path": "faiss_index",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_dim": 384,
            "max_batch_size": 32
        }
        _default_vector_search_tool = VectorSearchAgentTool(config=default_config)
        logger.info("Default vector search tool instance created")
    return _default_vector_search_tool


def reset_default_vector_search_tool() -> None:
    """Reset the default Vector Search tool instance.
    
    Use this to clear the singleton and create a fresh instance
    on the next call to get_default_vector_search_tool().
    """
    global _default_vector_search_tool
    _default_vector_search_tool = None
    logger.info("Default vector search tool instance reset")
