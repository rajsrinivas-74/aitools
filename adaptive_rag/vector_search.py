"""Vector Search RAG system using OpenAI embeddings and FAISS.

Ingest text documents into a FAISS vector database with OpenAI embeddings.

Behavior:
- Chunk a text document (default 500 character chunks with 100 character overlap)
- Generate embeddings for each chunk using OpenAI's embedding model
- Store embeddings and chunk metadata in FAISS index
- Query flow: embed the question, find similar chunks using FAISS,
  and call OpenAI LLM for an answer using the retrieved contexts.

Environment variables:
- OPENAI_API_KEY
"""

import os
import argparse
import logging
import pickle
from typing import List, Tuple

from rag_utils import (
    RAGIndexer, OpenAIEmbedding, chunk_text, log_calls, load_env_file, generate_response, generate_response_from_contexts
)

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    np = None
    FAISS_AVAILABLE = False


LOG_LEVEL = os.getenv("VS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Index Storage and Retrieval
# ============================================================================

class FAISSIndexStore:
    """Manages FAISS index storage and retrieval."""

    def __init__(self, index_path: str = "faiss_index"):
        """Initialize FAISS index store.
        
        Args:
            index_path: Base path for storing index files
        """
        self.index_path = index_path
        self.faiss_index = None
        self.chunk_metadata = []

    def add_embeddings(self, embeddings: List[List[float]], metadata: List[Tuple]) -> None:
        """Add embeddings to the index.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata tuples (chunk_text, doc_name, chunk_id)
        """
        if not embeddings:
            return

        embeddings_array = np.array(embeddings, dtype=np.float32)

        if self.faiss_index is None:
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            logger.info("Created new FAISS index with dimension %d", dimension)

        self.faiss_index.add(embeddings_array)
        self.chunk_metadata.extend(metadata)
        logger.info("Added %d embeddings to FAISS index", len(embeddings))

    def search(self, query_embedding: List[float], k: int) -> List[Tuple[str, float]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if self.faiss_index is None:
            raise RuntimeError("No index available")

        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.faiss_index.search(query_vector, min(k, len(self.chunk_metadata)))

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunk_metadata):
                chunk_text, doc_name, chunk_id = self.chunk_metadata[idx]
                similarity = 1 / (1 + distances[0][i])
                results.append((chunk_text, float(similarity)))

        return results

    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.faiss_index is None:
            logger.warning("No index to save")
            return

        faiss.write_index(self.faiss_index, f"{self.index_path}.index")
        with open(f"{self.index_path}.metadata", "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        logger.info("Index saved to %s", self.index_path)

    def load(self) -> bool:
        """Load index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.faiss_index = faiss.read_index(f"{self.index_path}.index")
            with open(f"{self.index_path}.metadata", "rb") as f:
                self.chunk_metadata = pickle.load(f)
            logger.info("Index loaded from %s", self.index_path)
            return True
        except Exception as ex:
            logger.warning("Failed to load index: %s", ex)
            self.faiss_index = None
            self.chunk_metadata = []
            return False

    def is_empty(self) -> bool:
        """Check if index is empty."""
        return self.faiss_index is None or len(self.chunk_metadata) == 0


# ============================================================================
# Vector Search Indexer
# ============================================================================

class VectorSearchIndexer(RAGIndexer):
    """Vector Search RAG system using OpenAI embeddings and FAISS."""

    def __init__(self, openai_api_key: str = None, embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-3.5-turbo", env_file: str = None, index_path: str = None):
        """Initialize the Vector Search RAG system.
        
        Args:
            openai_api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
            llm_model: LLM model to use for answering questions
            env_file: Path to .env file for loading environment variables
            index_path: Path to save/load FAISS index and metadata
        """
        super().__init__(llm_model=llm_model, env_file=env_file)
        
        self.openai_api_key = openai_api_key or self.env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "faiss_index")
        
        self.embedding_engine = OpenAIEmbedding(model=self.embedding_model, api_key=self.openai_api_key)
        self.index_store = FAISSIndexStore(self.index_path)
        
        if os.path.exists(f"{self.index_path}.index"):
            self.index_store.load()
        else:
            logger.info("No existing index found at %s", self.index_path)

    def _validate_dependencies(self):
        """Validate required dependencies."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu package is required (pip install faiss-cpu)")

    @log_calls
    def index_document(self, path: str, chunk_size: int = 500, overlap: int = 100):
        """Index a document by chunking and storing embeddings in FAISS.
        
        Args:
            path: Path to the text file to index
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        logger.info("Created %d chunks from %s", len(chunks), path)

        doc_name = os.path.basename(path)
        embeddings = []
        metadata = []

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.embedding_engine.embed(chunk)
                embeddings.append(embedding)
                chunk_id = f"{doc_name}::chunk::{i}"
                metadata.append((chunk, doc_name, chunk_id))
                logger.info("Embedded chunk %d/%d for document %s", i + 1, len(chunks), doc_name)
            except Exception as ex:
                logger.exception("Failed to embed chunk %d: %s", i, ex)
                raise

        self.index_store.add_embeddings(embeddings, metadata)
        logger.info("Indexing complete: document %s stored in FAISS", doc_name)

    @log_calls
    def query_index(self, question: str, k: int = 4) -> List[Tuple[str, float]]:
        """Retrieve top-k similar chunks using FAISS vector search.

        Args:
            question: The user's question
            k: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if self.index_store.is_empty():
            raise RuntimeError("No index available. Index a document first using index_document()")

        try:
            question_embedding = self.embedding_engine.embed(question)
            return self.index_store.search(question_embedding, k)
        except Exception as ex:
            logger.exception("Query failed: %s", ex)
            raise

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        self.index_store.save()

    def generate_response(self, question: str, k: int = 4) -> dict:
        """Query index and generate LLM response in one call.
        
        Args:
            question: The user's question
            k: Number of context blocks to retrieve
            
        Returns:
            Dictionary with:
            - response: LLM-generated answer
            - sources_used: List of sources
            - context_count: Number of context blocks
            - retrieved_contexts: List of retrieved contexts
        """
        # Query the index
        results = self.query_index(question, k=k)
        
        # Format context blocks with metadata
        context_blocks = []
        for i, (content, score) in enumerate(results):
            context_blocks.append({
                "content": content,
                "source": "vector_search",
                "score": float(score),
                "metadata": {
                    "chunk_index": i,
                    "retrieval_method": "semantic_similarity"
                }
            })
        
        # Generate response using common function
        response_data = generate_response_from_contexts(
            question=question,
            context_blocks=context_blocks,
            llm_model=self.llm_model,
            include_source_attribution=True
        )
        
        return response_data





def main():
    """CLI entry point for the Vector Search RAG system."""
    parser = argparse.ArgumentParser(description="Index documents using Vector Search RAG with FAISS")
    parser.add_argument("--doc", help="Path to text document to index")
    parser.add_argument("--ask", help="Ask a single question against the saved index")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks")
    parser.add_argument("--index-path", default="faiss_index", help="Path to save/load FAISS index")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="OpenAI LLM model")
    parser.add_argument("--show-env", action="store_true", help="Show variables loaded from .env")
    parser.add_argument("--env-file", help="Path to .env file")
    args = parser.parse_args()

    try:
        # Initialize the Vector Search RAG system
        vs = VectorSearchIndexer(
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            env_file=args.env_file,
            index_path=args.index_path
        )

        if args.show_env:
            print("Loaded environment variables:")
            for k, v in vs.env_vars.items():
                if v:
                    print(f"{k}={v[:20]}..." if len(str(v)) > 20 else f"{k}={v}")

        if args.doc:
            vs.index_document(args.doc, chunk_size=args.chunk_size, overlap=args.overlap)
            vs.save_index()

        if args.ask:
            results = vs.query_index(args.ask, k=args.k)
            contexts = [t for t, s in results]
            answer = vs.answer_with_llm(args.ask, contexts)
            print("\nAnswer:")
            print(answer)
            print("\nRetrieved contexts:")
            for i, (ctx, score) in enumerate(results, 1):
                preview = ctx[:400].replace("\n", " ")
                suffix = "..." if len(ctx) > 400 else ""
                print(f"[{i}] (similarity: {score:.4f}) {preview}{suffix}\n")

        if not args.doc and not args.ask:
            print("Provide --doc to index a document or --ask to query the index.")

    except Exception as e:
        logger.error("Error: %s", e)
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
