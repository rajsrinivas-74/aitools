"""Knowledge Graph RAG system using graphiti-core and Neo4j.

Ingest text documents into a knowledge graph using graphiti-core's API.

Behavior:
- Chunk a text document (default 500 token chars, 100 overlap)
- Use graphiti-core to add chunks and extract entities automatically
- Query flow: extract entities from question, find chunks mentioning those entities,
  and call OpenAI LLM for an answer using the retrieved contexts.

Environment variables:
- OPENAI_API_KEY
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import os
import argparse
import logging
import asyncio
from typing import List, Tuple, Dict, Any
from datetime import datetime

from rag_utils import RAGIndexer, chunk_text, log_calls, load_env_file, generate_response_from_contexts, BaseRetriever, ContextBlock
from app_config import get_config

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None

# Try graphiti-core first, then fall back to graphiti
try:
    from graphiti_core import Graphiti
    GRAPHITI_AVAILABLE = True
    GRAPHITI_CORE = True
except Exception:
    GRAPHITI_CORE = False
    try:
        import graphiti
        GRAPHITI_AVAILABLE = True
    except Exception:
        graphiti = None
        GRAPHITI_AVAILABLE = False


LOG_LEVEL = os.getenv("KG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Neo4j Database Connection
# ============================================================================

class Neo4jConnection:
    """Manages Neo4j database connection."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Neo4j database."""
        if not self.uri or not self.user or not self.password:
            raise RuntimeError(
                "Neo4j credentials missing. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD "
                "environment variables or pass them to __init__"
            )
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info("Connected to Neo4j at %s", self.uri)

    def execute_query(self, cypher: str, **parameters):
        """Execute a Cypher query.
        
        Args:
            cypher: Cypher query string
            **parameters: Query parameters
            
        Returns:
            Query results
        """
        if not self.driver:
            raise RuntimeError("Database connection not initialized")

        with self.driver.session() as session:
            return session.run(cypher, parameters)

    def close(self) -> None:
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")


# ============================================================================
# Knowledge Graph Indexer
# ============================================================================

class KnowledgeGraphIndexer(RAGIndexer):
    """Knowledge Graph RAG system using graphiti and Neo4j."""

    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None,
                 neo4j_password: str = None, openai_api_key: str = None,
                 llm_model: str = None, env_file: str = None):
        """Initialize the Knowledge Graph RAG system.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key
            llm_model: LLM model to use. If None, uses config default.
            env_file: Path to .env file for loading environment variables
        """
        super().__init__(llm_model=llm_model, env_file=env_file)

        self.neo4j_uri = neo4j_uri or self.env_vars.get("NEO4J_URI") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or self.env_vars.get("NEO4J_USER") or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or self.env_vars.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", "password")
        self.openai_api_key = openai_api_key or self.env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

        # Initialize graphiti-core client if available
        self.graphiti_client = None
        if GRAPHITI_CORE:
            try:
                # Initialize Graphiti with Neo4j connection
                self.graphiti_client = Graphiti(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password
                )
                logger.info("Graphiti-core client initialized with Neo4j connection")
            except Exception as ex:
                logger.warning("Failed to initialize graphiti-core client: %s", ex)
                raise
        else:
            raise RuntimeError("graphiti-core package is required (pip install graphiti-core)")
        
        # Keep Neo4j connection as backup
        try:
            self.db_connection = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
        except Exception as ex:
            logger.warning("Failed to create Neo4j backup connection: %s", ex)
            self.db_connection = None

    def _validate_dependencies(self):
        """Validate required dependencies."""
        if not GRAPHITI_CORE:
            raise RuntimeError(
                "graphiti-core package is required. "
                "Install with: pip install graphiti-core"
            )
        if GraphDatabase is None:
            logger.warning("neo4j package is not installed but may be required for backup Neo4j operations")
    
    def close(self):
        """Close connections and cleanup resources."""
        try:
            if self.graphiti_client:
                # graphiti-core client.close() is async, run it in an event loop
                try:
                    asyncio.run(self.graphiti_client.close())
                except RuntimeError as e:
                    # Already running in an event loop
                    if "This event loop is already running" in str(e):
                        # Schedule the coroutine to run later
                        logger.info("Async close scheduled for later (already in event loop)")
                    else:
                        logger.warning("Error closing graphiti client: %s", e)
                logger.info("Graphiti client closed")
        except Exception as ex:
            logger.warning("Error closing graphiti client: %s", ex)
        
        try:
            if self.db_connection:
                self.db_connection.close()
                logger.info("Neo4j connection closed")
        except Exception as ex:
            logger.warning("Error closing Neo4j connection: %s", ex)

    @log_calls
    async def index_document(self, path: str, chunk_size: int = 500, overlap: int = 100):
        """Index a document by chunking and adding episodes to the knowledge graph.
        
        Args:
            path: Path to the text file to index
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        if not self.graphiti_client:
            raise RuntimeError("Graphiti client not initialized")
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        logger.info("Created %d chunks from %s", len(chunks), path)

        doc_name = os.path.basename(path)
        doc_path = os.path.abspath(path)

        # Use graphiti-core to add episodes asynchronously
        try:
            tasks = []
            for i, chunk in enumerate(chunks):
                episode_name = f"{doc_name}::chunk::{i}"
                
                # Create async task for each episode
                task = self._add_episode_async(
                    episode_name=episode_name,
                    chunk=chunk,
                    doc_name=doc_name,
                    chunk_index=i,
                    total_chunks=len(chunks)
                )
                tasks.append(task)
            
            # Execute all add_episode calls concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for errors - allow some failures due to skipped/empty chunks
            errors = [r for r in results if isinstance(r, Exception)]
            successes = [r for r in results if r is not None and not isinstance(r, Exception)]
            skipped = [r for r in results if r is None]
            
            logger.info("Document %s: %d successful, %d skipped, %d errors", 
                       doc_name, len(successes), len(skipped), len(errors))
            
            if errors and len(successes) == 0:
                # Only raise error if ALL episodes failed
                for i, error in enumerate(errors):
                    logger.error("  Episode %d: %s", i, error)
                raise RuntimeError(f"Failed to add {len(errors)}/{len(chunks)} episodes")
                    
        except Exception as ex:
            logger.exception("Failed to add episodes: %s", ex)
            raise

        logger.info("Indexing complete: document %s stored using graphiti-core", doc_name)

    async def _add_episode_async(self, episode_name: str, chunk: str, doc_name: str, 
                                  chunk_index: int, total_chunks: int) -> None:
        """Helper method to add a single episode asynchronously.
        
        Args:
            episode_name: Name of the episode
            chunk: Chunk content
            doc_name: Document name
            chunk_index: Current chunk index
            total_chunks: Total number of chunks
        """
        try:
            # Validate parameters to avoid graphiti-core errors
            if chunk is None:
                logger.warning("Skipping episode %d: chunk is None", chunk_index)
                return None
            
            if episode_name is None:
                logger.warning("Skipping episode %d: episode_name is None", chunk_index)
                return None
            
            try:
                # Ensure chunk is a string and sanitize it
                chunk_str = str(chunk).strip()
            except Exception as str_ex:
                logger.warning("Skipping episode %d: cannot convert chunk to string: %s", chunk_index, str_ex)
                return None
                
            if not chunk_str:
                logger.warning("Skipping episode %d: chunk content is empty after sanitization", chunk_index)
                return None
            
            # Remove problematic characters that graphiti-core might not handle
            # Remove null bytes and control characters
            chunk_str = chunk_str.replace('\x00', '')  # Remove null bytes
            chunk_str = ''.join(char for char in chunk_str if ord(char) >= 32 or char in '\n\t\r')  # Remove control chars except whitespace
            chunk_str = chunk_str.strip()
            
            if not chunk_str:
                logger.warning("Skipping episode %d: chunk is empty after sanitization", chunk_index)
                return None
            
            # Add episode using graphiti-core async API
            response = await self.graphiti_client.add_episode(
                name=str(episode_name),
                episode_body=chunk_str,
                source_description=f"Document chunk {chunk_index} from {doc_name}",
                reference_time=datetime.now()
            )
            logger.info("Added episode %d/%d for document %s", chunk_index + 1, total_chunks, doc_name)
            return response
        except Exception as method_ex:
            logger.error("Failed to add episode %d: %s", chunk_index, method_ex)
            # Don't re-raise - allow other episodes to continue processing
            return None

    @log_calls
    async def query_index(self, question: str, k: int = 4) -> List[Tuple[str, float]]:
        """Retrieve top-k chunks using graphiti-core search.

        This uses graphiti-core's semantic search capabilities to find relevant episodes.
        
        Args:
            question: The user's question
            k: Number of results to return
            
        Returns:
            List of (chunk_text, score) tuples
        """
        if not self.graphiti_client:
            logger.warning("Graphiti client not initialized")
            return []
        
        results: List[Tuple[str, float]] = []
        
        try:
            # Use graphiti-core search functionality asynchronously
            # The search method returns episodes/relationships matching the query
            search_results = await self.graphiti_client.search(query=question)
            
            if search_results:
                for result in search_results:
                    # Handle EntityEdge and other objects returned by graphiti-core
                    content = None
                    score = 0.0
                    
                    # Try to extract content from various attribute options
                    if hasattr(result, 'fact'):
                        content = getattr(result, 'fact', None)
                    elif hasattr(result, 'name'):
                        content = getattr(result, 'name', None)
                    elif hasattr(result, 'content'):
                        content = getattr(result, 'content', None)
                    elif hasattr(result, '__dict__'):
                        # Fallback: try to get from dict representation
                        attrs = result.__dict__
                        content = attrs.get("fact") or attrs.get("name") or attrs.get("content")
                    else:
                        # Last resort: stringify the object
                        content = str(result)
                    
                    # Try to extract score from various attribute options
                    if hasattr(result, 'score'):
                        score = float(getattr(result, 'score', 0.0))
                    elif hasattr(result, 'similarity'):
                        score = float(getattr(result, 'similarity', 0.0))
                    elif hasattr(result, '__dict__') and 'score' in result.__dict__:
                        score = float(result.__dict__['score'])
                    
                    if content:
                        results.append((str(content), float(score)))
            
            if not results:
                logger.warning("No results found for query: %s", question)
                
        except Exception as ex:
            logger.error("Graphiti search failed: %s", ex)
            # Fall back to keyword search if available
            try:
                if self.db_connection:
                    qlower = question.lower()
                    cypher = (
                        "MATCH (e:Episode) WHERE toLower(e.content) CONTAINS $q "
                        "RETURN e.id AS id, e.content AS text LIMIT $k"
                    )
                    res = self.db_connection.execute_query(cypher, q=qlower, k=k)
                    for i, r in enumerate(res):
                        results.append((r.get("text"), float(k - i)))
            except Exception as fallback_ex:
                logger.warning("Fallback search also failed: %s", fallback_ex)

        return results[:k]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text.
        
        In graphiti-core, entity extraction happens automatically during add_episode.
        This method is kept for backward compatibility and returns key words from the text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entity names (simplified keyword extraction)
        """
        # Simple keyword extraction - return words longer than 4 characters
        # In production, use proper NLP entity extraction
        words = text.lower().split()
        # Filter common stop words and short words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
        entities = [w.strip(".,!?;:") for w in words 
                   if len(w) > 4 and w.lower() not in stop_words]
        # Return unique entities
        return list(set(entities[:10]))  # Limit to 10 unique entities

    def generate_response(self, question: str, k: int = 4) -> dict:
        """Query knowledge graph and generate LLM response in one call.
        
        DEPRECATED: Use generate_response_async() instead.
        This method now just wraps the async version for backward compatibility.
        
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
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, shouldn't happen in normal use
            raise RuntimeError("Use generate_response_async() when in async context")
        except RuntimeError:
            # No event loop, safe to use asyncio.run
            return asyncio.run(self.generate_response_async(question, k=k))

    async def generate_response_async(self, question: str, k: int = 4) -> dict:
        """Async version: Query knowledge graph and generate LLM response in one call.
        
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
        # Use await for async query_index
        results = await self.query_index(question, k=k)
        
        # Format context blocks with metadata
        context_blocks = []
        entities_used = self._extract_entities(question)
        
        for i, (content, score) in enumerate(results):
            context_blocks.append({
                "content": content,
                "source": "graph_search",
                "score": float(score),
                "metadata": {
                    "chunk_index": i,
                    "retrieval_method": "entity_relationship_matching",
                    "entities_matched": entities_used
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

    def close(self):
        """Close the database connection."""
        self.db_connection.close()


# ============================================================================
# Retriever Interface for Orchestrator
# ============================================================================

class GraphRetriever(BaseRetriever):
    """Graph-based retrieval using entity relationships and Neo4j."""

    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, 
                 neo4j_password: str = None, env_file: str = None):
        """
        Initialize GraphRetriever with Neo4j connection.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            env_file: Optional path to .env file
        """
        try:
            self.indexer = KnowledgeGraphIndexer(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                env_file=env_file
            )
            logger.info("GraphRetriever initialized with Neo4j connection")
        except Exception as e:
            logger.warning(f"GraphRetriever initialization failed: {e}")
            self.indexer = None

    def retrieve(self, query: str, top_k: int = 5, docs: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using graph-based entity search.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            docs: Optional list of paths to text documents to index before retrieval
            
        Returns:
            List of retrieved documents with content and entity match scores
        """
        logger.info(f"GraphRetriever: retrieving top {top_k} documents for query: {query}")
        
        if not self.indexer:
            logger.warning("GraphRetriever: Neo4j connection not initialized. Returning empty results.")
            return []
        
        try:
            # Run all async operations in a single event loop
            results = asyncio.run(self._retrieve_async(query, top_k, docs))
            logger.info(f"GraphRetriever: Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"GraphRetriever retrieval failed: {e}")
            return []
        finally:
            if self.indexer:
                self.indexer.close()
    
    async def _retrieve_async(self, query: str, top_k: int, docs: List[str] = None) -> List[Dict[str, Any]]:
        """Async helper for retrieve - batches all async operations.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            docs: Optional list of paths to text documents to index before retrieval
            
        Returns:
            List of retrieved documents with content and entity match scores
        """
        # Index documents if provided (execute concurrently)
        if docs:
            index_tasks = []
            for doc_path in docs:
                if not os.path.exists(doc_path):
                    logger.error(f"Document file not found: {doc_path}")
                    continue
                logger.info(f"Indexing document: {doc_path}")
                # Create async index task
                index_tasks.append(self.indexer.index_document(doc_path))
            
            if index_tasks:
                # Execute all indexing concurrently
                results = await asyncio.gather(*index_tasks, return_exceptions=True)
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(f"Failed to index {len(errors)}/{len(index_tasks)} documents")
        
        # Query the index
        results = await self.indexer.query_index(query, k=top_k)
        
        # Format results to match BaseRetriever interface
        formatted_results = []
        for i, (content, score) in enumerate(results):
            formatted_results.append({
                "id": f"graph_doc_{i}",
                "content": content,
                "score": float(score),
                "source": "graph_search"
            })
        
        return formatted_results
    
    def get_context_blocks(self, query: str, top_k: int = 5, docs: List[str] = None) -> List[ContextBlock]:
        """Retrieve context blocks from graph search.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            docs: Optional list of paths to text documents to index before retrieval
            
        Returns:
            List of ContextBlock objects
        """
        documents = self.retrieve(query, top_k=top_k, docs=docs)
        context_blocks = []
        
        for doc in documents:
            block = ContextBlock(
                content=doc.get("content", ""),
                source="graph_search",
                score=doc.get("score", 0.0),
                metadata={
                    "doc_id": doc.get("id", ""),
                    "retrieval_method": "entity_relationship_matching"
                }
            )
            context_blocks.append(block)
        
        return context_blocks

    def generate_response(self, query: str) -> str:
        """Generate a response for the query using graph-based retrieval.
        
        Args:
            query: User query
            
        Returns:
            Generated response string
        """
        if not self.indexer:
            return "No knowledge graph available for response generation."
        
        try:
            # Retrieve context blocks
            context_blocks = self.get_context_blocks(query, top_k=5)
            
            # Generate response from contexts (uses config default if llm_model is None)
            response = generate_response_from_contexts(
                question=query,
                context_blocks=context_blocks,
                llm_model=None,
                include_source_attribution=True
            )
            return response
        except Exception as e:
            logger.error(f"GraphRetriever generate_response failed: {e}")
            return f"Error generating response: {str(e)}"




def main():
    """CLI entry point for the Knowledge Graph RAG system."""
    parser = argparse.ArgumentParser(description="Index documents using Knowledge Graph RAG")
    parser.add_argument("--doc", help="Path to text document to index")
    parser.add_argument("--ask", help="Ask a single question against the saved graph")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks")
    parser.add_argument("--show-env", action="store_true", help="Show variables loaded from .env")
    parser.add_argument("--env-file", help="Path to .env file")
    args = parser.parse_args()

    try:
        # Initialize the Knowledge Graph RAG system
        kg = KnowledgeGraphIndexer(env_file=args.env_file)

        if args.show_env:
            print("Loaded environment variables:")
            for k, v in kg.env_vars.items():
                if v:
                    print(f"{k}={v[:20]}..." if len(str(v)) > 20 else f"{k}={v}")

        # Use single event loop for all async operations
        if args.doc or args.ask:
            asyncio.run(_main_async(kg, args))
        else:
            print("Provide --doc to index a document or --ask to query the graph.")

        kg.close()

    except Exception as e:
        logger.error("Error: %s", e)
        print(f"Error: {e}")
        return 1

    return 0


async def _main_async(kg: KnowledgeGraphIndexer, args):
    """Async helper for main() - batches all async operations in a single event loop.
    
    Args:
        kg: KnowledgeGraphIndexer instance
        args: Parsed command line arguments
    """
    if args.doc:
        print(f"Indexing document: {args.doc}")
        await kg.index_document(args.doc, chunk_size=args.chunk_size, overlap=args.overlap)

    if args.ask:
        results = await kg.query_index(args.ask, k=args.k)
        contexts = [t for t, s in results]
        answer = kg.answer_with_llm(args.ask, contexts)
        print("\nAnswer:")
        print(answer)
        print("\nRetrieved contexts:")
        for i, (ctx, score) in enumerate(results, 1):
            preview = ctx[:400].replace("\n", " ")
            suffix = "..." if len(ctx) > 400 else ""
            print(f"[{i}] (score: {score:.2f}) {preview}{suffix}\n")


if __name__ == "__main__":
    main()

