"""
Query Orchestrator for Adaptive RAG Systems

Coordinates retrieval and generation based on Query Analysis output.
Routes queries to appropriate retrieval strategies, handles multi-hop decomposition,
manages fallback strategies, and integrates with LLM generation.

Uses centralized configuration and dependency injection for single initialization.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from app_config import get_config
from enhance_prompt import PromptEnhancer
from query_analysis import QueryAnalyzer
from web_search_retriever import TavilySearch
from vector_search import VectorSearchIndexer
from graph_search import KnowledgeGraphIndexer

config = get_config()
logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retriever implementations."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents matching the query."""
        pass


class VectorRetriever(BaseRetriever):
    """Vector similarity-based retrieval using FAISS and OpenAI embeddings."""

    def __init__(self, index_path: str = "faiss_index", env_file: str = None):
        """
        Initialize VectorRetriever with FAISS index.

        Args:
            index_path: Path to FAISS index files
            env_file: Optional path to .env file
        """
        try:
            self.indexer = VectorSearchIndexer(
                index_path=index_path,
                env_file=env_file
            )
            logger.info("VectorRetriever initialized with FAISS index")
        except Exception as e:
            logger.warning(f"VectorRetriever initialization failed: {e}")
            self.indexer = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using vector similarity search.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with content and similarity scores
        """
        logger.info(f"VectorRetriever: retrieving top {top_k} documents for query: {query}")
        
        if not self.indexer or self.indexer.index_store.is_empty():
            logger.warning("VectorRetriever: No index available. Returning empty results.")
            return []
        
        try:
            results = self.indexer.query_index(query, k=top_k)
            
            # Format results to match BaseRetriever interface
            formatted_results = []
            for i, (content, score) in enumerate(results):
                formatted_results.append({
                    "id": f"vector_doc_{i}",
                    "content": content,
                    "score": float(score),
                    "source": "vector_search"
                })
            
            logger.info(f"VectorRetriever: Retrieved {len(formatted_results)} documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"VectorRetriever retrieval failed: {e}")
            return []


class HybridRetriever(BaseRetriever):
    """Hybrid search combining vector and keyword-based retrieval."""

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"HybridRetriever: retrieving top {top_k} documents for query: {query}")
        # Mock implementation - combine vector and BM25 retrieval
        return [
            {"id": f"hybrid_doc_{i}", "content": f"Hybrid document {i} for '{query}'", "score": 0.90 - i*0.08}
            for i in range(top_k)
        ]


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

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using graph-based entity search.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with content and entity match scores
        """
        logger.info(f"GraphRetriever: retrieving top {top_k} documents for query: {query}")
        
        if not self.indexer:
            logger.warning("GraphRetriever: Neo4j connection not initialized. Returning empty results.")
            return []
        
        try:
            results = self.indexer.query_index(query, k=top_k)
            
            # Format results to match BaseRetriever interface
            formatted_results = []
            for i, (content, score) in enumerate(results):
                formatted_results.append({
                    "id": f"graph_doc_{i}",
                    "content": content,
                    "score": float(score),
                    "source": "graph_search"
                })
            
            logger.info(f"GraphRetriever: Retrieved {len(formatted_results)} documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"GraphRetriever retrieval failed: {e}")
            return []
        finally:
            if self.indexer:
                self.indexer.close()


class SQLRetriever(BaseRetriever):
    """SQL-based retrieval from structured databases."""

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"SQLRetriever: querying structured database for: {query}")
        # Mock implementation - replace with actual SQL queries
        return [
            {"id": f"sql_doc_{i}", "content": f"SQL result {i} for '{query}'", "score": 0.88 - i*0.07}
            for i in range(top_k)
        ]


class WebSearchRetriever(BaseRetriever):
    """Web search-based retrieval using Tavily API."""

    def __init__(self, tavily_api_key: str = None):
        """
        Initialize WebSearchRetriever with Tavily API.

        Args:
            tavily_api_key: Tavily API key. If None, loads from environment.
        """
        try:
            self.search_engine = TavilySearch(api_key=tavily_api_key)
        except RuntimeError as e:
            logger.warning(f"Tavily initialization warning: {e}")
            self.search_engine = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve results from web search.

        Args:
            query: Search query
            top_k: Number of results to retrieve

        Returns:
            List of search results with id, content, and score
        """
        logger.info(f"WebSearchRetriever: searching web for: {query}")
        
        if not self.search_engine:
            logger.warning("Tavily search engine not initialized. Returning empty results.")
            return []
        
        try:
            # Perform Tavily search
            results = self.search_engine.search(query, max_results=top_k, search_depth="advanced")
            
            # Format results to match BaseRetriever interface
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "id": i,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": 1.0 - (i * 0.1),  # Ranking score
                })
            
            logger.info(f"Retrieved {len(formatted_results)} web search results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Web search retrieval failed: {e}")
            return []


class LLMGenerator:
    """LLM-based answer generation from retrieved context."""

    def __init__(self, llm=None):
        """
        Initialize LLMGenerator with optional injected LLM.

        Args:
            llm: Optional pre-initialized LLM instance for generation (higher temperature).
                 If None, creates one from app_config.
        """
        if llm is None:
            self.llm = config.get_llm_generator(temperature=0.7)
        else:
            self.llm = llm

    def generate(self, query: str, context: List[str]) -> str:
        """Generate an answer using LLM given query and context documents."""
        if not self.llm:
            return "Error: LLM not initialized."

        context_text = "\n\n".join(context) if context else "No context available."
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the information is not in the context, say so clearly."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        try:
            resp = self.llm.invoke(user_prompt)
            return getattr(resp, "content", str(resp))
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"


class QueryOrchestrator:
    """
    Orchestrates the Adaptive RAG pipeline.

    Routes queries to appropriate retrieval strategies, handles multi-hop decomposition,
    manages fallback strategies, and integrates with LLM generation.
    """

    # Mapping of strategy names to retriever classes
    RETRIEVER_MAP = {
        "vector search": VectorRetriever,
        "hybrid search": HybridRetriever,
        "graph retrieval": GraphRetriever,
        "sql retrieval": SQLRetriever,
        "web search": WebSearchRetriever,
        "multi-step retrieval": HybridRetriever,  # Default to hybrid for multi-step
    }

    def __init__(self, llm=None, prompt_enhancer=None, query_analyzer=None, 
                 llm_generator=None, confidence_threshold: float = None, 
                 min_docs_threshold: int = None, tavily_api_key: str = None,
                 vector_index_path: str = "faiss_index", neo4j_uri: str = None,
                 neo4j_user: str = None, neo4j_password: str = None, env_file: str = None):
        """
        Initialize the Query Orchestrator with optional injected dependencies.

        Args:
            llm: Optional pre-initialized LLM instance
            prompt_enhancer: Optional pre-initialized PromptEnhancer instance
            query_analyzer: Optional pre-initialized QueryAnalyzer instance
            llm_generator: Optional pre-initialized LLMGenerator instance
            confidence_threshold: Confidence threshold (uses config default if None)
            min_docs_threshold: Min docs threshold (uses config default if None)
            tavily_api_key: Optional Tavily API key for web search retriever
            vector_index_path: Path to FAISS index files
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            env_file: Optional path to .env file
        """
        # Use injected dependencies or create from config
        self.llm = llm or config.get_llm()
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer(llm=self.llm)
        self.query_analyzer = query_analyzer or QueryAnalyzer(llm=self.llm, prompt_enhancer=self.prompt_enhancer)
        self.generator = llm_generator or LLMGenerator(llm=config.get_llm_generator(temperature=0.7))
        
        # Configuration
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else config.confidence_threshold
        self.min_docs_threshold = min_docs_threshold if min_docs_threshold is not None else config.min_docs_threshold
        
        # Initialize retrievers
        self.retrievers: Dict[str, BaseRetriever] = {}
        
        # Vector Search Retriever
        self.retrievers["vector search"] = VectorRetriever(
            index_path=vector_index_path,
            env_file=env_file
        )
        
        # Graph Search Retriever
        self.retrievers["graph retrieval"] = GraphRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            env_file=env_file
        )
        
        # Web Search Retriever
        self.retrievers["web search"] = WebSearchRetriever(tavily_api_key=tavily_api_key)
        
        # Hybrid and SQL retrievers use defaults
        self.retrievers["hybrid search"] = HybridRetriever()
        self.retrievers["sql retrieval"] = SQLRetriever()
        self.retrievers["multi-step retrieval"] = HybridRetriever()
        
        logger.info("QueryOrchestrator initialized with Vector Search and Graph Search indexers")

    def orchestrate(self, query: str) -> Dict[str, Any]:
        """
        Orchestrate the full RAG pipeline from raw user query.

        Args:
            query: Raw user query string

        Returns:
            Final response with answer and metadata
        """
        # Step 1: Analyze the query
        logger.info(f"Analyzing query: {query}")
        query_analysis = self.query_analyzer.analyze(query)
        logger.info(f"Query analysis complete: {json.dumps(query_analysis, indent=2)}")

        # Extract analysis results
        query_type = query_analysis.get("query_type", "")
        strategy = query_analysis.get("recommended_retrieval_strategy", "hybrid search")
        confidence = query_analysis.get("confidence_score", 0.5)
        rewrite_query = query_analysis.get("rewrite_query", "")
        sub_queries = query_analysis.get("sub_queries", [])

        logger.info(f"Orchestrating query: {query} | Type: {query_type} | Strategy: {strategy}")

        # Handle query rewriting
        retrieval_query = self._select_retrieval_query(query, rewrite_query, confidence)

        # Execute retrieval strategy
        if query_type == "multi-hop" and sub_queries:
            documents = self._execute_multi_hop_retrieval(sub_queries, strategy)
            sub_queries_executed = sub_queries
        else:
            documents = self._execute_retrieval(retrieval_query, strategy)
            sub_queries_executed = []

        # Handle fallback if insufficient documents
        fallback_used = False
        if len(documents) < self.min_docs_threshold:
            logger.warning(f"Insufficient documents retrieved ({len(documents)}). Triggering fallback.")
            fallback_strategy = self._get_fallback_strategy(strategy)
            documents = self._execute_retrieval(retrieval_query, fallback_strategy)
            fallback_used = True

        # Generate answer using LLM
        context = [doc.get("content", "") for doc in documents]
        answer = self.generator.generate(query, context)

        # Compile response
        response = {
            "query": query,
            "retrieval_strategy": strategy,
            "documents_retrieved": documents,
            "answer": answer,
            "metadata": {
                "fallback_used": fallback_used,
                "sub_queries_executed": sub_queries_executed,
                "documents_count": len(documents),
                "confidence_score": confidence,
                "query_type": query_type,
            }
        }

        logger.info(f"Orchestration complete. Retrieved {len(documents)} documents.")
        return response

    def _select_retrieval_query(self, original_query: str, rewrite_query: str, confidence: float) -> str:
        """
        Select whether to use original or rewritten query based on confidence.
        If confidence is low, leverage enhance_prompt to improve the query further.

        Args:
            original_query: Original user query
            rewrite_query: Rewritten query from analysis
            confidence: Confidence score from analysis

        Returns:
            Query to use for retrieval
        """
        # If confidence is very low, use enhance_prompt to improve the query
        if confidence < self.confidence_threshold:
            logger.info(f"Low confidence ({confidence:.2f}). Enhancing query with PromptEnhancer.")
            
            # Get reflection on the original query
            reflection = self.prompt_enhancer.reflect_prompt_openai(original_query)
            logger.info(f"Reflection: {reflection}")
            
            # Use enhance_prompt to generate improved query
            improved = self.prompt_enhancer.improve_prompt_openai_with_confidence(
                original_query, 
                reflection, 
                confidence
            )
            
            if improved and improved.strip() != original_query.strip():
                logger.info(f"Enhanced query with confidence {confidence:.2f}: {improved}")
                return improved
            elif rewrite_query and rewrite_query.strip() != original_query.strip():
                logger.info(f"Using analysis-provided rewrite: {rewrite_query}")
                return rewrite_query
        
        logger.info(f"Confidence sufficient ({confidence:.2f}). Using original query.")
        return original_query

    def _execute_retrieval(self, query: str, strategy: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute retrieval using the specified strategy.

        Args:
            query: Query to retrieve for
            strategy: Retrieval strategy name
            top_k: Number of top documents to retrieve

        Returns:
            List of retrieved documents
        """
        retriever_class = self.RETRIEVER_MAP.get(strategy.lower())
        if not retriever_class:
            logger.warning(f"Unknown strategy '{strategy}'. Defaulting to hybrid search.")
            retriever_class = HybridRetriever

        retriever = retriever_class()
        documents = retriever.retrieve(query, top_k=top_k)
        logger.info(f"Retrieved {len(documents)} documents using {strategy}")
        return documents

    def _execute_multi_hop_retrieval(self, sub_queries: List[str], strategy: str,
                                      top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Execute retrieval for multi-hop queries by retrieving for each sub-query.

        Args:
            sub_queries: List of sub-queries
            strategy: Retrieval strategy name
            top_k: Number of documents per sub-query

        Returns:
            Aggregated list of documents
        """
        all_documents = []
        seen_ids = set()

        for i, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Executing sub-query {i}/{len(sub_queries)}: {sub_query}")
            docs = self._execute_retrieval(sub_query, strategy, top_k=top_k)
            for doc in docs:
                doc_id = doc.get("id")
                if doc_id not in seen_ids:
                    all_documents.append(doc)
                    seen_ids.add(doc_id)

        logger.info(f"Multi-hop retrieval complete. Aggregated {len(all_documents)} unique documents.")
        return all_documents

    @staticmethod
    def _get_fallback_strategy(primary_strategy: str) -> str:
        """
        Determine fallback strategy based on primary strategy.

        Args:
            primary_strategy: Primary retrieval strategy

        Returns:
            Fallback strategy name
        """
        fallback_map = {
            "vector search": "hybrid search",
            "hybrid search": "web search",
            "graph retrieval": "hybrid search",
            "sql retrieval": "hybrid search",
            "web search": "hybrid search",
            "multi-step retrieval": "web search",
        }
        return fallback_map.get(primary_strategy.lower(), "hybrid search")


def main():
    """Demonstrate Query Orchestrator with Vector Search and Graph Search integration."""
    # Initialize dependencies once
    llm = config.get_llm()
    prompt_enhancer = PromptEnhancer(llm=llm)
    query_analyzer = QueryAnalyzer(llm=llm, prompt_enhancer=prompt_enhancer)
    llm_generator = LLMGenerator(llm=config.get_llm_generator(temperature=0.7))
    
    # Initialize orchestrator with vector and graph search indexers
    orchestrator = QueryOrchestrator(
        llm=llm,
        prompt_enhancer=prompt_enhancer,
        query_analyzer=query_analyzer,
        llm_generator=llm_generator,
        vector_index_path="faiss_index",
        neo4j_uri=None,  # Set from environment or config
        neo4j_user=None,
        neo4j_password=None,
        env_file=".env"
    )

    # Sample raw user query
    query = "Which companies investing in AI are hiring machine learning engineers?"

    # Execute orchestration (analyzes query, then orchestrates retrieval and generation)
    result = orchestrator.orchestrate(query)

    # Print results
    print("\n" + "=" * 80)
    print("QUERY ORCHESTRATOR OUTPUT")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
