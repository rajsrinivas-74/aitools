"""
Query Orchestrator for Adaptive RAG Systems

Coordinates retrieval and generation based on Query Analysis output.
Routes queries to appropriate retrieval strategies, handles multi-hop decomposition,
manages fallback strategies, and integrates with LLM generation.

Uses centralized configuration and dependency injection for single initialization.
"""

import json
import logging
import argparse
import sys
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app_config import get_config
from enhance_prompt import PromptEnhancer
from query_analysis import QueryAnalyzer
from rag_utils import (
    BaseRetriever, ContextBlock, generate_response_from_contexts,
    validate_query, ValidationError, RetrieverError, LLMError
)
from vector_search import VectorRetriever
from graph_search import GraphRetriever
from web_search_retriever import WebSearchRetriever

config = get_config()
logger = logging.getLogger(__name__)


# ============================================================================
# Strategy Constants
# ============================================================================

STRATEGY_VECTOR_SEARCH = "vector search"
STRATEGY_GRAPH_SEARCH = "graph retrieval"
STRATEGY_WEB_SEARCH = "web search"

# Default primary strategies for multi-retriever mode
DEFAULT_MULTI_STRATEGIES = [STRATEGY_VECTOR_SEARCH, STRATEGY_GRAPH_SEARCH, STRATEGY_WEB_SEARCH]

# Fallback strategy map for single-retriever mode
FALLBACK_MAP = {
    STRATEGY_VECTOR_SEARCH: STRATEGY_WEB_SEARCH,
    STRATEGY_GRAPH_SEARCH: STRATEGY_VECTOR_SEARCH,
    STRATEGY_WEB_SEARCH: STRATEGY_VECTOR_SEARCH,
}


# ============================================================================
# Context Structures
# ============================================================================

@dataclass
class AggregatedContext:
    """Aggregates context blocks from multiple sources."""
    blocks: List[ContextBlock] = field(default_factory=list)
    
    def add_block(self, block: ContextBlock) -> None:
        """Add a context block."""
        self.blocks.append(block)
    
    def add_blocks(self, blocks: List[ContextBlock]) -> None:
        """Add multiple context blocks."""
        self.blocks.extend(blocks)
    
    def get_by_source(self, source: str) -> List[ContextBlock]:
        """Get all context blocks from a specific source."""
        return [b for b in self.blocks if b.source == source]
    
    def get_sources(self) -> List[str]:
        """Get list of unique sources."""
        return list(set(b.source for b in self.blocks))
    
    def get_formatted_context(self, include_sources: bool = True) -> str:
        """Format context for LLM consumption.
        
        Args:
            include_sources: Whether to include source attribution
            
        Returns:
            Formatted context string
        """
        if not self.blocks:
            return "No context available."
        
        formatted_parts = []
        
        if include_sources:
            # Group by source for organized output
            by_source = {}
            for block in self.blocks:
                if block.source not in by_source:
                    by_source[block.source] = []
                by_source[block.source].append(block)
            
            for source in sorted(by_source.keys()):
                formatted_parts.append(f"\n=== From {source.upper()} (confidence: {by_source[source][0].score:.2f}) ===\n")
                for i, block in enumerate(by_source[source], 1):
                    formatted_parts.append(f"[{source.upper()}-{i}] {block.content}\n")
        else:
            # Simple concatenation
            for i, block in enumerate(self.blocks, 1):
                formatted_parts.append(f"[Source {i}] {block.content}\n")
        
        return "\n".join(formatted_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "sources": self.get_sources(),
            "total_blocks": len(self.blocks)
        }


class QueryOrchestrator:
    """
    Orchestrates the Adaptive RAG pipeline.

    Routes queries to appropriate retrieval strategies, handles multi-hop decomposition,
    manages fallback strategies, and integrates with LLM generation.
    """

    # Mapping of strategy names to pre-initialized retrievers
    # (Populated in __init__)

    def __init__(self, llm=None, prompt_enhancer=None, query_analyzer=None, 
                 confidence_threshold: float = None, 
                 min_docs_threshold: int = None, tavily_api_key: str = None,
                 vector_index_path: str = "faiss_index", neo4j_uri: str = None,
                 neo4j_user: str = None, neo4j_password: str = None, env_file: str = None):
        """
        Initialize the Query Orchestrator with optional injected dependencies.

        Args:
            llm: Optional pre-initialized LLM instance
            prompt_enhancer: Optional pre-initialized PromptEnhancer instance
            query_analyzer: Optional pre-initialized QueryAnalyzer instance
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
        
        # Configuration
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else config.confidence_threshold
        self.min_docs_threshold = min_docs_threshold if min_docs_threshold is not None else config.min_docs_threshold
        
        # Initialize retrievers
        self.retrievers: Dict[str, BaseRetriever] = {}
        
        # Vector Search Retriever
        self.retrievers[STRATEGY_VECTOR_SEARCH] = VectorRetriever(
            index_path=vector_index_path,
            env_file=env_file
        )
        
        # Graph Search Retriever
        self.retrievers[STRATEGY_GRAPH_SEARCH] = GraphRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            env_file=env_file
        )
        
        # Web Search Retriever
        self.retrievers[STRATEGY_WEB_SEARCH] = WebSearchRetriever(tavily_api_key=tavily_api_key)
        
        logger.info("QueryOrchestrator initialized with Vector, Graph, and Web Search indexers")

    def orchestrate(self, query: str, context_doc_paths: List[str] = None, use_multiple_retrievers: bool = False) -> Dict[str, Any]:
        """
        Orchestrate the full RAG pipeline from raw user query.

        Args:
            query: Raw user query string (required) - The question/request from the user
            context_doc_paths: Optional list of file paths to RAG content documents
                              These documents are indexed for vector search and graph retrieval
                              Example: ["docs/file1.txt", "docs/file2.txt"]
            use_multiple_retrievers: If True, uses multiple retrievers (vector, graph, web) 
                                    and aggregates contexts. If False, uses single best strategy.

        Returns:
            Final response dictionary with answer and metadata. On error, returns error response with error details.
        """
        try:
            # Step 0: Validate input query
            logger.info(f"User query received: {query}")
            if context_doc_paths:
                logger.info(f"Context document paths provided: {context_doc_paths}")
            validated_query = validate_query(query)
            logger.info(f"Query validation passed for: {validated_query[:50]}...")
            
        except ValidationError as e:
            # Return error response with validation context
            logger.error(f"Query validation failed: {e.message}", extra=e.to_dict())
            return {
                "query": query,
                "error": True,
                "error_type": "ValidationError",
                "error_message": e.message,
                "error_code": e.error_code,
                "answer": None,
                "documents_retrieved": [],
                "aggregated_context": None,
                "metadata": {
                    "error_context": e.context,
                    "multi_retriever_used": use_multiple_retrievers,
                }
            }
        
        try:
            # Step 1: Analyze the query
            query_analysis = self._analyze_and_route_query(validated_query)
            
            # Step 2: Determine retrieval path and execute
            retrieval_query = query_analysis["retrieval_query"]
            strategy = query_analysis["strategy"]
            
            if use_multiple_retrievers:
                result = self._execute_multi_retriever_path(retrieval_query, query_analysis, context_doc_paths=context_doc_paths)
            else:
                result = self._execute_single_retriever_path(retrieval_query, strategy, query_analysis, context_doc_paths=context_doc_paths)
            
            # Step 3: Compile final response
            answer, aggregated_context, all_documents, fallback_used, sub_queries_executed = result
            
            response = {
                "query": query,
                "error": False,
                "retrieval_strategy": strategy,
                "documents_retrieved": all_documents,
                "aggregated_context": aggregated_context.to_dict() if aggregated_context else None,
                "answer": answer,
                "metadata": {
                    "fallback_used": fallback_used,
                    "sub_queries_executed": sub_queries_executed,
                    "documents_count": len(all_documents) if all_documents else 0,
                    "confidence_score": query_analysis["confidence"],
                    "query_type": query_analysis["query_type"],
                    "multi_retriever_used": use_multiple_retrievers,
                }
            }

            logger.info(f"Orchestration complete. Retrieved {len(all_documents) if all_documents else 0} documents.")
            return response
        
        except RetrieverError as e:
            # Retriever-specific error
            logger.error(f"Retrieval failed: {e.message}", extra=e.to_dict())
            return {
                "query": query,
                "error": True,
                "error_type": "RetrieverError",
                "error_message": e.message,
                "error_code": e.error_code,
                "answer": None,
                "documents_retrieved": [],
                "aggregated_context": None,
                "metadata": {
                    "error_context": e.context,
                    "multi_retriever_used": use_multiple_retrievers,
                }
            }
        
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error during orchestration: {str(e)}", exc_info=True)
            return {
                "query": query,
                "error": True,
                "error_type": "UnexpectedError",
                "error_message": f"An unexpected error occurred: {str(e)}",
                "answer": None,
                "documents_retrieved": [],
                "aggregated_context": None,
                "metadata": {
                    "multi_retriever_used": use_multiple_retrievers,
                }
            }
    
    def _analyze_and_route_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and determine routing strategy.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with analysis results and routing info
        """
        logger.info(f"Analyzing query: {query}")
        query_analysis = self.query_analyzer.analyze(query)
        logger.info(f"Query analysis complete: {json.dumps(query_analysis, indent=2)}")

        query_type = query_analysis.get("query_type", "")
        strategy = query_analysis.get("recommended_retrieval_strategy", STRATEGY_VECTOR_SEARCH)
        confidence = query_analysis.get("confidence_score", 0.5)
        rewrite_query = query_analysis.get("rewrite_query", "")
        sub_queries = query_analysis.get("sub_queries", [])

        logger.info(f"Orchestrating query: {query} | Type: {query_type} | Strategy: {strategy}")

        # Optimize the query using PromptEnhancer
        logger.info(f"Optimizing rewritten query: {rewrite_query}")
        try:
            optimization_result = self.prompt_enhancer.optimize_prompt(rewrite_query, max_iters=2, target_score=7)
            if optimization_result.get("final"):
                optimized_query = optimization_result["final"].get("improved", rewrite_query)
                logger.info(f"Query optimization complete. Optimized query: {optimized_query}")
                logger.info(f"Optimization history: {json.dumps([{'iteration': h['iteration'], 'score': h['score']} for h in optimization_result.get('history', [])], indent=2)}")
            else:
                optimized_query = rewrite_query
                logger.warning("Query optimization produced no improvements, using original rewritten query")
        except Exception as e:
            logger.error(f"Query optimization failed: {e}, falling back to rewritten query")
            optimized_query = rewrite_query

        # Use ONLY the optimized query for all subsequent retrieval processing
        retrieval_query = optimized_query
        logger.info(f"Using optimized query for retrieval: {retrieval_query}")
        
        return {
            "query_type": query_type,
            "strategy": strategy,
            "confidence": confidence,
            "retrieval_query": retrieval_query,
            "optimized_query": optimized_query,
            "sub_queries": sub_queries,
        }
    
    def _execute_single_retriever_path(self, query: str, strategy: str, 
                                      analysis: Dict[str, Any], context_doc_paths: List[str] = None) -> Tuple[str, Optional[AggregatedContext], List[Dict[str, Any]], bool, List[str]]:
        """Execute single-retriever path with fallback support.
        
        Args:
            query: Query to retrieve for
            strategy: Primary retrieval strategy
            analysis: Query analysis results
            context_doc_paths: Optional list of file paths to RAG content documents
            
        Returns:
            Tuple of (answer, aggregated_context, documents, fallback_used, sub_queries_executed)
        """
        query_type = analysis["query_type"]
        sub_queries = analysis["sub_queries"]
        
        # Execute retrieval
        if query_type == "multi-hop" and sub_queries:
            documents = self._execute_multi_hop_retrieval(sub_queries, strategy, context_doc_paths=context_doc_paths)
            sub_queries_executed = sub_queries
        else:
            documents = self._retrieve_with_strategy(query, strategy, context_doc_paths=context_doc_paths)
            sub_queries_executed = []
        
        # Handle fallback if insufficient documents
        fallback_used = False
        if len(documents) < self.min_docs_threshold:
            logger.warning(f"Insufficient documents retrieved ({len(documents)}). Triggering fallback.")
            fallback_strategy = FALLBACK_MAP.get(strategy, STRATEGY_VECTOR_SEARCH)
            documents = self._retrieve_with_strategy(query, fallback_strategy, context_doc_paths=context_doc_paths)
            fallback_used = True
            strategy = fallback_strategy
        
        # Delegate generation to the retriever
        try:
            retriever = self.retrievers[strategy]
            answer = retriever.generate_response(analysis["retrieval_query"])
            if isinstance(answer, dict):
                answer = answer.get("answer", str(answer))
        except LLMError as e:
            logger.error(f"Generation failed with LLM error: {e.message}", extra=e.to_dict())
            answer = f"Error generating answer: {e.message}"
        except RetrieverError as e:
            logger.error(f"Generation failed with retriever error: {e.message}", extra=e.to_dict())
            answer = f"Error generating answer: {e.message}"
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}", exc_info=True)
            answer = f"Error generating answer: {str(e)}"
        
        return answer, None, documents, fallback_used, sub_queries_executed
    
    def _execute_multi_retriever_path(self, query: str, analysis: Dict[str, Any], context_doc_paths: List[str] = None) -> Tuple[str, AggregatedContext, List[Dict[str, Any]], bool, List[str]]:
        """Execute multi-retriever path with context aggregation.
        
        Args:
            query: Query to retrieve for
            analysis: Query analysis results
            context_doc_paths: Optional list of file paths to RAG content documents
            
        Returns:
            Tuple of (answer, aggregated_context, documents, fallback_used, sub_queries_executed)
        """
        aggregated_context, all_documents = self._execute_multi_retriever(query, strategies=DEFAULT_MULTI_STRATEGIES, context_doc_paths=context_doc_paths)
        answer = self._synthesize_from_aggregated_context(analysis["retrieval_query"], aggregated_context)
        return answer, aggregated_context, all_documents, False, []
    
    def _execute_multi_retriever(self, query: str, strategies: List[str] = None, top_k: int = 5, context_doc_paths: List[str] = None) -> Tuple[AggregatedContext, List[Dict[str, Any]]]:
        """Execute multiple retrievers and aggregate context.
        
        Args:
            query: Query to retrieve for
            strategies: List of strategy names to use (if None, uses primary strategies)
            top_k: Number of results per retriever
            context_doc_paths: Optional list of file paths to RAG content documents
            
        Returns:
            Tuple of (AggregatedContext, list of documents)
        """
        if strategies is None:
            strategies = DEFAULT_MULTI_STRATEGIES
        
        aggregated_context = AggregatedContext()
        all_documents = []
        seen_ids = set()
        failed_strategies = []
        
        for strategy in strategies:
            logger.info(f"Executing {strategy} retriever")
            
            if strategy not in self.retrievers:
                logger.warning(f"Strategy '{strategy}' not available. Skipping.")
                continue
            
            try:
                retriever = self.retrievers[strategy]
                # Get context blocks - this is the primary interface
                # Pass context_doc_paths parameter to retriever if provided
                if context_doc_paths:
                    context_blocks = retriever.get_context_blocks(query, top_k=top_k, docs=context_doc_paths)
                else:
                    context_blocks = retriever.get_context_blocks(query, top_k=top_k)
                aggregated_context.add_blocks(context_blocks)
                
                # Extract documents from context blocks for backward compatibility
                for block in context_blocks:
                    doc = {
                        "id": block.metadata.get("doc_id", f"{strategy}_{len(all_documents)}"),
                        "content": block.content,
                        "score": block.score,
                        "source": block.source
                    }
                    doc_id = doc.get("id")
                    if doc_id not in seen_ids:
                        all_documents.append(doc)
                        seen_ids.add(doc_id)
                
                logger.info(f"Retrieved {len(context_blocks)} blocks from {strategy}")
                
            except RetrieverError as e:
                logger.warning(f"Retriever error in {strategy}: {e.message}", extra=e.to_dict())
                failed_strategies.append({
                    "strategy": strategy,
                    "error": e.message,
                    "error_code": e.error_code
                })
                continue
            except Exception as e:
                logger.error(f"Unexpected error in {strategy}: {e}", exc_info=True)
                failed_strategies.append({
                    "strategy": strategy,
                    "error": str(e),
                    "error_code": "UNKNOWN_ERROR"
                })
                continue
        
        # Log summary
        successful_count = len(strategies) - len(failed_strategies)
        logger.info(
            f"Multi-retriever complete. "
            f"Aggregated {len(aggregated_context.blocks)} blocks from {len(aggregated_context.get_sources())} sources. "
            f"Successful: {successful_count}/{len(strategies)}"
        )
        
        if failed_strategies:
            logger.warning(f"Failed strategies: {json.dumps(failed_strategies, indent=2)}")
        
        return aggregated_context, all_documents

    def _synthesize_from_aggregated_context(self, query: str, aggregated_context: AggregatedContext) -> str:
        """Synthesize answer from aggregated context across multiple sources.
        
        This is the only place where the orchestrator performs LLM synthesis,
        and only for multi-source aggregation where individual retrievers
        cannot handle cross-source synthesis.
        
        Args:
            query: The user's question
            aggregated_context: AggregatedContext object with blocks from multiple sources
            
        Returns:
            Synthesized answer. On error, returns error message (graceful fallback).
        """
        try:
            # Format context with source attribution
            context_list = [block.content for block in aggregated_context.blocks]
            
            if not context_list:
                return "No context available to answer the question."
            
            # Use the utility function for synthesis
            answer = generate_response_from_contexts(query, context_list, self.llm)
            if isinstance(answer, dict):
                answer = answer.get("answer", str(answer))
            
            logger.info(f"Multi-source synthesis complete for query: {query}")
            return answer
        except LLMError as e:
            # Structured LLM error handling
            logger.error(f"LLM synthesis failed: {e.message}", extra=e.to_dict())
            return f"Error synthesizing answer: {e.message}"
        except Exception as e:
            # Catch-all for unexpected errors during synthesis
            logger.error(f"Unexpected error during synthesis: {e}", exc_info=True)
            return f"Error synthesizing answer from multiple sources: {str(e)}"
    
    
    def _retrieve_with_strategy(self, query: str, strategy: str, top_k: int = 5, context_doc_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Execute retrieval using pre-initialized retriever for the specified strategy.

        Args:
            query: Query to retrieve for
            strategy: Retrieval strategy name
            top_k: Number of top documents to retrieve
            context_doc_paths: Optional list of file paths to RAG content documents

        Returns:
            List of retrieved documents
        """
        if strategy not in self.retrievers:
            logger.warning(f"Unknown strategy '{strategy}'. Defaulting to {STRATEGY_VECTOR_SEARCH}.")
            strategy = STRATEGY_VECTOR_SEARCH

        retriever = self.retrievers[strategy]
        # Pass context_doc_paths parameter to retriever if provided
        if context_doc_paths:
            documents = retriever.retrieve(query, top_k=top_k, docs=context_doc_paths)
        else:
            documents = retriever.retrieve(query, top_k=top_k)
        logger.info(f"Retrieved {len(documents)} documents using {strategy}")
        return documents

    def _execute_multi_hop_retrieval(self, sub_queries: List[str], strategy: str,
                                      top_k: int = 3, context_doc_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Execute retrieval for multi-hop queries by retrieving for each sub-query.

        Args:
            sub_queries: List of sub-queries
            strategy: Retrieval strategy name
            top_k: Number of documents per sub-query
            context_doc_paths: Optional list of file paths to RAG content documents

        Returns:
            Aggregated list of documents
        """
        all_documents = []
        seen_ids = set()

        for i, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Executing sub-query {i}/{len(sub_queries)}: {sub_query}")
            sub_docs = self._retrieve_with_strategy(sub_query, strategy, top_k=top_k, context_doc_paths=context_doc_paths)
            for doc in sub_docs:
                doc_id = doc.get("id")
                if doc_id not in seen_ids:
                    all_documents.append(doc)
                    seen_ids.add(doc_id)

        logger.info(f"Multi-hop retrieval complete. Aggregated {len(all_documents)} unique documents.")
        return all_documents




def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Adaptive RAG - Intelligent Retrieval-Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python adaptive_rag.py --query "What is artificial intelligence?"
  
  # With context documents
  python adaptive_rag.py --query "Best AI companies" \\
                         --context-docs docs/companies.txt docs/ai_trends.txt
  
  # With multiple retrievers
  python adaptive_rag.py --query "What is machine learning?" \\
                         --context-docs data/ml.txt \\
                         --multi-retriever
  
  # Query from file
  python adaptive_rag.py --query-file query.txt --context-docs data/docs.txt
        """
    )
    
    # Query arguments (required: either --query or --query-file)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        '--query', '-q',
        type=str,
        help='User query string',
        metavar='QUERY'
    )
    query_group.add_argument(
        '--query-file', '-qf',
        type=str,
        help='Path to file containing the query',
        metavar='FILE'
    )
    
    # Context documents (optional)
    parser.add_argument(
        '--context-docs', '-c',
        nargs='+',
        help='Paths to context document files (space-separated)',
        metavar='FILE'
    )
    
    # Retrieval mode
    parser.add_argument(
        '--multi-retriever', '-m',
        action='store_true',
        help='Use multiple retrievers (vector, graph, web) and aggregate contexts'
    )
    
    # Output format
    parser.add_argument(
        '--output-format', '-o',
        choices=['json', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    
    # Verbose logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def get_user_query(args: argparse.Namespace) -> Optional[str]:
    """
    Extract user query from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Query string or None if not found
    """
    if args.query:
        return args.query
    
    if args.query_file:
        try:
            query_file_path = Path(args.query_file)
            if not query_file_path.exists():
                logger.error(f"Query file not found: {args.query_file}")
                return None
            with open(query_file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading query file '{args.query_file}': {str(e)}")
            return None
    
    return None


def get_context_doc_paths(args: argparse.Namespace) -> Optional[List[str]]:
    """
    Extract and validate context document paths from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of validated file paths or None if not provided
    """
    if not args.context_docs:
        return None
    
    validated_paths = []
    for doc_path in args.context_docs:
        doc_file_path = Path(doc_path)
        if not doc_file_path.exists():
            logger.warning(f"Context document not found: {doc_path}")
        else:
            validated_paths.append(doc_path)
    
    return validated_paths if validated_paths else None


def format_output(result: Dict[str, Any], output_format: str) -> str:
    """
    Format the result based on output format preference.
    
    Args:
        result: Result dictionary from orchestrate()
        output_format: Output format ('json' or 'text')
        
    Returns:
        Formatted output string
    """
    if output_format == 'json':
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    # Text format (default)
    output = []
    output.append("\n" + "=" * 80)
    output.append("ADAPTIVE RAG RESPONSE")
    output.append("=" * 80)
    
    if result.get('error'):
        output.append(f"\n❌ ERROR: {result.get('error_message')}")
        output.append(f"Error Type: {result.get('error_type')}")
    else:
        output.append(f"\n📝 Query: {result.get('query')}")
        output.append(f"\n💡 Answer:\n{result.get('answer')}")
        
        metadata = result.get('metadata', {})
        output.append(f"\n📊 Metadata:")
        output.append(f"  - Strategy: {result.get('retrieval_strategy')}")
        output.append(f"  - Confidence: {metadata.get('confidence_score'):.2f}" if metadata.get('confidence_score') else "  - Confidence: N/A")
        output.append(f"  - Documents: {metadata.get('documents_count')}")
        output.append(f"  - Query Type: {metadata.get('query_type')}")
        output.append(f"  - Multi-Retriever: {metadata.get('multi_retriever_used')}")
    
    output.append("=" * 80)
    return "\n".join(output)


def main():
    """Main entry point - initializes RAG with CLI arguments and executes orchestration."""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Extract user query from CLI args
    user_query = get_user_query(args)
    if not user_query:
        logger.error("No query provided. Use --query or --query-file")
        sys.exit(1)
    
    # Extract context document paths from CLI args
    context_document_paths = get_context_doc_paths(args)
    
    # Log input parameters
    logger.info(f"User Query: {user_query}")
    if context_document_paths:
        logger.info(f"Context Documents: {context_document_paths}")
    logger.info(f"Multi-Retriever Mode: {args.multi_retriever}")
    
    # Initialize dependencies once
    llm = config.get_llm()
    prompt_enhancer = PromptEnhancer(llm=llm)
    query_analyzer = QueryAnalyzer(llm=llm, prompt_enhancer=prompt_enhancer)
    
    # Initialize orchestrator with vector and graph search indexers
    orchestrator = QueryOrchestrator(
        llm=llm,
        prompt_enhancer=prompt_enhancer,
        query_analyzer=query_analyzer,
        vector_index_path="faiss_index",
        neo4j_uri=None,  # Set from environment or config
        neo4j_user=None,
        neo4j_password=None,
        env_file=".env"
    )

    # Execute orchestration with user's choice of retrieval mode
    if args.multi_retriever:
        print("\n" + "=" * 80)
        print("MULTI-RETRIEVER MODE (AGGREGATED CONTEXT)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SINGLE RETRIEVER MODE")
        print("=" * 80)
    
    result = orchestrator.orchestrate(
        query=user_query,
        context_doc_paths=context_document_paths,
        use_multiple_retrievers=args.multi_retriever
    )
    
    # Format and print output
    output = format_output(result, args.output_format)
    print(output)
    
    # Exit with appropriate code
    sys.exit(0 if not result.get('error') else 1)


if __name__ == "__main__":
    main()
