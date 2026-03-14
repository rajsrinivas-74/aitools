"""
Example: Using the doc parameter with Adaptive RAG

The orchestrator now accepts a 'doc' parameter that specifies the path to a 
RAG content text file. This document will be indexed and used for retrieval
in both vector search and graph search strategies.

Features:
- Vector Search: Creates FAISS embeddings from the document
- Graph Search: Extracts entities and creates Neo4j knowledge graph
- Web Search: Ignores the doc parameter (searches live web)
"""

from adaptive_rag import QueryOrchestrator
import os


def example_single_document_retrieval():
    """Example 1: Query a single document using vector search (default)."""
    print("=" * 70)
    print("Example 1: Single Document Retrieval (Vector Search)")
    print("=" * 70)
    
    # Initialize orchestrator
    orchestrator = QueryOrchestrator()
    
    # Path to your content document
    doc_path = "rag_content.txt"  # Replace with actual path
    
    if not os.path.exists(doc_path):
        print(f"Note: Document not found at {doc_path}")
        print("To run this example, create a text file at that path")
        return
    
    # Query with document
    query = "What is machine learning?"
    response = orchestrator.orchestrate(
        query=query,
        use_multiple_retrievers=False,  # Just vector search
        doc=doc_path  # ← Pass document path here
    )
    
    # Check response
    if response.get("error"):
        print(f"Error: {response['error_message']}")
    else:
        print(f"Query: {query}")
        print(f"Answer: {response['answer']}")
        print(f"Documents retrieved: {response['metadata']['documents_count']}")
        print(f"Strategy used: {response['retrieval_strategy']}")


def example_multi_retriever_with_doc():
    """Example 2: Use multiple retrievers with a document."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Retriever with Document (Vector + Graph)")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    doc_path = "rag_content.txt"
    
    if not os.path.exists(doc_path):
        print(f"Note: Document not found at {doc_path}")
        return
    
    query = "How does deep learning relate to neural networks?"
    response = orchestrator.orchestrate(
        query=query,
        use_multiple_retrievers=True,  # Use all strategies
        doc=doc_path  # ← Document passed to vector + graph strategies
    )
    
    if response.get("error"):
        print(f"Error: {response['error_message']}")
    else:
        print(f"Query: {query}")
        print(f"Answer: {response['answer']}")
        print(f"Sources used: {response['aggregated_context']['sources']}")
        print(f"Total blocks retrieved: {response['aggregated_context']['total_blocks']}")


def example_without_doc():
    """Example 3: Query without document (uses pre-indexed content)."""
    print("\n" + "=" * 70)
    print("Example 3: Query Without Document (Pre-indexed Content)")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    
    # Query without doc parameter
    # This uses whatever is already indexed in FAISS or Neo4j
    query = "Tell me about recent developments"
    response = orchestrator.orchestrate(
        query=query,
        use_multiple_retrievers=False
        # doc parameter omitted - uses existing indexes
    )
    
    if response.get("error"):
        print(f"Error: {response['error_message']}")
    else:
        print(f"Query: {query}")
        print(f"Answer: {response['answer']}")


def example_error_handling_with_doc():
    """Example 4: Error handling when document doesn't exist."""
    print("\n" + "=" * 70)
    print("Example 4: Error Handling (Non-existent Document)")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    
    # Try to query with non-existent document
    query = "What is Python?"
    response = orchestrator.orchestrate(
        query=query,
        doc="/nonexistent/path/file.txt"
    )
    
    if response.get("error"):
        print(f"Error Type: {response['error_type']}")
        print(f"Error Message: {response['error_message']}")
        print(f"Error Code: {response['error_code']}")
        if response['metadata'].get('error_context'):
            print(f"Context: {response['metadata']['error_context']}")
    else:
        print("Query succeeded (but shouldn't in this case)")


# ============================================================================
# Usage Documentation
# ============================================================================

USAGE_GUIDE = """
DOC PARAMETER USAGE GUIDE
=========================

The orchestrator now accepts an optional 'doc' parameter in the orchestrate() method.

Signature:
----------
orchestrate(
    query: str,
    use_multiple_retrievers: bool = False,
    doc: str = None  # ← Path to your content document
) -> Dict[str, Any]

Parameters:
-----------
query (str):
    The user's question or query string

use_multiple_retrievers (bool, default=False):
    If True, uses Vector + Graph + Web search and aggregates results
    If False, uses only one strategy based on query analysis

doc (str, optional):
    Path to a text file containing the content to index and search
    
    Effects per strategy:
    - Vector Search: Indexes the document with FAISS embeddings
    - Graph Search: Extracts entities and builds Neo4j knowledge graph
    - Web Search: Ignores this parameter, searches live web
    
    If doc is not provided:
        - Uses pre-existing indexes (FAISS, Neo4j)
        - If no indexes exist, returns empty results

Response Format:
----------------
{
    "query": str,                    # Original query
    "error": bool,                   # Success/failure flag
    "error_type": str (optional),    # ValidationError, RetrieverError, etc.
    "error_message": str (optional), # Error description
    "error_code": str (optional),    # Error identifier
    "retrieval_strategy": str,       # Strategy used: vector search|graph retrieval|web search
    "documents_retrieved": List,     # Retrieved documents
    "aggregated_context": Dict,      # Context blocks (for multi-retriever)
    "answer": str,                   # Generated answer
    "metadata": {
        "documents_count": int,
        "confidence_score": float,
        "query_type": str,
        "fallback_used": bool,
        "multi_retriever_used": bool,
        "error_context": Dict (optional)
    }
}

Examples:
---------

# Example 1: Vector search with document
response = orchestrator.orchestrate(
    "What is machine learning?",
    doc="ml_content.txt"
)

# Example 2: Multi-retriever with document
response = orchestrator.orchestrate(
    "Compare supervised vs unsupervised learning",
    use_multiple_retrievers=True,
    doc="ml_content.txt"
)

# Example 3: Web-based (ignores doc parameter)
response = orchestrator.orchestrate(
    "Latest AI news",
    use_multiple_retrievers=False,
    doc="ml_content.txt"  # ← Ignored by web search strategy
)

# Example 4: Without document (uses pre-indexed content)
response = orchestrator.orchestrate(
    "Summarize the main topics"
)

Error Handling:
---------------
The doc parameter supports graceful error handling:

- If doc file doesn't exist → RetrieverError response
- If doc is invalid → ValidationError or RetrieverError
- If indexing fails → RetrieverError with context

Check response['error'] to detect failures.
"""

if __name__ == "__main__":
    print(USAGE_GUIDE)
    print("\n" + "=" * 70)
    print("Running Examples")
    print("=" * 70)
    
    example_single_document_retrieval()
    example_multi_retriever_with_doc()
    example_without_doc()
    example_error_handling_with_doc()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
