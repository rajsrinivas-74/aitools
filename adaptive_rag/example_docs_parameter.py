"""
Example: Using the docs parameter with Adaptive RAG

The orchestrator now accepts a 'docs' parameter that specifies a list of paths to
RAG content text files. All documents will be indexed and used for retrieval
in both vector search and graph search strategies.

Features:
- Vector Search: Creates FAISS embeddings from all documents, aggregated index
- Graph Search: Extracts entities from all documents, combines in Neo4j
- Web Search: Ignores the docs parameter (searches live web only)
"""

from adaptive_rag import QueryOrchestrator
import os


def example_single_query_multiple_docs():
    """Example 1: Query multiple documents with vector search."""
    print("=" * 70)
    print("Example 1: Multiple Documents with Vector Search")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    
    # Paths to multiple content documents
    doc_paths = [
        "chapter1.txt",
        "chapter2.txt",
        "chapter3.txt"
    ]
    
    # Verify at least some documents exist
    existing_docs = [d for d in doc_paths if os.path.exists(d)]
    if not existing_docs:
        print(f"Note: No documents found at {doc_paths}")
        print("To run this example, create text files at those paths")
        return
    
    print(f"Found {len(existing_docs)}/{len(doc_paths)} documents")
    
    # Query with multiple documents
    query = "Summarize the main concept"
    response = orchestrator.orchestrate(
        query=query,
        use_multiple_retrievers=False,
        docs=doc_paths  # ← Pass list of document paths
    )
    
    # Check response
    if response.get("error"):
        print(f"Error: {response['error_message']}")
    else:
        print(f"\nQuery: {query}")
        print(f"Answer: {response['answer']}")
        print(f"Documents retrieved: {response['metadata']['documents_count']}")
        print(f"Strategy: {response['retrieval_strategy']}")


def example_multi_retriever_multiple_docs():
    """Example 2: Multiple retrievers with multiple documents."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Retrievers with Multiple Documents")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    
    # Multiple documents to index
    docs = ["textbook.txt", "references.txt", "notes.txt"]
    
    query = "Compare concepts across all materials"
    response = orchestrator.orchestrate(
        query=query,
        use_multiple_retrievers=True,  # Use vector + graph + web
        docs=docs  # ← All three strategies receive docs list
    )
    
    if response.get("error"):
        print(f"Error: {response['error_message']}")
    else:
        print(f"Query: {query}")
        print(f"Answer: {response['answer']}")
        print(f"Sources: {response['aggregated_context']['sources']}")
        print(f"Total context blocks: {response['aggregated_context']['total_blocks']}")


def example_multiple_queries_same_docs():
    """Example 3: Run multiple queries against the same documents."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple Queries Against Same Documents")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    docs = ["source1.txt", "source2.txt"]
    
    queries = [
        "What is the definition?",
        "How does it work?",
        "What are the applications?"
    ]
    
    print(f"Indexing documents: {docs}")
    print(f"Running {len(queries)} queries...\n")
    
    for i, query in enumerate(queries, 1):
        response = orchestrator.orchestrate(
            query=query,
            docs=docs
        )
        
        status = "✓" if not response.get("error") else "✗"
        print(f"{status} Query {i}: {query}")
        if not response.get("error"):
            print(f"  Found {response['metadata']['documents_count']} documents")
        print()


def example_empty_and_none_docs():
    """Example 4: Handle edge cases."""
    print("\n" + "=" * 70)
    print("Example 4: Edge Cases - Empty and None docs")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    
    # Case 1: docs=None (use pre-indexed content)
    print("\nCase 1: docs=None (uses pre-indexed content)")
    response = orchestrator.orchestrate(
        query="What's available?",
        docs=None  # ← Explicitly None
    )
    print(f"Result: {response.get('error', False)} error")
    
    # Case 2: docs=[] (empty list - same as None)
    print("\nCase 2: docs=[] (empty list)")
    response = orchestrator.orchestrate(
        query="What's available?",
        docs=[]  # ← Empty list
    )
    print(f"Result: {response.get('error', False)} error")
    
    # Case 3: Mixed (some exist, some don't)
    print("\nCase 3: Mixed paths (some exist, some don't)")
    docs = ["existing.txt", "/nonexistent/file.txt", "another.txt"]
    response = orchestrator.orchestrate(
        query="Test query",
        docs=docs
    )
    print(f"Result: {response.get('error', False)} error")
    if not response.get('error'):
        print(f"Found {response['metadata']['documents_count']} documents despite missing files")


def example_error_handling():
    """Example 5: Error handling with docs parameter."""
    print("\n" + "=" * 70)
    print("Example 5: Error Handling with docs")
    print("=" * 70)
    
    orchestrator = QueryOrchestrator()
    
    # Non-existent documents
    print("\nAttempting to query non-existent documents...")
    response = orchestrator.orchestrate(
        query="Test",
        docs=["/nonexistent/file1.txt", "/nonexistent/file2.txt"]
    )
    
    if response.get("error"):
        print(f"Error Type: {response['error_type']}")
        print(f"Error Message: {response['error_message']}")
        if response['metadata'].get('error_context'):
            print(f"Context: {response['metadata']['error_context']}")
    else:
        print("Query handled gracefully (used fallback or returns empty)")


# ============================================================================
# Usage Documentation
# ============================================================================

USAGE_GUIDE = """
DOCS PARAMETER USAGE GUIDE
==========================

The orchestrator now accepts an optional 'docs' parameter (list of strings) for
specifying multiple content documents.

Signature:
----------
orchestrate(
    query: str,
    use_multiple_retrievers: bool = False,
    docs: List[str] = None  # ← List of document paths
) -> Dict[str, Any]

Parameters:
-----------
query (str):
    The user's question

use_multiple_retrievers (bool):
    If True: Uses Vector + Graph + Web and aggregates
    If False: Uses single best strategy

docs (List[str], optional):
    List of file paths to index and search
    
    Behavior per strategy:
    - Vector Search: Indexes all documents with FAISS (aggregated embeddings)
    - Graph Search: Extracts entities from all and builds combined Neo4j graph
    - Web Search: Ignores this parameter (searches live web)
    
    Handling:
    - Non-existent files: Logged as error, skipped, continues with others
    - Empty list: Same as docs=None (uses pre-indexed content)
    - None: Uses existing indexes

Examples:
---------

# Single document
docs = ["content.txt"]
response = orchestrator.orchestrate("Query", docs=docs)

# Multiple documents
docs = ["chapter1.txt", "chapter2.txt", "references.txt"]
response = orchestrator.orchestrate("Query", docs=docs)

# Multiple documents with multi-retriever
response = orchestrator.orchestrate(
    "Compare concepts",
    use_multiple_retrievers=True,
    docs=["source1.txt", "source2.txt", "source3.txt"]
)

# Using pre-indexed content
response = orchestrator.orchestrate("Query")  # docs=None by default

Response Format:
----------------
{
    "query": str,
    "error": bool,
    "error_type": str (optional),
    "error_message": str (optional),
    "error_code": str (optional),
    "retrieval_strategy": str,
    "documents_retrieved": List,
    "aggregated_context": Dict,
    "answer": str,
    "metadata": {
        "documents_count": int,
        "confidence_score": float,
        "multi_retriever_used": bool,
        "error_context": Dict (optional)
    }
}

Key Differences from Single doc:
---vs. doc (singular):
    - docs accepts List[str] instead of str
    - Multiple documents indexed in one call
    - All documents in aggregated index
    - Cleaner API for multi-document workflows

Error Handling:
---------------
- Missing files: Logged as error, skipped, continues
- Index failures: Logged, continues with queryable documents
- Empty docs list: Same as docs=None (uses pre-indexed)
- All docs fail: Returns empty results (fallback available)
"""

if __name__ == "__main__":
    print(USAGE_GUIDE)
    print("\n" + "=" * 70)
    print("Running Examples")
    print("=" * 70)
    
    example_single_query_multiple_docs()
    example_multi_retriever_multiple_docs()
    example_multiple_queries_same_docs()
    example_empty_and_none_docs()
    example_error_handling()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
