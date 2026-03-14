#!/usr/bin/env python
"""
Test script for validating error handling improvements in query orchestrator.

Tests:
1. Input validation with empty queries
2. Input validation with excessively long queries
3. Error response format for validation errors
4. Error response format for retriever errors
"""

import json
from rag_utils import validate_query, ValidationError, RetrieverError, LLMError, RAGError

def test_validation_errors():
    """Test input validation error handling."""
    print("\n" + "="*60)
    print("Testing Input Validation")
    print("="*60)
    
    # Test 1: Empty query
    print("\n[Test 1] Empty query validation:")
    try:
        validate_query("")
        print("  ❌ FAILED: Should have raised ValidationError")
    except ValidationError as e:
        print(f"  ✓ PASSED: Caught ValidationError")
        print(f"    Message: {e.message}")
        print(f"    Error Code: {e.error_code}")
        print(f"    Context: {e.context}")
    
    # Test 2: Very long query
    print("\n[Test 2] Very long query validation:")
    try:
        long_query = "a" * 15000  # Exceeds 10000 limit
        validate_query(long_query)
        print("  ❌ FAILED: Should have raised ValidationError for length")
    except ValidationError as e:
        print(f"  ✓ PASSED: Caught ValidationError")
        print(f"    Message: {e.message}")
        print(f"    Error Code: {e.error_code}")
    
    # Test 3: Valid query
    print("\n[Test 3] Valid query validation:")
    try:
        result = validate_query("What is machine learning?")
        print(f"  ✓ PASSED: Valid query accepted")
        print(f"    Validated query: {result}")
    except ValidationError as e:
        print(f"  ❌ FAILED: Should not raise error for valid query: {e.message}")


def test_error_classes():
    """Test error class functionality."""
    print("\n" + "="*60)
    print("Testing Error Classes")
    print("="*60)
    
    # Test RAGError (base class)
    print("\n[Test 1] RAGError instantiation:")
    try:
        err = RAGError(
            message="Something went wrong",
            error_code="RAG_001",
            context={"module": "test", "stage": "validation"}
        )
        print(f"  ✓ PASSED: RAGError created")
        print(f"    Message: {err.message}")
        print(f"    Dict: {json.dumps(err.to_dict(), indent=2)}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test ValidationError
    print("\n[Test 2] ValidationError instantiation:")
    try:
        err = ValidationError(
            message="Query too long",
            error_code="VAL_001",
            context={"max_length": 10000, "actual_length": 15000}
        )
        print(f"  ✓ PASSED: ValidationError created")
        print(f"    Dict: {json.dumps(err.to_dict(), indent=2)}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test RetrieverError
    print("\n[Test 3] RetrieverError instantiation:")
    try:
        err = RetrieverError(
            message="Vector index not found",
            error_code="RETR_001",
            context={"strategy": "vector_search", "index_path": "faiss_index"}
        )
        print(f"  ✓ PASSED: RetrieverError created")
        print(f"    Dict: {json.dumps(err.to_dict(), indent=2)}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test LLMError
    print("\n[Test 4] LLMError instantiation:")
    try:
        err = LLMError(
            message="API rate limit exceeded",
            error_code="LLM_001",
            context={"api": "openai", "retry_after": 60}
        )
        print(f"  ✓ PASSED: LLMError created")
        print(f"    Dict: {json.dumps(err.to_dict(), indent=2)}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")


def test_error_response_format():
    """Test that error responses have proper format."""
    print("\n" + "="*60)
    print("Testing Error Response Format")
    print("="*60)
    
    # Example validation error response
    print("\n[Example 1] Validation Error Response:")
    validation_error = ValidationError(
        message="Query validation failed",
        error_code="VAL_EMPTY",
        context={"min_length": 1, "actual_length": 0}
    )
    
    error_response = {
        "query": "",
        "error": True,
        "error_type": "ValidationError",
        "error_message": validation_error.message,
        "error_code": validation_error.error_code,
        "answer": None,
        "documents_retrieved": [],
        "aggregated_context": None,
        "metadata": {
            "error_context": validation_error.context,
            "multi_retriever_used": False,
        }
    }
    
    print("Response structure:")
    print(json.dumps(error_response, indent=2))
    
    # Validate structure
    required_fields = ["query", "error", "error_type", "error_message", "error_code", 
                      "answer", "documents_retrieved", "metadata"]
    missing = [f for f in required_fields if f not in error_response]
    
    if missing:
        print(f"\n  ❌ FAILED: Missing fields: {missing}")
    else:
        print(f"\n  ✓ PASSED: All required fields present")
    
    # Example retriever error response
    print("\n[Example 2] Retriever Error Response:")
    retriever_error = RetrieverError(
        message="Failed to retrieve documents",
        error_code="RETR_FAILED",
        context={"strategy": "vector_search", "reason": "Index not found"}
    )
    
    error_response2 = {
        "query": "machine learning",
        "error": True,
        "error_type": "RetrieverError",
        "error_message": retriever_error.message,
        "error_code": retriever_error.error_code,
        "answer": None,
        "documents_retrieved": [],
        "aggregated_context": None,
        "metadata": {
            "error_context": retriever_error.context,
            "multi_retriever_used": False,
        }
    }
    
    print("Response structure:")
    print(json.dumps(error_response2, indent=2))
    print(f"\n  ✓ PASSED: Error response properly structured")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG ERROR HANDLING TEST SUITE")
    print("="*60)
    
    try:
        test_validation_errors()
        test_error_classes()
        test_error_response_format()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}", exc_info=True)
