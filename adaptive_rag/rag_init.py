"""
Initialization example for the Adaptive RAG system.

This demonstrates the proper way to initialize all components with
centralized configuration and dependency injection.
"""

from app_config import get_config
from enhance_prompt import PromptEnhancer
from query_analysis import QueryAnalyzer
from adaptive_rag import QueryOrchestrator


def initialize_rag_system():
    """
    Initialize the entire RAG system once with all dependencies.
    
    This ensures:
    1. Environment variables are loaded once (in app_config)
    2. LLM is initialized once globally
    3. All components share the same LLM instance
    4. Configuration is consistent across the system
    
    Returns:
        A dict containing all initialized components ready to use.
    """
    # Get global configuration (singleton - initialized once)
    config = get_config()
    
    print(f"Initializing Adaptive RAG System")
    print(f"  Model: {config.model}")
    print(f"  Confidence Threshold: {config.confidence_threshold}")
    print(f"  Min Docs Threshold: {config.min_docs_threshold}")
    
    # Initialize LLM once at the top level (cached in config)
    llm = config.get_llm()
    print(f"✓ LLM initialized: {config.model}")
    
    # Initialize components with shared LLM instance
    prompt_enhancer = PromptEnhancer(llm=llm)
    print(f"✓ PromptEnhancer initialized")
    
    query_analyzer = QueryAnalyzer(llm=llm, prompt_enhancer=prompt_enhancer)
    print(f"✓ QueryAnalyzer initialized")
    
    # Initialize orchestrator with injected dependencies
    # Note: LLMGenerator was removed - generation is now delegated to individual retrievers
    orchestrator = QueryOrchestrator(
        llm=llm,
        prompt_enhancer=prompt_enhancer,
        query_analyzer=query_analyzer
    )
    print(f"✓ QueryOrchestrator initialized")
    
    # Return all components as a dict for convenient access
    return {
        "config": config,
        "llm": llm,
        "prompt_enhancer": prompt_enhancer,
        "query_analyzer": query_analyzer,
        "orchestrator": orchestrator,
    }


def main():
    """Example usage of the RAG system."""
    print("\n" + "="*80)
    print("ADAPTIVE RAG SYSTEM - INITIALIZATION EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize system once
    system = initialize_rag_system()
    
    print("\n" + "="*80)
    print("RUNNING QUERY...")
    print("="*80 + "\n")
    
    # Use the orchestrator to process a query
    query = "What are the latest developments in quantum computing?"
    result = system["orchestrator"].orchestrate(query)
    
    print(f"\nQuery: {result['query']}")
    print(f"Strategy: {result['retrieval_strategy']}")
    print(f"Documents Retrieved: {result['metadata']['documents_count']}")
    print(f"Confidence Score: {result['metadata']['confidence_score']:.2f}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
