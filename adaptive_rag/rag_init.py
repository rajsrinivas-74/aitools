"""
Initialization example for the Adaptive RAG system.

This demonstrates the proper way to initialize all components with
centralized configuration and dependency injection.
"""

from app_config import get_config
from enhance_prompt import PromptEnhancer
from query_analysis import QueryAnalyzer
from query_orchestrator import QueryOrchestrator, LLMGenerator


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
    
    # Initialize generation LLM (higher temperature for creativity)
    llm_gen = config.get_llm_generator(temperature=0.7)
    print(f"✓ Generation LLM initialized")
    
    # Initialize components with shared LLM instance
    prompt_enhancer = PromptEnhancer(llm=llm)
    print(f"✓ PromptEnhancer initialized")
    
    query_analyzer = QueryAnalyzer(llm=llm, prompt_enhancer=prompt_enhancer)
    print(f"✓ QueryAnalyzer initialized")
    
    llm_generator = LLMGenerator(llm=llm_gen)
    print(f"✓ LLMGenerator initialized")
    
    # Initialize orchestrator with all injected dependencies
    orchestrator = QueryOrchestrator(
        llm=llm,
        prompt_enhancer=prompt_enhancer,
        query_analyzer=query_analyzer,
        llm_generator=llm_generator
    )
    print(f"✓ QueryOrchestrator initialized")
    
    # Return all components as a dict for convenient access
    return {
        "config": config,
        "llm": llm,
        "llm_gen": llm_gen,
        "prompt_enhancer": prompt_enhancer,
        "query_analyzer": query_analyzer,
        "llm_generator": llm_generator,
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
