"""
Web Search Retriever Module

Performs hybrid web search using Tavily API and integrates results into LLM workflows.
Follows the Adaptive RAG architecture with dependency injection and centralized configuration.
"""

import logging
from typing import Dict, List, Any, Optional

from app_config import get_config

logger = logging.getLogger(__name__)


class TavilySearch:
    """
    Performs hybrid web searches using the Tavily API.
    
    Tavily provides advanced search capabilities with:
    - Semantic understanding
    - Advanced search ranking
    - Real-time indexing
    - Structured result extraction
    """

    def __init__(self, api_key: str = None):
        """
        Initialize TavilySearch with API credentials.

        Args:
            api_key: Tavily API key. If None, attempts to load from environment.

        Raises:
            RuntimeError: If Tavily client cannot be initialized.
        """
        import os
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable not set. "
                             "Get one at: https://tavily.com")
        
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("TavilySearch initialized successfully")
        except ImportError:
            raise RuntimeError(
                "Tavily client not installed. Install with: pip install tavily-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tavily client: {e}")

    def search(self, query: str, max_results: int = 5, search_depth: str = "advanced") -> List[Dict[str, Any]]:
        """
        Perform a hybrid web search using Tavily API.

        Args:
            query: Search query string
            max_results: Maximum number of results to retrieve (default: 5)
            search_depth: Search depth - "basic" or "advanced" (default: "advanced")

        Returns:
            List of search results with title, url, and content fields
        """
        logger.info(f"Performing Tavily search: {query} (depth: {search_depth}, max_results: {max_results})")
        
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=True  # Get full page content when available
            )
            
            # Extract relevant fields from Tavily response
            results = []
            for result in response.get("results", []):
                extracted = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                }
                results.append(extracted)
            
            logger.info(f"Retrieved {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise RuntimeError(f"Web search error: {e}")
    
    @staticmethod
    def format_results_for_context(results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a context string for LLM consumption.

        Args:
            results: List of search results from search()

        Returns:
            Formatted context string with all results
        """
        if not results:
            return "No search results found."
        
        formatted = "Web Search Results:\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", result.get("raw_content", ""))
            
            # Limit snippet length
            snippet = content[:500] + "..." if len(content) > 500 else content
            
            formatted += f"[{i}] {title}\n"
            if url:
                formatted += f"    URL: {url}\n"
            if snippet:
                formatted += f"    Content: {snippet}\n"
            formatted += "\n"
        
        return formatted


class WebSearchRetriever:
    """
    Retrieves information from the web using Tavily search.
    
    Integrates web search into the Adaptive RAG pipeline.
    """

    def __init__(self, tavily_api_key: str = None):
        """
        Initialize WebSearchRetriever.

        Args:
            tavily_api_key: Tavily API key (optional if set in environment)
        """
        self.search_engine = TavilySearch(api_key=tavily_api_key)
        logger.info("WebSearchRetriever initialized")

    def retrieve(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Retrieve web search results for a query.

        Args:
            query: Search query
            max_results: Maximum results to retrieve

        Returns:
            Dict with query, results, and formatted context
        """
        logger.info(f"Retrieving web search results for: {query}")
        
        try:
            # Perform search
            results = self.search_engine.search(query, max_results=max_results)
            
            # Format for LLM
            context = TavilySearch.format_results_for_context(results)
            
            return {
                "query": query,
                "results": results,
                "context": context,
                "error": None
            }
        except Exception as e:
            logger.error(f"Web search retrieval failed: {e}")
            return {
                "query": query,
                "results": [],
                "context": "",
                "error": str(e)
            }


class WebSearchLLMPipeline:
    """
    End-to-end pipeline for web search + LLM answer generation.
    
    Combines Tavily web search with LLM to generate answers based on
    current web information.
    """

    def __init__(self, llm=None, tavily_api_key: str = None):
        """
        Initialize the web search + LLM pipeline.

        Args:
            llm: Injected LLM instance (ChatOpenAI)
            tavily_api_key: Tavily API key
        """
        config = get_config()
        self.llm = llm or config.get_llm_generator(temperature=0.7)
        self.web_retriever = WebSearchRetriever(tavily_api_key=tavily_api_key)
        logger.info("WebSearchLLMPipeline initialized")

    def process_query(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Process a query using web search + LLM generation.

        Args:
            query: User query
            max_results: Maximum web results to retrieve

        Returns:
            Dict with query, web_results, context, and final answer
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve web results
        logger.info("Step 1: Performing web search...")
        retrieval_result = self.web_retriever.retrieve(query, max_results=max_results)
        
        if retrieval_result["error"]:
            logger.error(f"Web search error: {retrieval_result['error']}")
            return {
                "query": query,
                "web_results": [],
                "context": "",
                "answer": f"Web search failed: {retrieval_result['error']}",
                "error": retrieval_result["error"]
            }
        
        # Step 2: Format context
        logger.info("Step 2: Formatting context...")
        context = retrieval_result["context"]
        
        # Step 3: Generate answer using LLM
        logger.info("Step 3: Generating answer with LLM...")
        answer = self._generate_answer(query, context)
        
        # Return structured result
        result = {
            "query": query,
            "web_results": retrieval_result["results"],
            "context": context,
            "answer": answer,
            "error": None
        }
        
        logger.info("Query processing complete")
        return result

    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate an LLM answer using web search context.

        Args:
            query: Original user query
            context: Formatted web search results

        Returns:
            Generated answer from LLM
        """
        system_prompt = (
            "You are a helpful assistant with access to current web information. "
            "Use the provided web search results to answer the user's question. "
            "Cite sources when relevant and indicate if information is from web search results. "
            "Be concise and accurate."
        )
        
        user_prompt = f"""Web Search Results:
{context}

User Question: {query}

Please provide a comprehensive answer based on these web search results. 
Include relevant citations or source titles when applicable."""
        
        try:
            response = self.llm.invoke(user_prompt)
            answer = getattr(response, "content", str(response))
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"


def main():
    """
    Demonstrate the web search + LLM pipeline.
    """
    from rag_init import initialize_rag_system
    
    print("\n" + "="*80)
    print("WEB SEARCH + LLM PIPELINE DEMO")
    print("="*80 + "\n")
    
    # Initialize system
    system = initialize_rag_system()
    
    # Create pipeline with injected dependencies
    pipeline = WebSearchLLMPipeline(llm=system["llm_gen"])
    
    # Example query
    query = "What are the latest developments in artificial intelligence in 2026?"
    
    print(f"Query: {query}\n")
    
    # Process query
    result = pipeline.process_query(query, max_results=5)
    
    # Display results
    print("\n" + "-"*80)
    print("WEB SEARCH RESULTS")
    print("-"*80)
    for i, result_item in enumerate(result["web_results"], 1):
        print(f"[{i}] {result_item['title']}")
        print(f"    URL: {result_item['url']}")
        print(f"    Snippet: {result_item['content'][:200]}...")
        print()
    
    print("-"*80)
    print("FORMATTED CONTEXT")
    print("-"*80)
    print(result["context"])
    
    print("-"*80)
    print("LLM ANSWER")
    print("-"*80)
    print(result["answer"])


if __name__ == "__main__":
    main()
