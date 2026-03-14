import json
from typing import List, Dict, Any
import re

from app_config import get_config
from enhance_prompt import PromptEnhancer


class QueryAnalyzer:
    """
    Analyze user queries for Adaptive RAG systems: intent, type, entities, domain, complexity, retrieval strategy, rewrite, decomposition, and confidence.
 
    Dependencies are injected to ensure single initialization.
    """
    def __init__(self, llm=None, prompt_enhancer=None):
        """
        Initialize QueryAnalyzer with optional injected dependencies.

        Args:
            llm: Optional pre-initialized LLM instance
            prompt_enhancer: Optional pre-initialized PromptEnhancer instance
        """
        config = get_config()
        self.llm = llm or config.get_llm()
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer(llm=self.llm)

    @property
    def prompt_template(self) -> str:
        return (
            "You are an expert AI query analyst for a Retrieval-Augmented Generation (RAG) system. "
            "Given a user query, analyze it and return a JSON object with the following fields:\n"
            "- query: the original query\n"
            "- intent: one of [informational, analytical, lookup, action]\n"
            "- query_type: one of [factual, reasoning, multi-hop, ambiguous, comparison]\n"
            "- entities: list of important entities, organizations, technologies, or concepts\n"
            "- domain: the main domain (e.g., technology, finance, legal, healthcare, general knowledge)\n"
            "- complexity: one of [simple, moderate, complex]\n"
            "- recommended_retrieval_strategy: one of [vector search, hybrid search, graph retrieval, SQL retrieval, web search, multi-step retrieval]\n"
            "- rewrite_query: improved version of the query if needed (else repeat original)\n"
            "- sub_queries: list of sub-queries if multi-hop, else []\n"
            "- confidence_score: float between 0 and 1\n"
            "Analyze carefully and be concise. Only output valid JSON.\n"
            "Query: {query}"
        )

    def analyze(self, query: str) -> Dict[str, Any]:
        prompt = self.prompt_template.format(query=query)
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            json_str = self._extract_json(content)
            result = json.loads(json_str)
            # Use enhance_prompt to rewrite if needed
            if (not result.get("rewrite_query")) or (result["rewrite_query"].strip() == query.strip()):
                improved = self.prompt_enhancer.improve_prompt_openai(query, "")
                if improved and improved.strip() != query.strip():
                    result["rewrite_query"] = improved.strip()
            return result
        except Exception as e:
            return {
                "query": query,
                "intent": "",
                "query_type": "",
                "entities": [],
                "domain": "",
                "complexity": "",
                "recommended_retrieval_strategy": "",
                "rewrite_query": "",
                "sub_queries": [],
                "confidence_score": 0.0,
                "error": str(e)
            }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object from a string."""
        import re
        match = re.search(r'{.*}', text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON object found in LLM response.")


def main():
    config = get_config()
    example_query = "Which companies investing in AI are hiring machine learning engineers?"
    analyzer = QueryAnalyzer()
    result = analyzer.analyze(example_query)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
