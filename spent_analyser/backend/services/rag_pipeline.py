
"""
RAG Pipeline Orchestration.
Combines FAISS, Neo4j, and LLM for intelligent transaction analysis.
"""

import logging
from typing import List, Optional, Dict, Any

from backend.models import (
    ProcessedTransaction, TransactionChunk, AnalysisResult
)
from backend.services.faiss_store import FAISSStore
from backend.services.neo4j_store import Neo4jStore
from backend.services.llm_service import LLMService
from config.settings import settings

logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class RAGPipeline:
    """
    RAG Pipeline orchestrator.
    Combines vector search, graph queries, and LLM for analysis.
    """
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.logger = logging.getLogger(__name__)
        self.faiss_store = None
        self.neo4j_store = None
        self.llm_service = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize FAISS, Neo4j, and LLM services."""
        try:
            self.faiss_store = FAISSStore()
            self.logger.info("Initialized FAISS store")
        except Exception as e:
            self.logger.warning(f"Could not initialize FAISS: {str(e)}")
        
        try:
            self.neo4j_store = Neo4jStore()
            self.logger.info("Initialized Neo4j store")
        except Exception as e:
            self.logger.warning(f"Could not initialize Neo4j: {str(e)}")
        
        try:
            self.llm_service = LLMService()
            self.logger.info("Initialized LLM service")
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM: {str(e)}")
    
    def index_transactions(self, transactions: List[ProcessedTransaction]) -> None:
        """
        Index transactions in both FAISS and Neo4j.
        
        Args:
            transactions: Processed transactions to index
        """
        self.logger.info(f"Indexing {len(transactions)} transactions in RAG stores")
        if not transactions:
            self.logger.warning("No transactions to index")
            return
        
        try:
            # Index in FAISS
            if self.faiss_store and self.llm_service:
                chunks = self._chunk_transactions(transactions)
                embeddings = self._generate_embeddings(chunks)
                self.faiss_store.add_embeddings(chunks, embeddings)
                self.faiss_store.save_index()
            
            # Index in Neo4j
            if self.neo4j_store:
                self.neo4j_store.add_transactions(transactions)
            
            self.logger.info(f"Indexed {len(transactions)} transactions in RAG stores")
            
        except Exception as e:
            self.logger.error(f"Error indexing transactions: {str(e)}")
    
    def retrieve_context(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve context from FAISS and Neo4j.
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Number of results to retrieve
            
        Returns:
            Retrieved context dictionary
        """
        context = {
            "faiss_results": [],
            "neo4j_results": [],
            "combined_context": ""
        }
        
        try:
            # Retrieve from FAISS
            if self.faiss_store and self.llm_service:
                query_embedding = self.llm_service.generate_embedding(query)
                context["faiss_results"] = self.faiss_store.search(query_embedding, k=limit)
            
            # Retrieve from Neo4j
            if self.neo4j_store:
                context["neo4j_results"] = self.neo4j_store.get_spending_patterns(limit=limit)
            
            # Combine context
            context["combined_context"] = self._combine_context(context)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return context
    
    def generate_insights_with_rag(
        self,
        analysis_result: AnalysisResult,
        query: str = "Analyze spending patterns and provide insights"
    ) -> str:
        """
        Generate insights using RAG.
        
        Args:
            analysis_result: Current analysis result
            query: Analysis query
            
        Returns:
            Generated insights text
        """
        if not self.llm_service:
            return ""
        
        try:
            # Build context from analysis
            context = self._build_rag_context(analysis_result)
            
            # Generate insights using LLM
            insights = self.llm_service.generate_insights(context, query)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating RAG insights: {str(e)}")
            return ""
    
    def _chunk_transactions(self, transactions: List[ProcessedTransaction]) -> List[TransactionChunk]:
        """Chunk transactions for embedding."""
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        
        for i in range(0, len(transactions), chunk_size):
            chunk_txs = transactions[i:i+chunk_size]
            chunk_text = "\n".join([
                f"{t.canonical.date} - {t.normalized_description}: ${t.canonical.amount} ({t.category})"
                for t in chunk_txs
            ])
            
            chunk = TransactionChunk(
                chunk_id=f"chunk_{i//chunk_size}",
                text=chunk_text,
                transactions=chunk_txs
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(transactions)} transactions")
        return chunks
    
    def _generate_embeddings(self, chunks: List[TransactionChunk]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        if not self.llm_service:
            # Return zero embeddings as fallback
            return [[0.0] * settings.EMBEDDING_DIMENSION for _ in chunks]
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.llm_service.batch_generate_embeddings(texts)
        return embeddings
    
    def _combine_context(self, retrieval_result: Dict) -> str:
        """Combine FAISS and Neo4j results into context string."""
        context_parts = []
        
        # Add FAISS results
        if retrieval_result["faiss_results"]:
            context_parts.append("## Similar Transactions (Vector Search):")
            for result in retrieval_result["faiss_results"]:
                context_parts.append(f"- {result['text'][:100]}... (similarity: {result['similarity_score']:.2f})")
        
        # Add Neo4j results
        if retrieval_result["neo4j_results"]:
            context_parts.append("\n## Spending Patterns (Graph Analysis):")
            for result in retrieval_result["neo4j_results"]:
                context_parts.append(f"- {result['category']}: ${result['total']:.2f} ({result['count']} transactions)")
        
        return "\n".join(context_parts) if context_parts else "No context available"
    
    def _build_rag_context(self, analysis_result: AnalysisResult) -> str:
        """Build context for LLM from analysis results."""
        summary = analysis_result.summary
        context_parts = [
            "## Summary:",
            f"  Total Income: ${summary.get('total_income', 0):,.2f}",
            f"  Total Expenses: ${summary.get('total_expense', 0):,.2f}",
            f"  Net Savings: ${summary.get('net_savings', 0):,.2f}",
            f"  Savings Rate: {summary.get('savings_rate', 0):.1f}%",
            f"  Total Transactions: {summary.get('transaction_count', 0)}",
            "",
        ]
        
        # Handle new category structure
        categories = analysis_result.categories
        if isinstance(categories, dict):
            # New schema with separate income/expense
            if categories.get("expense"):
                context_parts.append("## Top Expense Categories:")
                for category in categories["expense"][:5]:
                    context_parts.append(
                        f"  - {category['category']}: ${category['amount']:,.2f} ({category['percentage']}%)"
                    )
            
            if categories.get("income"):
                context_parts.append("\n## Top Income Categories:")
                for category in categories["income"][:5]:
                    context_parts.append(
                        f"  - {category['category']}: ${category['amount']:,.2f} ({category['percentage']}%)"
                    )
        else:
            # Legacy format (list of categories)
            if categories:
                context_parts.append("## Top Categories:")
                for category in categories[:5]:
                    context_parts.append(
                        f"  - {category.get('name', category.get('category', 'Unknown'))}: "
                        f"${category['amount']:,.2f} ({category['percentage']}%)"
                    )
        
        if analysis_result.insights:
            context_parts.extend([
                "",
                "## Insights:",
            ])
            context_parts.extend(analysis_result.insights)
        
        return "\n".join(context_parts)


# Singleton instance - lazily initialized on first access
_rag_pipeline_instance = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the singleton RAGPipeline instance.
    This ensures Neo4j is initialized only once across the application.
    """
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance
