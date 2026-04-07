"""
Transaction Loader Service.
Converts unified Transaction format to database-compatible format and loads to FAISS/Neo4j.
Handles embedding generation, graph indexing, and analysis summary creation.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from backend.models.transaction import Transaction, TransactionBatch, TransactionType
from backend.models import ProcessedTransaction, CanonicalTransaction, TransactionChunk, AnalysisResult
from backend.services.rag_pipeline import RAGPipeline, get_rag_pipeline
from backend.services.llm_service import LLMService
from config.settings import settings

logger = logging.getLogger(__name__)


class TransactionLoader:
    """
    Unified Transaction Loader.
    Converts unified Transaction format to ProcessedTransaction and loads to databases.
    """
    
    def __init__(self):
        """Initialize transaction loader with RAG components."""
        self.logger = logging.getLogger(__name__)
        self.rag_pipeline = get_rag_pipeline()
        self.llm_service = None
        
        try:
            self.llm_service = LLMService()
        except Exception as e:
            self.logger.debug(f"LLM service not available: {e}")
    
    def load_batch_to_databases(self, batch: TransactionBatch) -> Dict[str, Any]:
        """
        Load a TransactionBatch to FAISS and Neo4j databases.
        
        Args:
            batch: TransactionBatch with unified transactions
            
        Returns:
            Dictionary with loading results and statistics
        """
        self.logger.info(f"Loading {len(batch.transactions)} transactions to databases")
        
        # Convert unified transactions to ProcessedTransaction format
        processed_transactions = self._convert_to_processed(batch.transactions)
        
        # Index in RAG pipeline (FAISS + Neo4j)
        try:
            self.rag_pipeline.index_transactions(processed_transactions)
            self.logger.info(f"✓ Indexed {len(processed_transactions)} transactions in FAISS and Neo4j")
        except Exception as e:
            self.logger.error(f"Failed to index transactions: {e}")
        
        # Generate analysis summary
        analysis = self._generate_analysis_summary(batch)
        
        return {
            'status': 'success',
            'total_transactions': len(batch.transactions),
            'indexed_count': len(processed_transactions),
            'faiss_indexed': bool(self.rag_pipeline.faiss_store),
            'neo4j_indexed': bool(self.rag_pipeline.neo4j_store),
            'analysis_summary': analysis,
            'source_file': batch.source_file,
            'average_confidence': batch.total_confidence,
            'high_confidence_count': batch.high_confidence_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def _convert_to_processed(self, transactions: List[Transaction]) -> List[ProcessedTransaction]:
        """
        Convert unified Transaction format to ProcessedTransaction.
        
        Args:
            transactions: List of unified Transaction objects
            
        Returns:
            List of ProcessedTransaction objects compatible with RAG pipeline
        """
        processed = []
        
        for i, trans in enumerate(transactions):
            # Create canonical transaction (legacy format compatibility)
            canonical = CanonicalTransaction(
                date=trans.date,  # Already in YYYY-MM-DD format
                description=trans.description,
                amount=trans.amount,
                type=trans.type.value,
                balance=trans.balance if trans.balance else None,
                source_file=trans.source_file,
                requires_review=trans.confidence_score < 0.8
            )
            
            # Determine category from description
            category = self._categorize_transaction(trans)
            
            # Create processed transaction
            processed_trans = ProcessedTransaction(
                canonical=canonical,
                category=category,
                normalized_description=trans.description,
                transaction_id=f"{trans.source_file}_{i:04d}",
                is_income=trans.type == TransactionType.CREDIT,
                metadata={
                    'confidence_score': trans.confidence_score,
                    'extraction_method': trans.extraction_method.value,
                    'source_file': trans.source_file,
                    'currency': trans.currency,
                    'llm_verified': trans.metadata.get('llm_verified', False) if trans.metadata else False,
                    'balance': float(trans.balance) if trans.balance else None
                }
            )
            
            processed.append(processed_trans)
        
        self.logger.debug(f"Converted {len(processed)} transactions to ProcessedTransaction format")
        return processed
    
    def _categorize_transaction(self, trans: Transaction) -> str:
        """
        Categorize transaction based on description and amount.
        
        Args:
            trans: Transaction to categorize
            
        Returns:
            Category string
        """
        desc_lower = trans.description.lower()
        
        # Payment/Transfer categories
        if any(kw in desc_lower for kw in ['upi', 'transfer', 'paid to', 'payment']):
            return 'Payment'
        
        # Shopping
        if any(kw in desc_lower for kw in ['amazon', 'flipkart', 'mall', 'store', 'shop']):
            return 'Shopping'
        
        # Bills & Utilities
        if any(kw in desc_lower for kw in ['bill', 'utility', 'electric', 'water', 'internet', 'mobile']):
            return 'Bills & Utilities'
        
        # Travel
        if any(kw in desc_lower for kw in ['travel', 'flight', 'hotel', 'uber', 'cab', 'taxi']):
            return 'Travel'
        
        # Food & Dining
        if any(kw in desc_lower for kw in ['restaurant', 'cafe', 'food', 'dining', 'pizza', 'burger']):
            return 'Food & Dining'
        
        # Income
        if trans.type == TransactionType.CREDIT:
            return 'Income'
        
        # Default
        return 'Other'
    
    def _generate_analysis_summary(self, batch: TransactionBatch) -> Dict[str, Any]:
        """
        Generate analysis summary of the transaction batch.
        
        Args:
            batch: TransactionBatch to analyze
            
        Returns:
            Analysis summary dictionary
        """
        # Calculate basic statistics
        total_debit = sum(float(t.amount) for t in batch.transactions if t.type == TransactionType.DEBIT)
        total_credit = sum(float(t.amount) for t in batch.transactions if t.type == TransactionType.CREDIT)
        
        transactions_by_category = self._group_by_category(batch.transactions)
        
        summary = {
            'total_transactions': len(batch.transactions),
            'total_debit': round(total_debit, 2),
            'total_credit': round(total_credit, 2),
            'net_balance': round(total_credit - total_debit, 2),
            'average_transaction': round((total_debit + total_credit) / len(batch.transactions), 2) if batch.transactions else 0,
            'transactions_by_category': {cat: len(trans) for cat, trans in transactions_by_category.items()},
            'average_confidence': round(batch.total_confidence, 2),
            'high_confidence_transactions': batch.high_confidence_count,
            'date_range': self._get_date_range(batch.transactions),
            'top_categories': self._get_top_categories(transactions_by_category),
            'high_value_transactions': self._get_high_value_transactions(batch.transactions, limit=5)
        }
        
        self.logger.debug(f"Generated analysis summary: {summary['total_transactions']} transactions analyzed")
        return summary
    
    def _group_by_category(self, transactions: List[Transaction]) -> Dict[str, List[Transaction]]:
        """Group transactions by category."""
        categories = {}
        for trans in transactions:
            cat = self._categorize_transaction(trans)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(trans)
        return categories
    
    def _get_date_range(self, transactions: List[Transaction]) -> Dict[str, str]:
        """Get earliest and latest transaction dates."""
        if not transactions:
            return {'from': None, 'to': None}
        
        dates = sorted([t.date for t in transactions])
        return {'from': dates[0], 'to': dates[-1]}
    
    def _get_top_categories(self, categories_dict: Dict[str, List[Transaction]], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top spending categories."""
        category_totals = []
        
        for cat, trans in categories_dict.items():
            total = sum(float(t.amount) for t in trans if t.type == TransactionType.DEBIT)
            category_totals.append({
                'category': cat,
                'count': len(trans),
                'total': round(total, 2),
                'average': round(total / len(trans), 2) if trans else 0
            })
        
        # Sort by total amount and return top categories
        return sorted(category_totals, key=lambda x: x['total'], reverse=True)[:limit]
    
    def _get_high_value_transactions(self, transactions: List[Transaction], limit: int = 5) -> List[Dict[str, Any]]:
        """Get highest value transactions with category information."""
        high_value = sorted(
            transactions,
            key=lambda t: float(t.amount),
            reverse=True
        )[:limit]
        
        return [
            {
                'date': t.date,
                'description': t.description[:50],
                'amount': round(float(t.amount), 2),
                'type': t.type.value,
                'category': t.category,
                'confidence': round(t.confidence_score, 2),
                'currency': t.currency
            }
            for t in high_value
        ]


class TransactionQueryEngine:
    """
    Query engine for user queries on transactions.
    Uses RAG pipeline for semantic search and knowledge graphs.
    """
    
    def __init__(self):
        """Initialize query engine."""
        self.logger = logging.getLogger(__name__)
        self.rag_pipeline = get_rag_pipeline()
        self.llm_service = None
        
        try:
            self.llm_service = LLMService()
        except Exception as e:
            self.logger.debug(f"LLM service not available: {e}")
    
    def query_transactions(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Execute user query on loaded transactions.
        Uses semantic search via FAISS and spending patterns via Neo4j.
        
        Args:
            query: Natural language query (e.g., "How much did I spend on food?")
            limit: Maximum results to return
            
        Returns:
            Query results with transactions and insights
        """
        self.logger.info(f"Processing user query: {query}")
        
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'semantic_results': [],
            'graph_results': [],
            'insights': [],
            'error': None
        }
        
        try:
            # Try semantic search via FAISS
            if self.rag_pipeline.faiss_store:
                try:
                    context = self.rag_pipeline.retrieve_context(query, limit=limit)
                    results['semantic_results'] = context.get('context', [])
                except Exception as e:
                    self.logger.debug(f"FAISS search failed: {e}")
            
            # Try spending patterns via Neo4j
            if self.rag_pipeline.neo4j_store and self.rag_pipeline.neo4j_store.connected:
                try:
                    graph_results = self.rag_pipeline.neo4j_store.get_spending_patterns(limit=limit)
                    results['graph_results'] = graph_results or []
                except Exception as e:
                    self.logger.debug(f"Neo4j query failed: {e}")
            
            # Generate LLM insights from results
            if self.llm_service and (results['semantic_results'] or results['graph_results']):
                insights = self._generate_query_insights(query, results)
                results['insights'] = insights
            
            results['total_results'] = len(results['semantic_results']) + len(results['graph_results'])
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_query_insights(self, query: str, results: Dict[str, Any]) -> List[str]:
        """
        Generate insights from query results using LLM.
        
        Args:
            query: Original query
            results: Query results from FAISS/Neo4j
            
        Returns:
            List of insight strings
        """
        if not self.llm_service:
            return []
        
        try:
            # Prepare context from results
            context_text = f"User Query: {query}\n\n"
            
            # Add semantic results
            if results['semantic_results']:
                context_text += f"Found {len(results['semantic_results'])} related transactions:\n"
                for r in results['semantic_results'][:3]:
                    if isinstance(r, dict):
                        context_text += f"  - {r.get('text', str(r)[:100])}\n"
                    else:
                        context_text += f"  - {str(r)[:100]}\n"
            
            # Add graph results
            if results['graph_results']:
                context_text += f"\nSpending patterns:\n"
                for r in results['graph_results'][:3]:
                    if isinstance(r, dict):
                        cat = r.get('category', 'Unknown')
                        total = r.get('total', 0)
                        count = r.get('count', 0)
                        context_text += f"  - {cat}: ${total:,.2f} ({count} transactions)\n"
            
            # Generate insights
            insight_prompt = f"""{context_text}

Based on this transaction data, provide 2-3 brief, actionable insights.
Focus on: spending patterns, notable observations, or helpful observations.
Be specific and practical.
Return insights as bullet points."""
            
            response = self.llm_service.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": insight_prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            insights_text = response.choices[0].message.content.strip()
            # Split by bullet points or numbers
            insights = []
            for line in insights_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or (len(line) > 0 and line[0].isdigit())):
                    # Remove bullet/number and clean up
                    clean_line = line.lstrip('•-0123456789. ').strip()
                    if clean_line:
                        insights.append(clean_line)
            
            return insights[:3]  # Return top 3 insights
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
            return []