"""
Unified Transaction Extraction Example.
Demonstrates using the new Transaction data model with confidence scores.
Includes LLM-based confidence boosting for low-confidence transactions.
"""

import sys
import os
import logging
from pathlib import Path
from decimal import Decimal
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from backend.services.parser import DocumentParser
from backend.models.transaction import Transaction, TransactionBatch, ExtractionMethod, TransactionType
from backend.services.transaction_loader import TransactionLoader, TransactionQueryEngine
from backend.services.spent_analysis_service import SpentAnalysisService


class UnifiedExtractor:
    """Wraps DocumentParser and converts output to unified Transaction format."""
    
    def __init__(self):
        """Initialize with DocumentParser."""
        self.parser = DocumentParser()
        self.logger = logging.getLogger(__name__)
        self.llm_service = None
        self.transaction_loader = TransactionLoader()
        self.query_engine = TransactionQueryEngine()
        self.analysis_service = SpentAnalysisService()
        try:
            from backend.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            self.logger.debug(f"LLM service not available: {e}")
    
    def extract_with_confidence(self, file_path: Path) -> TransactionBatch:
        """
        Extract transactions and convert to unified format with confidence scores.
        
        Confidence scoring logic:
        - Schema confidence from schema adapter × 0.9 = base confidence
        - High confidence (>= 0.8): High quality extraction
        - Medium confidence (0.5-0.8): Good extraction with some uncertainty
        - Low confidence (< 0.5): Low quality, needs review
        """
        transactions, schema_mapping = self.parser.parse_and_normalize(str(file_path))
        
        # Determine extraction method from filename patterns
        file_name = file_path.name.lower()
        if 'icici' in file_name:
            method = ExtractionMethod.REGEX  # ICICI is parsed with structured regex
        elif 'gpay' in file_name or 'google' in file_name:
            method = ExtractionMethod.REGEX  # Google Pay is parsed with structured regex
        else:
            method = ExtractionMethod.REGEX
        
        # Convert to unified Transaction format with confidence scores
        unified_transactions: List[Transaction] = []
        
        for canonical_trans in transactions:
            # Calculate confidence: base schema confidence scaled appropriately
            # Schema detection itself is reliable, so keep it high for well-matched schemas
            confidence = schema_mapping.confidence_score * 0.90
            
            # Adjust confidence based on transaction quality flags
            # Only significantly reduce if has actual issues (not just 'requires_review' which is informational)
            if hasattr(canonical_trans, 'has_errors') and canonical_trans.has_errors:
                confidence *= 0.8  # Reduce by 20% if actual extraction errors detected
            
            # Create unified transaction
            trans = Transaction(
                date=canonical_trans.date.strftime('%Y-%m-%d') if hasattr(canonical_trans.date, 'strftime') else str(canonical_trans.date),
                description=canonical_trans.description,
                amount=Decimal(str(canonical_trans.amount)),
                type=TransactionType.DEBIT if canonical_trans.type == 'debit' else TransactionType.CREDIT,
                currency=getattr(canonical_trans, 'currency', '$'),
                confidence_score=min(confidence, 0.99),  # Cap at 0.99 to avoid false certainty
                extraction_method=method,
                source_file=file_path.name,
                balance=Decimal(str(canonical_trans.balance)) if hasattr(canonical_trans, 'balance') and canonical_trans.balance else None,
                metadata={
                    'schema_confidence': schema_mapping.confidence_score,
                    'source_schema_mapping': schema_mapping.to_dict() if hasattr(schema_mapping, 'to_dict') else {}
                }
            )
            unified_transactions.append(trans)
        
        # Create batch
        batch = TransactionBatch(
            transactions=unified_transactions,
            source_file=file_path.name,
            extraction_method=method
        )
        
        return batch
    
    def boost_confidence_with_llm(self, batch: TransactionBatch, confidence_threshold: float = 0.80) -> TransactionBatch:
        """
        Use LLM to verify and boost confidence of low-confidence transactions and update categorization.
        
        Processes two types of transactions:
        1. Low-confidence transactions (< threshold): LLM re-parses/verifies extraction
        2. "Other" category transactions: LLM suggests better category classification
        
        If LLM confirms the data, confidence is boosted to min(0.95, original + 0.15).
        If LLM suggests a category, transaction is re-categorized and confidence adjusted.
        
        Transactions with confidence >= 0.80 (80%) are only processed if category is "Other".
        
        Args:
            batch: TransactionBatch with transactions to boost
            confidence_threshold: Minimum confidence to skip LLM boosting (default 0.80)
                                 If category is "Other", LLM is called regardless of confidence
            
        Returns:
            New TransactionBatch with boosted confidences and updated categories
        """
        if not self.llm_service:
            self.logger.warning("LLM service not available, skipping confidence boosting")
            return batch
        
        boosted_transactions = []
        boosted_count = 0
        category_updated_count = 0
        
        for trans in batch.transactions:
            # Check if transaction needs LLM processing
            is_low_confidence = trans.confidence_score < confidence_threshold
            is_other_category = trans.category.lower() in ('other', 'other expense', 'other income')
            
            # Only process if low confidence OR category is "Other"
            if not is_low_confidence and not is_other_category:
                boosted_transactions.append(trans)
                continue
            
            # Get LLM verification and categorization suggestion
            llm_result = self._verify_and_categorize_with_llm(trans)
            
            if llm_result['verified']:
                # Create new transaction with updates
                boosted_confidence = min(trans.confidence_score + 0.15, 0.95)
                new_category = llm_result.get('suggested_category', trans.category)
                
                boosted_trans = Transaction(
                    date=trans.date,
                    description=trans.description,
                    amount=trans.amount,
                    type=trans.type,
                    currency=trans.currency,
                    category=new_category,  # Updated category
                    confidence_score=boosted_confidence,
                    extraction_method=trans.extraction_method,
                    source_file=trans.source_file,
                    balance=trans.balance,
                    raw_data=trans.raw_data,
                    metadata={
                        **(trans.metadata or {}),
                        'llm_verified': True,
                        'original_confidence': trans.confidence_score,
                        'original_category': trans.category,
                        'boost_reason': 'LLM verification confirmed extraction',
                        'category_suggested': llm_result.get('suggested_category') is not None,
                        'category_suggestion_reason': llm_result.get('category_reason', '')
                    }
                )
                boosted_transactions.append(boosted_trans)
                boosted_count += 1
                
                if is_other_category and llm_result.get('suggested_category'):
                    category_updated_count += 1
            else:
                # LLM couldn't verify, keep original
                # But still update category if it was "Other" and LLM suggested something
                if is_other_category and llm_result.get('suggested_category'):
                    trans.category = llm_result.get('suggested_category')
                    trans.metadata = {
                        **(trans.metadata or {}),
                        'original_category': trans.category,
                        'category_suggested': True,
                        'category_suggestion_reason': llm_result.get('category_reason', '')
                    }
                    category_updated_count += 1
                
                boosted_transactions.append(trans)
        
        if boosted_count > 0 or category_updated_count > 0:
            self.logger.debug(f"✓ LLM processing complete:")
            if boosted_count > 0:
                self.logger.debug(f"  - Confidence boosted: {boosted_count} transactions")
            if category_updated_count > 0:
                self.logger.debug(f"  - Categories updated: {category_updated_count} transactions")
        
        # Return new batch with boosted transactions
        return TransactionBatch(
            transactions=boosted_transactions,
            source_file=batch.source_file,
            extraction_method=batch.extraction_method
        )
    
    def _verify_and_categorize_with_llm(self, trans: Transaction) -> dict:
        """
        Use LLM to verify transaction extraction AND suggest a better category if "Other".
        
        Returns both verification result and category suggestion.
        
        Args:
            trans: Transaction to verify and categorize
            
        Returns:
            Dictionary with:
            - verified: bool (True if LLM confirms extraction)
            - suggested_category: str (better category if available)
            - category_reason: str (reason for suggestion)
        """
        try:
            from backend.services.llm_service import LLMService
            
            llm = self.llm_service or LLMService()
            
            # Prepare verification and categorization prompt
            is_other_category = trans.category.lower() in ('other', 'other expense', 'other income')
            
            if is_other_category:
                # For "Other" category transactions, ask for both verification and categorization
                verification_prompt = f"""Verify and categorize this transaction:

Transaction Data:
- Date: {trans.date}
- Description: {trans.description}
- Amount: {trans.currency}{trans.amount:.2f}
- Type: {trans.type.value}
- Current Category: {trans.category}
- Confidence: {trans.confidence_score:.2f}

First, verify if the extraction is correct (reasonable date, plausible amount, matching type).
Then, suggest a better category based on the description.

Common categories: Food & Dining, Groceries, Transportation, Travel, Shopping, Entertainment, 
Utilities, Rent/Mortgage, Insurance, Healthcare, Education, Personal Care, Home & Garden, Pet Care, 
Subscriptions, Fees & Charges, Salary, Bonus, Refund, Interest, Other Income.

Respond in JSON format:
{{
  "verification": "YES" or "NO",
  "suggested_category": "category name" or null,
  "reason": "brief reason"
}}

Output only valid JSON, no other text."""
            else:
                # For low-confidence transactions, just verify
                verification_prompt = f"""Verify if this transaction extraction is correct:

Transaction Data:
- Date: {trans.date}
- Description: {trans.description}
- Amount: {trans.currency}{trans.amount:.2f}
- Type: {trans.type.value}
- Category: {trans.category}
- Confidence: {trans.confidence_score:.2f}

Respond in JSON format:
{{
  "verification": "YES" or "NO",
  "suggested_category": null,
  "reason": "brief reason"
}}

Output only valid JSON, no other text."""
            
            response = llm.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            import json
            try:
                result = json.loads(response_text)
                return {
                    'verified': result.get('verification', 'NO').upper() == 'YES',
                    'suggested_category': result.get('suggested_category'),
                    'category_reason': result.get('reason', '')
                }
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                self.logger.warning(f"Failed to parse LLM response as JSON: {response_text}")
                return {
                    'verified': False,
                    'suggested_category': None,
                    'category_reason': ''
                }
            
        except Exception as e:
            self.logger.error(f"LLM verification/categorization failed: {e}")
            return {
                'verified': False,
                'suggested_category': None,
                'category_reason': ''
            }
    
    def _verify_transaction_with_llm(self, trans: Transaction) -> bool:
        """
        Use LLM to verify a transaction is correctly extracted (legacy method).
        
        Sends the transaction data to Claude with a verification prompt.
        Returns True if LLM confirms the extraction is reasonable.
        
        Args:
            trans: Transaction to verify
            
        Returns:
            True if LLM verification passed, False otherwise
        """
        result = self._verify_and_categorize_with_llm(trans)
        return result['verified']
    
    def load_to_databases(self, batch: TransactionBatch) -> dict:
        """
        Load transactions to vector and graph databases (FAISS + Neo4j).
        
        ⚠️  IMPORTANT: Design Intent - Required for Comprehensive Analysis
        ────────────────────────────────────────────────────────────────
        This method is REQUIRED if you plan to use comprehensive analysis.
        Calling this is completely OPTIONAL for simple analysis.
        
        When to call:
        - ✅ Before generate_llm_analysis_summary() (comprehensive analysis)
        - ✅ Before query_transactions() (user queries)
        - ❌ NOT needed for generate_simple_analysis() (transaction-only)
        
        What happens:
        1. Indexes transactions to FAISS (semantic embeddings)
        2. Indexes transactions to Neo4j (spending patterns graph)
        3. Enables rich analysis using embeddings + graph data
        
        This is called explicitly because:
        - Different use cases may not need both databases
        - Caller controls performance trade-offs
        - Simple analysis can run without this overhead
        
        Args:
            batch: TransactionBatch to load
            
        Returns:
            Dictionary with loading results and analysis summary
            
        See Also:
            - ANALYSIS_MODES_ARCHITECTURE.md (design documentation)
            - generate_simple_analysis (no database loading needed)
            - generate_llm_analysis_summary (requires this to be called first)
        """
        self.logger.debug(f"Loading {len(batch.transactions)} to FAISS (embeddings) + Neo4j (graph) for comprehensive analysis")
        
        try:
            results = self.transaction_loader.load_batch_to_databases(batch)
            
            self.logger.debug(f"✓ Database loading complete:")
            self.logger.debug(f"  - Total indexed: {results['indexed_count']}")
            self.logger.debug(f"  - FAISS indexed: {results['faiss_indexed']}")
            self.logger.debug(f"  - Neo4j indexed: {results['neo4j_indexed']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load transactions to databases: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_transactions': len(batch.transactions)
            }
    
    def query_transactions(self, query: str, limit: int = 10) -> dict:
        """
        Execute user query on loaded transactions.
        
        Args:
            query: Natural language query
            limit: Maximum results to return
            
        Returns:
            Query results with semantic and graph results
        """
        self.logger.debug(f"Processing user query: {query}")
        return self.query_engine.query_transactions(query, limit=limit)
    
    def generate_llm_analysis_summary(self, batch: TransactionBatch, user_id: str = "default_user") -> dict:
        """
        Generate comprehensive LLM-powered analysis using embeddings and graph data.
        
        ⚠️  IMPORTANT: Design Intent - Database Loading is Required
        ────────────────────────────────────────────────────────────
        This method REQUIRES that load_to_databases() was called BEFORE this.
        The comprehensive analysis queries FAISS and Neo4j for rich insights.
        
        Call sequence:
            1. extract_with_confidence() - extract transactions
            2. load_to_databases() - index to FAISS + Neo4j ⭐ REQUIRED
            3. generate_llm_analysis_summary() - generate comprehensive analysis
        
        Combines:
        - Transaction embeddings from FAISS (semantic understanding)
        - Graph relationships from Neo4j (spending patterns)
        - Statistical analysis
        - LLM insights for actionable recommendations
        
        Args:
            batch: TransactionBatch with analyzed transactions
            user_id: User identifier for graph queries
            
        Returns:
            Comprehensive analysis summary with key findings, spending habits,
            recommendations, and risk alerts
            
        See Also:
            - ANALYSIS_MODES_ARCHITECTURE.md (design documentation)
            - generate_simple_analysis (transaction-only, no DB needed)
        """
        self.logger.debug(f"Generating comprehensive LLM-powered analysis (with embeddings + graph)")
        
        return self.analysis_service.generate_analysis_summary(batch, user_id)
    
    def generate_simple_analysis(self, batch: TransactionBatch) -> dict:
        """
        Generate simple LLM-powered analysis using ONLY transaction data.
        
        ✅ IMPORTANT: Design Intent - No Database Loading Required
        ──────────────────────────────────────────────────────────
        This method is INTENTIONALLY lightweight and does NOT use FAISS or Neo4j.
        By design, simple analysis uses only transaction data for fast insights.
        
        Use this when:
        - ✅ Speed is important (fast: 2-9 seconds)
        - ✅ Transaction data alone provides sufficient insights
        - ✅ Embeddings/graph context not needed
        - ✅ Databases not set up (faster startup)
        
        Fast analysis mode that uses only:
        - Transaction data (amounts, dates, descriptions)
        - Statistical analysis
        - LLM insights (no embeddings or graph context)
        
        Ideal for quick analysis without overhead of vector/graph lookups.
        Embeddings + graph context reserved for user queries + comprehensive analysis.
        
        Call sequence:
            1. extract_with_confidence() - extract transactions
            2. [Optional] boost_confidence_with_llm() - improve low-confidence
            3. generate_simple_analysis() - fast insights ⭐ No DB required
        
        Args:
            batch: TransactionBatch with analyzed transactions
            
        Returns:
            Simple analysis summary with transaction data only
            
        See Also:
            - ANALYSIS_MODES_ARCHITECTURE.md (design documentation)
            - generate_llm_analysis_summary (comprehensive, requires load_to_databases)
        """
        self.logger.debug(f"Generating simple LLM-powered analysis (transaction data only, no DB)")
        
        return self.analysis_service.generate_simple_analysis(batch)


def main():
    """Test unified extraction on sample files."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    extractor = UnifiedExtractor()
    
    test_files = [('icici.pdf', 'ICICI Bank'), ('gpay.pdf', 'Google Pay')]
    
    for file_name, label in test_files:
        file_path = Path(file_name)
        if not file_path.exists():
            print(f"⚠ {label} ({file_name}) not found, skipping\n")
            continue
        
        print(f"\n{'='*90}")
        print(f"UNIFIED EXTRACTION: {label} ({file_name})")
        print(f"{'='*90}\n")
        
        try:
            batch = extractor.extract_with_confidence(file_path)
            
            print(f"📊 EXTRACTION SUMMARY")
            print(f"   Source File: {batch.source_file}")
            print(f"   Method: {batch.extraction_method.value}")
            print(f"   Total Transactions: {len(batch.transactions)}")
            print(f"   Average Confidence: {batch.total_confidence:.2f}")
            print(f"   High Confidence (>=0.8): {batch.high_confidence_count}")
            print(f"   Medium Confidence (0.5-0.8): {batch.medium_confidence_count}")
            print(f"   Low Confidence (<0.5): {batch.low_confidence_count}\n")
            
            # Try LLM confidence boosting for low-confidence transactions
            low_conf_count = sum(1 for t in batch.transactions if t.confidence_score < 0.80)
            if low_conf_count > 0 and extractor.llm_service:
                print(f"🚀 ATTEMPTING LLM CONFIDENCE BOOST")
                print(f"   Identified {low_conf_count} transactions below 0.80 confidence")
                print(f"   Using LLM verification to boost confidence...\n")
                batch = extractor.boost_confidence_with_llm(batch, confidence_threshold=0.80)
                
                print(f"📊 AFTER LLM BOOST")
                print(f"   Average Confidence: {batch.total_confidence:.2f}")
                print(f"   High Confidence (>=0.8): {batch.high_confidence_count}")
                print(f"   Medium Confidence (0.5-0.8): {batch.medium_confidence_count}")
                print(f"   Low Confidence (<0.5): {batch.low_confidence_count}\n")
            
            print(f"📝 TRANSACTIONS (showing first 5)")
            print(f"{'─'*90}")
            
            for i, trans in enumerate(batch.transactions[:5], 1):
                confidence_icon = "✓" if trans.is_high_confidence() else "⚠" if trans.is_medium_confidence() else "✗"
                llm_boost_note = " [LLM verified]" if trans.metadata and trans.metadata.get('llm_verified') else ""
                print(f"{i:2}. {confidence_icon} {trans.date} | {trans.type.value.upper():6} | "
                      f"{trans.currency}{float(trans.amount):>10.2f} | {trans.description[:40]:40} | "
                      f"Confidence: {trans.confidence_score:.2f}{llm_boost_note}")
            
            if len(batch.transactions) > 5:
                print(f"\n... and {len(batch.transactions) - 5} more transactions\n")
            
            # Print statistics by confidence level
            print(f"\n📈 CONFIDENCE BREAKDOWN")
            print(f"   High:   {batch.high_confidence_count:3} ({batch.high_confidence_count/len(batch.transactions)*100:5.1f}%)")
            print(f"   Medium: {batch.medium_confidence_count:3} ({batch.medium_confidence_count/len(batch.transactions)*100:5.1f}%)")
            print(f"   Low:    {batch.low_confidence_count:3} ({batch.low_confidence_count/len(batch.transactions)*100:5.1f}%)")
            
            # Print sample transaction as JSON
            if batch.transactions:
                print(f"\n💾 SAMPLE TRANSACTION (JSON format)")
                import json
                sample = batch.transactions[0].to_json_dict()
                print(json.dumps(sample, indent=2))
            
            # Load to Vector and Graph Databases
            print(f"\n{'='*90}")
            print(f"📦 LOADING TO DATABASES")
            print(f"{'='*90}\n")
            
            db_results = extractor.load_to_databases(batch)
            
            if db_results.get('status') == 'success':
                print(f"✓ Successfully loaded {db_results['indexed_count']} transactions to databases")
                print(f"  - FAISS (Vector DB): {db_results['faiss_indexed']}")
                print(f"  - Neo4j (Graph DB): {db_results['neo4j_indexed']}\n")
                
                # Display analysis summary
                analysis = db_results.get('analysis_summary', {})
                print(f"📊 ANALYSIS SUMMARY")
                print(f"   Total Debit: ${analysis.get('total_debit', 0):,.2f}")
                print(f"   Total Credit: ${analysis.get('total_credit', 0):,.2f}")
                print(f"   Net Balance: ${analysis.get('net_balance', 0):,.2f}")
                print(f"   Average Transaction: ${analysis.get('average_transaction', 0):,.2f}\n")
                
                # Top categories
                print(f"🏷️  TOP SPENDING CATEGORIES")
                for cat in analysis.get('top_categories', [])[:3]:
                    print(f"   {cat['category']:20} ${cat['total']:>10,.2f} ({cat['count']} transactions)")
                print()
                
                # High value transactions
                print(f"💰 HIGH VALUE TRANSACTIONS")
                for trans in analysis.get('high_value_transactions', [])[:3]:
                    print(f"   {trans['date']} | {trans['description'][:35]:35} | ${trans['amount']:>10,.2f}")
                print()
                
                # Generate LLM-powered analysis with embeddings and graph data
                print(f"{'='*90}")
                print(f"🤖 LLM-POWERED ANALYSIS (using embeddings + graph data)")
                print(f"{'='*90}\n")
                
                llm_analysis = extractor.generate_llm_analysis_summary(batch, user_id="default_user")
                
                print(f"📋 EXECUTIVE SUMMARY")
                print(f"   {llm_analysis.get('executive_summary', 'Unable to generate summary')}\n")
                
                print(f"🔍 KEY FINDINGS")
                for i, finding in enumerate(llm_analysis.get('key_findings', [])[:3], 1):
                    print(f"   {i}. {finding}")
                print()
                
                print(f"💡 SPENDING HABITS")
                for i, habit in enumerate(llm_analysis.get('spending_habits', [])[:3], 1):
                    print(f"   {i}. {habit}")
                print()
                
                print(f"💪 RECOMMENDATIONS")
                for i, rec in enumerate(llm_analysis.get('recommendations', [])[:3], 1):
                    print(f"   {i}. {rec}")
                print()
                
                if llm_analysis.get('risk_alerts'):
                    print(f"⚠️  RISK ALERTS")
                    for alert in llm_analysis.get('risk_alerts', []):
                        print(f"   ⚠ {alert}")
                    print()
                
                # Display semantic clusters (from embeddings)
                print(f"📊 SEMANTIC TRANSACTION CLUSTERS (from embeddings)")
                for cluster in llm_analysis.get('semantic_clusters', [])[:5]:
                    print(f"   • {cluster['name']}: ${cluster['total']:,.2f} ({cluster['count']} txns, "
                          f"{cluster['average']:.2f} avg)")
                print()
                
                # Display spending patterns (from graph)
                print(f"📈 SPENDING PATTERNS (from graph DB)")
                for pattern in llm_analysis.get('spending_patterns', [])[:5]:
                    print(f"   • {pattern['category']}: ${pattern['total_amount']:,.2f} "
                          f"({pattern['transaction_count']} transactions)")
                print()
            else:
                print(f"⚠ Database loading failed: {db_results.get('error', 'Unknown error')}\n")
            
            # Example user queries
            print(f"{'='*90}")
            print(f"🔍 EXAMPLE USER QUERIES")
            print(f"{'='*90}\n")
            
            sample_queries = [
                "How much did I spend on UPI payments?",
                "What are my top spending categories?"
            ]
            
            for user_query in sample_queries:
                print(f"Query: '{user_query}'")
                
                query_results = extractor.query_transactions(user_query, limit=5)
                
                if query_results.get('error'):
                    print(f"  ⚠ Query failed: {query_results['error']}")
                else:
                    print(f"  Results: {query_results.get('total_results', 0)} matches found")
                    
                    if query_results.get('insights'):
                        print(f"  Insights:")
                        for insight in query_results['insights'][:2]:
                            print(f"    • {insight}")
                
                print()
        
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}\n")


if __name__ == '__main__':
    main()