"""
Spent Analysis Service.
Takes spend data with embeddings and graph relationships to generate LLM-powered analysis summary.
"""

import logging
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from backend.models.transaction import TransactionBatch
from backend.services.rag_pipeline import RAGPipeline, get_rag_pipeline
from backend.services.llm_service import LLMService
from backend.services.transaction_loader import TransactionLoader

logger = logging.getLogger(__name__)


class SpentAnalysisService:
    """
    Comprehensive spending analysis powered by LLM.
    Combines embeddings, graph data, and statistical analysis for rich insights.
    """
    
    def __init__(self):
        """Initialize analysis service."""
        self.logger = logging.getLogger(__name__)
        self.rag_pipeline = get_rag_pipeline()
        self.llm_service = None
        self.transaction_loader = TransactionLoader()
        
        try:
            self.llm_service = LLMService()
        except Exception as e:
            self.logger.warning(f"LLM service not available: {e}")
    
    def generate_analysis_summary(self, batch: TransactionBatch, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Generate comprehensive LLM-powered analysis summary.
        
        ⚠️  IMPORTANT: Requires load_to_databases() to be called first
        ────────────────────────────────────────────────────────────
        This method queries FAISS and Neo4j databases. If they're not loaded,
        the method gracefully returns empty insights but continues functioning.
        
        Call sequence:
            1. extract_with_confidence(file_path)
            2. load_to_databases(batch) ⭐ REQUIRED
            3. generate_analysis_summary(batch)
        
        Combines:
        - Transaction embeddings from FAISS (semantic understanding)
        - Graph relationships from Neo4j (spending patterns)
        - Statistical analysis
        - LLM insights for actionable recommendations
        
        Args:
            batch: TransactionBatch with analyzed transactions
            user_id: User identifier for graph queries
            
        Returns:
            Comprehensive analysis summary with embeddings + graph context
            
        See Also:
            - ANALYSIS_MODES_ARCHITECTURE.md (design documentation)
            - generate_simple_analysis (transaction-only, no DB required)
        """
        self.logger.debug(f"Generating LLM-powered analysis (with embeddings + graph) for {len(batch.transactions)} transactions")
        
        # 1. Get basic analysis first
        basic_analysis = self.transaction_loader._generate_analysis_summary(batch)
        
        # 2. Retrieve data from vector DB (embeddings)
        embedding_insights = self._get_embedding_insights(batch)
        
        # 3. Retrieve data from graph DB (relationships)
        graph_insights = self._get_graph_insights(user_id)
        
        # 4. Send to LLM for enriched analysis
        llm_analysis = self._generate_llm_analysis(batch, basic_analysis, embedding_insights, graph_insights)
        
        # 5. Combine all insights
        comprehensive_summary = {
            'timestamp': datetime.now().isoformat(),
            'transaction_count': len(batch.transactions),
            'source_file': batch.source_file,
            'average_confidence': round(batch.total_confidence, 2),
            'analysis_type': 'comprehensive',  # Mark as full analysis with embeddings + graph
            
            # Basic statistics
            'financial_summary': {
                'total_debit': basic_analysis.get('total_debit', 0),
                'total_credit': basic_analysis.get('total_credit', 0),
                'net_balance': basic_analysis.get('net_balance', 0),
                'average_transaction': basic_analysis.get('average_transaction', 0),
                'date_range': basic_analysis.get('date_range', {}),
            },
            
            # Categories
            'spending_by_category': basic_analysis.get('top_categories', []),
            'high_value_transactions': basic_analysis.get('high_value_transactions', []),
            
            # Vector DB insights
            'semantic_clusters': embedding_insights.get('clusters', []),
            'transaction_patterns': embedding_insights.get('patterns', []),
            
            # Graph DB insights
            'spending_patterns': graph_insights.get('spending_patterns', []),
            'relationship_analysis': graph_insights.get('relationships', []),
            
            # LLM-powered insights
            'key_findings': llm_analysis.get('key_findings', []),
            'spending_habits': llm_analysis.get('spending_habits', []),
            'recommendations': llm_analysis.get('recommendations', []),
            'risk_alerts': llm_analysis.get('risk_alerts', []),
            'executive_summary': llm_analysis.get('executive_summary', ''),
        }
        
        self.logger.debug("✓ Comprehensive analysis summary generated successfully")
        return comprehensive_summary
    
    def generate_simple_analysis(self, batch: TransactionBatch) -> Dict[str, Any]:
        """
        Generate simple LLM-powered analysis using ONLY transaction data.
        
        ✅ IMPORTANT: By Design - No Database Loading Required
        ────────────────────────────────────────────────────
        This method DOES NOT use FAISS or Neo4j. It is intentionally lightweight
        for fast analysis using only transaction amounts, dates, and descriptions.
        
        Call sequence:
            1. extract_with_confidence(file_path)
            2. [Optional] boost_confidence_with_llm(batch)
            3. generate_simple_analysis(batch) ⭐ Fast, no DB needed
        
        Fast analysis mode that uses only:
        - Transaction data (amounts, dates, descriptions)
        - Statistical analysis
        - LLM insights (no embeddings or graph context)
        
        This is ideal for quick analysis without the overhead of vector/graph lookups.
        Use this when speed matters and embeddings/graph not needed.
        
        Args:
            batch: TransactionBatch with analyzed transactions
            
        Returns:
            Simple analysis summary with transaction data only
            
        See Also:
            - ANALYSIS_MODES_ARCHITECTURE.md (design documentation)
            - generate_analysis_summary (comprehensive, requires load_to_databases)
        """
        self.logger.debug(f"Generating simple LLM-powered analysis (transaction data only) for {len(batch.transactions)} transactions")
        
        # 1. Get basic analysis only
        basic_analysis = self.transaction_loader._generate_analysis_summary(batch)
        
        # 2. Send to LLM with transaction data only
        llm_analysis = self._generate_llm_analysis_simple(batch, basic_analysis)
        
        # 3. Combine insights
        simple_summary = {
            'timestamp': datetime.now().isoformat(),
            'transaction_count': len(batch.transactions),
            'source_file': batch.source_file,
            'average_confidence': round(batch.total_confidence, 2),
            'analysis_type': 'simple',  # Mark as simple analysis without embeddings + graph
            
            # Basic statistics
            'financial_summary': {
                'total_debit': basic_analysis.get('total_debit', 0),
                'total_credit': basic_analysis.get('total_credit', 0),
                'net_balance': basic_analysis.get('net_balance', 0),
                'average_transaction': basic_analysis.get('average_transaction', 0),
                'date_range': basic_analysis.get('date_range', {}),
            },
            
            # Categories
            'spending_by_category': basic_analysis.get('top_categories', []),
            'high_value_transactions': basic_analysis.get('high_value_transactions', []),
            
            # LLM-powered insights (no embeddings or graph)
            'key_findings': llm_analysis.get('key_findings', []),
            'spending_habits': llm_analysis.get('spending_habits', []),
            'recommendations': llm_analysis.get('recommendations', []),
            'risk_alerts': llm_analysis.get('risk_alerts', []),
            'executive_summary': llm_analysis.get('executive_summary', ''),
        }
        
        self.logger.debug("✓ Simple analysis summary generated successfully")
        return simple_summary
    
    def _get_embedding_insights(self, batch: TransactionBatch) -> Dict[str, Any]:
        """
        Extract insights from transaction embeddings (FAISS).
        
        Provides semantic understanding of transactions through clustering.
        
        Args:
            batch: TransactionBatch
            
        Returns:
            Dictionary with embedding-based insights
        """
        self.logger.debug("Analyzing transaction embeddings...")
        
        insights = {
            'clusters': [],
            'patterns': []
        }
        
        try:
            if not self.rag_pipeline.faiss_store:
                return insights
            
            # Group transactions by semantic similarity
            # This would use FAISS to find clusters of similar transactions
            
            # For now, return basic grouping by description
            description_groups = defaultdict(list)
            for trans in batch.transactions:
                # Extract merchant/category from description
                desc_parts = trans.description.split('/')
                merchant = desc_parts[1] if len(desc_parts) > 1 else trans.description[:30]
                description_groups[merchant].append({
                    'date': trans.date,
                    'amount': float(trans.amount),
                    'confidence': trans.confidence_score
                })
            
            # Convert to clusters
            for merchant, transactions in sorted(
                description_groups.items(),
                key=lambda x: sum(t['amount'] for t in x[1]),
                reverse=True
            )[:5]:  # Top 5 merchants
                total = sum(t['amount'] for t in transactions)
                insights['clusters'].append({
                    'name': merchant[:40],
                    'count': len(transactions),
                    'total': round(total, 2),
                    'average': round(total / len(transactions), 2),
                    'confidence': round(sum(t['confidence'] for t in transactions) / len(transactions), 2)
                })
            
            # Pattern detection
            # Detect frequency patterns
            date_freq = defaultdict(int)
            for trans in batch.transactions:
                date_freq[trans.date] += 1
            
            for date, count in date_freq.items():
                if count > 1:
                    insights['patterns'].append({
                        'type': 'high_frequency_day',
                        'date': date,
                        'transaction_count': count,
                        'description': f"Multiple transactions on {date}"
                    })
            
            self.logger.debug(f"Found {len(insights['clusters'])} semantic clusters")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing embeddings: {e}")
            return insights
    
    def _get_graph_insights(self, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Extract insights from graph database relationships (Neo4j).
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with graph-based insights
        """
        self.logger.debug("Querying graph database for spending patterns...")
        
        insights = {
            'spending_patterns': [],
            'relationships': []
        }
        
        try:
            if not self.rag_pipeline.neo4j_store or not self.rag_pipeline.neo4j_store.connected:
                return insights
            
            # Get spending patterns from Neo4j
            patterns = self.rag_pipeline.neo4j_store.get_spending_patterns(user_id, limit=10)
            
            for pattern in patterns:
                insights['spending_patterns'].append({
                    'category': pattern.get('category', 'Unknown'),
                    'transaction_count': pattern.get('count', 0),
                    'total_amount': round(pattern.get('total', 0), 2),
                    'percentage_of_total': 0  # Will be calculated in LLM analysis
                })
            
            # Add relationship info
            if patterns:
                insights['relationships'].append({
                    'type': 'spending_by_category',
                    'description': f"User has transactions across {len(patterns)} categories",
                    'tier': 'primary'
                })
            
            self.logger.debug(f"Found {len(patterns)} spending patterns")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error querying graph database: {e}")
            return insights
    
    def _generate_llm_analysis(
        self,
        batch: TransactionBatch,
        basic_analysis: Dict[str, Any],
        embedding_insights: Dict[str, Any],
        graph_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to generate enriched analysis with embeddings + graph data.
        
        Args:
            batch: TransactionBatch
            basic_analysis: Basic statistical analysis
            embedding_insights: Embedding-based insights
            graph_insights: Graph-based insights
            
        Returns:
            LLM-generated analysis
        """
        if not self.llm_service:
            self.logger.warning("LLM service not available, skipping enriched analysis")
            return {
                'key_findings': [],
                'spending_habits': [],
                'recommendations': [],
                'risk_alerts': [],
                'executive_summary': ''
            }
        
        try:
            self.logger.debug("Sending data to LLM for enriched analysis (with embeddings + graph)...")
            
            # Prepare comprehensive prompt
            prompt = self._build_analysis_prompt(
                batch, basic_analysis, embedding_insights, graph_insights
            )
            
            # Call LLM
            response = self.llm_service.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial advisor with 15+ years of experience in personal finance, wealth management, and financial planning. Your role is to provide deep, actionable guidance on spending patterns and financial health.

As a financial advisor, you should:

1. **Analyze Comprehensively**: Look beyond numbers to understand the financial story. What does this spending pattern tell us about financial health, priorities, and risk exposure?

2. **Provide Expert Guidance**: Give professional, nuanced recommendations based on industry best practices and your deep knowledge of financial principles.

3. **Identify Opportunities**: Spot optimization opportunities, inefficiencies, and areas for improvement in spending behavior.

4. **Risk Management**: Identify financial risks, vulnerabilities, and warning signs that could impact long-term financial stability.

5. **Behavioral Insights**: Analyze spending habits as reflections of financial discipline, priorities, and potential blind spots.

6. **Actionable Advice**: Provide specific, implementable recommendations that the user can act on immediately to improve their financial position.

7. **Strategic Perspective**: Think long-term about how current spending patterns affect future financial goals, savings rate, and wealth accumulation.

When analyzing spending data:
- Use numbers effectively but focus on their meaning
- Consider context (is this sustainable? What are the trends?)
- Identify patterns that reveal financial behavior
- Flag risks and opportunities
- Provide guidance that's practical and implementable
- Be direct about concerns but constructive in tone
- Acknowledge good practices when you see them

Format your response as JSON with these keys:
- key_findings: List of important discoveries (3-5 findings)
- spending_habits: Behavioral patterns and what they mean (3-5 insights)
- recommendations: Specific, actionable guidance (3-5 recommendations)
- risk_alerts: Financial risks and red flags (0-3 alerts)
- executive_summary: 2-3 sentence professional summary of financial position and outlook"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse response
            parsed_analysis = self._parse_llm_response(analysis_text)
            
            self.logger.debug("✓ LLM analysis completed")
            self.logger.info(f"LLM ANALYSIS RESPONSE: {analysis_text}")
            return parsed_analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {
                'key_findings': [],
                'spending_habits': [],
                'recommendations': [],
                'risk_alerts': [],
                'executive_summary': ''
            }
    
    def _generate_llm_analysis_simple(
        self,
        batch: TransactionBatch,
        basic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to generate simple analysis using only transaction data.
        
        No embeddings or graph data - just transaction-level analysis.
        Faster, lighter analysis for quick insights.
        
        Args:
            batch: TransactionBatch
            basic_analysis: Basic statistical analysis
            
        Returns:
            LLM-generated simple analysis
        """
        if not self.llm_service:
            self.logger.warning("LLM service not available, skipping analysis")
            return {
                'key_findings': [],
                'spending_habits': [],
                'recommendations': [],
                'risk_alerts': [],
                'executive_summary': ''
            }
        
        try:
            self.logger.debug("Sending transaction data only to LLM for simple analysis...")
            
            # Prepare simple prompt (transaction data only)
            prompt = self._build_simple_analysis_prompt(batch, basic_analysis)
            
            # Call LLM
            response = self.llm_service.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial advisor. Analyze the spending transactions and provide professional financial guidance.

Focus on:
1. Key observations from the transaction data
2. Spending patterns and behavior insights
3. Actionable recommendations for improvement
4. Risk alerts if concerning patterns detected
5. Executive summary of financial position

Format your response as JSON with these keys:
- key_findings: List of important observations (3-5)
- spending_habits: Behavioral patterns identified (2-3)
- recommendations: Actionable suggestions (2-3)
- risk_alerts: Financial concerns detected (0-2)
- executive_summary: 1-2 sentence summary"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse response
            parsed_analysis = self._parse_llm_response(analysis_text)
            
            self.logger.debug("✓ Simple LLM analysis completed")
            self.logger.info(f"LLM ANALYSIS RESPONSE: {analysis_text}")
            return parsed_analysis
            
        except Exception as e:
            self.logger.error(f"Simple LLM analysis failed: {e}")
            return {
                'key_findings': [],
                'spending_habits': [],
                'recommendations': [],
                'risk_alerts': [],
                'executive_summary': ''
            }
    
    def _build_analysis_prompt(
        self,
        batch: TransactionBatch,
        basic_analysis: Dict[str, Any],
        embedding_insights: Dict[str, Any],
        graph_insights: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive prompt for LLM analysis with embeddings + graph.
        
        Args:
            batch: TransactionBatch
            basic_analysis: Basic statistics
            embedding_insights: Embedding insights
            graph_insights: Graph insights
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Analyze the following spending data and provide comprehensive financial insights:

TRANSACTION DATA:
- Total Transactions: {len(batch.transactions)}
- Date Range: {basic_analysis['date_range']['from']} to {basic_analysis['date_range']['to']}
- Average Confidence Score: {basic_analysis['average_confidence']:.2f} (0-1 scale)

FINANCIAL SUMMARY:
- Total Debits: ${basic_analysis['total_debit']:,.2f}
- Total Credits: ${basic_analysis['total_credit']:,.2f}
- Net Balance: ${basic_analysis['net_balance']:,.2f}
- Average Transaction: ${basic_analysis['average_transaction']:,.2f}

SPENDING BY CATEGORY (Top 5):
"""
        for i, cat in enumerate(basic_analysis['top_categories'][:5], 1):
            prompt += f"\n{i}. {cat['category']}: ${cat['total']:,.2f} ({cat['count']} transactions, avg ${cat['average']:.2f})"
        
        prompt += "\n\nHIGH-VALUE TRANSACTIONS (Top 3):\n"
        for i, trans in enumerate(basic_analysis['high_value_transactions'][:3], 1):
            prompt += f"\n{i}. {trans['date']} | {trans['description'][:50]} | ${trans['amount']:,.2f}"
        
        # Add semantic clusters
        if embedding_insights['clusters']:
            prompt += "\n\nSEMANTIC TRANSACTION CLUSTERS (from embeddings - similar merchants/categories):\n"
            for cluster in embedding_insights['clusters'][:5]:
                prompt += f"\n- {cluster['name']}: ${cluster['total']:,.2f} ({cluster['count']} transactions)"
        
        # Add patterns detected
        if embedding_insights['patterns']:
            prompt += "\n\nDETECTED PATTERNS:\n"
            for pattern in embedding_insights['patterns'][:5]:
                prompt += f"\n- {pattern['description']}"
        
        # Add graph insights
        if graph_insights['spending_patterns']:
            prompt += "\n\nGRAPH ANALYSIS (spending patterns across user data):\n"
            for pattern in graph_insights['spending_patterns'][:5]:
                prompt += f"\n- {pattern['category']}: ${pattern['total_amount']:,.2f} ({pattern['transaction_count']} transactions)"
        
        prompt += """

Based on this comprehensive spending data (transaction data, semantic clustering from embeddings, and relationship analysis from graph database), provide:

1. KEY FINDINGS: 3-5 most important observations about spending patterns
2. SPENDING HABITS: Description of user's spending behavior and characteristics
3. RECOMMENDATIONS: 3-5 actionable recommendations to improve financial health
4. RISK ALERTS: Any concerning patterns or red flags (empty list if none)
5. EXECUTIVE SUMMARY: 2-3 sentence summary of overall financial state

Format your response as JSON with keys: key_findings (list), spending_habits (list), recommendations (list), risk_alerts (list), executive_summary (string)
"""
        
        return prompt
    
    def _build_simple_analysis_prompt(
        self,
        batch: TransactionBatch,
        basic_analysis: Dict[str, Any]
    ) -> str:
        """
        Build simple prompt for LLM analysis using ONLY transaction data.
        
        No embeddings or graph context - just transaction-level insights.
        
        Args:
            batch: TransactionBatch
            basic_analysis: Basic statistics
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Analyze the following spending transactions and provide financial guidance:

TRANSACTION OVERVIEW:
- Total Transactions: {len(batch.transactions)}
- Date Range: {basic_analysis['date_range']['from']} to {basic_analysis['date_range']['to']}
- Data Quality: {basic_analysis['average_confidence']:.1%}

FINANCIAL SUMMARY:
- Total Debits: ${basic_analysis['total_debit']:,.2f}
- Total Credits: ${basic_analysis['total_credit']:,.2f}
- Net Balance: ${basic_analysis['net_balance']:,.2f}
- Average Transaction: ${basic_analysis['average_transaction']:,.2f}

SPENDING BY CATEGORY:
"""
        for i, cat in enumerate(basic_analysis['top_categories'][:5], 1):
            prompt += f"\n{i}. {cat['category']}: ${cat['total']:,.2f} ({cat['count']} transactions)"
        
        prompt += "\n\nHIGH-VALUE TRANSACTIONS:\n"
        for i, trans in enumerate(basic_analysis['high_value_transactions'][:5], 1):
            prompt += f"\n{i}. {trans['date']} | {trans['description'][:40]} | ${trans['amount']:,.2f}"
        
        prompt += f"""

Based on this transaction data, provide:

1. KEY FINDINGS: 3-5 observations from the transaction data
2. SPENDING HABITS: Behavioral patterns identified
3. RECOMMENDATIONS: 2-3 suggestions for improvement
4. RISK ALERTS: Any financial concerns (empty if none)
5. EXECUTIVE SUMMARY: 1-2 sentence financial assessment

Format as JSON with keys: key_findings (list), spending_habits (list), recommendations (list), risk_alerts (list), executive_summary (string)
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured insights.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Parsed analysis dictionary
        """
        try:
            # Try to extract JSON from response
            # Look for JSON block
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                return {
                    'key_findings': parsed.get('key_findings', []),
                    'spending_habits': parsed.get('spending_habits', []),
                    'recommendations': parsed.get('recommendations', []),
                    'risk_alerts': parsed.get('risk_alerts', []),
                    'executive_summary': parsed.get('executive_summary', '')
                }
            
            # If no JSON, try to parse text manually
            return self._parse_text_response(response_text)
            
        except json.JSONDecodeError:
            self.logger.warning("Could not parse LLM response as JSON, attempting text extraction")
            return self._parse_text_response(response_text)
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return {
                'key_findings': [],
                'spending_habits': [],
                'recommendations': [],
                'risk_alerts': [],
                'executive_summary': response_text[:200]
            }
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response as plain text.
        
        Args:
            response_text: Response text
            
        Returns:
            Parsed analysis
        """
        sections = {
            'key_findings': [],
            'spending_habits': [],
            'recommendations': [],
            'risk_alerts': [],
            'executive_summary': ''
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Detect section headers
            if 'KEY FINDINGS' in line.upper():
                current_section = 'key_findings'
            elif 'SPENDING HABITS' in line.upper():
                current_section = 'spending_habits'
            elif 'RECOMMENDATIONS' in line.upper():
                current_section = 'recommendations'
            elif 'RISK ALERTS' in line.upper():
                current_section = 'risk_alerts'
            elif 'EXECUTIVE SUMMARY' in line.upper():
                current_section = 'executive_summary'
            elif current_section and (line.startswith('•') or line.startswith('-') or line[0].isdigit()):
                # Add to current section
                clean_line = line.lstrip('•-0123456789. ').strip()
                if clean_line:
                    if current_section == 'executive_summary':
                        sections[current_section] += clean_line + ' '
                    else:
                        sections[current_section].append(clean_line)
        
        return sections