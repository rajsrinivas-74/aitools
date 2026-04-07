"""
LLM Service for embedding generation and insight generation.
Uses OpenAI for embeddings and text generation.
Integrates Financial Analysis and Spend Intelligence system prompt.
"""

import logging
import json
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from backend.models import ProcessedTransaction
from backend.services.financial_analysis_prompt import (
    FINANCIAL_ANALYSIS_SYSTEM_PROMPT,
    build_financial_analysis_prompt,
    validate_financial_analysis_output,
    extract_insights_from_analysis
)
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Custom exception for LLM operations."""
    pass


class LLMService:
    """OpenAI-based LLM service for embeddings and text generation."""
    
    def __init__(self):
        """Initialize LLM service."""
        if OpenAI is None:
            raise LLMServiceError("OpenAI not installed")
        
        if not settings.OPENAI_API_KEY:
            raise LLMServiceError("OPENAI_API_KEY not set")
        
        self.logger = logging.getLogger(__name__)
        
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        except TypeError as e:
            # Handle httpx compatibility issues with older OpenAI versions
            if "proxies" in str(e):
                self.logger.warning(f"OpenAI initialization failed due to httpx compatibility: {str(e)}")
                self.logger.info("Attempting alternative initialization...")
                import httpx
                http_client = httpx.Client()
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY, http_client=http_client)
            else:
                raise LLMServiceError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        self.logger.debug(f"Generating embedding for text: {text[:50]}...")
        try:
            response = self.client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise LLMServiceError(f"Failed to generate embedding: {str(e)}")
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        for text in texts:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Error embedding text: {str(e)}")
                embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
        
        return embeddings
    
    def generate_insights(self, context: str, query: str) -> str:
        """
        Generate insights using LLM with provided context.
        Uses Financial Analysis and Spend Intelligence system prompt.
        CRITICAL: Uses only retrieved context to avoid hallucination.
        
        Args:
            context: Retrieved context from RAG
            query: User query/analysis request
            
        Returns:
            Generated insights (JSON or text based on query)
        """
        try:
            prompt = build_financial_analysis_prompt(context, query)
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": FINANCIAL_ANALYSIS_SYSTEM_PROMPT
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Very low temperature for factual output
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            raise LLMServiceError(f"Failed to generate insights: {str(e)}")
    
    def detect_ambiguous_descriptions(self, descriptions: List[str]) -> List[Dict]:
        """
        Use LLM to detect ambiguous transaction descriptions.
        
        Args:
            descriptions: List of transaction descriptions
            
        Returns:
            List of ambiguities with explanations
        """
        try:
            descriptions_text = "\n".join([f"- {desc}" for desc in descriptions[:10]])
            
            prompt = f"""Review these transaction descriptions and identify any that are ambiguous or lack clarity:

{descriptions_text}

For each ambiguous description, explain why it's unclear."""
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return [{"analysis": response.choices[0].message.content}]
            
        except Exception as e:
            self.logger.warning(f"Error detecting ambiguities: {str(e)}")
            return []
    
    def generate_financial_analysis_json(self, transactions_context: str) -> Dict:
        """
        Generate structured financial analysis in JSON format.
        Enforces compliance with Financial Analysis and Spend Intelligence schema.
        
        Args:
            transactions_context: Formatted transaction data and context
            
        Returns:
            Dictionary with financial analysis following strict JSON schema
        """
        try:
            prompt = build_financial_analysis_prompt(transactions_context)
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": FINANCIAL_ANALYSIS_SYSTEM_PROMPT
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Extremely low temperature for precise JSON
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            analysis_output = json.loads(json_str)
            
            # Validate output schema
            is_valid, errors = validate_financial_analysis_output(analysis_output)
            if not is_valid:
                self.logger.warning(f"Financial analysis output validation errors: {errors}")
                # Still return the output, but flag issues
                analysis_output["schema_validation_errors"] = errors
            
            return analysis_output
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing financial analysis JSON: {str(e)}")
            raise LLMServiceError(f"Failed to parse analysis output as JSON: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating financial analysis: {str(e)}")
            raise LLMServiceError(f"Failed to generate financial analysis: {str(e)}")