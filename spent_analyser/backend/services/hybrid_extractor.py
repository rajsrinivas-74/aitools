"""
Hybrid PDF Transaction Extractor.
Tries multiple extraction methods and returns transactions with confidence scores.
Priority: Tabula → LLM → Regex Pattern Matching
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from decimal import Decimal
import pandas as pd
import re

from backend.models.transaction import Transaction, TransactionBatch, ExtractionMethod, TransactionType
from config.settings import Settings

logger = logging.getLogger(__name__)


class HybridTransactionExtractor:
    """
    Hybrid extractor that combines multiple PDF extraction strategies with confidence scoring.
    """
    
    def __init__(self):
        """Initialize hybrid extractor."""
        self.logger = logging.getLogger(__name__)
    
    def extract(self, file_path: Path) -> Optional[TransactionBatch]:
        """
        Extract transactions using hybrid approach.
        
        Tries in order:
        1. Tabula (for structured tables)
        2. LLM/Claude (for any format)
        3. Regex patterns (for simple formats)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            TransactionBatch with extracted transactions and confidence scores
        """
        file_path = Path(file_path)
        self.logger.info(f"Starting hybrid extraction for: {file_path.name}")
        
        # Try Tabula first
        batch = self._try_tabula(file_path)
        if batch and len(batch.transactions) > 0:
            self.logger.info(f"Tabula extracted {len(batch.transactions)} transactions")
            return batch
        
        # Try LLM extraction
        batch = self._try_llm(file_path)
        if batch and len(batch.transactions) > 0:
            self.logger.info(f"LLM extracted {len(batch.transactions)} transactions")
            return batch
        
        # Fallback to regex
        batch = self._try_regex(file_path)
        if batch and len(batch.transactions) > 0:
            self.logger.info(f"Regex extracted {len(batch.transactions)} transactions")
            return batch
        
        self.logger.warning(f"No transactions extracted from {file_path.name}")
        return None
    
    def _try_tabula(self, file_path: Path) -> Optional[TransactionBatch]:
        """
        Try extracting tables using Tabula.
        
        Tabula is excellent at:
        - Bank statements with structured tables
        - Invoices with tabular data
        - Any PDF with clear column structure
        """
        try:
            import tabula
        except ImportError:
            self.logger.debug("Tabula not installed")
            return None
        
        try:
            self.logger.debug(f"Attempting Tabula extraction on {file_path.name}")
            
            # Extract all tables from PDF
            tables = tabula.read_pdf(
                str(file_path),
                pages='all',
                multiple_tables=True,
                pandas_options={'header': 0}
            )
            
            if not tables:
                self.logger.debug("Tabula found no tables")
                return None
            
            transactions = []
            
            # Process each table
            for table_idx, df in enumerate(tables):
                self.logger.debug(f"Processing Tabula table {table_idx + 1}: {df.shape}")
                
                # Try to detect columns automatically
                extracted = self._process_dataframe(
                    df,
                    extraction_method=ExtractionMethod.TABULA,
                    source_file=file_path.name
                )
                transactions.extend(extracted)
            
            if transactions:
                batch = TransactionBatch(
                    transactions=transactions,
                    source_file=file_path.name,
                    extraction_method=ExtractionMethod.TABULA
                )
                return batch
            
        except Exception as e:
            self.logger.debug(f"Tabula extraction failed: {str(e)}")
        
        return None
    
    def _try_llm(self, file_path: Path) -> Optional[TransactionBatch]:
        """
        Try extracting using Claude API (LLM).
        
        LLM is excellent at:
        - Any PDF format
        - Complex layouts
        - Handling OCR text
        - Understanding context
        
        Returns highest confidence scores when text quality is good.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            self.logger.debug("PyMuPDF not installed for LLM extraction")
            return None
        
        # Check if OPENAI_API_KEY or Claude API key is available
        if not Settings.OPENAI_API_KEY:
            self.logger.debug("LLM extraction skipped: No API key configured")
            return None
        
        try:
            from openai import OpenAI
            
            self.logger.debug(f"Attempting LLM extraction on {file_path.name}")
            
            # Extract text from PDF
            doc = fitz.open(file_path)
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text() + "\n"
            doc.close()
            
            if not pdf_text or len(pdf_text) < 100:
                self.logger.debug("PDF text too short for LLM extraction")
                return None
            
            # Send to Claude for extraction
            client = OpenAI(api_key=Settings.OPENAI_API_KEY)
            
            prompt = f"""Extract all financial transactions from this bank statement or financial document.
Return a JSON array with transactions in this format:
[
  {{
    "date": "YYYY-MM-DD",
    "description": "transaction description",
    "amount": "numeric amount only",
    "type": "credit or debit",
    "currency": "currency symbol"
  }}
]

Only return valid JSON, no other text.

Document text:
{pdf_text[:5000]}"""  # Limit to 5000 chars to avoid token limits
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            transactions_data = json.loads(response_text)
            
            transactions = []
            for trans_data in transactions_data:
                try:
                    trans = Transaction(
                        date=trans_data.get('date', ''),
                        description=trans_data.get('description', ''),
                        amount=Decimal(str(trans_data.get('amount', 0))),
                        type=TransactionType(trans_data.get('type', 'unknown').lower()),
                        currency=trans_data.get('currency', Settings.DEFAULT_CURRENCY_SYMBOL),
                        confidence_score=0.85,  # LLM extraction confidence
                        extraction_method=ExtractionMethod.LLM,
                        source_file=file_path.name
                    )
                    transactions.append(trans)
                except Exception as e:
                    self.logger.debug(f"Failed to parse LLM transaction: {str(e)}")
            
            if transactions:
                batch = TransactionBatch(
                    transactions=transactions,
                    source_file=file_path.name,
                    extraction_method=ExtractionMethod.LLM
                )
                return batch
            
        except Exception as e:
            self.logger.debug(f"LLM extraction failed: {str(e)}")
        
        return None
    
    def _try_regex(self, file_path: Path) -> Optional[TransactionBatch]:
        """
        Fallback: Try regex-based extraction.
        
        Universal regex patterns for:
        - Dates (DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, DD MMM YYYY)
        - Amounts (currency symbol + number)
        - Common transaction patterns
        
        Returns lower confidence scores when using regex.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            self.logger.debug("PyMuPDF not installed for regex extraction")
            return None
        
        try:
            self.logger.debug(f"Attempting regex extraction on {file_path.name}")
            
            # Extract raw text
            doc = fitz.open(file_path)
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text() + "\n"
            doc.close()
            
            transactions = []
            
            # Generic transaction pattern:
            # Date | Description | Amount
            # Regex patterns for common formats
            date_patterns = [
                r'(\d{1,2}\.\d{2}\.\d{4})',  # DD.MM.YYYY (ICICI style)
                r'(\d{1,2}/\d{2}/\d{4})',    # DD/MM/YYYY
                r'(\d{4}-\d{2}-\d{2})',      # YYYY-MM-DD
                r'(\d{1,2}\s+\w+,?\s+\d{4})',  # DD MMM YYYY (Google Pay style)
            ]
            
            amount_pattern = r'[₹$€£¥]?\s*[\d,]+\.?\d*'
            
            lines = pdf_text.split('\n')
            
            for i, line in enumerate(lines):
                # Look for date in line
                date_match = None
                for pattern in date_patterns:
                    date_match = re.search(pattern, line)
                    if date_match:
                        break
                
                if date_match:
                    date_str = date_match.group(1)
                    
                    # Look for amount in same or next lines
                    amount_match = re.search(amount_pattern, line)
                    
                    if amount_match:
                        amount_str = amount_match.group(0)
                        amount_clean = re.sub(r'[^\d.]', '', amount_str)
                        
                        # Extract description (remaining text in line or use next line)
                        description = re.sub(
                            fr'{re.escape(date_str)}|{re.escape(amount_str)}',
                            '',
                            line
                        ).strip()
                        
                        if amount_clean and description:
                            try:
                                trans = Transaction(
                                    date=date_str,
                                    description=description,
                                    amount=Decimal(amount_clean),
                                    type=TransactionType.DEBIT,  # Default
                                    confidence_score=0.6,  # Lower confidence for regex
                                    extraction_method=ExtractionMethod.REGEX,
                                    source_file=file_path.name
                                )
                                transactions.append(trans)
                            except Exception as e:
                                self.logger.debug(f"Failed to create transaction from regex match: {str(e)}")
            
            if transactions:
                batch = TransactionBatch(
                    transactions=transactions,
                    source_file=file_path.name,
                    extraction_method=ExtractionMethod.REGEX
                )
                return batch
            
        except Exception as e:
            self.logger.debug(f"Regex extraction failed: {str(e)}")
        
        return None
    
    def _process_dataframe(
        self,
        df: pd.DataFrame,
        extraction_method: ExtractionMethod,
        source_file: str
    ) -> List[Transaction]:
        """
        Process pandas DataFrame and extract transactions.
        
        Auto-detects column names and types.
        """
        transactions = []
        
        try:
            # Auto-detect columns
            columns = df.columns.tolist()
            
            # Look for standard column names (case-insensitive)
            date_col = self._find_column(columns, ['date', 'transaction date', 'posted date'])
            desc_col = self._find_column(columns, ['description', 'details', 'transaction', 'narration'])
            amount_col = self._find_column(columns, ['amount', 'debit', 'credit', 'withdrawal', 'deposit'])
            
            if not (date_col and desc_col and amount_col):
                self.logger.debug("Could not auto-detect required columns")
                return []
            
            # Extract rows
            for _, row in df.iterrows():
                try:
                    date_str = str(row[date_col]).strip()
                    description = str(row[desc_col]).strip()
                    amount_str = str(row[amount_col]).strip()
                    
                    # Clean amount
                    amount_clean = re.sub(r'[^\d.]', '', amount_str)
                    
                    if amount_clean and description and date_str:
                        trans = Transaction(
                            date=date_str,
                            description=description,
                            amount=Decimal(amount_clean),
                            type=TransactionType.DEBIT,  # Default
                            confidence_score=0.9 if extraction_method == ExtractionMethod.TABULA else 0.75,
                            extraction_method=extraction_method,
                            source_file=source_file
                        )
                        transactions.append(trans)
                except Exception as e:
                    self.logger.debug(f"Failed to process DataFrame row: {str(e)}")
            
        except Exception as e:
            self.logger.debug(f"DataFrame processing failed: {str(e)}")
        
        return transactions
    
    @staticmethod
    def _find_column(columns: list, keywords: list) -> Optional[str]:
        """Find column matching any keyword (case-insensitive)."""
        columns_lower = [c.lower() for c in columns]
        for keyword in keywords:
            for i, col in enumerate(columns_lower):
                if keyword.lower() in col:
                    return columns[i]
        return None