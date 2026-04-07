"""
Schema Detection and Normalization Layer (CRITICAL)
Handles heterogeneous financial data from CSV and PDF formats.
Dynamically detects schema and maps all inputs to canonical format.
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import re
from decimal import Decimal

import pandas as pd

from backend.models import CanonicalTransaction, SchemaMapping, ConfidenceLevel
from config.settings import settings

logger = logging.getLogger(__name__)


class SchemaDetectionError(Exception):
    """Custom exception for schema detection errors."""
    pass


class SchemaAdapter:
    """
    Intelligently detects and adapts heterogeneous financial data schemas.
    Maps all input variations to the canonical transaction model.
    """
    
    # Common patterns for various column types
    DATE_PATTERNS = {
        "date": [
            "^date$", "^transaction.?date$", "^trans.?date$", "^posting.?date$",
            "^value.?date$", "^when$", "^transaction.?date$"
        ],
        "debit_date": ["^debit.?date$", "^withdrawal.?date$"],
        "credit_date": ["^credit.?date$", "^deposit.?date$"]
    }
    
    AMOUNT_PATTERNS = {
        "amount": ["^amount$", "^value$", "^transaction.?amount$", "^trans.?amount$"],
        "debit": ["^debit$", "^withdrawal$", "^paid$", "^expenses?$"],
        "credit": ["^credit$", "^deposit$", "^income$", "^received$"],
        "balance": ["^balance$", "^account.?balance$", "^closing.?balance$"]
    }
    
    DESCRIPTION_PATTERNS = {
        "description": [
            "^description$", "^desc$", "^memo$", "^notes$", "^narration$",
            "^remarks$", "^transaction.?details?$", "^payee$", "^merchant$",
            "^reference$", "^transaction.?description$"
        ]
    }
    
    def __init__(self):
        """Initialize schema adapter."""
        self.logger = logging.getLogger(__name__)
        self.detected_schemas: Dict[str, SchemaMapping] = {}
    
    def detect_schema(self, df: pd.DataFrame, filename: str) -> SchemaMapping:
        """
        Detect schema from DataFrame by analyzing column names and sample data.
        
        Args:
            df: DataFrame to analyze
            filename: Source filename
            
        Returns:
            SchemaMapping with detected schema
        """
        if df.empty:
            raise SchemaDetectionError("DataFrame is empty")
        
        try:
            # Normalize column names
            df.columns = [str(col).lower().strip() for col in df.columns]
            
            # Detect column purposes
            column_mapping = self._detect_column_mapping(df)
            
            # Calculate confidence
            confidence_score = self._calculate_confidence_score(column_mapping, df)
            
            # Check for ambiguities
            ambiguities = self._detect_ambiguities(column_mapping, df)
            
            requires_llm = confidence_score < 0.7 or bool(ambiguities)
            
            mapping = SchemaMapping(
                file_name=filename,
                detected_schema={
                    "columns": list(df.columns),
                    "sample_rows": df.head(3).to_dict('records'),
                    "total_rows": len(df)
                },
                column_mapping=column_mapping,
                confidence_score=confidence_score,
                requires_llm=requires_llm,
                ambiguities=ambiguities
            )
            
            self.detected_schemas[filename] = mapping
            self.logger.info(f"Detected schema for {filename}: confidence={confidence_score:.2f}")
            
            return mapping
            
        except SchemaDetectionError:
            raise
        except Exception as e:
            self.logger.error(f"Error detecting schema: {str(e)}")
            raise SchemaDetectionError(f"Schema detection failed: {str(e)}")
    
    def normalize_to_canonical(
        self,
        df: pd.DataFrame,
        schema_mapping: SchemaMapping
    ) -> List[CanonicalTransaction]:
        """
        Normalize DataFrame rows to canonical transaction format.
        
        Args:
            df: Source DataFrame
            schema_mapping: Detected schema mapping
            
        Returns:
            List of CanonicalTransaction objects
        """
        transactions = []
        column_mapping = schema_mapping.column_mapping
        
        # Normalize column names
        df.columns = [str(col).lower().strip() for col in df.columns]
        
        for idx, row in df.iterrows():
            try:
                # Extract values using detected mapping
                date_str = self._extract_date(row, column_mapping)
                description = self._extract_description(row, column_mapping)
                amount, tx_type = self._extract_amount_and_type(row, column_mapping)
                balance = self._extract_balance(row, column_mapping)
                
                # Determine confidence level
                confidence = self._assess_row_confidence(
                    row, column_mapping, schema_mapping.confidence_score
                )
                
                # Create canonical transaction
                transaction = CanonicalTransaction(
                    date=date_str,
                    description=description,
                    amount=amount,
                    type=tx_type,
                    balance=balance,
                    source_file=schema_mapping.file_name,
                    confidence=confidence,
                    requires_review=confidence != ConfidenceLevel.HIGH,
                    review_reason=f"Low confidence in mapping" if confidence != ConfidenceLevel.HIGH else None,
                    parsed_metadata={
                        "row_index": idx,
                        "original_values": row.to_dict()
                    }
                )
                
                transactions.append(transaction)
                
            except Exception as e:
                self.logger.warning(f"Skipping row {idx} in {schema_mapping.file_name}: {str(e)}")
                continue
        
        self.logger.info(f"Normalized {len(transactions)} transactions from {schema_mapping.file_name}")
        return transactions
    
    def _detect_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect mapping from physical columns to logical purposes.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Mapping of purpose -> column_name
        """
        mapping = {}
        matched_cols = set()
        
        # Try to match columns to purposes
        cols_lower = [str(col).lower() for col in df.columns]
        
        # Match date column
        date_col = self._find_matching_column(cols_lower, self.DATE_PATTERNS.get("date", []))
        if date_col:
            mapping["date"] = date_col
            matched_cols.add(date_col)
        
        # Match description column
        desc_col = self._find_matching_column(cols_lower, self.DESCRIPTION_PATTERNS.get("description", []))
        if desc_col:
            mapping["description"] = desc_col
            matched_cols.add(desc_col)
        
        # Match amount columns
        amount_col = self._find_matching_column(cols_lower, self.AMOUNT_PATTERNS.get("amount", []))
        if amount_col:
            mapping["amount"] = amount_col
            mapping["amount_type"] = "unified"
            matched_cols.add(amount_col)
        else:
            # Try separate debit/credit
            debit_col = self._find_matching_column(cols_lower, self.AMOUNT_PATTERNS.get("debit", []))
            credit_col = self._find_matching_column(cols_lower, self.AMOUNT_PATTERNS.get("credit", []))
            
            if debit_col or credit_col:
                mapping["amount_type"] = "split"
                if debit_col:
                    mapping["debit"] = debit_col
                    matched_cols.add(debit_col)
                if credit_col:
                    mapping["credit"] = credit_col
                    matched_cols.add(credit_col)
        
        # Match balance column
        balance_col = self._find_matching_column(cols_lower, self.AMOUNT_PATTERNS.get("balance", []))
        if balance_col:
            mapping["balance"] = balance_col
            matched_cols.add(balance_col)
        
        return mapping
    
    def _find_matching_column(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """
        Find column matching any regex pattern.
        
        Args:
            columns: List of column names
            patterns: List of regex patterns
            
        Returns:
            Matching column name or None
        """
        for pattern in patterns:
            for col in columns:
                if re.match(pattern, col):
                    return col
        return None
    
    def _calculate_confidence_score(self, mapping: Dict[str, str], df: pd.DataFrame) -> float:
        """
        Calculate confidence score based on detected mapping quality.
        
        Args:
            mapping: Column mapping
            df: DataFrame
            
        Returns:
            Confidence score 0.0-1.0
        """
        score = 0.0
        
        # Must have date
        if "date" in mapping:
            score += 0.3
        elif "debit_date" in mapping or "credit_date" in mapping:
            score += 0.2
        
        # Must have description
        if "description" in mapping:
            score += 0.2
        
        # Must have amount (unified or split)
        if "amount" in mapping:
            score += 0.3
        elif mapping.get("amount_type") == "split":
            score += 0.25
        
        # Bonus for balance
        if "balance" in mapping:
            score += 0.15
        
        # Check data quality
        if "date" in mapping:
            date_col = mapping["date"]
            try:
                pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = df[date_col].notna().sum()
                if valid_dates / len(df) > 0.9:
                    score += 0.1
            except:
                score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _detect_ambiguities(self, mapping: Dict[str, str], df: pd.DataFrame) -> List[str]:
        """
        Detect ambiguities in the schema.
        
        Args:
            mapping: Column mapping
            df: DataFrame
            
        Returns:
            List of ambiguity descriptions
        """
        ambiguities = []
        
        if "date" not in mapping:
            ambiguities.append("Could not reliably detect date column")
        
        if "description" not in mapping:
            ambiguities.append("Could not reliably detect description column")
        
        if "amount" not in mapping and mapping.get("amount_type") != "split":
            ambiguities.append("Could not reliably detect amount column")
        
        if mapping.get("amount_type") == "split":
            if "debit" not in mapping or "credit" not in mapping:
                ambiguities.append("Split debit/credit detected but columns unclear")
        
        return ambiguities
    
    def _extract_date(self, row: pd.Series, mapping: Dict[str, str]) -> str:
        """Extract and normalize date."""
        if "date" not in mapping:
            return datetime.now().strftime("%Y-%m-%d")
        
        try:
            date_col = mapping["date"]
            if pd.isna(row[date_col]):
                return datetime.now().strftime("%Y-%m-%d")
            
            date_obj = pd.to_datetime(row[date_col])
            return date_obj.strftime("%Y-%m-%d")
        except:
            return datetime.now().strftime("%Y-%m-%d")
    
    def _extract_description(self, row: pd.Series, mapping: Dict[str, str]) -> str:
        """Extract description."""
        if "description" not in mapping:
            return "Unknown Transaction"
        
        try:
            desc_col = mapping["description"]
            if pd.isna(row[desc_col]):
                return "Unknown Transaction"
            return str(row[desc_col]).strip()
        except:
            return "Unknown Transaction"
    
    def _extract_amount_and_type(self, row: pd.Series, mapping: Dict[str, str]) -> Tuple[Decimal, str]:
        """Extract amount and transaction type."""
        if "amount" in mapping:
            # Unified amount column
            try:
                amount_str = str(row[mapping["amount"]]).replace("$", "").replace(",", "").strip()
                amount = Decimal(amount_str)
                
                if amount >= 0:
                    return abs(amount), "debit"
                else:
                    return abs(amount), "credit"
            except:
                raise ValueError("Invalid amount format")
        
        elif mapping.get("amount_type") == "split":
            # Separate debit/credit columns
            try:
                debit_val = Decimal(0)
                credit_val = Decimal(0)
                
                if "debit" in mapping:
                    debit_str = str(row[mapping["debit"]]).replace("$", "").replace(",", "").strip()
                    if debit_str and debit_str != "0":
                        debit_val = Decimal(debit_str)
                
                if "credit" in mapping:
                    credit_str = str(row[mapping["credit"]]).replace("$", "").replace(",", "").strip()
                    if credit_str and credit_str != "0":
                        credit_val = Decimal(credit_str)
                
                if debit_val > 0:
                    return debit_val, "debit"
                elif credit_val > 0:
                    return credit_val, "credit"
                else:
                    raise ValueError("No valid amount found")
            except:
                raise ValueError("Invalid debit/credit format")
        
        else:
            raise ValueError("No amount information found")
    
    def _extract_balance(self, row: pd.Series, mapping: Dict[str, str]) -> Optional[Decimal]:
        """Extract account balance if available."""
        if "balance" not in mapping:
            return None
        
        try:
            balance_col = mapping["balance"]
            if pd.isna(row[balance_col]):
                return None
            
            balance_str = str(row[balance_col]).replace("$", "").replace(",", "").strip()
            return Decimal(balance_str)
        except:
            return None
    
    def _assess_row_confidence(
        self,
        row: pd.Series,
        mapping: Dict[str, str],
        schema_confidence: float
    ) -> ConfidenceLevel:
        """Assess confidence level for a specific row."""
        # Start with schema confidence
        if schema_confidence > 0.9:
            base_confidence = ConfidenceLevel.HIGH
        elif schema_confidence > 0.7:
            base_confidence = ConfidenceLevel.MEDIUM
        else:
            base_confidence = ConfidenceLevel.LOW
        
        # Adjust based on row data quality
        null_count = row.isna().sum()
        if null_count > len(row) * 0.3:
            if base_confidence == ConfidenceLevel.HIGH:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        
        return base_confidence