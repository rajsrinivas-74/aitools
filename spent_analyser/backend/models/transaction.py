"""
Unified Transaction Data Model with Confidence Scoring.
Provides a common data structure for all transaction extraction sources.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from decimal import Decimal


class ExtractionMethod(str, Enum):
    """Extraction method used to obtain the transaction data."""
    TABULA = "tabula"  # Table extraction library
    LLM = "llm"  # Claude/GPT extraction
    REGEX = "regex"  # Pattern matching
    PYMUPDF = "pymupdf"  # PyMuPDF text extraction
    PDFPLUMBER = "pdfplumber"  # Pdfplumber extraction
    MANUAL = "manual"  # Manual entry
    CSV = "csv"  # CSV file
    UNKNOWN = "unknown"


class TransactionType(str, Enum):
    """Transaction type."""
    CREDIT = "credit"
    DEBIT = "debit"
    TRANSFER = "transfer"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    """
    Unified transaction data structure with confidence scoring.
    
    Attributes:
        date: Transaction date (YYYY-MM-DD format)
        description: Transaction description/narration
        amount: Transaction amount (absolute value, no sign)
        type: Transaction type (credit/debit)
        currency: Currency symbol or code (default: $)
        confidence_score: Confidence score 0.0-1.0 for extraction accuracy
        category: Transaction category (e.g., "Food & Dining", "Other Expense")
        extraction_method: How the transaction was extracted
        source_file: Source PDF/CSV filename
        balance: Account balance after transaction (if available)
        raw_data: Original extracted data before normalization
        metadata: Additional metadata from source (llm_verified, original_category, etc.)
    """
    
    date: str  # YYYY-MM-DD format
    description: str
    amount: Decimal
    type: TransactionType
    currency: str = "$"
    confidence_score: float = 0.8  # Default confidence
    category: str = "Other"  # Category assignment (e.g., "Food & Dining", "Other Expense")
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    source_file: str = ""
    balance: Optional[Decimal] = None
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize data."""
        # Ensure amount is Decimal
        if isinstance(self.amount, (int, float)):
            self.amount = Decimal(str(self.amount))
        
        # Ensure confidence score is between 0-1
        if self.confidence_score < 0:
            self.confidence_score = 0.0
        elif self.confidence_score > 1:
            self.confidence_score = 1.0
        
        # Ensure type is TransactionType enum
        if isinstance(self.type, str):
            try:
                self.type = TransactionType(self.type.lower())
            except ValueError:
                self.type = TransactionType.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling Decimal and Enum types."""
        data = asdict(self)
        data['amount'] = float(self.amount)
        data['balance'] = float(self.balance) if self.balance else None
        data['type'] = self.type.value
        data['extraction_method'] = self.extraction_method.value
        return data
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'date': self.date,
            'description': self.description,
            'amount': f"{self.currency}{float(self.amount):.2f}",
            'type': self.type.value,
            'currency': self.currency,
            'category': self.category,
            'confidence_score': round(self.confidence_score, 2),
            'extraction_method': self.extraction_method.value,
            'source_file': self.source_file,
            'balance': f"{self.currency}{float(self.balance):.2f}" if self.balance else None,
        }
    
    def is_high_confidence(self) -> bool:
        """Check if confidence score is high (>= 0.8)."""
        return self.confidence_score >= 0.8
    
    def is_medium_confidence(self) -> bool:
        """Check if confidence score is medium (0.5-0.8)."""
        return 0.5 <= self.confidence_score < 0.8
    
    def is_low_confidence(self) -> bool:
        """Check if confidence score is low (< 0.5)."""
        return self.confidence_score < 0.5
    
    def __str__(self) -> str:
        """Human-readable transaction string."""
        confidence_emoji = "✓" if self.is_high_confidence() else "⚠" if self.is_medium_confidence() else "✗"
        return (
            f"{confidence_emoji} {self.date} | {self.type.value.upper():6} | "
            f"{self.currency}{self.amount:>10.2f} | {self.description[:40]:40} | "
            f"[{self.extraction_method.value}] ({self.confidence_score:.2f})"
        )


@dataclass
class TransactionBatch:
    """
    Batch of extracted transactions with aggregate statistics.
    """
    transactions: list[Transaction]
    source_file: str
    extraction_method: ExtractionMethod
    total_confidence: float = 0.0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    
    def __post_init__(self):
        """Calculate statistics."""
        if self.transactions:
            self.total_confidence = sum(t.confidence_score for t in self.transactions) / len(self.transactions)
            self.high_confidence_count = sum(1 for t in self.transactions if t.is_high_confidence())
            self.medium_confidence_count = sum(1 for t in self.transactions if t.is_medium_confidence())
            self.low_confidence_count = sum(1 for t in self.transactions if t.is_low_confidence())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'transactions': [t.to_dict() for t in self.transactions],
            'source_file': self.source_file,
            'extraction_method': self.extraction_method.value,
            'total_confidence': round(self.total_confidence, 2),
            'high_confidence_count': self.high_confidence_count,
            'medium_confidence_count': self.medium_confidence_count,
            'low_confidence_count': self.low_confidence_count,
            'total_transactions': len(self.transactions),
        }
    
    def __str__(self) -> str:
        """String representation of batch."""
        return (
            f"TransactionBatch(source='{self.source_file}', method={self.extraction_method.value}, "
            f"count={len(self.transactions)}, avg_confidence={self.total_confidence:.2f}, "
            f"high={self.high_confidence_count}, medium={self.medium_confidence_count}, low={self.low_confidence_count})"
        )