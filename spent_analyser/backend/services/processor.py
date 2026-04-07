"""
Transaction Processor - Categorization and Analysis.
Converts canonical transactions to processed form with categorization.
"""

import logging
import re
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from decimal import Decimal
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.models import CanonicalTransaction, ProcessedTransaction, AnalysisResult, ConfidenceLevel
from config.settings import settings

logger = logging.getLogger(__name__)


class TransactionProcessor:
    """
    Processes canonical transactions for categorization and analysis.
    Assigns categories, generates IDs, and computes insights.
    """
    
    # Category keyword mappings
    CATEGORY_KEYWORDS = {
        "Food & Dining": [
            "restaurant", "cafe", "pizza", "burger", "food", "dining",
            "coffee", "lunch", "dinner", "breakfast", "takeout", "delivery",
            "doordash", "ubereats", "grubhub", "diner", "grill", "bar",
            "bakery", "bistro", "canteen", "eatery"
        ],
        "Groceries": [
            "grocery", "supermarket", "trader joe", "whole foods", "costco",
            "safeway", "kroger", "walmart", "target", "market", "organic",
            "instacart", "amazon fresh", "asda", "tesco"
        ],
        "Transportation": [
            "uber", "lyft", "taxi", "gas", "fuel", "parking", "transit",
            "metro", "bus", "train", "public transport", "toll", "car",
            "vehicle", "auto", "gas station"
        ],
        "Travel": [
            "hotel", "airbnb", "flight", "airline", "motel", "resort",
            "vacation", "trip", "booking", "expedia", "airfare", "train",
            "cruise", "accommodation"
        ],
        "Shopping": [
            "amazon", "ebay", "shop", "store", "mall", "retail", "boutique",
            "clothes", "apparel", "fashion", "department store", "purchase"
        ],
        "Entertainment": [
            "movie", "cinema", "theater", "concert", "spotify", "netflix",
            "hulu", "game", "entertainment", "show", "ticket", "event",
            "album", "streaming", "cinema"
        ],
        "Utilities": [
            "electric", "water", "gas", "internet", "phone", "utility",
            "power", "bill", "connection", "wireless", "telephone"
        ],
        "Rent/Mortgage": [
            "rent", "mortgage", "landlord", "lease", "housing", "property",
            "apartment", "tenant"
        ],
        "Insurance": [
            "insurance", "policy", "premium", "claim", "coverage",
            "geico", "state farm", "allstate"
        ],
        "Healthcare": [
            "doctor", "hospital", "pharmacy", "medical", "dental", "clinic",
            "health", "medicine", "prescription", "cvs", "walgreens",
            "physician", "surgeon", "healthcare"
        ],
        "Education": [
            "school", "university", "tuition", "course", "education",
            "book", "training", "academy", "lesson", "class", "udemy",
            "college", "college", "student"
        ],
        "Personal Care": [
            "gym", "haircut", "salon", "spa", "massage", "fitness",
            "beauty", "personal", "care", "barber"
        ],
        "Home & Garden": [
            "home", "garden", "furniture", "decor", "hardware", "home depot",
            "lowes", "plants", "tools", "home improvement", "ikea"
        ],
        "Pet Care": [
            "pet", "vet", "veterinary", "dog", "cat", "animal", "petco",
            "petsmart", "animal hospital"
        ],
        "Subscriptions": [
            "subscription", "recurring", "membership", "annual", "monthly",
            "subscription charge", "auto-renew", "renew"
        ],
        "Fees & Charges": [
            "fee", "charge", "service charge", "overdraft", "penalty",
            "interest charge", "commission", "bank fee"
        ],
        "Salary": [
            "salary", "paycheck", "wage", "income", "direct deposit",
            "payroll", "compensation", "earnings", "pay", "wages"
        ],
        "Bonus": [
            "bonus", "commission", "incentive", "performance bonus",
            "annual bonus", "extra pay", "award"
        ],
        "Refund": [
            "refund", "return credit", "reimbursement", "credit",
            "refunded", "chargeback", "return"
        ],
        "Interest": [
            "interest", "dividend", "yield", "return on investment",
            "interest payment", "interest income", "interest earned"
        ],
        "Other Income": [
            "income", "transfer in", "deposit", "received",
            "payment received", "incoming transfer"
        ]
    }
    
    def __init__(self):
        """Initialize processor."""
        self.logger = logging.getLogger(__name__)
        self._build_reverse_mapping()
    
    def _build_reverse_mapping(self) -> None:
        """Build keyword to category reverse mapping."""
        self.keyword_to_category = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                self.keyword_to_category[keyword.lower()] = category
    
    def process_transactions(
        self,
        canonical_transactions: List[CanonicalTransaction]
    ) -> List[ProcessedTransaction]:
        """
        Process canonical transactions to add categorization and metadata.
        
        Args:
            canonical_transactions: List of canonical transactions
            
        Returns:
            List of processed transactions
        """
        processed = []
        
        for transaction in canonical_transactions:
            try:
                # Normalize description
                normalized_desc = self._normalize_description(transaction.description)
                
                # Determine if income or expense
                is_income = transaction.type.lower() == "credit"
                
                # Categorize transaction
                category = self._categorize_transaction(transaction, is_income)
                
                # Generate transaction ID
                tx_id = self._generate_transaction_id(transaction)
                
                # Create processed transaction
                processed_tx = ProcessedTransaction(
                    canonical=transaction,
                    category=category,
                    normalized_description=normalized_desc,
                    transaction_id=tx_id,
                    is_income=is_income,
                    metadata={
                        "original_description": transaction.description,
                        "confidence": transaction.confidence.value
                    }
                )
                
                processed.append(processed_tx)
                
            except Exception as e:
                self.logger.error(f"Error processing transaction: {str(e)}")
                continue
        
        self.logger.debug(f"Processed {len(processed)} transactions successfully")
        return processed
    
    def filter_by_date_range(
        self,
        transactions: List[ProcessedTransaction],
        start_date: datetime,
        end_date: datetime
    ) -> List[ProcessedTransaction]:
        """Filter transactions by date range."""
        self.logger.debug(f"Filtering {len(transactions)} transactions between {start_date.date()} and {end_date.date()}")
        filtered = []
        
        for transaction in transactions:
            tx_date = datetime.strptime(transaction.canonical.date, "%Y-%m-%d")
            if start_date <= tx_date <= end_date:
                filtered.append(transaction)
        
        self.logger.debug(f"Filtered to {len(filtered)} transactions in date range")
        return filtered
    
    def analyze_transactions(self, transactions: List[ProcessedTransaction]) -> AnalysisResult:
        """
        Generate comprehensive analysis of transactions.
        Outputs follow Financial Analysis and Spend Intelligence schema format.
        
        Args:
            transactions: List of processed transactions
            
        Returns:
            AnalysisResult object with schema-compliant structure
        """
        self.logger.debug(f"Starting analysis of {len(transactions)} transactions")
        if not transactions:
            self.logger.warning("No transactions provided for analysis")
            return AnalysisResult(
                summary={
                    "total_income": 0,
                    "total_expense": 0,
                    "net_savings": 0,
                    "savings_rate": 0,
                    "transaction_count": 0
                },
                categories=[],
                trends=[],
                insights=["No transactions to analyze"],
                requires_human_review=[]
            )
        
        # Calculate summary
        total_income = sum(
            t.canonical.amount for t in transactions if t.is_income
        )
        total_expense = sum(
            t.canonical.amount for t in transactions if not t.is_income
        )
        net_savings = total_income - total_expense
        
        # Calculate savings rate
        savings_rate = 0.0
        if total_income > 0:
            savings_rate = (float(net_savings) / float(total_income)) * 100
        
        summary = {
            "total_income": float(total_income),
            "total_expense": float(total_expense),
            "net_savings": float(net_savings),
            "savings_rate": round(savings_rate, 2),
            "transaction_count": len(transactions)
        }
        
        # Analyze categories (separates income/expense)
        category_breakdown = self._analyze_categories_separated(transactions)
        
        # Analyze trends
        trends = self._analyze_trends(transactions)
        
        # Generate insights
        insights = self._generate_insights(transactions, summary, category_breakdown)
        
        # Find transactions requiring review
        requires_review = self._get_review_transactions(transactions)
        
        # Metadata
        metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "transaction_count": len(transactions),
            "date_range": {
                "start": min(t.canonical.date for t in transactions),
                "end": max(t.canonical.date for t in transactions)
            },
            "requires_review_count": len(requires_review)
        }
        
        return AnalysisResult(
            summary=summary,
            categories=category_breakdown,
            trends=trends,
            insights=insights,
            requires_human_review=requires_review,
            metadata=metadata
        )
    
    def _analyze_categories_separated(self, transactions: List[ProcessedTransaction]) -> Dict:
        """
        Analyze spending by category, separating income and expense.
        Returns schema-compliant category breakdown structure.
        
        Args:
            transactions: List of processed transactions
            
        Returns:
            Dictionary with 'income' and 'expense' arrays
        """
        income_data: Dict[str, Tuple[Decimal, int]] = defaultdict(lambda: (Decimal(0), 0))
        expense_data: Dict[str, Tuple[Decimal, int]] = defaultdict(lambda: (Decimal(0), 0))
        
        # Separate by income/expense
        for transaction in transactions:
            if transaction.is_income:
                current_amount, count = income_data[transaction.category]
                income_data[transaction.category] = (
                    current_amount + transaction.canonical.amount,
                    count + 1
                )
            else:
                current_amount, count = expense_data[transaction.category]
                expense_data[transaction.category] = (
                    current_amount + transaction.canonical.amount,
                    count + 1
                )
        
        # Calculate totals
        total_income = sum(amount for amount, _ in income_data.values())
        total_expense = sum(amount for amount, _ in expense_data.values())
        
        # Build income categories
        income_categories = []
        for category_name, (amount, count) in sorted(
            income_data.items(),
            key=lambda x: x[1][0],
            reverse=True
        ):
            percentage = (float(amount) / float(total_income) * 100) if total_income > 0 else 0
            income_categories.append({
                "category": category_name,
                "amount": round(float(amount), 2),
                "percentage": round(percentage, 2)
            })
        
        # Build expense categories
        expense_categories = []
        for category_name, (amount, count) in sorted(
            expense_data.items(),
            key=lambda x: x[1][0],
            reverse=True
        ):
            percentage = (float(amount) / float(total_expense) * 100) if total_expense > 0 else 0
            expense_categories.append({
                "category": category_name,
                "amount": round(float(amount), 2),
                "percentage": round(percentage, 2)
            })
        
        return {
            "income": income_categories,
            "expense": expense_categories
        }
    
    def _analyze_categories(self, transactions: List[ProcessedTransaction]) -> List[Dict]:
        """Analyze spending by category."""
        category_data: Dict[str, Tuple[Decimal, int]] = defaultdict(lambda: (Decimal(0), 0))
        
        for transaction in transactions:
            current_amount, count = category_data[transaction.category]
            category_data[transaction.category] = (
                current_amount + transaction.canonical.amount,
                count + 1
            )
        
        # Calculate total
        total_amount = sum(amount for amount, _ in category_data.values())
        
        # Build results
        categories = []
        for category_name, (amount, count) in sorted(
            category_data.items(),
            key=lambda x: x[1][0],
            reverse=True
        ):
            percentage = (float(amount) / float(total_amount) * 100) if total_amount > 0 else 0
            categories.append({
                "name": category_name,
                "amount": float(amount),
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return categories
    
    def _analyze_trends(self, transactions: List[ProcessedTransaction]) -> List[Dict]:
        """Analyze trends over time."""
        trend_data: Dict[str, Dict] = defaultdict(
            lambda: {"income": Decimal(0), "expense": Decimal(0), "count": 0}
        )
        
        for transaction in transactions:
            month_key = transaction.canonical.date[:7]  # YYYY-MM
            data = trend_data[month_key]
            
            if transaction.is_income:
                data["income"] += transaction.canonical.amount
            else:
                data["expense"] += transaction.canonical.amount
            
            data["count"] += 1
        
        # Build sorted results
        trends = []
        for month_key in sorted(trend_data.keys()):
            data = trend_data[month_key]
            net = data["income"] - data["expense"]
            trends.append({
                "date": month_key,
                "income": float(data["income"]),
                "expense": float(data["expense"]),
                "net": float(net),
                "transaction_count": data["count"]
            })
        
        return trends
    
    def _generate_insights(
        self,
        transactions: List[ProcessedTransaction],
        summary: Dict,
        category_breakdown: Dict
    ) -> List[str]:
        """
        Generate text insights from transaction analysis.
        Uses schema-compliant summary fields.
        """
        insights = []
        
        total_income = summary.get("total_income", 0)
        total_expense = summary.get("total_expense", 0)
        net_savings = summary.get("net_savings", 0)
        savings_rate = summary.get("savings_rate", 0)
        
        if total_income > 0:
            insights.append(f"Total income: ${total_income:,.2f}")
        
        if total_expense > 0:
            insights.append(f"Total expenses: ${total_expense:,.2f}")
        
        if net_savings > 0:
            insights.append(f"Net savings: ${net_savings:,.2f} ({savings_rate:.1f}% savings rate)")
        elif net_savings < 0:
            insights.append(f"Net deficit: ${abs(net_savings):,.2f}")
        
        # Get top expense category
        expense_categories = category_breakdown.get("expense", [])
        if expense_categories:
            top_category = expense_categories[0]
            insights.append(
                f"Top expense category: {top_category['category']} "
                f"(${top_category['amount']:,.2f}, {top_category['percentage']}%)"
            )
        
        # Get top income category
        income_categories = category_breakdown.get("income", [])
        if income_categories:
            top_income = income_categories[0]
            insights.append(
                f"Top income source: {top_income['category']} "
                f"(${top_income['amount']:,.2f})"
            )
        
        if transactions:
            avg_amount = sum(t.canonical.amount for t in transactions) / len(transactions)
            insights.append(f"Average transaction: ${avg_amount:,.2f}")
        
        return insights
    
    def _get_review_transactions(self, transactions: List[ProcessedTransaction]) -> List[Dict]:
        """Get transactions requiring human review."""
        review_list = []
        
        for transaction in transactions:
            if transaction.canonical.requires_review:
                review_list.append({
                    "transaction_id": transaction.transaction_id,
                    "date": transaction.canonical.date,
                    "description": transaction.canonical.description,
                    "amount": float(transaction.canonical.amount),
                    "confidence": transaction.canonical.confidence.value,
                    "reason": transaction.canonical.review_reason
                })
        
        return review_list
    
    def _normalize_description(self, description: str) -> str:
        """Normalize transaction description."""
        text = description.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[a-z]{2}\d{3,}', '', text)
        return text.title().strip()
    
    def _categorize_transaction(
        self,
        transaction: CanonicalTransaction,
        is_income: bool
    ) -> str:
        """Categorize transaction based on description."""
        description = transaction.description.lower()
        
        # Search keywords
        for keyword, category in self.keyword_to_category.items():
            if keyword in description:
                return category
        
        # Default based on type
        return "Other Income" if is_income else "Other Expense"
    
    def _generate_transaction_id(self, transaction: CanonicalTransaction) -> str:
        """Generate unique transaction ID."""
        id_str = f"{transaction.date}{transaction.amount}{transaction.description}{transaction.source_file}"
        return hashlib.md5(id_str.encode()).hexdigest()[:12]

# ============================================================================
# TEST/DEMO SECTION - Run this file directly to test processor functionality
# ============================================================================

def create_sample_transactions() -> List[CanonicalTransaction]:
    """Create sample transactions for testing."""
    return [
        # Income
        CanonicalTransaction(
            date="2026-01-05",
            description="Salary Deposit",
            amount=Decimal("5000"),
            type="credit",
            balance=Decimal("5000"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        CanonicalTransaction(
            date="2026-01-15",
            description="Bonus Payment",
            amount=Decimal("500"),
            type="credit",
            balance=Decimal("5500"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        
        # Food & Dining
        CanonicalTransaction(
            date="2026-01-10",
            description="Starbucks Coffee",
            amount=Decimal("5.50"),
            type="debit",
            balance=Decimal("4994.50"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        CanonicalTransaction(
            date="2026-01-12",
            description="Pizza Restaurant",
            amount=Decimal("25.00"),
            type="debit",
            balance=Decimal("4969.50"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        CanonicalTransaction(
            date="2026-01-18",
            description="McDonald's",
            amount=Decimal("12.50"),
            type="debit",
            balance=Decimal("4957.00"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        
        # Groceries
        CanonicalTransaction(
            date="2026-01-11",
            description="Whole Foods Market",
            amount=Decimal("85.00"),
            type="debit",
            balance=Decimal("4872.00"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        CanonicalTransaction(
            date="2026-01-20",
            description="Costco Grocery",
            amount=Decimal("120.00"),
            type="debit",
            balance=Decimal("4752.00"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        
        # Transportation
        CanonicalTransaction(
            date="2026-01-13",
            description="Uber Trip",
            amount=Decimal("18.50"),
            type="debit",
            balance=Decimal("4733.50"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        CanonicalTransaction(
            date="2026-01-19",
            description="Gas Station Fuel",
            amount=Decimal("45.00"),
            type="debit",
            balance=Decimal("4688.50"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        
        # Subscriptions
        CanonicalTransaction(
            date="2026-01-25",
            description="Netflix Subscription",
            amount=Decimal("15.99"),
            type="debit",
            balance=Decimal("4672.51"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
        
        # Entertainment
        CanonicalTransaction(
            date="2026-01-22",
            description="Movie Theater Ticket",
            amount=Decimal("15.00"),
            type="debit",
            balance=Decimal("4657.51"),
            source_file="test_transactions.csv",
            confidence=ConfidenceLevel.HIGH
        ),
    ]


def test_processor():
    """Test the transaction processor with sample data."""
    print("\n" + "="*80)
    print("TRANSACTION PROCESSOR - STANDALONE TEST")
    print("="*80)
    
    # Initialize processor
    processor = TransactionProcessor()
    print("\n✓ TransactionProcessor initialized")
    
    # Create sample transactions
    canonical_transactions = create_sample_transactions()
    print(f"\n✓ Created {len(canonical_transactions)} sample transactions")
    
    # Display input transactions
    print("\n" + "-"*80)
    print("INPUT TRANSACTIONS:")
    print("-"*80)
    for i, tx in enumerate(canonical_transactions, 1):
        print(f"{i:2d}. {tx.date} | {tx.description:30s} | ${tx.amount:>8.2f} | {tx.type}")
    
    # Process transactions
    print("\n" + "-"*80)
    print("PROCESSING TRANSACTIONS...")
    print("-"*80)
    processed = processor.process_transactions(canonical_transactions)
    print(f"✓ Processed {len(processed)} transactions")
    
    # Display processed transactions with categories
    print("\n" + "-"*80)
    print("PROCESSED TRANSACTIONS (WITH CATEGORIES):")
    print("-"*80)
    for i, tx in enumerate(processed, 1):
        income_str = "INCOME" if tx.is_income else "EXPENSE"
        print(f"{i:2d}. {tx.canonical.date} | {tx.normalized_description:25s} | "
              f"${tx.canonical.amount:>8.2f} | {income_str:7s} | {tx.category}")
    
    # Analyze transactions
    print("\n" + "-"*80)
    print("ANALYZING TRANSACTIONS...")
    print("-"*80)
    analysis = processor.analyze_transactions(processed)
    print("✓ Analysis complete")
    
    # Display summary
    print("\n" + "="*80)
    print("FINANCIAL SUMMARY")
    print("="*80)
    summary = analysis.summary
    print(f"Total Income:      ${summary['total_income']:>10.2f}")
    print(f"Total Expenses:    ${summary['total_expense']:>10.2f}")
    print(f"Net Savings:       ${summary['net_savings']:>10.2f}")
    print(f"Savings Rate:      {summary['savings_rate']:>10.1f}%")
    print(f"Transaction Count: {summary['transaction_count']:>10d}")
    
    # Display category breakdown
    print("\n" + "="*80)
    print("INCOME BREAKDOWN")
    print("="*80)
    if analysis.categories.get('income'):
        for cat in analysis.categories['income']:
            print(f"  {cat['category']:25s}: ${cat['amount']:>10.2f} ({cat['percentage']:>6.2f}%)")
    else:
        print("  No income transactions")
    
    print("\n" + "="*80)
    print("EXPENSE BREAKDOWN")
    print("="*80)
    if analysis.categories.get('expense'):
        for cat in analysis.categories['expense']:
            print(f"  {cat['category']:25s}: ${cat['amount']:>10.2f} ({cat['percentage']:>6.2f}%)")
    else:
        print("  No expense transactions")
    
    # Display trends
    print("\n" + "="*80)
    print("MONTHLY TRENDS")
    print("="*80)
    if analysis.trends:
        print(f"{'Month':<12} | {'Income':>10} | {'Expense':>10} | {'Net':>10} | {'Count':>5}")
        print("-" * 55)
        for trend in analysis.trends:
            print(f"{trend['date']:<12} | ${trend['income']:>9.2f} | ${trend['expense']:>9.2f} | "
                  f"${trend['net']:>9.2f} | {trend['transaction_count']:>5d}")
    else:
        print("No trend data")
    
    # Display insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    for i, insight in enumerate(analysis.insights, 1):
        print(f"{i}. {insight}")
    
    # Display human review items
    if analysis.requires_human_review:
        print("\n" + "="*80)
        print("TRANSACTIONS REQUIRING HUMAN REVIEW")
        print("="*80)
        for item in analysis.requires_human_review:
            print(f"  ID: {item['transaction_id']}")
            print(f"  Date: {item['date']}")
            print(f"  Description: {item['description']}")
            print(f"  Amount: ${item['amount']:.2f}")
            print(f"  Confidence: {item['confidence']}")
            print(f"  Reason: {item['reason']}")
            print()
    else:
        print("\n✓ No transactions require human review")
    
    # Display metadata
    print("\n" + "="*80)
    print("ANALYSIS METADATA")
    print("="*80)
    metadata = analysis.metadata
    print(f"Analysis Timestamp: {metadata['analysis_timestamp']}")
    print(f"Date Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    print(f"Total Transactions: {metadata['transaction_count']}")
    print(f"Requires Review: {metadata['requires_review_count']}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_processor()