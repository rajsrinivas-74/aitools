"""
Document Parser for CSV and PDF files.
Extracts raw data and passes to schema adapter for normalization.
"""

import logging
import sys
import argparse
import csv
from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
import pdfplumber

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.services.schema_adapter import SchemaAdapter

from backend.models import CanonicalTransaction, SchemaMapping
from backend.services.schema_adapter import SchemaDetectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class ParserError(Exception):
    """Custom exception for parsing errors."""
    pass


class DocumentParser:
    """
    Parses CSV and PDF files into structured DataFrames.
    Works with schema adapter to normalize heterogeneous data.
    """
    
    def __init__(self):
        """Initialize parser with schema adapter."""
        self.schema_adapter = SchemaAdapter()
        self.logger = logging.getLogger(__name__)
    
    def parse_and_normalize(self, file_path: str) -> Tuple[List[CanonicalTransaction], SchemaMapping]:
        """
        Parse file and normalize to canonical format.
        
        Args:
            file_path: Path to CSV or PDF fiTle
            
        Returns:
            Tuple of (canonical transactions, schema mapping)
        """
        self.logger.info(f"Parsing file: {file_path}")
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = self._parse_csv(file_path)
            elif file_path.suffix.lower() == '.pdf':
                df = self._parse_pdf(file_path)
            else:
                raise ParserError(f"Unsupported format: {file_path.suffix}")
            
            if df is None or df.empty:
                raise ParserError(f"No data found in {file_path.name}")
            
            # Detect schema
            schema_mapping = self.schema_adapter.detect_schema(df, file_path.name)
            
            # Normalize to canonical format
            transactions = self.schema_adapter.normalize_to_canonical(df, schema_mapping)
            
            if not transactions:
                raise ParserError(f"No valid transactions extracted from {file_path.name}")
            
            self.logger.info(
                f"Successfully parsed {len(transactions)} transactions from {file_path.name} "
                f"(schema confidence: {schema_mapping.confidence_score:.2f})"
            )
            
            return transactions, schema_mapping
            
        except ParserError:
            raise
        except Exception as e:
            logger.error(f"Error parsing {file_path.name}: {str(e)}")
            raise ParserError(f"Failed to parse {file_path.name}: {str(e)}")
    
    def _parse_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse CSV file with multiple encoding fallbacks.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame or None if parsing fails
        """
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully parsed CSV with {encoding} encoding")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        raise ParserError("Could not parse CSV with any encoding")
    
    def _parse_pdf(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract tables from PDF file using multiple strategies.
        Priority order for bank statement PDFs:
        1. PyMuPDF - Better for text extraction from structured bank documents
        2. PyPDF2 - Simple but effective text extraction
        3. pdfplumber - Last resort (best for pure table structures)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            DataFrame combining all tables or None
        """
        # Try PyMuPDF first (best for bank statements with mixed text/tables)
        try:
            df = self._parse_pdf_pymupdf(file_path)
            if self._is_valid_extraction(df):
                return df
        except Exception as e:
            self.logger.debug(f"PyMuPDF parsing failed: {str(e)}")
        
        # Try PyPDF2 as second option (simple but effective)
        try:
            df = self._parse_pdf_pypdf2(file_path)
            if self._is_valid_extraction(df):
                return df
        except Exception as e:
            self.logger.debug(f"PyPDF2 parsing failed: {str(e)}")
        
        # Try pdfplumber as last resort (good for pure table structures)
        df = self._parse_pdf_pdfplumber(file_path)
        if self._is_valid_extraction(df):
            return df
        
        self.logger.warning(f"No usable content found in PDF {file_path.name}")
        return None
    
    def _is_valid_extraction(self, df: Optional[pd.DataFrame]) -> bool:
        """
        Check if extracted DataFrame has useful data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if DataFrame has substantive data, False otherwise
        """
        if df is None or df.empty:
            return False
        
        # Check minimum dimensions
        if df.shape[0] < 2:  # At least 2 rows (header + data)
            return False
        
        # Check average column width (to detect single corrupted column)
        avg_col_width = df.shape[1]
        if avg_col_width < 3:  # At least 3 columns expected
            self.logger.debug(f"Extracted DataFrame has only {avg_col_width} columns (too narrow)")
            return False
        
        # Check for reasonable data content
        # Convert to string and check average cell length
        total_chars = df.astype(str).applymap(len).values.sum() if hasattr(df.astype(str), 'applymap') else 0
        if total_chars == 0:
            return False
        
        avg_cell_length = total_chars / (df.shape[0] * df.shape[1]) if (df.shape[0] * df.shape[1]) > 0 else 0
        self.logger.debug(f"Extraction quality: {df.shape[0]} rows × {df.shape[1]} cols, avg cell length: {avg_cell_length:.1f}")
        
        return True

    
    def _parse_pdf_pdfplumber(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract tables using pdfplumber (best for structured tables).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            DataFrame or None
        """
        all_tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                self.logger.debug(f"pdfplumber: Processing {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    if tables:
                        self.logger.debug(f"pdfplumber: Found {len(tables)} tables on page {page_num + 1}")
                        
                        for table_idx, table in enumerate(tables):
                            try:
                                # Convert table to DataFrame
                                if len(table) > 1:
                                    df_table = pd.DataFrame(table[1:], columns=table[0])
                                    all_tables.append(df_table)
                            except Exception as e:
                                self.logger.debug(f"pdfplumber: Could not parse table {table_idx} on page {page_num + 1}: {str(e)}")
                                continue
            
            if all_tables:
                # Concatenate all tables
                df = pd.concat(all_tables, ignore_index=True)
                self.logger.info(f"✓ pdfplumber: Extracted {len(all_tables)} tables from PDF")
                return df
            
            self.logger.debug(f"pdfplumber: No tables found in PDF {file_path.name}")
            return None
            
        except Exception as e:
            self.logger.debug(f"pdfplumber error: {str(e)}")
            return None
    
    def _parse_pdf_pymupdf(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract text and tables using PyMuPDF (fitz).
        Better at handling scanned PDFs and complex layouts.
        Specialized parsing for Google Pay statements.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            DataFrame or None
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            self.logger.debug("PyMuPDF (fitz) not installed")
            return None
        
        all_data = []
        all_text = ""  # Collect all text from all pages
        
        try:
            doc = fitz.open(file_path)
            self.logger.debug(f"PyMuPDF: Processing {len(doc)} pages")
            
            for page_num, page in enumerate(doc):
                # Try to extract tables using PyMuPDF's find_tables
                try:
                    tabs = page.find_tables()
                    if tabs:
                        self.logger.debug(f"PyMuPDF: Found {len(tabs)} tables on page {page_num + 1}")
                        for table in tabs:
                            try:
                                table_data = table.extract()
                                if table_data and len(table_data) > 1:
                                    all_data.extend(table_data)
                            except Exception as e:
                                self.logger.debug(f"PyMuPDF table extraction error: {str(e)}")
                                continue
                except Exception as e:
                    self.logger.debug(f"PyMuPDF: No tables on page {page_num + 1}: {str(e)}")
                
                # Collect text from all pages for fallback parsing
                text = page.get_text()
                if text:
                    all_text += text + "\n"
            
            doc.close()
            
            # If no tables found, try parsing all collected text as Google Pay or ICICI statement
            if not all_data and all_text:
                # Try ICICI bank statement parser first
                transactions = self._parse_icici_statement(all_text)
                if transactions:
                    all_data.extend(transactions)
                    self.logger.debug(f"ICICI parser extracted {len(transactions)} transactions")
                
                # Fallback to Google Pay parser if not ICICI
                if not all_data:
                    transactions = self._parse_google_pay_statement(all_text, Settings.DEFAULT_CURRENCY_SYMBOL)
                    if transactions:
                        all_data.extend(transactions)
                        self.logger.debug(f"Google Pay parser extracted {len(transactions)} transactions")
            
            if all_data:
                try:
                    df = pd.DataFrame(all_data)
                    self.logger.info(f"✓ PyMuPDF: Extracted data from PDF ({len(all_data)} rows)")
                    return df
                except Exception as e:
                    self.logger.debug(f"PyMuPDF DataFrame creation error: {str(e)}")
            
            return None
            
        except Exception as e:
            self.logger.debug(f"PyMuPDF error: {str(e)}")
            return None
    
    def _parse_google_pay_statement(self, text: str, currency_symbol: str = "$") -> Optional[list]:
        """
        Parse Google Pay transaction statements with structured text parsing.
        Extracts: Date, Description, Amount from formatted transaction blocks.
        Supports multiple currency symbols and auto-detects if not found.
        
        Args:
            text: Raw text from PDF page
            currency_symbol: Currency symbol to match (default: "$", auto-detects common symbols)
            
        Returns:
            List of transaction dictionaries or None
        """
        import re
        
        transactions = []
        
        # Common currency symbols to try if primary doesn't match
        currency_symbols = [currency_symbol, '$', '₹', '€', '£', '¥', '₽']
        
        # Try to find which currency symbol exists in the text
        active_symbol = currency_symbol
        for sym in currency_symbols:
            if sym in text:
                active_symbol = sym
                self.logger.debug(f"Google Pay Parser: Detected currency symbol: {sym}")
                break
        
        # Pattern for transactions with detected currency symbol
        escaped_symbol = re.escape(active_symbol)
        amount_pattern = f'{escaped_symbol}[\\d,.]+'
        
        # Split text into lines
        lines = text.strip().split('\n')
        
        # Remove header/footer lines, keep only actual content
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Skip page headers, footers, and empty lines
            if (line and 
                not line.startswith('Transaction statement') and
                not line.startswith('Note:') and
                not line.startswith('Page ') and
                not line.startswith('Transaction details') and
                not line.startswith('Date & time') and
                not line.startswith('Amount') and
                not line.startswith('Sent') and
                not line.startswith('Received')):
                filtered_lines.append(line)
        
        # Parse transactions - match date, then collect lines until finding amount
        i = 0
        while i < len(filtered_lines):
            line = filtered_lines[i].strip()
            
            # Look for date pattern (DD MMM, YYYY or DD Mon YYYY)
            if re.match(r'\d{1,2}\s+\w+,?\s+\d{4}', line):
                date_str = line
                time_str = ""
                description_lines = []
                amount = ""
                
                # Check next line for time
                if i + 1 < len(filtered_lines):
                    next_line = filtered_lines[i + 1].strip()
                    if re.match(r'\d{1,2}:\d{2}\s*(AM|PM)', next_line):
                        time_str = next_line
                        i += 1
                
                # Collect description and amount lines
                i += 1
                while i < len(filtered_lines):
                    current = filtered_lines[i].strip()
                    
                    # Check if this line contains an amount
                    amount_match = re.search(amount_pattern, current)
                    if amount_match:
                        # Extract amount
                        amount = amount_match.group().replace(active_symbol, '').replace(',', '')
                        # Extract any text before amount as description
                        before_amount = current[:amount_match.start()].strip()
                        if before_amount:
                            description_lines.append(before_amount)
                        break
                    else:
                        # This line is part of the description
                        if current:
                            description_lines.append(current)
                    i += 1
                
                # Only add valid transactions (has date and amount)
                if amount and date_str:
                    description = " ".join(description_lines).strip()
                    if not description:
                        description = "Transaction"
                    
                    transactions.append({
                        'Date': date_str,
                        'Time': time_str,
                        'Description': description,
                        'Amount': amount
                    })
            
            i += 1
        
        if transactions:
            self.logger.debug(f"Google Pay Parser: Extracted {len(transactions)} transactions")
        
        return transactions if transactions else None
    
    def _parse_icici_statement(self, text: str) -> Optional[list]:
        """
        Parse ICICI bank statement with multi-line structured transaction format.
        ICICI transactions can span multiple lines with descriptions wrapping.
        
        Format pattern:
        S_NO (line 1)
        DATE (line 2, DD.MM.YYYY)
        DESCRIPTION... (lines 3+)
        WITHDRAWAL_AMOUNT or "" (line N)
        DEPOSIT_AMOUNT or "" (line N+1)
        BALANCE (line N+2)
        
        Args:
            text: Raw text from PDF
            
        Returns:
            List of transaction dictionaries or None
        """
        import re
        
        transactions = []
        
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for S No. line (transaction serial number)
            if re.match(r'^\d+$', line) and i + 1 < len(lines):
                s_no = line
                
                # Next line should be date
                date_line = lines[i + 1].strip()
                date_match = re.match(r'^(\d{1,2}\.\d{2}\.\d{4})$', date_line)
                
                if date_match:
                    date_str = date_match.group(1)
                    
                    # Collect description lines until we find amounts
                    description_lines = []
                    j = i + 2
                    amounts_found = False
                    withdrawal = ""
                    deposit = ""
                    balance = ""
                    
                    while j < len(lines) and not amounts_found:
                        current_line = lines[j].strip()
                        
                        # Check if this line contains an amount (number possibly with comma/decimal)
                        if re.match(r'^[\d,]+\.?\d*$', current_line):
                            # This looks like an amount
                            # In ICICI: first amount is withdrawal, second is deposit, third is balance
                            if not withdrawal:
                                withdrawal = current_line
                            elif not deposit:
                                deposit = current_line
                            elif not balance:
                                balance = current_line
                                amounts_found = True  # Got withdrawal, deposit, and balance
                        elif current_line and current_line.upper() not in ['WITHDRAWAL', 'DEPOSIT', 'AMOUNT', 'BALANCE', 'CHEQUE NUMBER', 'TRANSACTION', 'REMARKS', 'DATE']:
                            # This is part of description
                            description_lines.append(current_line)
                        
                        j += 1
                    
                    description = " ".join(description_lines).strip()
                    
                    # Clean amounts
                    withdrawal_clean = re.sub(r'[^\d.]', '', withdrawal) if withdrawal else ""
                    deposit_clean = re.sub(r'[^\d.]', '', deposit) if deposit else ""
                    balance_clean = re.sub(r'[^\d.]', '', balance) if balance else ""
                    
                    # Determine amount and type
                    amount = ""
                    trans_type = "debit"
                    
                    try:
                        if withdrawal_clean and float(withdrawal_clean) > 0:
                            amount = withdrawal_clean
                            trans_type = "debit"
                        elif deposit_clean and float(deposit_clean) > 0:
                            amount = deposit_clean
                            trans_type = "credit"
                    except ValueError:
                        pass
                    
                    # Convert date format from DD.MM.YYYY to YYYY-MM-DD
                    try:
                        date_parts = date_str.split('.')
                        date_formatted = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                    except:
                        date_formatted = date_str
                    
                    # Add transaction if valid
                    if amount and description:
                        transactions.append({
                            'Date': date_formatted,
                            'Description': description,
                            'Amount': amount,
                            'Type': trans_type,
                            'Balance': balance_clean if balance_clean else ""
                        })
                        i = j - 1  # Move past processed lines
                
                i += 1
            else:
                i += 1
        
        if transactions:
            self.logger.debug(f"ICICI Parser: Extracted {len(transactions)} transactions")
        
        return transactions if transactions else None
    
    def _parse_pdf_pypdf2(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract text using PyPDF2.
        Simple text extraction from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            DataFrame or None
        """
        try:
            import PyPDF2
        except ImportError:
            self.logger.debug("PyPDF2 not installed")
            return None
        
        all_text = []
        
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                self.logger.debug(f"PyPDF2: Processing {len(reader.pages)} pages")
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        lines = text.strip().split('\n')
                        all_text.extend([line.split() for line in lines if line.strip()])
            
            if all_text:
                try:
                    df = pd.DataFrame(all_text)
                    self.logger.info(f"✓ PyPDF2: Extracted text from PDF ({len(all_text)} rows)")
                    return df
                except Exception as e:
                    self.logger.debug(f"PyPDF2 DataFrame creation error: {str(e)}")
            
            return None
            
        except Exception as e:
            self.logger.debug(f"PyPDF2 error: {str(e)}")
            return None


# ============================================================================
# TEST/DEMO SECTION - Run this file directly to test parser functionality
# ============================================================================

def create_sample_csv(output_path: str) -> str:
    """
    Create a sample CSV file with transaction data for testing.
    
    Args:
        output_path: Path where to save the sample CSV
        
    Returns:
        Path to the created CSV file
    """
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample transactions with various column names
    sample_data = [
        ['Date', 'Description', 'Amount', 'Type', 'Balance'],
        ['2026-01-05', 'Salary Deposit', '5000.00', 'credit', '5000.00'],
        ['2026-01-10', 'Starbucks Coffee', '5.50', 'debit', '4994.50'],
        ['2026-01-11', 'Whole Foods Market', '85.00', 'debit', '4909.50'],
        ['2026-01-12', 'Pizza Restaurant', '25.00', 'debit', '4884.50'],
        ['2026-01-13', 'Uber Trip', '18.50', 'debit', '4866.00'],
        ['2026-01-15', 'Bonus Payment', '500.00', 'credit', '5366.00'],
        ['2026-01-18', 'McDonald\'s', '12.50', 'debit', '5353.50'],
        ['2026-01-19', 'Gas Station Fuel', '45.00', 'debit', '5308.50'],
        ['2026-01-20', 'Costco Grocery', '120.00', 'debit', '5188.50'],
        ['2026-01-22', 'Movie Theater Ticket', '15.00', 'debit', '5173.50'],
        ['2026-01-25', 'Netflix Subscription', '15.99', 'debit', '5157.51'],
    ]
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)
    
    logger.info(f"Created sample CSV file: {csv_path}")
    return str(csv_path)


def test_parser(file_path: str, verbose: bool = False):
    """
    Test the document parser with a file.
    
    Args:
        file_path: Path to CSV or PDF file to parse
        verbose: Print verbose logging output
    """
    print("\n" + "="*80)
    print("DOCUMENT PARSER - STANDALONE TEST")
    print("="*80)
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Initialize parser
    try:
        parser = DocumentParser()
        logger.info("✓ DocumentParser initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize parser: {str(e)}")
        return
    
    # Parse file
    print("\n" + "-"*80)
    print("PARSING FILE")
    print("-"*80)
    
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        logger.error(f"❌ File not found: {file_path}")
        return
    
    try:
        logger.info(f"Parsing file: {file_path}")
        transactions, schema_mapping = parser.parse_and_normalize(file_path)
        logger.info(f"✓ Successfully parsed {len(transactions)} transactions")
    except ParserError as e:
        logger.error(f"❌ Parser error: {str(e)}")
        return
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return
    
    # Display schema information
    print("\n" + "="*80)
    print("SCHEMA MAPPING INFORMATION")
    print("="*80)
    print(f"File Name:           {schema_mapping.file_name}")
    print(f"Confidence Score:    {schema_mapping.confidence_score:.2f}")
    print(f"Requires LLM:        {schema_mapping.requires_llm}")
    
    if schema_mapping.llm_explanation:
        print(f"LLM Explanation:     {schema_mapping.llm_explanation}")
    
    print(f"\nDetected Schema:")
    for field, dtype in schema_mapping.detected_schema.items():
        print(f"  {field:20s}: {dtype}")
    
    print(f"\nColumn Mapping:")
    for from_col, to_col in schema_mapping.column_mapping.items():
        print(f"  {from_col:30s} → {to_col}")
    
    if schema_mapping.ambiguities:
        print(f"\nAmbiguities Detected:")
        for i, ambiguity in enumerate(schema_mapping.ambiguities, 1):
            print(f"  {i}. {ambiguity}")
    
    # Display parsed transactions
    print("\n" + "="*80)
    print(f"PARSED TRANSACTIONS ({len(transactions)} total)")
    print("="*80)
    
    # Show first few transactions in detail
    display_limit = min(5, len(transactions))
    for i, tx in enumerate(transactions[:display_limit], 1):
        print(f"\nTransaction {i}:")
        print(f"  Date:              {tx.date}")
        print(f"  Description:       {tx.description}")
        print(f"  Amount:            ${tx.amount:.2f}")
        print(f"  Type:              {tx.type} ({'Income' if tx.type == 'credit' else 'Expense'})")
        print(f"  Balance:           ${tx.balance:.2f}" if tx.balance else "  Balance:           N/A")
        print(f"  Source File:       {tx.source_file}")
        print(f"  Confidence:        {tx.confidence.value}")
        print(f"  Requires Review:   {tx.requires_review}")
        if tx.review_reason:
            print(f"  Review Reason:     {tx.review_reason}")
    
    if len(transactions) > display_limit:
        print(f"\n... and {len(transactions) - display_limit} more transactions")
    
    # Display summary statistics
    print("\n" + "="*80)
    print("TRANSACTION SUMMARY")
    print("="*80)
    
    income_count = sum(1 for tx in transactions if tx.type == "credit")
    expense_count = sum(1 for tx in transactions if tx.type == "debit")
    
    print(f"Total Transactions:  {len(transactions)}")
    print(f"Income Transactions: {income_count}")
    print(f"Expense Transactions: {expense_count}")
    print(f"Date Range:          {transactions[0].date} to {transactions[-1].date}")
    
    # Calculate totals
    from decimal import Decimal
    total_income = sum(tx.amount for tx in transactions if tx.type == "credit")
    total_expense = sum(tx.amount for tx in transactions if tx.type == "debit")
    
    print(f"Total Income:        ${total_income:.2f}")
    print(f"Total Expense:       ${total_expense:.2f}")
    print(f"Net:                 ${total_income - total_expense:.2f}")
    
    # Quality check
    print("\n" + "="*80)
    print("QUALITY CHECK")
    print("="*80)
    
    high_confidence = sum(1 for tx in transactions if tx.confidence.value == "high")
    medium_confidence = sum(1 for tx in transactions if tx.confidence.value == "medium")
    low_confidence = sum(1 for tx in transactions if tx.confidence.value == "low")
    requires_review = sum(1 for tx in transactions if tx.requires_review)
    
    print(f"High Confidence:     {high_confidence} ({high_confidence/len(transactions)*100:.1f}%)")
    print(f"Medium Confidence:   {medium_confidence} ({medium_confidence/len(transactions)*100:.1f}%)")
    print(f"Low Confidence:      {low_confidence} ({low_confidence/len(transactions)*100:.1f}%)")
    print(f"Requires Review:     {requires_review}")
    
    if requires_review > 0:
        print("\nTransactions Requiring Review:")
        for tx in transactions:
            if tx.requires_review:
                print(f"  - {tx.date} | {tx.description} | ${tx.amount:.2f} | {tx.review_reason}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80 + "\n")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Parse CSV or PDF transaction files and extract canonical transactions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a specific CSV file
  python parser.py path/to/transactions.csv
  
  # Parse a specific PDF file
  python parser.py path/to/statements.pdf
  
  # Create and parse a sample CSV file
  python parser.py --sample
  
  # Enable verbose logging
  python parser.py transactions.csv --verbose
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Path to CSV or PDF file to parse'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Create and parse a sample CSV file for testing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG level) logging'
    )
    
    parser.add_argument(
        '--output',
        default='sample_transactions.csv',
        help='Output path for sample CSV file (default: sample_transactions.csv)'
    )
    
    args = parser.parse_args()
    
    # Handle sample file creation
    if args.sample:
        logger.info("Creating sample CSV file for testing...")
        sample_file = create_sample_csv(args.output)
        test_parser(sample_file, verbose=args.verbose)
    elif args.file:
        # Parse provided file
        test_parser(args.file, verbose=args.verbose)
    else:
        # Default: create and parse sample
        logger.info("No file provided. Creating sample CSV file...")
        sample_file = create_sample_csv(args.output)
        test_parser(sample_file, verbose=args.verbose)


if __name__ == "__main__":
    main()