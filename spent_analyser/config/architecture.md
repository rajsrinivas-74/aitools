# Spend Analyzer - Architecture Documentation

## Overview

Spend Analyzer is a production-grade, AI-powered financial transaction analysis system designed to handle heterogeneous data inputs (CSV and PDF files with varying schemas) and provide actionable insights through intelligent RAG (Retrieval-Augmented Generation).

**Key Principle:** The system does NOT assume a fixed input schema. Instead, it dynamically detects and normalizes all inputs to a canonical transaction model.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                     │
│  - File Upload (CSV/PDF)  - Date Range Selector             │
│  - Visualizations (Charts) - Insights Display               │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP API Calls
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  BACKEND (Flask API)                        │
│  Routes: /upload, /analyze, /results, /health              │
└────────────────┬────────────────────────┬────────────────────┘
                 │                        │
        ┌────────▼────────┐      ┌───────▼────────┐
        │  Parser Layer   │      │  Processor     │
        │  - CSV parser   │      │  - Categorize │
        │  - PDF extractor│      │  - Normalize   │
        │  - Raw extraction       │  - Analyze     │
        └────────┬────────┘      └───────┬────────┘
                 │                        │
        ┌────────▼──────────────────────▼────────┐
        │  SCHEMA ADAPTER (Critical Component)   │
        │  🔑 Dynamic Schema Detection           │
        │  - Column mapping (date, amount, desc) │
        │  - Confidence scoring                  │
        │  - Ambiguity detection                 │
        └────────┬───────────────────────────────┘
                 │
        ┌────────▼────────────────────────────┐
        │  Canonical Transactions             │
        │  {date, description, amount,        │
        │   type, balance, confidence}        │
        └────────┬───────────────────────────┘
                 │
        ┌────────┴────────────┬──────────────┐
        │                     │              │
   ┌────▼──────┐      ┌──────▼─────┐  ┌────▼─────┐
   │  FAISS    │      │   Neo4j    │  │    LLM   │
   │  Vector   │      │   Graph    │  │  Service │
   │   Store   │      │   Store    │  │  (OpenAI)│
   │           │      │            │  │          │
   │ - Embed   │      │ Nodes:     │  │Generate: │
   │ - Index   │      │ User       │  │Embeddings│
   │ - Search  │      │ Txn        │  │Insights  │
   │           │      │ Category   │  │Ambiguity │
   │           │      │            │  │Detection │
   └────────────┘      │ Rel:       │  └──────────┘
                       │ MADE       │
                       │ BELONGS_TO │
                       └────────────┘
                            ▲
                            │
        ┌───────────────────┴──────────────────┐
        │     RAG PIPELINE (Orchestrator)      │
        │  - Index transactions                │
        │  - Retrieve similar context (FAISS)  │
        │  - Graph insights (Neo4j)            │
        │  - Generate insights (LLM)           │
        │  - Avoid hallucination               │
        └───────────────────┬──────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Structured     │
                    │ Output (JSON)  │
                    │- summary       │
                    │- categories    │
                    │- trends        │
                    │- insights      │
                    │- review_items  │
                    └────────────────┘
```

---

## Core Components

### 1. **Schema Adapter** (Critical Component)

**File:** `backend/services/schema_adapter.py`

**Responsibility:** Dynamically detect and normalize heterogeneous input schemas.

#### Why It's Critical

- Financial institutions use **varying formats** for transaction exports
- CSV column names vary: `Date`, `Transaction Date`, `Txn Date`, `Value Date`
- Amount representations: single `Amount` column vs. separate `Debit`/`Credit` columns
- Without schema detection, the system would fail on non-standard data

#### Key Features

1. **Pattern Matching for Column Detection**
   - Uses regex patterns to detect date, amount, and description columns
   - Handles variations in naming conventions
   - Fallback mechanisms for ambiguous columns

2. **Confidence Scoring**
   - Scores schema detection quality (0.0-1.0)
   - HIGH confidence: >0.9
   - MEDIUM confidence: 0.7-0.9
   - LOW confidence: <0.7

3. **Ambiguity Detection**
   - Reports unclear column mappings
   - Flags for manual review
   - Triggers optional LLM analysis for complex schemas

4. **Canonical Normalization**
   - Converts all transactions to standard format:
     ```python
     CanonicalTransaction(
         date: YYYY-MM-DD,
         description: str,
         amount: Decimal,
         type: "credit" | "debit",
         balance: Optional[Decimal],
         confidence: "high" | "medium" | "low"
     )
     ```

#### Schema Detection Flow

```
Raw CSV/PDF
    │
    ▼
Parse Raw Data (Parser)
    │
    ▼
Detect Schema (SchemaAdapter)
|- Analyze column names with regex
|- Calculate confidence score
|- Identify ambiguities
|- Flag for LLM if needed
    │
    ▼
Normalize to Canonical Format
|- Extract fields using detected mapping
|- Type conversion (date, amount)
|- Confidence propagation
    │
    ▼
Canonical Transactions Ready for Processing
```

### 2. **Parser Module**

**File:** `backend/services/parser.py`

**Duties:**
- Extract raw data from CSV/PDF files
- Handle multiple encodings
- Extract tables from PDFs
- Pass to schema adapter for normalization

**Key Methods:**
- `parse_and_normalize()` - Single entry point
- `_parse_csv()` - CSV parsing with encoding fallback
- `_parse_pdf()` - PDF table extraction using pdfplumber

### 3. **Transaction Processor**

**File:** `backend/services/processor.py`

**Duties:**
- Categorize transactions into predefined categories
- Normalize descriptions
- Generate unique transaction IDs
- Compute analysis (trends, summaries, insights)

**Categories:**

Income:
- Salary, Bonus, Refund, Interest, Investment Return, Gift, Reimbursement

Expense:
- Food & Dining, Groceries, Transportation, Travel, Shopping, Entertainment, Utilities, Rent/Mortgage, Insurance, Healthcare, Education, Personal Care, Home & Garden, Pet Care, Subscriptions, Fees & Charges

**Categorization Strategy:**
- Keyword matching on normalized description
- Fallback to "Other Income" or "Other Expense"
- Confidence based on keyword match quality

### 4. **FAISS Vector Store**

**File:** `backend/services/faiss_store.py`

**Purpose:** Semantic similarity search for transactions

**Capabilities:**
- Store transaction embeddings (OpenAI embedding-3-small, 1536-dim)
- Retrieve similar transactions based on query
- Persistent index storage

**Usage in RAG:**
- When analyzing transactions → retrieve similar historical patterns
- When generating insights → find contextually similar transactions

### 5. **Neo4j Graph Store**

**File:** `backend/services/neo4j_store.py`

**Purpose:** Model transaction relationships and spending patterns

**Data Model:**

```
(User) -[:MADE]-> (Transaction) -[:BELONGS_TO]-> (Category)
```

**Queries:**
- Spending patterns by category
- Category co-occurrence (what categories appear together in user spending)
- User summary statistics

### 6. **LLM Service**

**File:** `backend/services/llm_service.py`

**Duties:**
- Generate embeddings using OpenAI
- Generate insights using GPT-4
- Detect ambiguous descriptions (optional)

**CRITICAL: Hallucination Prevention**
```python
# ONLY use retrieved context, never make up data
context = retrieve_from_faiss_and_neo4j()
insights = llm.generate(context_only=True)
```

### 7. **RAG Pipeline**

**File:** `backend/services/rag_pipeline.py`

**Orchestrates:**
1. Index transactions in FAISS and Neo4j
2. Retrieve context on analysis
3. Combine vector + graph + LLM insights
4. Generate structured output

**RAG Flow:**
```
Analysis Request
    │
    ├─▶ Retrieve from FAISS (vector similarity)
    │
    ├─▶ Query Neo4j (graph patterns)
    │
    ├─▶ Combine context
    │
    └─▶ Send to LLM with context
        │
        ▼
        Generated Insights (grounded, no hallucination)
```

---

## Data Flow

### 1. **Upload Phase**

```
User selects files
         │
         ▼
POST /upload
         │
         ├─▶ Parser.parse_and_normalize()
         │        │
         │        ├─▶ Parse raw (CSV/PDF)
         │        └─▶ SchemaAdapter.detect_schema()
         │             └─▶ Canonical transactions
         │
         ▼
Response: File metadata, transaction count, schema confidence
```

### 2. **Analysis Phase**

```
POST /analyze with date range
         │
         ├─▶ Collect canonical transactions
         │
         ├─▶ Processor.process_transactions()
         │        │
         │        ├─▶ Categorize
         │        ├─▶ Normalize descriptions
         │        └─▶ Generate IDs
         │
         ├─▶ RAGPipeline.index_transactions()
         │        │
         │        ├─▶ Generate embeddings
         │        ├─▶ Store in FAISS
         │        └─▶ Store relationships in Neo4j
         │
         ├─▶ Processor.analyze_transactions()
         │        │
         │        ├─▶ Calculate summary (income, expense, net)
         │        ├─▶ Analyze categories
         │        ├─▶ Analyze trends
         │        └─▶ Generate basic insights
         │
         ├─▶ RAGPipeline.generate_insights_with_rag()
         │        │
         │        ├─▶ Build context from analysis
         │        ├─▶ Call LLM with context
         │        └─▶ Append LLM insights
         │
         ▼
Response: AnalysisResult (summary, categories, trends, insights, requires_review)
```

### 3. **Data Quality & Review**

```
For each transaction:
    ├─ Schema confidence propagated
    ├─ Row-level confidence assessed
    ├─ Ambiguities flagged
    └─ Marked for human review if:
         - Confidence < HIGH
         - Requires clarification
```

---

## Output Format

All analysis results conform to this strict JSON structure:

```json
{
  "summary": {
    "income": 12000.00,
    "expense": 8500.00,
    "net": 3500.00,
    "transaction_count": 150
  },
  "categories": [
    {
      "name": "Food & Dining",
      "amount": 2500.00,
      "count": 45,
      "percentage": 29.41
    }
  ],
  "trends": [
    {
      "date": "2024-01",
      "income": 4000.00,
      "expense": 2800.00,
      "net": 1200.00,
      "transaction_count": 50
    }
  ],
  "insights": [
    "Total income: $12,000.00",
    "Total expenses: $8,500.00",
    "Top category: Food & Dining ($2,500.00, 29.41%)"
  ],
  "requires_human_review": [
    {
      "transaction_id": "abc123",
      "date": "2024-01-15",
      "description": "Generic transaction",
      "amount": 150.00,
      "confidence": "low",
      "reason": "Low confidence in mapping"
    }
  ],
  "metadata": {
    "analysis_timestamp": "2024-01-20T10:30:00",
    "transaction_count": 150,
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    },
    "requires_review_count": 3
  }
}
```

---

## Error Handling & Robustness

### Schema Detection Failures

If schema cannot be reliably detected:
1. Set `confidence = LOW`
2. Mark transaction `requires_review = True`
3. Include `review_reason` explanation
4. Continue processing (don't fail)

### Missing Data

- Missing date → Use upload date
- Missing description → Use "Unknown Transaction"
- Invalid amount → Skip row with warning

### Backend Service Failures

- FAISS unavailable → Continue without vector search
- Neo4j unavailable → Continue without graph
- LLM unavailable → Continue without RAG insights

### Data Validation

All inputs validated using Pydantic:
```python
@dataclass
class CanonicalTransaction:
    date: str  # Must be YYYY-MM-DD
    amount: Decimal  # Must be positive
    type: str  # Must be "credit" or "debit"
```

---

## Configuration

All configuration through environment variables (`.env`):

```env
OPENAI_API_KEY=your_key
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=50
```

---

## Deployment Guide

### Prerequisites

- Python 3.9+
- OpenAI API key
- Optional: Neo4j instance
- Optional: Docker

### Setup

1. **Clone and install:**
   ```bash
   cd spent_analysis
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Run Backend:**
   ```bash
   python backend/app.py
   # Server runs on http://localhost:5000
   ```

3. **Run Frontend (separate terminal):**
   ```bash
   streamlit run frontend/app.py
   # Opens on http://localhost:8501
   ```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000 8501

CMD ["python", "backend/app.py"]
```

---

## Performance Considerations

### Embedding Generation
- Uses `text-embedding-3-small` (1536 dimensions)
- Batch processing for efficiency
- Cached embeddings in FAISS index

### Graph Queries
- Neo4j for complex relationship queries
- Pre-computed patterns for speed
- Graph caching for repeated queries

### File Parsing
- Streaming for large PDFs
- Multiple encoding fallbacks
- Error recovery without halting

---

## Security

- No hardcoded credentials (env variables)
- File upload size limits (configurable)
- Input validation on all routes
- CORS enabled for frontend communication
- Flask SECRET_KEY for session security

---

## Extensibility

### Adding New Categories

Edit `backend/services/processor.py`:
```python
CATEGORY_KEYWORDS = {
    "New Category": ["keyword1", "keyword2"],
    ...
}
```

### Custom Analysis

Extend `Processor.analyze_transactions()` with custom metrics.

### LLM Integration

Swap OpenAI for other LLM:
```python
# backend/services/llm_service.py
class LLMService:
    def __init__(self, provider="openai"):
        if provider == "anthropic":
            self.client = AnthropicClient()
```

---

## Monitoring & Logging

All components log at INFO, WARNING, and ERROR levels:

```python
logger = logging.getLogger(__name__)
logger.info(f"Processed {len(transactions)} transactions")
```

Monitor logs for:
- Schema detection failures
- Missing embeddings
- Graph connection issues

---

## Future Enhancements

1. **Multi-user support** with authentication
2. **Scheduled analysis** via message queues
3. **Predictive spending** using time-series models
4. **Expense forecasting** with ML pipelines
5. **Custom category hierarchies** per user
6. **Data export** (PDF reports, CSV)
7. **Real-time notifications** for spending alerts

---

## Summary

Spend Analyzer is a **modular, scalable, production-ready system** that handles heterogeneous financial data through intelligent schema detection and normalization, powered by RAG for context-grounded insights. The architecture ensures **no hallucination**, **high reliability**, and **easy extensibility**.