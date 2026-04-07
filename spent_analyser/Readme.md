# FinSight AI - Spending Analysis Application

A comprehensive financial analysis application that leverages AI and advanced data processing to provide deep insights into spending patterns, transaction analysis, and financial health assessment.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Database Setup](#database-setup)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

FinSight AI is an intelligent financial analysis platform designed to help users understand their spending patterns through advanced AI-powered insights. The application processes financial transactions, analyzes spending behavior, identifies trends, and provides actionable recommendations.

**Core Capabilities:**
- Transaction data upload and parsing (PDF/CSV support)
- AI-powered financial analysis with GPT-4 integration
- Real-time transaction classification and categorization
- Spending pattern visualization and reporting
- Neo4j graph database for relationship analysis
- FAISS vector embeddings for semantic search
- Comprehensive financial health assessment

## ✨ Key Features

### Frontend
- **Interactive Dashboard**: Streamlit-based responsive UI with dark theme
- **Financial Summary**: Card-based visualization of key metrics (debits, credits, net balance, averages)
- **Transaction Analysis**: Detailed transaction breakdown with filtering and search
- **Spending Patterns**: Visual charts showing spending trends and category analysis
- **Executive Summaries**: AI-generated insights and recommendations
- **Real-time Processing**: 120-second timeout for comprehensive analysis operations

### Backend
- **REST API**: Flask-based API with multiple analytical endpoints
- **LLM Integration**: OpenAI GPT-4 for intelligent analysis and insights
- **Vector Search**: FAISS-powered semantic search on transactions
- **Graph Analytics**: Neo4j integration for relationship and pattern analysis
- **Multi-mode Analysis**: Simple vs. Comprehensive analysis modes
- **Singleton Pattern**: Optimized Neo4j initialization to prevent redundant connections

### Data Processing
- **PDF Parser**: Extract financial transactions from PDF documents
- **Multi-format Support**: Handle various transaction formats
- **Data Validation**: Comprehensive transaction validation and cleaning
- **Schema Adaptation**: Flexible schema handling for different data sources
- **Batch Processing**: Efficient processing of large transaction datasets

## 🏗️ Architecture

### Quick Overview

FinSight AI uses a **modular, layered architecture** with clear separation between data processing, analysis, and presentation layers.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                      │
│  - UI Components  - Session Management  - Data Display      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────────┐
│              Backend API (Flask)                             │
│  - /analyze    - /query    - /upload    - /health           │
└──────┬────────────────────────┬──────────────────┬───────────┘
       │                        │                  │
   ┌───▼──────┐          ┌──────▼────────┐    ┌──▼────────────┐
   │  Services │          │  Data Layer   │    │ Auth Layer   │
   │  - RAG    │          │  - Parser     │    │ - JWT        │
   │  - LLM    │          │  - Processor  │    │ - Session    │
   │  - Vector │          │  - Adapter    │    │ - Credentials│
   └───┬──────┘          └──────┬────────┘    └──┬────────────┘
       │                        │                  │
   ┌───▼───────────────┬────────▼──────┐         │
   │                   │                │         │
┌──▼─────────┐  ┌─────▼────────┐  ┌───▼──────┐  │
│   Neo4j     │  │  FAISS       │  │ OpenAI   │  │
│  (Graph DB) │  │ (Vector DB)  │  │  (LLM)   │  │
└─────────────┘  └──────────────┘  └──────────┘  │
                                                  │
                                          └───────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit, HTML/CSS | User interface and visualization |
| **Backend** | Flask, Python | REST API and business logic |
| **LLM** | OpenAI GPT-4 | AI-powered analysis and insights |
| **Graph DB** | Neo4j | Transaction relationships and patterns |
| **Vector DB** | FAISS | Semantic search and embeddings |
| **Logging** | Python logging | System monitoring (ERROR level default) |

### Detailed Architecture

For comprehensive architecture documentation including component breakdown, data flows, design patterns, and deployment strategies, see the [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md) document.

## 📦 Prerequisites

- **Python**: 3.9 or higher
- **Neo4j**: 4.0+ (local or cloud instance)
- **OpenAI API Key**: For GPT-4 access
- **System Memory**: 4GB+ recommended
- **Disk Space**: 2GB+ for embeddings and cache

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd spent_analysis
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Setup Script

```bash
bash setup.sh
```

This script will:
- Create necessary directories (indexes, uploads)
- Initialize configuration files
- Set up logging structure
- Validate environment setup

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Application Settings
LOG_LEVEL=ERROR
FLASK_ENV=production
STREAMLIT_SERVER_PORT=8501
```

### Settings File

Configure application behavior in `config/settings.py`:

```python
# Logging
LOG_LEVEL = "ERROR"  # ERROR, WARNING, INFO, DEBUG

# Timeouts
ANALYSIS_TIMEOUT = 120  # seconds
QUERY_TIMEOUT = 120     # seconds

# LLM Configuration
LLM_MODEL = "gpt-4"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# Neo4j Configuration
NEO4J_BATCH_SIZE = 100
FAISS_INDEX_DIMENSION = 1536
```

## 💻 Usage

### Starting the Application

#### Backend Server

```bash
# In one terminal
python -m flask --app backend.app run
# Server runs on http://localhost:5000
```

#### Frontend Application

```bash
# In another terminal
streamlit run frontend/app.py
# UI accessible at http://localhost:8501
```

### Quick Start Workflow

1. **Upload Data**
   - Click "Upload Transaction Data" in the sidebar
   - Select PDF or CSV file containing transactions
   - System extracts and validates transactions

2. **Analyze Spending**
   - Select analysis mode (Simple or Comprehensive)
   - Choose LLM enhancement (optional for comprehensive)
   - Click "Analyze" button
   - View Financial Summary, patterns, and insights

3. **Query Transactions**
   - Use the Query section to ask questions about spending
   - Natural language queries processed by LLM
   - Results returned with semantic search

4. **Export Reports**
   - Download analysis results
   - Export transaction classifications
   - Generate spending reports

### API Usage Examples

#### Analyze Endpoint

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [...],
    "mode": "comprehensive",
    "use_llm_for_enhancement": true
  }'
```

#### Query Endpoint

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are my top spending categories?",
    "context": {...}
  }'
```

## 🔌 API Endpoints

### Analysis Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/analyze` | Comprehensive spending analysis |
| POST | `/query` | Natural language transaction queries |
| POST | `/upload` | Upload transaction data |
| GET | `/health` | API health check |

### Request/Response Format

**Analysis Request:**
```json
{
  "transactions": [
    {
      "date": "2024-01-01",
      "amount": "150.00",
      "description": "Grocery Store",
      "type": "debit"
    }
  ],
  "mode": "comprehensive",
  "use_llm_for_enhancement": true
}
```

**Analysis Response:**
```json
{
  "financial_summary": {
    "total_debit": 5000.00,
    "total_credit": 8000.00,
    "net_balance": 3000.00,
    "average_transaction": 125.50
  },
  "executive_summary": "Positive spending pattern with...",
  "key_findings": ["Finding 1", "Finding 2"],
  "spending_habits": ["Habit 1", "Habit 2"],
  "recommendations": ["Recommendation 1"],
  "risk_alerts": ["Alert 1"]
}
```

## 📁 Project Structure

```
spent_analysis/
├── backend/                          # Flask backend server
│   ├── app.py                       # Main Flask application
│   ├── auth.py                      # Authentication logic
│   ├── models/                      # Data models
│   ├── routes/                      # API route handlers
│   ├── services/                    # Business logic services
│   │   ├── llm_service.py          # LLM (GPT-4) integration
│   │   ├── rag_pipeline.py         # RAG and Neo4j pipeline
│   │   ├── faiss_store.py          # Vector search with FAISS
│   │   ├── parser.py               # Parsing logic
│   │   ├── processor.py            # Data processing
│   │   └── schema_adapter.py       # Schema conversion
│   └── utils/                       # Utility functions
│
├── frontend/                         # Streamlit frontend
│   └── app.py                       # Main Streamlit interface
│
├── config/                           # Configuration files
│   └── settings.py                  # Application settings
│
├── indexes/                          # FAISS vector indexes
├── uploads/                          # Uploaded data storage
│
├── docs/                             # Documentation
│   ├── architecture.md              # System architecture
│   ├── FINANCIAL_ANALYSIS_GUIDE.md  # Analysis methodology
│   └── E2E_FLOW_REVIEW.md          # End-to-end flow documentation
│
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Initial setup script
├── config.json                       # Configuration file
└── README.md                         # This file
```

## 🗄️ Database Setup

### Neo4j Setup

1. **Local Installation**
   ```bash
   # Install Neo4j Community Edition
   # Download from https://neo4j.com/download-center/
   # Start the service
   neo4j start
   ```

2. **Cloud Setup (Neo4j Aura)**
   ```bash
   # Create instance at https://neo4j.com/cloud/aura/
   # Use connection string in .env file
   ```

3. **Initialize Database**
   ```python
   # Automatic schema creation on first connection
   # Transaction nodes and relationships created as needed
   ```

### FAISS Index Setup

- Indexes created automatically in `indexes/` directory
- Vector embeddings (1536 dimensions) generated via OpenAI
- Persistent serialization for quick startup

## 🐛 Troubleshooting

### Common Issues

#### 1. Financial Summary Not Displaying
**Solution**: Clear caches and restart Streamlit
```bash
streamlit cache clear
streamlit run frontend/app.py
```

#### 2. Neo4j Connection Errors
**Solution**: Verify Neo4j URI and credentials
```bash
# Check connection
neo4j status
# Update .env with correct credentials
```

#### 3. OpenAI API Errors
**Solution**: Verify API key and rate limits
```bash
# Test API connectivity
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 4. Timeout Issues
**Solution**: Increase timeout in settings
```python
# In config/settings.py
ANALYSIS_TIMEOUT = 180  # Increase from 120
```

#### 5. Memory Issues with Large Files
**Solution**: Process in batches or increase system memory
```bash
# Increase virtual memory
sudo fallocate -l 2G /swapfile
```

### Logging and Debugging

**Enable Debug Logging:**
```python
# In config/settings.py
LOG_LEVEL = "DEBUG"
```

**View Backend Logs:**
```bash
tail -f backend.log
```

**Frontend Console:**
Press 'c' in Streamlit terminal and check the console output

## 🤝 Contributing

### Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Run syntax validation: `python -m py_compile <file>`
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push and create Pull Request

### Code Standards

- **Python**: PEP 8 compliance
- **Naming**: Descriptive, snake_case for functions/variables
- **Comments**: Document complex logic
- **Error Handling**: Use try-catch with meaningful messages
- **Logging**: Use appropriate log levels (ERROR, WARNING, INFO, DEBUG)

### Testing

```bash
# Run tests
python -m pytest tests/

# Test syntax
python -m py_compile frontend/app.py
python -m py_compile backend/*.py
```

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Support

For issues, questions, or suggestions:
1. Check the [documentation](docs/) directory
2. Review [architecture guide](docs/architecture.md)
3. Check [troubleshooting section](#troubleshooting)
4. Open an issue with detailed description

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Core analysis pipeline
- ✅ Neo4j graph integration
- ✅ FAISS vector search
- ✅ Streamlit frontend
- ✅ LLM-powered insights
- ✅ Singleton pattern optimization
- ✅ Logging enhancements (ERROR level default)
- ✅ Financial Summary visual redesign

---

**Last Updated**: March 2026  
**Status**: Production Ready