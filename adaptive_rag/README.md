# Adaptive RAG System

A production-ready Retrieval-Augmented Generation (RAG) framework that intelligently routes queries to optimal retrieval strategies while providing advanced prompt enhancement and LLM integration.

## Overview

The Adaptive RAG system combines multiple retrieval methods with intelligent query analysis to provide accurate, context-aware responses. It adapts its retrieval strategy based on query characteristics, ensuring optimal performance across diverse use cases.

### Key Features

- **Intelligent Query Analysis**: Analyzes query intent, type, entities, domain, and complexity to recommend optimal retrieval strategies
- **Multiple Retrieval Strategies**: 
  - Vector similarity search (FAISS)
  - Hybrid search (vector + keyword)
  - Graph-based retrieval (Neo4j knowledge graphs)
  - SQL-based structured data retrieval
  - Web search (Tavily API)
- **Adaptive Routing**: Automatically selects the best retrieval strategy based on query analysis
- **Prompt Enhancement**: Uses LLM to improve and iterate on prompts for better results
- **Centralized Configuration**: Singleton pattern ensures single LLM initialization and consistent configuration
- **Dependency Injection**: Clean architecture pattern for testability and modularity
- **Multi-hop Query Decomposition**: Handles complex queries by breaking them into sub-queries
- **Fallback Strategies**: Graceful degradation when primary retrieval methods fail
- **Confidence Scoring**: Evaluates answer quality with configurable thresholds

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) Neo4j instance for knowledge graph features
- (Optional) Tavily API key for web search features

### Setup

1. **Clone the repository**
   ```bash
   cd adaptive_rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your credentials:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LLM_MODEL=gpt-3.5-turbo
   EMBED_MODEL=text-embedding-3-small
   CONFIDENCE_THRESHOLD=0.6
   MIN_DOCS_THRESHOLD=3
   
   # Optional: Knowledge Graph
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   
   # Optional: Web Search
   TAVILY_API_KEY=your_tavily_api_key
   
   # Logging
   LOG_LEVEL=INFO
   ```

## Quick Start

### Basic Usage

```python
from rag_init import initialize_rag_system

# Initialize the system once
system = initialize_rag_system()

# Use the orchestrator to process queries
query = "What are the latest developments in AI?"
result = system["orchestrator"].orchestrate(query)

print(f"Query: {result['query']}")
print(f"Retrieved Strategy: {result['retrieval_strategy']}")
print(f"Answer: {result['answer']}")
```

### Component Usage

```python
from app_config import get_config
from enhance_prompt import PromptEnhancer
from query_analysis import QueryAnalyzer
from query_orchestrator import QueryOrchestrator

# Get global configuration
config = get_config()
llm = config.get_llm()

# Initialize components
enhancer = PromptEnhancer(llm=llm)
analyzer = QueryAnalyzer(llm=llm)
orchestrator = QueryOrchestrator(llm=llm)

# Analyze a query
query = "How does quantum computing work?"
analysis = analyzer.analyze(query)
print(f"Intent: {analysis['intent']}")
print(f"Recommended Strategy: {analysis['recommended_retrieval_strategy']}")

# Enhance a prompt
original_prompt = "Explain quantum computing"
enhanced = enhancer.enhance_prompt(original_prompt)
print(f"Enhanced: {enhanced}")
```

## Project Structure

```
adaptive_rag/
├── README.md                    # This file
├── ARCHITECTURE.md              # System architecture and design
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── app_config.py               # Centralized configuration & singleton LLM
├── query_analysis.py           # Query intent & strategy analysis
├── enhance_prompt.py           # Prompt improvement and scoring
├── query_orchestrator.py       # Query routing and orchestration
├── rag_init.py                 # System initialization example
├── vector_search.py            # FAISS-based vector similarity search
├── graph_search.py             # Neo4j knowledge graph retrieval
└── web_search_retriever.py     # Tavily web search integration
```

## Core Modules

### `app_config.py`
Centralized configuration management using the singleton pattern. Ensures LLM is initialized once and reused throughout the application.

**Key Features:**
- Environment variable loading via `python-dotenv`
- Singleton LLM instance caching
- Separate generation LLM with configurable temperature
- Logging configuration

**Usage:**
```python
from app_config import get_config

config = get_config()
llm = config.get_llm()  # Access cached LLM instance
```

### `query_analysis.py`
Analyzes user queries to determine optimal retrieval strategy and understand query characteristics.

**Analysis Outputs:**
- `intent`: informational, analytical, lookup, action
- `query_type`: factual, reasoning, multi-hop, ambiguous, comparison
- `entities`: extracted entities and concepts
- `domain`: technology, finance, legal, healthcare, general knowledge
- `complexity`: simple, moderate, complex
- `recommended_retrieval_strategy`: optimal strategy for the query
- `rewrite_query`: improved version of the query
- `sub_queries`: breakdown for multi-hop queries
- `confidence_score`: analysis confidence

**Usage:**
```python
from query_analysis import QueryAnalyzer

analyzer = QueryAnalyzer(llm=llm)
analysis = analyzer.analyze("What is the impact of quantum computing on cryptography?")
```

### `enhance_prompt.py`
Improves user prompts using LLM reflection and iteration.

**Features:**
- Weakness analysis
- Prompt rewriting
- Score-based improvement
- Multi-iteration enhancement
- Streamlit UI integration

**Usage:**
```python
from enhance_prompt import PromptEnhancer

enhancer = PromptEnhancer(llm=llm)
improved = enhancer.enhance_prompt("Explain AI")
```

### `query_orchestrator.py`
Coordinates the entire retrieval and generation pipeline based on query analysis.

**Retrieval Strategies:**
- `VectorRetriever`: FAISS-based similarity search
- `HybridRetriever`: Combined vector and keyword search
- `GraphRetriever`: Neo4j knowledge graph traversal
- `SQLRetriever`: Structured database queries
- `WebSearchRetriever`: Tavily web search

**Generation:**
- `LLMGenerator`: Creates answers from retrieved context

**Features:**
- Strategy routing based on query analysis
- Multi-hop decomposition
- Fallback strategy management
- Context aggregation
- Answer generation with confidence scoring

**Usage:**
```python
from query_orchestrator import QueryOrchestrator

orchestrator = QueryOrchestrator(llm=llm, query_analyzer=analyzer)
result = orchestrator.orchestrate("What are recent AI breakthroughs?")
print(result['answer'])
```

### `vector_search.py`
Implements vector similarity search using FAISS and OpenAI embeddings.

**Features:**
- Document chunking with configurable overlap
- FAISS index creation and management
- OpenAI text embeddings
- Similarity-based document retrieval

### `graph_search.py`
Implements knowledge graph-based retrieval using Neo4j and graphiti.

**Features:**
- Automatic entity extraction
- Knowledge graph ingestion via graphiti
- Entity-based query routing
- Relationship-aware traversal

### `web_search_retriever.py`
Integrates Tavily API for real-time web search.

**Features:**
- Semantic web search
- Real-time indexing
- Advanced search ranking
- Structured result extraction

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `LLM_MODEL` | gpt-3.5-turbo | LLM model name |
| `EMBED_MODEL` | text-embedding-3-small | Embedding model for vector search |
| `CONFIDENCE_THRESHOLD` | 0.6 | Minimum confidence score (0-1) |
| `MIN_DOCS_THRESHOLD` | 3 | Minimum documents to retrieve |
| `NEO4J_URI` | - | Neo4j connection URI |
| `NEO4J_USER` | neo4j | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| `TAVILY_API_KEY` | - | Tavily API key |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Usage Examples

### Example 1: Multi-hop Query
```python
query = "Compare quantum computing approaches and their applications in cryptography"
result = system["orchestrator"].orchestrate(query)

# System automatically:
# 1. Detects multi-hop nature
# 2. Decomposes into sub-queries
# 3. Retrieves relevant documents for each sub-query
# 4. Generates comprehensive answer
```

### Example 2: Domain-Specific Query
```python
query = "What are the FDA approval processes for new medications?"
result = system["orchestrator"].orchestrate(query)

# System detects healthcare domain and uses appropriate retrieval strategy
```

### Example 3: Prompt Enhancement Workflow
```python
original = "Tell me about machine learning"
enhanced = system["prompt_enhancer"].enhance_prompt(original)
# Returns: "Provide a comprehensive overview of machine learning including supervised learning, 
# unsupervised learning, and reinforcement learning with key algorithms and applications."
```

## Architecture Highlights

### Dependency Injection
All components accept injected dependencies, enabling:
- Easy testing with mock objects
- Single LLM initialization
- Flexible configuration
- Consistent state management

### Singleton Pattern
The `AppConfig` class uses the singleton pattern to ensure:
- LLM initialized once
- Configuration consistent across application
- Minimal resource usage

### Strategy Pattern
Retrieval strategies implement a common interface (`BaseRetriever`), allowing:
- Easy addition of new strategies
- Runtime strategy selection
- Graceful fallbacks

## Running Tests

```bash
# Unit tests (when available)
pytest tests/

# Integration tests
python -m pytest tests/integration/
```

## Performance Considerations

- **LLM Caching**: Single LLM instance reused across components
- **Embedding Caching**: Consider caching embeddings for repeated queries
- **Index Optimization**: FAISS indexes can be optimized for specific hardware
- **Batch Processing**: Support for batch query processing
- **Timeout Management**: Configurable timeouts for LLM calls

## Troubleshooting

### "LangChain ChatOpenAI not available"
- Ensure `langchain` and `langchain-openai` are installed
- Verify OpenAI API key is set
- Check Python version compatibility

### "TAVILY_API_KEY not set"
- Web search will not be available
- Other retrieval strategies will still work
- Get API key from https://tavily.com

### "Neo4j connection failed"
- Knowledge graph features will not be available
- Other retrieval strategies will still work
- Check Neo4j connection URI and credentials

### Low confidence scores
- Increase `MIN_DOCS_THRESHOLD` for more context
- Adjust `CONFIDENCE_THRESHOLD` based on use case
- Consider domain-specific fine-tuning

## API Reference

### Core Classes

#### AppConfig
```python
config = AppConfig()
llm = config.get_llm()  # Get cached LLM instance
llm_gen = config.get_llm_generator(temperature=0.7)  # New generation LLM
```

#### QueryAnalyzer
```python
analyzer = QueryAnalyzer(llm=llm)
analysis = analyzer.analyze(query: str) -> Dict[str, Any]
```

#### PromptEnhancer
```python
enhancer = PromptEnhancer(llm=llm)
improved = enhancer.enhance_prompt(prompt: str) -> str
score = enhancer.score_prompt(prompt: str) -> float
```

#### QueryOrchestrator
```python
orchestrator = QueryOrchestrator(llm=llm, query_analyzer=analyzer)
result = orchestrator.orchestrate(query: str) -> Dict[str, Any]
# result contains: query, retrieval_strategy, documents, answer, metadata
```

## Contributing

To extend the system:

1. **Add a new retrieval strategy**: Implement `BaseRetriever`
2. **Customize LLM behavior**: Override `app_config.py` methods
3. **Add new analysis features**: Extend `query_analysis.py`
4. **Integrate new data sources**: Create new retriever classes

## License

[Add your license information here]

## Support

For issues and questions:
- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design
- Review example usage in [rag_init.py](rag_init.py)
- Check logs with appropriate `LOG_LEVEL` environment variable

## References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Tavily API](https://tavily.com/)
