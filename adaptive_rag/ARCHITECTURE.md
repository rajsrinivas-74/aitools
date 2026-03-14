# Adaptive RAG System - Architecture Guide

This document provides a comprehensive overview of the Adaptive RAG system's architecture, design patterns, and component interactions.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Configuration Management](#configuration-management)
7. [Extensibility](#extensibility)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)

## System Overview

The Adaptive RAG system is a modular, component-based architecture designed to:

- **Analyze** incoming queries to understand their characteristics
- **Route** queries to optimal retrieval strategies based on analysis
- **Retrieve** relevant documents from multiple sources
- **Generate** accurate answers using LLM with retrieved context
- **Score** confidence and quality of results

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Query Analysis       │
        │  (Intent, Domain,      │
        │   Complexity...)       │
        └────────┬───────────────┘
                 │
                 ▼
     ┌───────────────────────────┐
     │  Strategy Recommendation  │
     │  • Vector Search          │
     │  • Hybrid Search          │
     │  • Graph Retrieval        │
     │  • SQL Queries            │
     │  • Web Search             │
     └────────┬──────────────────┘
              │
              ▼
    ┌─────────────────────────────┐
    │  Query Orchestrator         │
    │                             │
    │  Route & Execute            │
    │  Fallback Management        │
    │  Context Aggregation        │
    └────────┬────────────────────┘
             │
    ┌────────┴────────────────┬──────────────┬────────────┐
    │                         │              │            │
    ▼                         ▼              ▼            ▼
┌──────────┐          ┌──────────────┐ ┌──────────┐ ┌──────────┐
│ Vector   │          │ Graph DB     │ │ SQL DB   │ │ Web      │
│ Index    │          │ (Neo4j)      │ │ (SQLite) │ │ Search   │
│ (FAISS)  │          │              │ │          │ │ (Tavily) │
└──────────┘          └──────────────┘ └──────────┘ └──────────┘
    │                         │              │            │
    └────────┬────────────────┴──────────────┴────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ LLM Generator       │
    │                     │
    │ Generate Answer     │
    │ with Context        │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Confidence Scoring  │
    │                     │
    │ Quality Assessment  │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Result Bundle       │
    │                     │
    │ {query, strategy,   │
    │  documents,         │
    │  answer,            │
    │  confidence}        │
    └─────────────────────┘
```

## Core Architecture

### Layered Architecture

The system follows a clean, layered architecture:

```
┌─────────────────┐
│   User Layer    │  (Streamlit UI, CLI)
├─────────────────┤
│ Orchestration   │  (QueryOrchestrator, LLMGenerator)
│    Layer        │
├─────────────────┤
│  Core Logic     │  (QueryAnalyzer, PromptEnhancer)
│    Layer        │
├─────────────────┤
│  Retrieval      │  (VectorRetriever, GraphRetriever, etc.)
│    Layer        │
├─────────────────┤
│ Integration     │  (FAISS, Neo4j, OpenAI, Tavily)
│    Layer        │
├─────────────────┤
│ Configuration   │  (AppConfig, Environment)
│    Layer        │
└─────────────────┘
```

### Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                    AppConfig (Singleton)                 │
│              (LLM, Configuration, Environment)           │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌─────────┐  ┌──────────┐  ┌──────────────┐
    │ QueryAn │  │ Prompt   │  │ QueryOrch.   │
    │ alyzer  │  │ Enhancer │  │              │
    └────┬────┘  └─────────┘   └──────┬───────┘
         │                            │
         │        ┌──────────────────┴──────────┐
         │        │                             │
         ▼        ▼                             ▼
    ┌──────────────────────────┐      ┌─────────────────┐
    │   Retriever Classes      │      │ LLM Generator   │
    │  • VectorRetriever       │      │                 │
    │  • HybridRetriever       │      └─────────────────┘
    │  • GraphRetriever        │
    │  • SQLRetriever          │
    │  • WebSearchRetriever    │
    └──────────────────────────┘
```

## Component Design

### 1. AppConfig - Singleton Configuration Manager

**Purpose**: Centralized configuration and LLM management

**Pattern**: Singleton

**Key Responsibilities**:
- Load environment variables
- Initialize LLM once and cache globally
- Provide configuration access to all components
- Setup logging

**Implementation Details**:
```python
class AppConfig:
    _instance = None  # Singleton instance
    _llm = None       # Cached LLM
    _llm_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
```

**Initialization Guarantee**:
- LLM initialized exactly once (first call)
- All subsequent calls return cached instance
- Thread-safe lazy initialization

### 2. QueryAnalyzer - Intent and Strategy Detection

**Purpose**: Analyze queries to understand characteristics and recommend strategies

**Pattern**: Strategy Pattern (recommends different approaches)

**Key Responsibilities**:
- Analyze query intent (informational, analytical, lookup, action)
- Determine query type (factual, reasoning, multi-hop, etc.)
- Extract entities and concepts
- Identify domain (technology, finance, healthcare, etc.)
- Assess complexity (simple, moderate, complex)
- Recommend optimal retrieval strategy
- Rewrite queries for clarity
- Decompose multi-hop queries
- Provide confidence scores

**Output Structure**:
```python
{
    "query": "original_query",
    "intent": "informational|analytical|lookup|action",
    "query_type": "factual|reasoning|multi-hop|ambiguous|comparison",
    "entities": ["entity1", "entity2"],
    "domain": "technology|finance|legal|healthcare|general",
    "complexity": "simple|moderate|complex",
    "recommended_retrieval_strategy": "vector|hybrid|graph|sql|web|multi-step",
    "rewrite_query": "improved_query",
    "sub_queries": ["subquery1", "subquery2"],
    "confidence_score": 0.85
}
```

**LLM Integration**:
- Uses LLM for intelligent analysis
- Parses JSON responses
- Handles parsing errors gracefully

### 3. PromptEnhancer - Prompt Optimization

**Purpose**: Improve user prompts for better results

**Pattern**: Reflection and Iteration

**Key Responsibilities**:
- Analyze prompt weaknesses
- Generate improved versions
- Score prompt quality
- Iterate on improvements
- Handle Streamlit UI integration

**Process Flow**:
```
Input Prompt
    │
    ├─→ Weakness Analysis (LLM reflection)
    │
    ├─→ Improvement Generation (LLM rewrite)
    │
    ├─→ Quality Scoring (LLM evaluation)
    │
    └─→ Output Prompt
```

**Caching Strategy**:
- Maintains model cache for reuse
- Avoids redundant LLM calls

### 4. QueryOrchestrator - Pipeline Orchestration

**Purpose**: Coordinate entire retrieval and generation pipeline

**Pattern**: Orchestrator Pattern with Strategy Pattern

**Key Responsibilities**:
- Route queries to appropriate retrievers
- Manage falling back strategies
- Aggregate context from multiple sources
- Handle multi-hop decomposition
- Generate answers via LLMGenerator
- Provide result metadata

**Core Components**:

#### BaseRetriever (Abstract)
```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        pass
```

#### Concrete Retrievers

1. **VectorRetriever**
   - Uses FAISS for similarity search
   - Requires pre-built vector index
   - Fast retrieval (O(log n))
   - Best for semantic similarity

2. **HybridRetriever**
   - Combines vector + keyword search
   - Uses BM25 + vector scores
   - Better recall than vector alone
   - Balanced precision/recall

3. **GraphRetriever**
   - Leverages Neo4j knowledge graphs
   - Entity-driven retrieval
   - Relationship-aware results
   - Best for interconnected data

4. **SQLRetriever**
   - Queries structured databases
   - Supports complex WHERE clauses
   - Best for tabular/structured data
   - High precision

5. **WebSearchRetriever**
   - Real-time web search via Tavily
   - Latest information
   - Structured result extraction
   - Best for current events/recent data

**Orchestration Logic**:
```
1. Analyze Query (QueryAnalyzer)
2. Select Strategy (from analysis result)
3. Execute Retrieval:
   - Try primary retriever
   - On failure, try fallback
   - Aggregate results
4. Generate Answer (LLMGenerator)
5. Score Confidence
6. Return Result Bundle
```

**Output Structure**:
```python
{
    "query": "original_query",
    "retrieval_strategy": "selected_strategy",
    "documents": [
        {
            "id": "doc_id",
            "content": "document_content",
            "score": 0.95,
            "source": "vector|graph|web|sql"
        }
    ],
    "answer": "generated_answer",
    "metadata": {
        "documents_count": 5,
        "confidence_score": 0.87,
        "strategy_used": "hybrid",
        "fallback_used": False,
        "processing_time_ms": 1234
    }
}
```

### 5. LLMGenerator - Answer Generation

**Purpose**: Generate answers from retrieved context

**Key Responsibilities**:
- Format context for LLM
- Generate responses
- Confidence scoring
- Handle edge cases

**Design**:
- Higher temperature (0.7) than analysis LLM (0.0)
- Separate instance for generation
- Context-aware prompting

## Data Flow

### Query Processing Flow

```
1. INPUT PHASE
   ┌──────────────────┐
   │  Raw User Query  │
   └────────┬─────────┘

2. ANALYSIS PHASE
   ┌──────────────────────────────────┐
   │  QueryAnalyzer.analyze(query)    │
   │  • Intent detection              │
   │  • Entity extraction             │
   │  • Strategy recommendation       │
   │  • Query rewriting               │
   │  • Sub-query decomposition       │
   └────────┬───────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────┐
   │  Analysis Result                  │
   │  {intent, domain, strategy, ...}  │
   └────────┬───────────────────────────┘

3. ROUTING PHASE
   ┌──────────────────────────┐
   │  Select Retriever        │
   │  Based on Strategy       │
   └────────┬─────────────────┘

4. RETRIEVAL PHASE
   ┌──────────────────────┐
   │  Execute Retriever   │
   │  • Vector Search     │
   │  • Graph Query       │
   │  • Web Search        │
   │  • SQL Query         │
   │  • or Hybrid         │
   └────────┬──────────────┘
            │
            ▼
   ┌──────────────────────┐
   │  Retrieved Documents │
   │  [doc1, doc2, doc3]  │
   └────────┬──────────────┘

5. CONTEXT PHASE
   ┌────────────────────────────┐
   │  Aggregate & Format        │
   │  Context for LLM           │
   │  • Deduplicate docs        │
   │  • Rank by relevance       │
   │  • Truncate to token limit │
   └────────┬───────────────────┘

6. GENERATION PHASE
   ┌────────────────────────────┐
   │  LLMGenerator.generate()   │
   │  • Format prompt           │
   │  • Call LLM                │
   │  • Parse response          │
   │  • Score confidence        │
   └────────┬───────────────────┘

7. OUTPUT PHASE
   ┌──────────────────────────────┐
   │  Result Bundle               │
   │  {query, documents, answer,  │
   │   metadata}                  │
   └──────────────────────────────┘
```

### Dependency Initialization Flow

```
1. Application Start
   │
   ▼
2. Import modules
   │
   ├─→ app_config.py imported
   │   └─→ AppConfig class loaded
   │   └─→ Singleton pattern ready
   │
   ├─→ Other modules imported
   │   └─→ get_config() available
   │
   ▼
3. First get_config() call
   │
   ├─→ AppConfig.__new__() called
   │
   ├─→ Check: _instance is None?
   │   ├─→ YES: Create instance
   │   │   └─→ Call _initialize()
   │   │       └─→ Load ENV vars
   │   │       └─→ Load LLM (0.0 temp)
   │   │
   │   └─→ NO: Return existing instance
   │
   ▼
4. All modules get same AppConfig instance
   │
   ├─→ QueryAnalyzer uses config.get_llm()
   ├─→ PromptEnhancer uses config.get_llm()
   ├─→ QueryOrchestrator uses config.get_llm()
   │
   ▼
5. Cached LLM instance reused
```

## Design Patterns

### 1. Singleton Pattern

**Implementation**: AppConfig

**Benefits**:
- Ensures single LLM initialization
- Consistent configuration across app
- Resource efficiency
- Thread-safe lazy loading

**Usage**:
```python
config1 = AppConfig()
config2 = AppConfig()  # Same instance as config1
```

### 2. Strategy Pattern

**Implementation**: BaseRetriever and concrete classes

**Benefits**:
- Easy to add new retrieval strategies
- Runtime strategy selection
- Loose coupling
- Open/Closed Principle

**Usage**:
```python
strategies = {
    "vector": VectorRetriever(),
    "graph": GraphRetriever(),
    "web": WebSearchRetriever()
}

strategy = strategies[recommended_strategy]
documents = strategy.retrieve(query)
```

### 3. Dependency Injection

**Implementation**: Constructor injection throughout

**Benefits**:
- Testability with mock objects
- Loose coupling
- Single responsibility
- Configuration flexibility

**Usage**:
```python
class QueryAnalyzer:
    def __init__(self, llm=None, prompt_enhancer=None):
        self.llm = llm or get_config().get_llm()
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer()
```

### 4. Orchestrator Pattern

**Implementation**: QueryOrchestrator

**Benefits**:
- Coordinates complex workflows
- Encapsulates business logic
- Easy process modification
- Clear separation of concerns

### 5. Template Method Pattern

**Implementation**: LLMGenerator.generate()

**Benefits**:
- Consistent answer generation flow
- Reusable steps
- Easy to override specific steps

## Configuration Management

### Environment Variables

```
OPENAI_API_KEY           # Required
LLM_MODEL               # Default: gpt-3.5-turbo
EMBED_MODEL             # Default: text-embedding-3-small
CONFIDENCE_THRESHOLD    # Default: 0.6
MIN_DOCS_THRESHOLD      # Default: 3
NEO4J_URI               # Optional
NEO4J_USER              # Default: neo4j
NEO4J_PASSWORD          # Optional
TAVILY_API_KEY          # Optional
LOG_LEVEL               # Default: INFO
```

### Configuration Loading

```python
# 1. Find .env file in current directory
_ = load_dotenv(find_dotenv())

# 2. Access via os.environ
openai_key = os.environ.get("OPENAI_API_KEY")

# 3. Config available via AppConfig
config = get_config()
print(config.openai_api_key)
```

### Dynamic Configuration

Override defaults in code:

```python
# Example: Using different model
from app_config import AppConfig
config = AppConfig()
config.model = "gpt-4"
```

## Extensibility

### Adding a New Retrieval Strategy

1. **Implement BaseRetriever**:
```python
class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Implementation
        return [{"id": "...", "content": "...", "score": 0.95}]
```

2. **Register in QueryOrchestrator**:
```python
self.retrievers = {
    "vector": VectorRetriever(),
    "custom": CustomRetriever(),
    # ...
}
```

3. **Update QueryAnalyzer** to recommend new strategy:
```python
# Modify prompt template to include new strategy
```

### Extending Query Analysis

1. **Add new fields to analysis output**:
```python
# In query_analysis.py, extend prompt template
"- new_field: description"
```

2. **Parse new field in orchestrator**:
```python
# Use analysis["new_field"] for routing
```

### Custom LLM Integration

1. **Create custom config class**:
```python
class CustomConfig(AppConfig):
    def _init_llm(self):
        # Use different LLM (Claude, LLaMA, etc.)
        from custom_llm import CustomChatModel
        return CustomChatModel()
```

2. **Use in app**:
```python
from custom_config import CustomConfig
config = CustomConfig()
```

## Error Handling

### LLM Initialization Errors

```python
try:
    llm = config.get_llm()
except RuntimeError as e:
    logger.error(f"LLM initialization failed: {e}")
    # Fallback or exit gracefully
```

### Retrieval Failures

```python
try:
    documents = retriever.retrieve(query)
except Exception as e:
    logger.warning(f"Primary strategy failed, trying fallback")
    documents = fallback_retriever.retrieve(query)
```

### JSON Parsing Errors

```python
try:
    analysis = json.loads(response)
except json.JSONDecodeError:
    # Return default analysis
    logger.warning("Failed to parse LLM response")
    analysis = default_analysis()
```

## Performance Optimization

### 1. LLM Caching

**Current**: Single instance per temperature
```python
llm_analysis = config.get_llm()  # temperature=0.0
llm_generation = config.get_llm_generator()  # temperature=0.7
```

**Future**: Token caching, batch processing

### 2. Embedding Caching

**Current**: OpenAI embeddings computed on-demand
```python
# In vector_search.py
embeddings = openai.Embedding.create(input=texts)
```

**Future**: Cache embeddings in local store

### 3. Index Optimization

**Current**: FAISS index creation
```python
# In vector_search.py
index = faiss.IndexFlatL2(dimension)
```

**Future**: GPU support, HNSW approximation

### 4. Parallel Retrieval

**Current**: Sequential strategy execution
```python
documents = selected_retriever.retrieve(query)
```

**Future**: Parallel multi-strategy retrieval
```python
documents = await asyncio.gather(
    vector_retriever.retrieve(query),
    graph_retriever.retrieve(query),
    web_retriever.retrieve(query)
)
```

### 5. Result Ranking

**Current**: Individual strategy ranking
```python
result["score"] = 0.95 - i*0.1
```

**Future**: Cross-strategy ranking
```python
# Merge and rank results from multiple strategies
ranked = rank_cross_strategy(results)
```

## Monitoring and Logging

### Logging Strategy

**Configured in AppConfig**:
```python
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)
```

**Log Levels**:
- `DEBUG`: Detailed component behavior
- `INFO`: High-level operations
- `WARNING`: Recoverable issues
- `ERROR`: Critical failures

**Key Log Points**:
```
INFO: config initialized
INFO: LLM initialized
INFO: QueryAnalyzer: analyzing query
INFO: QueryOrchestrator: executing strategy
INFO: LLMGenerator: generating answer
ERROR: Retrieval failed, using fallback
```

## Security Considerations

1. **API Key Management**:
   - Load from environment only
   - Never log API keys
   - Use .env files, not hardcoded

2. **Input Validation**:
   - Validate query format
   - Sanitize LLM responses
   - Protect against prompt injection

3. **Access Control**:
   - Implement authentication (if exposed as API)
   - Rate limiting
   - Usage monitoring

## Future Enhancements

1. **Async/Concurrent Processing**
   - Parallel strategy execution
   - Async LLM calls
   - Stream responses

2. **Advanced Caching**
   - Query result caching
   - Embedding caching
   - LLM response caching

3. **Multi-LLM Support**
   - Switch between models
   - Ensemble approaches
   - Cost optimization

4. **User Feedback Loop**
   - Collect answer ratings
   - Fine-tune strategy selection
   - Improve confidence scoring

5. **Observability**
   - Tracing requests
   - Performance metrics
   - Error tracking

6. **Advanced Retrieval**
   - Re-ranking algorithms
   - Query expansion
   - Semantic search improvements

7. **Streaming Responses**
   - Stream LLM output
   - Real-time result updates
   - Improved UX

## Deployment Considerations

### Single Container
```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "enhance_prompt.py"]
```

### Microservices
```
- API Service (FastAPI)
- Orchestration Worker
- Cache Layer (Redis)
- Message Queue (Celery)
```

### Scalability
- Stateless design
- Shared configuration
- Horizontal scaling support
