# Module Dependency Architecture

## Dependency Graph (After Refactoring)

```
┌─────────────────────────────────────────────────────────────────┐
│                       DEPENDENCIES                              │
│                    (Few, Well-Defined)                          │
└─────────────────────────────────────────────────────────────────┘

LAYER 0: CONFIGURATION & UTILITIES (Foundation)
═══════════════════════════════════════════════════════════════
        
        ┌──────────────────┐
        │   app_config     │  (Singleton: Environment, LLM setup)
        │  (Central Hub)   │
        └────────┬─────────┘
                 │
                 │ provides: config, llm instances
                 │
        ┌────────▼─────────────────────────────┐
        │                                       │
        │       rag_utils.py (Utilities)       │
        │  ┌─────────────────────────────┐    │
        │  │ ▸ chunk_text()           │    │
        │  │ ▸ load_env_file()        │    │
        │  │ ▸ log_calls decorator    │    │
        │  │ ▸ EmbeddingModel ABC     │    │
        │  │ ▸ OpenAIEmbedding        │    │
        │  │ ▸ ContextBlock (data)    │    │
        │  │ ▸ BaseRetriever ABC      │    │
        │  │ ▸ generate_response...() │    │
        │  └─────────────────────────────┘    │
        │       (No external dependencies)     │
        └──────────────────────────────────────┘


LAYER 1: CORE LOGIC (Business Logic)
═══════════════════════════════════════════════════════════════

        ┌────────────────────┐    ┌────────────────────┐
        │ enhance_prompt.py  │    │ query_analysis.py  │
        │  ▸ PromptEnhancer  │    │  ▸ QueryAnalyzer   │
        │  ▸ Reflection      │    │  ▸ Domain inference│
        │  ▸ Improvement     │    │  ▸ Strategy suggest│
        │                    │    │                    │
        │  (Uses: config,    │    │ (Uses: config,     │
        │   rag_utils)       │    │  rag_utils,        │
        │                    │    │  enhance_prompt)   │
        └────────┬───────────┘    └────────┬───────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │   retriever_factory.py (Factory) NEW!    │
        │  ┌──────────────────────────────────┐   │
        │  │ ▸ RetrieverFactory (Registry)    │   │
        │  │ ▸ BaseRetriever (ABC)            │   │
        │  │ ▸ .register(strategy, class)     │   │
        │  │ ▸ .create_retriever(strategy)    │   │
        │  └──────────────────────────────────┘   │
        │                                          │
        │ (Decouples Orchestrator from concrete   │
        │  retriever implementations)              │
        └──────────────────────────────────────────┘


LAYER 2: CONCRETE IMPLEMENTATIONS (Pluggable Strategy Classes)
═══════════════════════════════════════════════════════════════

    ┌────────────────────────┐  ┌────────────────────────┐  ┌──────────────────┐
    │  vector_search.py      │  │  graph_search.py       │  │web_search_        │
    │                        │  │                        │  │retriever.py       │
    │ ┌────────────────────┐ │  │ ┌────────────────────┐ │  │                  │
    │ │ VectorRetriever    │ │  │ │ GraphRetriever     │ │  │WebSearchRetriever│
    │ │  (Implements ABC)  │ │  │ │  (Implements ABC)  │ │  │ (Implements ABC) │
    │ │                    │ │  │ │                    │ │  │                  │
    │ │ ▸ retrieve()       │ │  │ ▸ retrieve()       │ │  │▸ retrieve()      │
    │ │ ▸ get_context_...()│  │ │ ▸ get_context_...()│ │  │▸ get_context_...()
    │ │ ▸ generate_...()   │ │  │ ▸ generate_...()   │ │  │▸ generate_...()  │
    │ │                    │ │  │ │                    │ │  │                  │
    │ │ Uses: FAISS, OpenAI│ │  │ │ Uses: Neo4j, OpenAI│ │  │Uses: Tavily, OpenAI
    │ └────────────────────┘ │  │ └────────────────────┘ │  │                  │
    │                        │  │                        │  │                  │
    │ (Uses: rag_utils,      │  │ (Uses: rag_utils,     │  │(Uses: rag_utils, │
    │  app_config)           │  │  app_config)          │  │ app_config)      │
    └────────────────────────┘  └────────────────────────┘  └──────────────────┘
             ▲                              ▲                        ▲
             │                              │                        │
             └──────────────┬───────────────┴────────────────────────┘
                            │
                            │ registered with factory
                            │


LAYER 3: ORCHESTRATION (Routing & Coordination)
═══════════════════════════════════════════════════════════════

        ┌────────────────────────────────────────────────────┐
        │         query_orchestrator.py                       │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ QueryOrchestrator                            │  │
        │  │  ▸ orchestrate(query)                        │  │
        │  │  ▸ _analyze_and_route_query()                │  │
        │  │  ▸ _execute_single_retriever_path()          │  │
        │  │  ▸ _execute_multi_retriever_path()           │  │
        │  │  ▸ _synthesize_from_aggregated_context()     │  │
        │  │  ▸ _retrieve_with_strategy()                 │  │
        │  │  ▸ _select_retrieval_query()                 │  │
        │  │  ▸ AggregatedContext (data class)            │  │
        │  │                                               │  │
        │  │ NO HARD DEPENDENCIES ON CONCRETE CLASSES!    │  │
        │  │ Uses Factory for dynamic retriever creation  │  │
        │  └──────────────────────────────────────────────┘  │
        │                                                     │
        │ (Uses: app_config, enhance_prompt, query_analysis,│
        │  rag_utils, retriever_factory)                    │
        └──────────────────────────────────────────────────┬─┘
                                                           │
                                                           │ Creates via factory
                                                           │ (decoupled pattern)
                                                           ▼


LAYER 4: INITIALIZATION & USAGE
═══════════════════════════════════════════════════════════════

        ┌─────────────────────────────────────────┐
        │        rag_init.py (Example)             │
        │  ▸ initialize_rag_system()               │
        │  ▸ Wires all components together         │
        │                                          │
        │ (Uses: app_config, enhance_prompt,      │
        │  query_analysis, query_orchestrator)    │
        └──────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
DEPENDENCY STATISTICS
═══════════════════════════════════════════════════════════════════════════════

BEFORE REFACTORING:
  - query_orchestrator imports: VectorRetriever, GraphRetriever, WebSearchRetriever
  - Coupling: TIGHT (concrete imports)
  - Extensibility: HARD (must edit orchestrator for new retrievers)
  - Dead code: ~300 lines
  - Cohesion: LOW (mixed concerns in rag_utils)

AFTER REFACTORING:
  - query_orchestrator imports: None (uses factory)
  - Coupling: LOOSE (abstract factory pattern)
  - Extensibility: EASY (register with factory, no changes elsewhere)
  - Dead code: ZERO
  - Cohesion: HIGH (7 focused sections)


═══════════════════════════════════════════════════════════════════════════════
KEY ARCHITECTURAL IMPROVEMENTS
═══════════════════════════════════════════════════════════════════════════════

✅ FACTORY PATTERN
   Benefit: Add new retrievers without modifying orchestrator
   
   Example:
   ```python
   # In your_new_retriever.py
   class ElasticsearchRetriever(BaseRetriever):
       ...
   
   # Register it anywhere
   RetrieverFactory.register("elasticsearch", ElasticsearchRetriever)
   
   # Orchestrator automatically works with it!
   result = orchestrator.orchestrate(query)
   ```

✅ NO CIRCULAR DEPENDENCIES
   Benefit: Clear dependency flow, easy to test
   
   Before: retriever_factory ← → query_orchestrator (circular)
   After:  retriever_factory ← query_orchestrator (one-way)

✅ CLEAN SEPARATION OF CONCERNS
   Benefit: Each module has single responsibility
   
   vvv Routing Logic
   ┌──────────────────────────┐
   │  query_orchestrator.py   │
   └──────────────────────────┘
              ↓
   ┌──────────────────────────┐
   │  retriever_factory.py    │ ← Creation Logic
   └──────────────────────────┘
              ↓
   ┌──────────────────────────┐
   │  vector/graph/web search │ ← Retrieval Logic
   └──────────────────────────┘

✅ TESTABILITY IMPROVED
   Benefit: Easy to mock, test in isolation
   
   ```python
   # Mock retriever for testing
   factory.register("test", MockRetriever)
   orchestrator = QueryOrchestrator(...)
   # Test without external dependencies
   ```

═══════════════════════════════════════════════════════════════════════════════
ADDING A NEW RETRIEVER (After Refactoring)
═══════════════════════════════════════════════════════════════════════════════

1. Create new file: pinecone_search.py
   ```python
   from rag_utils import BaseRetriever, ContextBlock
   
   class PineconeRetriever(BaseRetriever):
       def retrieve(self, query, top_k=5):
           # Implementation
           pass
       
       def get_context_blocks(self, query, top_k=5):
           # Implementation
           pass
       
       def generate_response(self, query):
           # Implementation
           pass
   ```

2. Register with factory: pinecone_search.py
   ```python
   from retriever_factory import RetrieverFactory
   
   RetrieverFactory.register("pinecone", PineconeRetriever)
   ```

3. Use in config or app:
   ```python
   config = {
       "STRATEGY": "pinecone",
       "PINECONE_API_KEY": "..."
   }
   orchestrator = QueryOrchestrator(**config)
   ```

4. DONE! No changes to orchestrator, factory uses reflection!

This is the power of the Factory Pattern!

═══════════════════════════════════════════════════════════════════════════════
