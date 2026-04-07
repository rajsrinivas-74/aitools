# AI Orchestrator - LiveKit Vector Search Agent
## Application Summary

---

## Overview

The AI Orchestrator is a real-time voice-enabled AI agent built on the LiveKit platform that integrates semantic document search with conversational AI. The application enables intelligent voice-based interactions with a knowledge base through vector search, allowing users to ask questions and receive contextual answers backed by indexed documents.

**Primary Purpose:** Combine real-time voice communication with semantic document retrieval to create an intelligent conversational agent that can answer questions using knowledge from indexed documents.

---

## Core Features

### 1. Voice-Based Interaction
- Real-time speech-to-text (STT) using Deepgram
- LLM processing using OpenAI GPT-4
- Text-to-speech (TTS) using Cartesia
- Complete voice pipeline integration with LiveKit

### 2. Vector Search Capabilities
- Semantic document search across knowledge base
- Multi-similarity algorithms for document retrieval
- Query embedding and vector matching
- Configurable top-k result retrieval (default: 4 results)

### 3. Knowledge Base Integration
- Automatic document loading from `documents/` folder
- Multi-format support: PDF, TXT, MD, JSON, YAML
- Automatic indexing and vector embedding generation
- Persistent index storage in JSON format

### 4. LLM Integration
- Context-aware response generation
- Integration with OpenAI GPT-4
- Knowledge base context injection into prompts
- Noise cancellation and audio processing

### 5. Dynamic Operations
- Works with any LiveKit room without hardcoded configuration
- Session lifecycle management
- Async concurrent operations
- Persistent vector index (auto-generated)

---

## Architecture Overview

### Data Flow
```
User (Voice Call)
    ↓
LiveKit Cloud (Real-time Communication)
    ├→ STT (Deepgram): Voice → Text
    ├→ Agent Logic: Processing & Context
    ├→ TTS (Cartesia): Text → Voice
    └→ Vector Search Tool: Context Retrieval

Vector Search Pipeline
    ├→ Query Processing
    ├→ Embedding Generation (sentence-transformers)
    ├→ Vector Similarity Search
    ├→ Document Retrieval
    └→ Result Ranking
```

### Component Architecture
- **client.py** - Main agent entry point, orchestrates STT/LLM/TTS pipeline
- **VectorSearchAgentTool** - Core service for semantic search with dependency injection
- **SimpleVectorStore** - In-memory vector storage with JSON persistence
- **SimpleEmbeddingProvider** - Embedding generation using sentence-transformers
- **Tools Package** - Modular tools for document indexing and search

---

## Technology Stack

### Runtime & Framework
- **Language:** Python 3.9+
- **Real-time Communication:** LiveKit Agents (v1.5.1)
- **Async Runtime:** asyncio (stdlib)
- **Environment:** Python venv

### Voice & Communication
- **STT:** Deepgram
- **LLM:** OpenAI (GPT-4)
- **TTS:** Cartesia
- **Noise Cancellation:** Silero

### Vector & Search
- **Vector Store:** SimpleVectorStore (in-memory with JSON persistence)
- **Embeddings:** sentence-transformers
- **Similarity:** Cosine similarity
- **Indexing:** Custom implementation

### Audio Processing
- **sounddevice** - Audio I/O
- **soundfile** - Audio file operations
- **librosa** - Audio analysis
- **numpy/scipy** - Numerical operations

---

## Directory Structure

```
livekit/
├── client.py                           # Main agent entry point
├── README.md                           # User documentation
├── requirements.txt                    # Python dependencies
├── docs/
│   └── ARCHITECTURE.md                # Detailed architecture docs
├── src/
│   └── tools/
│       ├── __init__.py                # Package exports
│       ├── vector_search_agent_tool.py    # Core vector search
│       ├── simple_vector_store.py         # Vector storage implementation
│       ├── simple_embedding_provider.py   # Embedding generation
│       └── demo_vector_search.py          # Standalone demo
├── tests/
│   ├── test_flow.py                   # Test workflows
│   └── run_test.sh                    # Test execution script
├── documents/                          # Knowledge base (user-provided documents)
├── config/                             # Configuration files
└── live_agent_index.json              # Auto-generated vector index
```

---

## Key Classes & Components

### VectorSearchAgentTool
**Purpose:** Central service for semantic search operations
- Loads and indexes documents from knowledge base
- Generates embeddings using SimpleEmbeddingProvider
- Performs vector similarity search
- Returns ranked search results with scores

**Key Methods:**
- `search(query, top_k)` - Semantic search with ranking
- `index_documents()` - Build vector index from documents
- `add_document()` - Add single document to index
- Supports async operations for concurrent processing

### SimpleVectorStore
**Purpose:** In-memory vector storage backend
- Stores document vectors and metadata
- Implements cosine similarity search
- Persists index to JSON file
- Provides CRUD operations for vectors

**Features:**
- Cosine similarity calculation
- Top-k search retrieval
- JSON serialization/deserialization
- Vector metadata tracking

### SimpleEmbeddingProvider
**Purpose:** Generate embeddings for documents and queries
- Uses sentence-transformers library
- Provides consistent embedding generation
- Fallback mechanisms for edge cases
- Integration with vector store

### Assistant (Agent Class)
**Purpose:** Main LiveKit agent implementation
- Extends Agent base class from LiveKit SDK
- Registers function tools for vector search
- Manages STT/LLM/TTS pipeline
- Handles session lifecycle and agent commands

---

## Workflow & Execution Flow

### Agent Initialization
1. Load environment variables (API keys, LiveKit config)
2. Initialize vector search tool with document folder
3. Build vector index from documents in `documents/` folder
4. Register the agent with LiveKit using AgentServer
5. Wait for incoming voice calls

### Call Handling
1. User initiates voice call to LiveKit room
2. Agent joins room using room credentials
3. Enable STT for speech-to-text conversion
4. Pass user query to LLM with vector search context
5. LLM generates response using retrieved documents
6. Convert response to speech via TTS
7. Stream audio back to user through LiveKit

### Vector Search Process
1. User query received from transcript
2. Generate embedding for query
3. Search vector store for similar vectors
4. Retrieve top-k documents based on cosine similarity
5. Rank results by relevance score (0-1)
6. Return context blocks to LLM for answer generation
7. LLM synthesizes response using document context

---

## Configuration & Requirements

### Environment Variables
```
LIVEKIT_URL=wss://your-livekit-cloud-url
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
OPENAI_API_KEY=your-openai-key
GREETING_MESSAGE=custom-greeting (optional)
```

### Dependencies
- livekit (1.1.5) - LiveKit SDK
- livekit-agents (1.5.1) - Agents framework
- livekit-plugins-openai - OpenAI integration
- livekit-plugins-deepgram - Speech-to-text
- livekit-plugins-cartesia - Text-to-speech
- livekit-plugins-silero - Noise cancellation
- python-dotenv - Environment management
- sentence-transformers - Embedding generation
- numpy/scipy - Numerical operations
- librosa - Audio processing

---

## Use Cases

### 1. Customer Support Agent
- Training documents indexed as knowledge base
- Customers call agent with questions
- Agent searches relevant docs and provides answers
- Reduces support workload

### 2. Document Q&A System
- PDF/document-based knowledge base
- Users query information verbally
- Agent retrieves and synthesizes answers
- Multi-format document support

### 3. Internal Knowledge Assistant
- Company policies and procedures indexed
- Employees call for guidance on processes
- Agent provides context-aware information
- Real-time voice interface

### 4. Domain Expert Assistant
- Research papers or technical documentation indexed
- Users ask questions about domain topics
- Agent provides citations and context
- Semantic search ensures relevance

---

## Performance Characteristics

### Scalability
- Single-process agent (per room)
- In-memory vector store for fast search
- JSON persistence for index durability
- Async operations for concurrent processing

### Search Performance
- Vector search: O(n) cosine similarity comparison
- Typical query response: < 2 seconds
- Configurable top-k (default: 4) for relevance/speed tradeoff
- JSON persistence minimal impact on retrieval

### Limitations
- Single vector store instance per process
- In-memory storage limited by available RAM
- No distributed indexing
- JSON index file for persistence

---

## Integration Points

### LiveKit Integration
- AgentServer registration
- AgentSession management
- room_io for audio streaming
- STT/LLM/TTS plugin system

### Document Loading
- Supports local filesystem paths
- Multi-format file parsing
- Automatic format detection
- Extensible document loader architecture

### External Services
- OpenAI API for LLM
- Deepgram API for STT
- Cartesia API for TTS
- LiveKit Cloud for real-time communication

---

## Development Workflow

### Local Testing
```bash
# Activate environment
source .livekit/bin/activate

# Run standalone demo
python3 -m src.tools.demo_vector_search

# Run full agent
python3 src/client.py

# Run test suite
bash tests/run_test.sh
```

### Adding Documents
1. Place PDF/TXT/MD files in `documents/` folder
2. Agent automatically reindexes on startup
3. New documents available for search immediately

### Extending Search
- Implement custom VectorStore (ABC)
- Replace SimpleVectorStore in initialization
- Custom similarity algorithms
- Different embedding providers

---

## Key Design Patterns

### 1. Dependency Injection
- VectorSearchAgentTool accepts VectorStore and EmbeddingProvider
- Enables flexible backend swapping
- Easy testing with mock implementations

### 2. Abstract Base Classes
- VectorStore (ABC) - pluggable storage backends
- EmbeddingProvider (ABC) - pluggable embedding sources
- Loose coupling between components

### 3. Async-First Design
- Async/await throughout pipeline
- Non-blocking I/O operations
- Concurrent query processing
- Responsive agent performance

### 4. Dataclass Models
- SearchResult - structured search output
- ContextBlock - formatted context for LLM
- Type safety and IDE support

---

## Security Considerations

### API Keys
- Loaded from environment variables
- Never hardcoded in source
- .env.local for local development
- Environment-based configuration for deployment

### Document Access
- Documents stored locally in `documents/` folder
- No external document repository access
- Vector index stored locally (JSON file)
- Search scoped to indexed documents

### Voice & Privacy
- Audio processed in-memory
- No persistent call recording
- Transcripts used for context only
- LiveKit room isolation

---

## Future Enhancement Opportunities

1. **Distributed Vector Store** - Multi-node indexing
2. **Persistent Database** - Replace JSON with database
3. **Advanced Embeddings** - Fine-tuned embedding models
4. **Multi-Language Support** - Translation integration
5. **Document Updates** - Hot reload without restart
6. **Analytics** - Query logging and performance metrics
7. **Custom Tools** - Extensible function_tool system
8. **Memory** - Session history and context retention

---

## Troubleshooting & Common Issues

### Vector Index Not Loading
- Verify `documents/` folder exists and contains files
- Check file formats are supported
- Review logs for embedding errors

### Poor Search Results
- Verify documents have relevant content
- Adjust top-k parameter
- Check similarity threshold
- Consider embedding model limitations

### Agent Not Responding
- Verify OpenAI API key is valid
- Check LiveKit credentials
- Confirm network connectivity
- Review agent logs for errors

### Performance Issues
- Monitor vector store size
- Profile similarity search timing
- Consider caching frequent queries
- Reduce top-k if necessary

---

## Conclusion

The AI Orchestrator LiveKit Vector Search Agent provides a production-ready platform for building intelligent voice-based conversational AI systems with semantic document search. The modular architecture, comprehensive technology integration, and flexible configuration options make it suitable for a wide range of voice-enabled applications requiring knowledge-base integration.
