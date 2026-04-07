import logging
import asyncio
import sys
import os
import argparse

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, room_io, llm, function_tool
from livekit.plugins import noise_cancellation, silero

# Import Vector Search Components from tools package
from src.tools import (
    get_default_vector_search_tool,
    create_basic_vector_search_tool,
    SimpleVectorStore,
    SimpleEmbeddingProvider,
)


# Get custom greeting from CLI argument or environment variable
GREETING = os.getenv("GREETING_MESSAGE", "Hello! I'm an AI Orchestrator assistant. How can I help you today?")

# Logger will be configured with command-line log level in main
logger = logging.getLogger(__name__)

# Configure logging in the module itself so it's available in worker processes
def configure_logging():
    """Configure logging - called in both main and worker processes"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        force=True
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Ensure our loggers are INFO
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger('livekit.agents').setLevel(logging.INFO)
    logging.getLogger('livekit').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

# Configure at module load time
logger = configure_logging()

# Global vector store - loaded once and reused
vector_tool = None
vector_store_initialized = False


class Assistant(Agent):
    def __init__(self, vector_tool_instance=None, tools=None) -> None:
        self.vector_tool = vector_tool_instance
        super().__init__(
            instructions="""
            

Here is the **revised AI Orchestrator Agent System Prompt** with the **Knowledge Base expanded** to include **company vision, approach, methodology, and platform positioning**.

---

# AI Orchestrator Agent — System Instruction

You are the **AI Orchestrator Agent**, designed to support customers only with information related to **AI Orchestrator offerings**.

Your responses must be strictly limited to the following four service areas:

1. AI Governance Orchestrator
2. Agentic AI Design and Development
3. Training and Coaching
4. AI Orchestrator Knowledge Base (Vector Search)

You must not provide information outside these areas.

---

# PRIMARY RESPONSIBILITIES

You assist customers with the following:

---

# 1. AI Governance Orchestrator

Provide guidance on:

• AI governance frameworks and operating models
• ISO 42001 aligned governance using AI Orchestrator
• AI risk management and control mapping
• AI policy orchestration
• Human-in-the-loop governance workflows
• Responsible AI controls
• Audit readiness and compliance workflows
• AI lifecycle governance
• Enterprise AI oversight dashboards
• Governance automation using AI Orchestrator
• AI inventory and model registry governance
• AI approval workflows and control gates
• AI risk scoring and classification
• Governance for agentic AI systems
• Enterprise AI governance architecture

---

# 2. Agentic AI Design and Development

Provide guidance on:

• Agentic AI solution design
• Multi-agent orchestration architecture
• Human + AI workflow design
• AI agent use case identification
• Enterprise AI orchestration patterns
• Integration with enterprise systems
• MVP to enterprise rollout approach
• AI orchestration platform architecture
• AI agent lifecycle management
• AI solution implementation approach
• AI orchestrator reference architecture
• Tool + agent + human orchestration
• Agent governance and control design
• AI workflow orchestration patterns
• Enterprise AI operating model design

---

# 3. Training and Coaching

Provide guidance on:

• AI leadership workshops
• AI governance training
• Agentic AI training programs
• AI adoption coaching
• Enterprise AI enablement sessions
• AI Center of Excellence setup guidance
• AI literacy programs
• Role-based AI training (business, IT, audit)
• ISO 42001 awareness training
• Hands-on AI Orchestrator workshops
• AI governance implementation coaching
• AI operating model training
• Executive AI strategy sessions
• AI orchestrator platform training

---

# 4. AI Orchestrator Knowledge Base (Vector Search)

Use the knowledge base tools to retrieve and provide information about:

## Company Vision & Positioning

• AI Orchestrator vision and mission
• AI Orchestrator platform positioning
• Enterprise AI orchestration approach
• Human + AI orchestration philosophy
• AI Orchestrator value proposition
• Target customers and industries
• AI Orchestrator differentiation
• Governance-first AI adoption model

## AI Orchestrator Methodology

• AI adoption maturity model
• AI orchestration lifecycle
• Governance-first implementation approach
• MVP to scale framework
• AI operating model design
• AI CoE setup methodology
• AI transformation roadmap
• AI orchestrator implementation phases

## Platform & Architecture

• AI Orchestrator platform overview
• AI Governance Orchestrator architecture
• Multi-agent orchestration architecture
• Human-AI workflow engine
• AI governance control layer
• Enterprise integration architecture
• AI lifecycle management architecture

## Services & Offerings

• AI Orchestrator service offerings
• Implementation packages
• Advisory services
• Governance services
• Agentic AI development services
• Training offerings
• Deployment approach

## Use Cases

• Enterprise AI governance use cases
• Agentic AI use cases
• Human + AI workflow use cases
• AI automation with governance
• AI audit and compliance use cases

---

# KNOWLEDGE BASE TOOL USAGE

You have access to vector search tools:

Use when:
• Customer asks about AI Orchestrator
• Customer asks about platform
• Customer asks about methodology
• Customer asks about approach
• Customer asks about company vision
• Customer asks about services
• Customer asks about architecture

Available tools:

search_knowledge_base
→ Retrieve relevant documents

get_answer
→ Generate contextual answer with sources

Requirements:
• Use knowledge base for factual responses
• Prefer knowledge base over assumptions
• Include source attribution when relevant
• Provide structured enterprise responses

---

# OUT OF SCOPE

If a question is not related to:

• AI Governance Orchestrator
• Agentic AI Design and Development
• Training and Coaching
• AI Orchestrator knowledge base

Respond with:

"I can assist only with AI Orchestrator services including AI Governance Orchestrator, Agentic AI Design & Development, and Training & Coaching. Please ask a related question."

---

# BEHAVIOR RULES

You must:

• Stay within AI Orchestrator offerings only
• Position AI Orchestrator as the primary approach
• Avoid generic AI consulting outside AI Orchestrator
• Avoid unrelated technology discussions
• Avoid speculation outside defined services
• Ask clarifying questions if needed
• Keep responses structured and enterprise-focused
• Emphasize governance-first AI adoption
• Emphasize human + AI orchestration
• Align responses to enterprise AI transformation

---

# RESPONSE STYLE

Tone:
Professional, structured, enterprise-focused, advisory.

Response Structure:

When appropriate, use:

• Overview
• How AI Orchestrator Helps
• Approach
• Architecture (if relevant)
• Deliverables
• Next Steps

Always anchor answers to **AI Orchestrator platform + services**.


           """,
            tools=tools or [],
        )


server = AgentServer()


async def initialize_vector_store():
    """Initialize vector store and load documents once at startup"""
    global vector_tool, vector_store_initialized
    
    logger.info("DEBUG: Entering initialize_vector_store()")
    
    if vector_store_initialized:
        logger.info("Vector store already initialized, reusing...")
        return vector_tool
    
    logger.info("=" * 60)
    logger.info("🔧 INITIALIZING VECTOR STORE")
    logger.info("=" * 60)
    
    try:
        # Create a new vector search tool with backends
        logger.info("DEBUG: Creating vector search tool...")
        vector_tool = create_basic_vector_search_tool(config={
            "index_path": "live_agent_index",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_dim": 384,
        })
        logger.info("DEBUG: Vector search tool created")
        
        # Configure backends
        logger.info("DEBUG: Configuring backends...")
        vector_store = SimpleVectorStore(save_path="live_agent_index.json")
        embedding_provider = SimpleEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        logger.info("DEBUG: Backends created")
        
        logger.info("DEBUG: Setting backends to vector tool...")
        vector_tool.set_vector_store(vector_store)
        vector_tool.set_embedding_provider(embedding_provider)
        logger.info("✓ Vector search tool initialized with backends")
        
        # Load documents from the documents folder
        docs_folder = os.getenv("DOCS_FOLDER", "./documents")
        logger.info(f"DEBUG: Checking docs folder: {docs_folder}")
        
        if os.path.exists(docs_folder):
            logger.info(f"Loading documents from {docs_folder}...")
            try:
                logger.info("DEBUG: Calling load_documents_from_folder()...")
                result = await vector_tool.load_documents_from_folder(docs_folder)
                logger.info(f"DEBUG: load_documents_from_folder returned: {result[:100] if result else 'None'}")
                
                import json
                result_data = json.loads(result)
                
                if result_data.get("loaded", 0) > 0:
                    logger.info(f"✓ Successfully loaded {result_data['loaded']} documents")
                    logger.info(f"  Failed: {result_data.get('failed', 0)}")
                else:
                    logger.warning(f"No documents loaded from {docs_folder}")
                    if "error" in result_data:
                        logger.warning(f"  Error: {result_data['error']}")
                        
            except Exception as e:
                logger.warning(f"Failed to load documents: {e}", exc_info=True)
        else:
            logger.info(f"Documents folder not found at {docs_folder}")
        
        vector_store_initialized = True
        logger.info("✓ Vector store initialization complete\n")
        return vector_tool
        
    except Exception as e:
        logger.error(f"❌ Error in initialize_vector_store: {e}", exc_info=True)
        raise


async def load_documents(vector_tool):
    """Load documents using the vector tool's folder loading capability"""
    logger.info("\n" + "=" * 60)
    logger.info("📚 LOADING DOCUMENTS")
    logger.info("=" * 60)
    
    docs_folder = os.getenv("DOCS_FOLDER", "./documents")
    if os.path.exists(docs_folder):
        logger.info(f"Loading documents from {docs_folder}...")
        try:
            result = await vector_tool.load_documents_from_folder(docs_folder)
            import json
            result_data = json.loads(result)
            
            if result_data.get("loaded", 0) > 0:
                logger.info(f"✓ Successfully loaded {result_data['loaded']} documents")
            else:
                logger.warning(f"No documents loaded from {docs_folder}")
                if "error" in result_data:
                    logger.warning(f"  Error: {result_data['error']}")
                    
        except Exception as e:
            logger.warning(f"Failed to load documents: {e}")
    else:
        logger.info(f"Documents folder not found at {docs_folder}")


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    """Default handler for all incoming RTC sessions"""
    global vector_tool
    
    logger.info("=" * 80)
    logger.info("🤖 AGENT SESSION STARTED")
    logger.info("=" * 80)
    logger.info(f"📍 Room: {ctx.room.name}")
    logger.info("")
    
    logger.info("🔧 Initializing agent components...")
    logger.info("  ✓ STT: deepgram/nova-3:multi")
    logger.info("  ✓ LLM: openai/gpt-4o")
    logger.info("  ✓ TTS: cartesia/sonic-3")
    logger.info("  ✓ Vector Search: SimpleVectorStore (AI Orchestrator Knowledge Base)")
    logger.info("")
    
    # Initialize vector store once (first session only)
    logger.info(f"DEBUG: vector_tool global state: {vector_tool is None}")
    if vector_tool is None:
        logger.info("📚 Loading vector search tool (first time)...")
        logger.info("DEBUG: About to call initialize_vector_store()")
        try:
            result = await initialize_vector_store()
            logger.info(f"DEBUG: initialize_vector_store() returned: {result is not None}")
            logger.info("✓ Vector store loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}", exc_info=True)
            raise
    else:
        logger.info("✓ Using cached vector store")
    
    logger.info(f"DEBUG: vector_tool after init: {vector_tool is not None}")
    
    # Define tool functions for the LLM using the shared vector store
    logger.info("Creating LLM tools...")
    
    # Create tools using the correct @function_tool() decorator pattern
    @function_tool()
    async def search_knowledge_base(query: str) -> str:
        """Search the AI Orchestrator knowledge base for relevant information using semantic similarity."""
        try:
            logger.info(f"🔍 Searching knowledge base: '{query}'")
            result = await vector_tool.search_knowledge_base(query, top_k=3)
            logger.info(f"✓ Found results ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search error: {str(e)}"
    
    @function_tool()
    async def get_answer_with_sources(query: str) -> str:
        """Search knowledge base and generate an answer with source attribution."""
        try:
            logger.info(f"💡 Generating answer for: '{query}'")
            result = await vector_tool.get_answer(query)
            logger.info(f"✓ Answer generated ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error: {str(e)}"
    
    logger.info("✓ Vector search tools created successfully")
    
    # Create assistant with vector tool instance AND tools passed to Agent.__init__
    assistant = Assistant(
        vector_tool_instance=vector_tool,
        tools=[search_knowledge_base, get_answer_with_sources],
    )
    
    session = AgentSession(
        stt="deepgram/nova-3:multi",
        llm="openai/gpt-4o",
        tts="cartesia/sonic-3",
        vad=silero.VAD.load(),
    )
    
    logger.info("✓ Session created with vector search tools enabled (via Agent)")

    logger.info("")
    logger.info("🔌 Starting session on room...")
    try:
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=lambda params: noise_cancellation.BVC(),
                ),
            ),
        )
        logger.info("✓ Session successfully started on room")
    except Exception as e:
        logger.error(f"❌ Failed to start session: {e}", exc_info=True)
        raise
    
    logger.info("")
    logger.info("🎤 Agent is now LISTENING for user input...")
    logger.info("  ✓ STT listening enabled")
    logger.info("  ✓ LLM processing enabled")
    logger.info("  ✓ TTS response enabled")
    logger.info("  ✓ Vector search tool available")
    logger.info("")
    
    # Session.start() sets up the full interaction loop internally
    # It automatically:
    # - Listens for speech via STT
    # - Processes with LLM + vector search
    # - Responds with TTS
    # Just keep it alive
    try:
        logger.info("Entering main session loop...")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Agent session interrupted by user")
    except Exception as e:
        logger.error(f"Error in agent session: {e}", exc_info=True)
    finally:
        logger.info("Agent session ended")


if __name__ == "__main__":
    # Re-configure logging here as well to ensure it's set up
    logger = configure_logging()
    
    logger.info("=" * 80)
    logger.info("🚀 Starting AI Orchestrator Agent")
    logger.info("=" * 80)
    logger.info(f"Environment: LiveKit URL = {os.getenv('LIVEKIT_URL')}")
    logger.info(f"Documents folder: {os.getenv('DOCS_FOLDER', './documents')}")
    logger.info("")
    
    # Run the LiveKit agent server
    agents.cli.run_app(server)