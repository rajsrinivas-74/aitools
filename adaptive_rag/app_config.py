"""
Centralized Configuration Module

Manages environment variables, LLM initialization, and global configuration
using the singleton pattern to ensure single initialization and reuse.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class AppConfig:
    """Singleton configuration manager for the Adaptive RAG system."""
    
    _instance = None
    _llm_instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        if self._initialized:
            return
        
        # Load environment variables
        env_file = os.getenv("ENV_FILE", ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from system environment
        
        # Core configuration - LLM Models
        # Default models for different components (can be overridden via environment)
        self.llm_model = os.getenv("LLM_MODEL", "gpt-5")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Deprecated: Use self.llm_model instead
        self.model = self.llm_model
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # RAG configuration
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        self.min_docs_threshold = int(os.getenv("MIN_DOCS_THRESHOLD", "3"))
        
        # Generation configuration
        self.generation_temperature = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
        
        # External services
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # Paths
        self.vector_index_path = os.getenv("VECTOR_INDEX_PATH", "faiss_index")
        
        # Logging
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        self._initialized = True
        logger.info(f"AppConfig initialized with model: {self.llm_model}")
    
    def get_default_llm_model(self) -> str:
        """Get the default LLM model from configuration."""
        return self.llm_model
    
    def get_default_embedding_model(self) -> str:
        """Get the default embedding model from configuration."""
        return self.embedding_model
    
    def get_llm(self):
        """
        Get or create the singleton LLM instance.
        
        Returns:
            Initialized LLM instance (cached)
        """
        if AppConfig._llm_instance is None:
            try:
                from langchain_openai import ChatOpenAI
                
                AppConfig._llm_instance = ChatOpenAI(
                    model_name=self.llm_model,
                    temperature=self.generation_temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=self.openai_api_key,
                )
                logger.info(f"LLM instance created: {self.llm_model}")
            except ImportError:
                logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
                raise RuntimeError("LangChain OpenAI is not installed")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise RuntimeError(f"LLM initialization failed: {e}")
        
        return AppConfig._llm_instance
    
    def reset_llm(self):
        """Reset the cached LLM instance (useful for testing)."""
        AppConfig._llm_instance = None
        logger.info("LLM instance reset")


def get_config() -> AppConfig:
    """
    Get the singleton AppConfig instance.
    
    Returns:
        AppConfig singleton instance
    """
    return AppConfig()
