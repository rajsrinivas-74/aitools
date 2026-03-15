"""Factory for creating retriever instances.

This module decouples the QueryOrchestrator from concrete retriever implementations.
It provides a single place to instantiate retrievers based on strategy names.

Design Pattern: Factory Pattern
- Reduces coupling between orchestrator and concrete retrievers
- Single responsibility: retriever instantiation
- Easy to add new retrievers without modifying orchestrator
"""

import logging
from typing import Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retriever implementations.
    
    Defines the interface that all retrievers must implement.
    Enables polymorphism and decoupling from concrete implementations.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        pass

    @abstractmethod
    def get_context_blocks(self, query: str, top_k: int = 5) -> list:
        """Get structured context blocks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of ContextBlock objects
        """
        pass

    @abstractmethod
    def generate_response(self, query: str) -> str:
        """Generate a response for a query using this retriever.
        
        Args:
            query: User query
            
        Returns:
            Generated response string
        """
        pass


class RetrieverFactory:
    """Factory for creating retriever instances.
    
    This class manages the instantiation of retrievers and maintains
    a registry of available retriever types. It decouples the
    QueryOrchestrator from concrete retriever implementations.
    
    Usage:
        factory = RetrieverFactory()
        vector_retriever = factory.create_retriever("vector search", config_dict)
        graph_retriever = factory.create_retriever("graph retrieval", config_dict)
    """

    # Registry of retriever constructors
    _retrievers: Dict[str, callable] = {}

    @classmethod
    def register(cls, strategy_name: str, retriever_class):
        """Register a retriever class for a strategy.
        
        Args:
            strategy_name: Name of the strategy (e.g., "vector search")
            retriever_class: The retriever class to instantiate
            
        Example:
            RetrieverFactory.register("vector search", VectorRetriever)
        """
        cls._retrievers[strategy_name] = retriever_class
        logger.info(f"Registered retriever for strategy: {strategy_name}")

    @classmethod
    def create_retriever(cls, strategy_name: str, **config) -> BaseRetriever:
        """Create a retriever instance for the given strategy.
        
        Args:
            strategy_name: Name of the retrieval strategy
            **config: Configuration parameters for the retriever
            
        Returns:
            Configured retriever instance
            
        Raises:
            ValueError: If strategy is not registered
        """
        if strategy_name not in cls._retrievers:
            available = ", ".join(cls._retrievers.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available strategies: {available}"
            )

        retriever_class = cls._retrievers[strategy_name]
        instance = retriever_class(**config)
        logger.info(f"Created retriever for strategy: {strategy_name}")
        return instance

    @classmethod
    def list_strategies(cls) -> list:
        """Get list of registered strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._retrievers.keys())

    @classmethod
    def reset(cls):
        """Clear all registered retrievers. Useful for testing."""
        cls._retrievers.clear()
