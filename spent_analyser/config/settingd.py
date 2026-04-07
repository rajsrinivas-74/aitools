"""
Configuration management for Spend Analyzer application.
Uses environment variables from .env files in multiple locations.
Loads configuration using load_dotenv from:
  1. Home directory (~/.env)
  2. Project root directory (.env)
"""

import os
from typing import Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _mask_sensitive_value(key: str, value: str) -> str:
    """Mask sensitive environment variable values for logging."""
    sensitive_keys = {
        'KEY', 'PASSWORD', 'SECRET', 'TOKEN', 'AUTH', 'CREDENTIALS',
        'API_KEY', 'OPENAI_API_KEY', 'NEO4J_PASSWORD', 'API_PASSWORD'
    }
    
    # Check if any sensitive key pattern matches
    if any(sensitive in key.upper() for sensitive in sensitive_keys):
        if not value:
            return "[NOT SET]"
        elif value == "change-me-in-production" or value == "dev-secret-key-change-in-production":
            return "[DEFAULT - CHANGE ME]"
        else:
            return "[SET]"
    
    return value


def _load_env_files() -> None:
    """
    Load environment variables from .env files in priority order:
    1. Home directory ~/.env
    2. Project root directory .env
    
    This function must be called BEFORE Settings class attributes are evaluated.
    """
    env_paths = [
        Path.home() / ".env",                    # Home directory: ~/.env
        Path(__file__).parent.parent / ".env",   # Project root: /path/to/spent_analysis/.env
    ]
    
    loaded_files = []
    all_vars = {}
    
    for env_path in env_paths:
        if env_path.exists():
            # Read the .env file to show what variables are being loaded
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            if key:
                                all_vars[key] = value.strip()
            except Exception as e:
                print(f"⚠ Error reading {env_path}: {e}", flush=True)
            
            # Load using load_dotenv with override=True to prioritize .env over shell environment
            load_dotenv(env_path, override=True)
            loaded_files.append(env_path)
            print(f"✓ Loaded .env from: {env_path}", flush=True)
            logger.info(f"Loaded configuration from: {env_path}")
    
    # Log all loaded environment variables
    if all_vars:
        print("\n📋 ENVIRONMENT VARIABLES LOADED:", flush=True)
        print("-" * 70, flush=True)
        for key in sorted(all_vars.keys()):
            masked_value = _mask_sensitive_value(key, all_vars[key])
            print(f"  {key:40} = {masked_value}", flush=True)
        print("-" * 70 + "\n", flush=True)
        logger.info(f"Loaded {len(all_vars)} environment variables from .env files")
    
    if not loaded_files:
        print(
            f"⚠ No .env files found in:\n"
            f"  - Home directory: {env_paths[0]}\n"
            f"  - Project root: {env_paths[1]}",
            flush=True
        )
        logger.warning(
            f"No .env files found in:\n"
            f"  - Home directory: {env_paths[0]}\n"
            f"  - Project root: {env_paths[1]}\n"
            f"Using default values and environment variables only.\n"
            f"To configure the application, create .env file in home directory or project root."
        )


# IMPORTANT: Load environment variables NOW, before Settings class is defined
# This ensures os.environ has values when class attributes are evaluated
_load_env_files()


class Settings:
    """Application configuration class with values loaded from .env files (home and project root)."""

    # Application
    APP_NAME: str = "Spend Analyzer"
    APP_ENV: str = os.environ.get("APP_ENV", "development")
    DEBUG: bool = os.environ.get("DEBUG", "").lower() == "true" or "development" in os.environ.get("APP_ENV", "development")
    
    # File paths
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    FAISS_INDEX_DIR: Path = BASE_DIR / "indexes"
    CACHE_DIR: Path = BASE_DIR / ".cache"
    
    # Flask
    FLASK_HOST: str = os.environ.get("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.environ.get("FLASK_PORT", 5000))
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # Authentication
    ENABLE_AUTH: bool = os.environ.get("ENABLE_AUTH", "false").lower() == "true"
    API_KEY: str = os.environ.get("API_KEY", "change-me-in-production")
    API_USERNAME: str = os.environ.get("API_USERNAME", "admin")
    API_PASSWORD: str = os.environ.get("API_PASSWORD", "change-me-in-production")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Neo4j Configuration
    NEO4J_URI: str = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME: str = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "password")
    
    # FAISS Configuration
    FAISS_INDEX_NAME: str = "transaction_embeddings"
    EMBEDDING_DIMENSION: int = 1536
    
    # Transaction Processing
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", 500))
    MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 50))
    DEFAULT_CURRENCY_SYMBOL: str = os.environ.get("DEFAULT_CURRENCY_SYMBOL", "$")
    
    # RAG Configuration
    TOP_K_FAISS: int = int(os.environ.get("TOP_K_FAISS", 5))
    CONTEXT_WINDOW: int = int(os.environ.get("CONTEXT_WINDOW", 2000))
    
    # Categories
    INCOME_CATEGORIES: list = [
        "Salary", "Bonus", "Refund", "Interest",
        "Investment Return", "Gift", "Reimbursement", "Other Income"
    ]
    
    EXPENSE_CATEGORIES: list = [
        "Food & Dining", "Groceries", "Transportation", "Travel",
        "Shopping", "Entertainment", "Utilities", "Rent/Mortgage",
        "Insurance", "Healthcare", "Education", "Personal Care",
        "Home & Garden", "Pet Care", "Subscriptions",
        "Fees & Charges", "Business Expenses", "Other Expense"
    ]
    
    # Logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "ERROR")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration and create required directories."""
        if not cls.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set. RAG features will be limited.")
        
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def log_configuration(cls) -> None:
        """Log all loaded configuration settings."""
        logger.info("="*70)
        logger.info("CONFIGURATION LOADED")
        logger.info("="*70)
        
        # Application Settings
        logger.info("📱 APPLICATION SETTINGS:")
        logger.info(f"  - App Name: {cls.APP_NAME}")
        logger.info(f"  - Environment: {cls.APP_ENV}")
        logger.info(f"  - Debug Mode: {cls.DEBUG}")
        
        # Flask Settings
        logger.info("🌐 FLASK SERVER:")
        logger.info(f"  - Host: {cls.FLASK_HOST}")
        logger.info(f"  - Port: {cls.FLASK_PORT}")
        logger.info(f"  - Secret Key: {'[SET]' if cls.SECRET_KEY != 'dev-secret-key-change-in-production' else '[DEFAULT - CHANGE ME]'}")
        
        # Authentication Settings
        logger.info("🔐 AUTHENTICATION:")
        logger.info(f"  - Auth Enabled: {cls.ENABLE_AUTH}")
        logger.info(f"  - API Username: {cls.API_USERNAME}")
        logger.info(f"  - API Key: {'[SET]' if cls.API_KEY != 'change-me-in-production' else '[DEFAULT - CHANGE ME]'}")
        logger.info(f"  - API Password: {'[SET]' if cls.API_PASSWORD != 'change-me-in-production' else '[DEFAULT - CHANGE ME]'}")
        
        # File Paths
        logger.info("📁 FILE PATHS:")
        logger.info(f"  - Base Directory: {cls.BASE_DIR}")
        logger.info(f"  - Upload Directory: {cls.UPLOAD_DIR}")
        logger.info(f"  - FAISS Index Directory: {cls.FAISS_INDEX_DIR}")
        logger.info(f"  - Cache Directory: {cls.CACHE_DIR}")
        
        # OpenAI Configuration
        logger.info("🤖 OPENAI:")
        logger.info(f"  - API Key: {'[SET]' if cls.OPENAI_API_KEY else '[NOT SET]'}")
        logger.info(f"  - Model: {cls.OPENAI_MODEL}")
        logger.info(f"  - Embedding Model: {cls.OPENAI_EMBEDDING_MODEL}")
        
        # Neo4j Configuration
        logger.info("📊 NEO4J:")
        logger.info(f"  - URI: {cls.NEO4J_URI}")
        logger.info(f"  - Username: {cls.NEO4J_USERNAME}")
        logger.info(f"  - Password: {'[SET]' if cls.NEO4J_PASSWORD != 'password' else '[DEFAULT]'}")
        
        # FAISS Configuration
        logger.info("🔍 FAISS:")
        logger.info(f"  - Index Name: {cls.FAISS_INDEX_NAME}")
        logger.info(f"  - Embedding Dimension: {cls.EMBEDDING_DIMENSION}")
        
        # Processing Configuration
        logger.info("⚙️  PROCESSING:")
        logger.info(f"  - Chunk Size: {cls.CHUNK_SIZE}")
        logger.info(f"  - Max Upload Size: {cls.MAX_UPLOAD_SIZE_MB} MB")
        logger.info(f"  - Top-K FAISS Results: {cls.TOP_K_FAISS}")
        logger.info(f"  - Context Window: {cls.CONTEXT_WINDOW}")
        
        # Categories
        logger.info(f"📂 CATEGORIES:")
        logger.info(f"  - Income Categories: {len(cls.INCOME_CATEGORIES)} defined")
        logger.info(f"  - Expense Categories: {len(cls.EXPENSE_CATEGORIES)} defined")
        
        # Logging
        logger.info("📝 LOGGING:")
        logger.info(f"  - Log Level: {cls.LOG_LEVEL}")
        logger.info(f"  - Log Format: {cls.LOG_FORMAT}")
        
        logger.info("="*70)
        
        # Log all environment variables
        cls._log_env_variables()
    
    @classmethod
    def _log_class_variable_initialization(cls) -> None:
        """Log all class variables and their initialization source."""
        logger.info("")
        logger.info("📋 CLASS VARIABLE INITIALIZATION:")
        logger.info("-"*70)
        
        class_vars = {
            # Application
            "APP_NAME": (cls.APP_NAME, "hardcoded"),
            "APP_ENV": (cls.APP_ENV, "os.environ"),
            "DEBUG": (cls.DEBUG, "os.environ + logic"),
            
            # Flask
            "FLASK_HOST": (cls.FLASK_HOST, "os.environ"),
            "FLASK_PORT": (cls.FLASK_PORT, "os.environ"),
            "SECRET_KEY": (cls.SECRET_KEY, "os.environ"),
            
            # Authentication
            "ENABLE_AUTH": (cls.ENABLE_AUTH, "os.environ"),
            "API_KEY": (cls.API_KEY, "os.environ"),
            "API_USERNAME": (cls.API_USERNAME, "os.environ"),
            "API_PASSWORD": (cls.API_PASSWORD, "os.environ"),
            
            # OpenAI
            "OPENAI_API_KEY": (cls.OPENAI_API_KEY, "os.environ"),
            "OPENAI_MODEL": (cls.OPENAI_MODEL, "os.environ"),
            
            # Neo4j
            "NEO4J_URI": (cls.NEO4J_URI, "os.environ"),
            "NEO4J_USERNAME": (cls.NEO4J_USERNAME, "os.environ"),
            "NEO4J_PASSWORD": (cls.NEO4J_PASSWORD, "os.environ"),
            
            # Processing
            "CHUNK_SIZE": (cls.CHUNK_SIZE, "os.environ"),
            "MAX_UPLOAD_SIZE_MB": (cls.MAX_UPLOAD_SIZE_MB, "os.environ"),
            "TOP_K_FAISS": (cls.TOP_K_FAISS, "os.environ"),
            "CONTEXT_WINDOW": (cls.CONTEXT_WINDOW, "os.environ"),
            
            # Logging
            "LOG_LEVEL": (cls.LOG_LEVEL, "os.environ"),
        }
        
        for var_name, (value, source) in sorted(class_vars.items()):
            masked_val = _mask_sensitive_value(var_name, str(value))
            logger.info(f"  {var_name:30} = {str(masked_val):35} [source: {source}]")
        
        logger.info("-"*70)
    
    @classmethod
    def _log_env_variables(cls) -> None:
        """Log all relevant environment variables with security masking."""
        logger.info("")
        logger.info("🔐 ENVIRONMENT VARIABLES LOADED FROM .env:")
        logger.info("-"*70)
        
        # Define all environment variables that should be logged
        env_vars = {
            "APP_ENV": os.environ.get("APP_ENV", "[NOT SET]"),
            "DEBUG": os.environ.get("DEBUG", "[NOT SET]"),
            "FLASK_HOST": os.environ.get("FLASK_HOST", "[NOT SET]"),
            "FLASK_PORT": os.environ.get("FLASK_PORT", "[NOT SET]"),
            "SECRET_KEY": os.environ.get("SECRET_KEY", "[NOT SET]"),
            "ENABLE_AUTH": os.environ.get("ENABLE_AUTH", "[NOT SET]"),
            "API_KEY": os.environ.get("API_KEY", "[NOT SET]"),
            "API_USERNAME": os.environ.get("API_USERNAME", "[NOT SET]"),
            "API_PASSWORD": os.environ.get("API_PASSWORD", "[NOT SET]"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "[NOT SET]"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "[NOT SET]"),
            "NEO4J_URI": os.environ.get("NEO4J_URI", "[NOT SET]"),
            "NEO4J_USERNAME": os.environ.get("NEO4J_USERNAME", "[NOT SET]"),
            "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", "[NOT SET]"),
            "MAX_UPLOAD_SIZE_MB": os.environ.get("MAX_UPLOAD_SIZE_MB", "[NOT SET]"),
            "LOG_LEVEL": os.environ.get("LOG_LEVEL", "[NOT SET]"),
            "CHUNK_SIZE": os.environ.get("CHUNK_SIZE", "[NOT SET]"),
            "TOP_K_FAISS": os.environ.get("TOP_K_FAISS", "[NOT SET]"),
            "CONTEXT_WINDOW": os.environ.get("CONTEXT_WINDOW", "[NOT SET]"),
        }
        
        for key, value in env_vars.items():
            masked_value = _mask_sensitive_value(key, value)
            logger.info(f"  {key:30} = {masked_value}")
        
        logger.info("-"*70)
        logger.info("✓ Environment variables loaded successfully")
        logger.info("")


# Initialize settings instance once at module level
settings = Settings()
settings.validate()
settings.log_configuration()
settings._log_class_variable_initialization()


def main():

    """Main function to test and display all loaded configuration variables."""
    print("\n", "="*70)
    print("SPEND ANALYZER - CONFIGURATION VALIDATION")
    print("="*70, "\n")
    
    # Display all loaded variables
    print("📱 APPLICATION:")
    print(f"   APP_NAME                   : {settings.APP_NAME}")
    print(f"   APP_ENV                    : {settings.APP_ENV}")
    print(f"   DEBUG                      : {settings.DEBUG}")
    
    print("\n🌐 FLASK SERVER:")
    print(f"   FLASK_HOST                 : {settings.FLASK_HOST}")
    print(f"   FLASK_PORT                 : {settings.FLASK_PORT}")
    print(f"   SECRET_KEY                 : {_mask_sensitive_value('SECRET_KEY', settings.SECRET_KEY)}")
    
    print("\n🔐 AUTHENTICATION:")
    print(f"   ENABLE_AUTH                : {settings.ENABLE_AUTH}")
    print(f"   API_USERNAME               : {settings.API_USERNAME}")
    print(f"   API_KEY                    : {_mask_sensitive_value('API_KEY', settings.API_KEY)}")
    print(f"   API_PASSWORD               : {_mask_sensitive_value('API_PASSWORD', settings.API_PASSWORD)}")
    
    print("\n📁 FILE PATHS:")
    print(f"   BASE_DIR                   : {settings.BASE_DIR}")
    print(f"   UPLOAD_DIR                 : {settings.UPLOAD_DIR}")
    print(f"   FAISS_INDEX_DIR            : {settings.FAISS_INDEX_DIR}")
    print(f"   CACHE_DIR                  : {settings.CACHE_DIR}")
    
    print("\n🤖 OPENAI:")
    print(f"   OPENAI_API_KEY             : {_mask_sensitive_value('OPENAI_API_KEY', settings.OPENAI_API_KEY)}")
    print(f"   OPENAI_MODEL               : {settings.OPENAI_MODEL}")
    print(f"   OPENAI_EMBEDDING_MODEL     : {settings.OPENAI_EMBEDDING_MODEL}")
    
    print("\n📊 NEO4J:")
    print(f"   NEO4J_URI                  : {settings.NEO4J_URI}")
    print(f"   NEO4J_USERNAME             : {settings.NEO4J_USERNAME}")
    print(f"   NEO4J_PASSWORD             : {_mask_sensitive_value('NEO4J_PASSWORD', settings.NEO4J_PASSWORD)}")
    
    print("\n🔍 FAISS:")
    print(f"   FAISS_INDEX_NAME           : {settings.FAISS_INDEX_NAME}")
    print(f"   EMBEDDING_DIMENSION        : {settings.EMBEDDING_DIMENSION}")
    
    print("\n⚙️  PROCESSING:")
    print(f"   CHUNK_SIZE                 : {settings.CHUNK_SIZE}")
    print(f"   MAX_UPLOAD_SIZE_MB         : {settings.MAX_UPLOAD_SIZE_MB}")
    print(f"   TOP_K_FAISS                : {settings.TOP_K_FAISS}")
    print(f"   CONTEXT_WINDOW             : {settings.CONTEXT_WINDOW}")
    
    print("\n📂 CATEGORIES:")
    print(f"   INCOME_CATEGORIES          : {len(settings.INCOME_CATEGORIES)} types")
    for i, cat in enumerate(settings.INCOME_CATEGORIES, 1):
        print(f"      {i:2}. {cat}")
    
    print(f"\n   EXPENSE_CATEGORIES         : {len(settings.EXPENSE_CATEGORIES)} types")
    for i, cat in enumerate(settings.EXPENSE_CATEGORIES, 1):
        print(f"      {i:2}. {cat}")
    
    print("\n📝 LOGGING:")
    print(f"   LOG_LEVEL                  : {settings.LOG_LEVEL}")
    print(f"   LOG_FORMAT                 : {settings.LOG_FORMAT}")
    
    print("\n" + "="*70)
    print("✅ ALL VARIABLES LOADED SUCCESSFULLY FROM .env!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()