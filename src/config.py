"""Configuration management"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # API
    API_KEY: str = os.getenv("API_KEY", "dev-key")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Databases
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = "documents"
    
    # LLM
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    
    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    COLBERT_WEIGHT: float = float(os.getenv("COLBERT_WEIGHT", "0.5"))
    GRAPH_WEIGHT: float = float(os.getenv("GRAPH_WEIGHT", "0.2"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    
    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    
    # Metadata DB
    METADATA_DB_PATH: str = os.getenv("METADATA_DB_PATH", "metadata.db")


config = Config()
