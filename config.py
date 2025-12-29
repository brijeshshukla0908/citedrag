# config.py
"""
CitedRAG Configuration
======================
Central configuration file for the CitedRAG application.
All settings, paths, and constants are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# Base Paths
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"

# Ensure critical directories exist
STORAGE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
(STORAGE_DIR / "chroma_db").mkdir(exist_ok=True)
(STORAGE_DIR / "cache").mkdir(exist_ok=True)
(STORAGE_DIR / "logs").mkdir(exist_ok=True)

# ==========================================
# Application Settings
# ==========================================
APP_NAME = os.getenv("APP_NAME", "CitedRAG")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
APP_DESCRIPTION = "Production-ready RAG system with hybrid search and verifiable source citations"

# ==========================================
# LLM Configuration - Groq
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = 0.1  # Low temperature for factual responses
GROQ_MAX_TOKENS = 1024
GROQ_TIMEOUT = 30  # seconds

# Aliases for compatibility
LLM_MODEL = GROQ_MODEL
LLM_TEMPERATURE = GROQ_TEMPERATURE
LLM_MAX_TOKENS = GROQ_MAX_TOKENS

# ==========================================
# Ollama Configuration (Local Fallback)
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 1024

# ==========================================
# Embedding Configuration
# ==========================================
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if GPU available

# ==========================================
# Document Processing
# ==========================================
CHUNK_SIZE = 500  # tokens per chunk
CHUNK_OVERLAP = 100  # token overlap between chunks
MAX_DOCUMENT_SIZE_MB = 50
SUPPORTED_FILE_TYPES = [".pdf"]

# ==========================================
# Retrieval Configuration
# ==========================================
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve
BM25_WEIGHT = 0.5  # Weight for BM25 score
VECTOR_WEIGHT = 0.5  # Weight for vector similarity
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
RERANK_ENABLED = True

# ==========================================
# ChromaDB Settings
# ==========================================
CHROMA_PERSIST_DIR = str(STORAGE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "citedrag_documents")
CHROMA_DISTANCE_METRIC = "cosine"

# ==========================================
# Cache Settings
# ==========================================
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_DIR = str(STORAGE_DIR / "cache")
CACHE_TTL_HOURS = 24
CACHE_MAX_SIZE = 1000  # Maximum number of cached responses

# ==========================================
# Rate Limiting Configuration
# ==========================================
MAX_QUERIES_PER_HOUR = int(os.getenv("MAX_QUERIES_PER_HOUR", 10))
MAX_QUERIES_PER_DAY = int(os.getenv("MAX_QUERIES_PER_DAY", 25))
GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", 500))
BURST_LIMIT_PER_MINUTE = 5

# ==========================================
# Logging Configuration
# ==========================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(STORAGE_DIR / "logs" / "app.log")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "1 week"

# ==========================================
# RAGAS Evaluation
# ==========================================
RAGAS_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
RAGAS_TEST_FILE = str(DATA_DIR / "test_questions.json")
RAGAS_OUTPUT_FILE = str(OUTPUTS_DIR / "evaluation_results.csv")
RAGAS_REPORT_FILE = str(OUTPUTS_DIR / "evaluation_report.md")
RAGAS_MIN_SCORE = 0.75  # Minimum acceptable score

# ==========================================
# Demo Questions
# ==========================================
DEMO_QUESTIONS_FILE = str(DATA_DIR / "demo_questions.json")
MAX_DEMO_QUESTIONS = 15

# ==========================================
# UI Settings (Streamlit)
# ==========================================
PAGE_TITLE = "CitedRAG - RAG with Source Citations"
PAGE_ICON = "📚"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"
THEME = "light"

# ==========================================
# Cost Calculation (for display purposes)
# ==========================================
COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.002,
    "claude-3-opus": 0.015,
    "claude-3-sonnet": 0.003,
    "groq-llama-3.1-8b": 0.0,  # Free tier
    "ollama-local": 0.0,  # Local compute
}

# ==========================================
# Feature Flags
# ==========================================
ENABLE_DEMO_MODE = True
ENABLE_COST_ANALYSIS = True
ENABLE_USAGE_STATS = True
ENABLE_LOCAL_FALLBACK = True
ENABLE_MULTI_DOCUMENT = True  # Allow multiple document upload
ENABLE_RAGAS_EVAL = True

# ==========================================
# System Messages & Templates
# ==========================================
WELCOME_MESSAGE = """
# Welcome to CitedRAG 📚

Upload a document and ask questions. Every answer includes **verifiable source citations** 
to prevent AI hallucinations.

**Key Features:**
- 🔍 Hybrid search (BM25 + Vector)
- 📝 Source citations for every answer
- ✅ RAGAS evaluation metrics
- ⚡ Response caching
- 🛡️ Rate limiting for fair access
"""

ERROR_MESSAGE_RATE_LIMIT = """
⏱️ **Query Limit Reached**

You've used your {limit} queries per hour. This ensures fair access for all visitors.

**Try instead:**
- Explore pre-computed demo questions
- Upload a different document
- View evaluation results

Resets in: {reset_time} minutes
"""

ERROR_MESSAGE_NO_DOCUMENT = """
📄 **No Document Loaded**

Please upload a PDF document first using the sidebar.
"""

ERROR_MESSAGE_API_FAILURE = """
⚠️ **API Temporarily Unavailable**

Switching to local processing (may take 10-15 seconds longer).
"""

# ==========================================
# Validation
# ==========================================
def validate_config():
    """Validate critical configuration settings"""
    errors = []
    
    if not GROQ_API_KEY and ENVIRONMENT == "production":
        errors.append("GROQ_API_KEY must be set in production environment")
    
    if not (0 <= BM25_WEIGHT <= 1) or not (0 <= VECTOR_WEIGHT <= 1):
        errors.append("BM25_WEIGHT and VECTOR_WEIGHT must be between 0 and 1")
    
    if BM25_WEIGHT + VECTOR_WEIGHT != 1.0:
        errors.append("BM25_WEIGHT + VECTOR_WEIGHT must equal 1.0")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))

# Run validation on import
validate_config()

# ==========================================
# Display Configuration (for debugging)
# ==========================================
def print_config():
    """Print current configuration (for debugging)"""
    print(f"\n{'='*50}")
    print(f"CitedRAG Configuration - {ENVIRONMENT.upper()}")
    print(f"{'='*50}")
    print(f"App Version: {APP_VERSION}")
    print(f"LLM Model: {GROQ_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Chunk Size: {CHUNK_SIZE} tokens (overlap: {CHUNK_OVERLAP})")
    print(f"Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"Hybrid Weights: BM25={BM25_WEIGHT}, Vector={VECTOR_WEIGHT}")
    print(f"Rate Limits: {MAX_QUERIES_PER_HOUR}/hr, {MAX_QUERIES_PER_DAY}/day")
    print(f"Cache Enabled: {CACHE_ENABLED}")
    print(f"Demo Mode: {ENABLE_DEMO_MODE}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    print_config() 