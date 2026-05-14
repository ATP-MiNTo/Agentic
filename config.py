# Configuration file for Medical RAG PoC
# Edit this file to customize behavior without changing code

import os
from pathlib import Path

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_documents"
FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, FAISS_INDEX_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== MODEL CONFIGURATION ====================

# LLM Configuration
LLM_MODEL = "llama3.1:8b"  # Via Ollama
LLM_ENDPOINT = "http://localhost:11434"  # Ollama local endpoint
LLM_TIMEOUT = 60  # seconds

# Embedding Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # From HuggingFace
EMBEDDING_DIMENSION = 384  # Dimension of embeddings

# ==================== RAG CONFIGURATION ====================

# Retrieval Settings
TOP_K = 5  # Number of documents to retrieve
CHUNK_SIZE = 300  # Approximate words per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks in words

# FAISS Index Settings
FAISS_INDEX_TYPE = "flat"  # Type of FAISS index (flat, ivf)
USE_GPU = False  # Use GPU for FAISS if available

# ==================== LLM PARAMETERS ====================

# Generation Settings
MAX_TOKENS = 512  # Maximum tokens in response
MIN_TOKENS = 50  # Minimum tokens in response
TEMPERATURE = 0.7  # 0.0=deterministic, 1.0=very creative
TOP_P = 0.9  # Nucleus sampling
TOP_K_SAMPLING = 50  # Top-k sampling

# ==================== LOGGING CONFIGURATION ====================

# Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Save logs to file
SAVE_LOGS_TO_FILE = True
LOG_FORMAT = "[{timestamp}] [{level}] [{module}] {message}"
LOG_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5  # Number of backup log files to keep

# ==================== DOCUMENT PROCESSING ====================

# Supported document extensions
SUPPORTED_EXTENSIONS = [".txt", ".md"]

# Auto-rebuild index if documents are older than N days
AUTO_REBUILD_THRESHOLD_DAYS = 7

# ==================== UI CONFIGURATION ====================

# Gradio UI Settings
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False  # Share link for public access
GRADIO_DEBUG = False  # Debug mode

# UI Theme
GRADIO_THEME = "soft"  # Options: soft, base, monochrome, glass

# ==================== MEDICAL DATA SETTINGS ====================

# Disclaimer for medical information
MEDICAL_DISCLAIMER = """
⚠️ DISCLAIMER: This information is for educational purposes only and 
should NOT be used for medical diagnosis, treatment, or advice. 
Always consult with a qualified healthcare professional before making 
any medical decisions. This AI system is not a substitute for professional 
medical judgment.
"""

# Minimum confidence score for retrieval
MIN_RETRIEVAL_SCORE = 0.5

# ==================== PERFORMANCE SETTINGS ====================

# Batch processing
BATCH_SIZE = 32  # For embedding documents

# Caching
USE_EMBEDDING_CACHE = True  # Cache embeddings to avoid recomputation
CACHE_DIR = PROJECT_ROOT / ".cache"

# ==================== SYSTEM SETTINGS ====================

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of threads for CPU operations
NUM_THREADS = 4

# Device: "cpu", "cuda", "mps"
DEVICE = "cpu"

# ==================== FEATURE FLAGS ====================

# Enable/disable features
ENABLE_STREAMING = False  # Stream LLM responses
ENABLE_CACHING = True  # Cache LLM responses for identical queries
ENABLE_METRICS = True  # Track performance metrics
ENABLE_USER_FEEDBACK = False  # Allow users to rate responses

# Agentic loop behavior
ENABLE_AGENTIC_LOOP = True
ENABLE_SESSION_MEMORY = True
AGENT_MAX_STEPS = 4
AGENT_MIN_EVIDENCE_DOCS = 2
AGENT_CONFIDENCE_THRESHOLD = 0.7
AGENT_FORCE_CLARIFY_ON_LOW_EVIDENCE = True
AGENT_MAX_SUBQUERIES = 3

# ==================== DEBUGGING ====================

# Verbose output
VERBOSE = False

# Show retrieved document chunks (useful for debugging)
SHOW_DOCUMENT_CHUNKS = False

# Keep intermediate data for debugging
DEBUG_MODE = False

# ==================== ADVANCED SETTINGS ====================

# Prompt templates location
PROMPT_TEMPLATES_DIR = PROJECT_ROOT / "src" / "utils" / "prompts"

# System prompt
SYSTEM_PROMPT_FILE = PROMPT_TEMPLATES_DIR / "system_prompt.txt"

# RAG context window size
CONTEXT_WINDOW_SIZE = 4096  # Max tokens for context

# Response post-processing
POST_PROCESS_RESPONSE = True  # Clean up response formatting

# ==================== VALIDATION ====================

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOG_LEVEL not in valid_log_levels:
        errors.append(f"Invalid LOG_LEVEL: {LOG_LEVEL}")
    
    # Check temperature range
    if not (0.0 <= TEMPERATURE <= 1.0):
        errors.append(f"TEMPERATURE must be between 0.0 and 1.0, got {TEMPERATURE}")
    
    # Check chunk size
    if CHUNK_SIZE < 50:
        errors.append(f"CHUNK_SIZE should be at least 50, got {CHUNK_SIZE}")
    
    # Check top_k
    if TOP_K < 1:
        errors.append(f"TOP_K must be at least 1, got {TOP_K}")

    if AGENT_MAX_STEPS < 1:
        errors.append(f"AGENT_MAX_STEPS must be at least 1, got {AGENT_MAX_STEPS}")

    if AGENT_MAX_SUBQUERIES < 1:
        errors.append(f"AGENT_MAX_SUBQUERIES must be at least 1, got {AGENT_MAX_SUBQUERIES}")

    if not (0.0 <= AGENT_CONFIDENCE_THRESHOLD <= 1.0):
        errors.append(
            f"AGENT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {AGENT_CONFIDENCE_THRESHOLD}"
        )
    
    if errors:
        print("⚠️ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Configuration validation failed")
    
    return True

# Run validation on import
try:
    validate_config()
except Exception as e:
    print(f"Warning: {e}")
