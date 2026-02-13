"""Configuration for the Manual Maker RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# MUST set SSL environment variables BEFORE importing any libraries
if os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true":
    print("⚠️  SSL verification disabled")
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "anthropic" or "ollama"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")  # Default local model
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# Chunking Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks to preserve context

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality local embeddings

# Retrieval Configuration
INITIAL_RETRIEVAL_K = 10  # Number of chunks to retrieve initially
REFINED_RETRIEVAL_K = 5   # Number of chunks after re-ranking
RELEVANCE_THRESHOLD = 0.5  # Minimum similarity score

# Progressive RAG Configuration
MAX_ITERATIONS = 3  # Maximum retrieval refinement iterations
