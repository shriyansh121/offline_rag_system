import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_minilm_db")

# Embeddings
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu" 

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# LLM / Ollama
LLM_MODEL = "gemma3:4b"
LLM_TEMPERATURE = 0.1

# Retrieval
K = 4
FETCH_K = 12  # for MMR

# Other
APP_NAME = "Offline RAG Chat (Advanced)"
