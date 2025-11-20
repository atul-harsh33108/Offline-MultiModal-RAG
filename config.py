"""
Configuration file for the Multimodal RAG System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"

# Create directories if they don't exist
for dir_path in [DATA_DIR, DOCUMENTS_DIR, IMAGES_DIR, AUDIO_DIR, VECTOR_DB_DIR, CHAT_HISTORY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS = {
    "phi3-mini": "phi3:mini",
    "gemma2-2b": "gemma2:2b",
    "qwen2.5-1.5b": "qwen2.5:1.5b"
}
DEFAULT_LLM = "phi3-mini"

# Embedding Models
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# Whisper Configuration
WHISPER_MODEL = "base"  # Options: "base" (74M) or "small" (244M)

# Vector Database Configuration
VECTOR_DB_NAME = "multimodal_rag"
COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 384  # MiniLM-L6-v2 dimension

# RAG Configuration
TOP_K_RETRIEVAL = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# OCR Configuration (Tesseract)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# UI Configuration
MAX_FILE_SIZE_MB = 100
SUPPORTED_DOCUMENT_FORMATS = [".pdf", ".docx"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

# Processing Configuration
ENABLE_GPU = True  # Set to False if GPU is not available
MAX_WORKERS = 4

