# Multimodal RAG System (Offline Mode)

A comprehensive Retrieval-Augmented Generation system that supports multiple data formats (PDF, DOCX, Images, Audio) with cross-modal semantic search capabilities.

## Features

- **Multimodal Ingestion**: Supports PDF, DOCX, Images (with OCR), and Audio files
- **Cross-Modal Search**: Text-to-image, image-to-text, and audio-to-others search
- **Multiple LLM Support**: Phi-3-mini, Gemma 3, Qwen3 (4-bit quantized via Ollama)
- **Citation Transparency**: Every answer includes source citations
- **Low-End Device Optimized**: Designed for systems with limited VRAM (4GB)

## Setup

### Prerequisites

1. **Python 3.10+** (recommended: use virtual environment)
2. **Ollama** installed and running locally
3. **Tesseract OCR** installed (for image text extraction)

### Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Install and setup Ollama models:
```bash
# Install Ollama from https://ollama.ai
# Then pull the models:
ollama pull phi3:mini
ollama pull gemma2:2b
ollama pull qwen2.5:1.5b
```

4. Install Tesseract OCR:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or update config in `config.py`

### Running the Application

```bash
streamlit run app.py
```

## Project Structure

```
Offline_rag/
├── app.py                 # Streamlit UI
├── config.py              # Configuration settings
├── loaders/               # Data loaders
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── image_loader.py
│   └── audio_loader.py
├── embeddings/            # Embedding models
│   ├── __init__.py
│   ├── text_embedder.py
│   └── image_embedder.py
├── vectorstore/          # Vector database
│   ├── __init__.py
│   └── chroma_store.py
├── llm/                  # LLM integration
│   ├── __init__.py
│   └── ollama_llm.py
├── rag/                  # RAG pipeline
│   ├── __init__.py
│   └── rag_pipeline.py
├── utils/                # Utilities
│   ├── __init__.py
│   └── file_manager.py
└── data/                 # Uploaded files storage
    ├── documents/
    ├── images/
    └── audio/
```

## Usage

1. **Upload Files**: Use the upload area to add PDF, DOCX, Images, or Audio files
2. **Wait for Processing**: Monitor the processing status indicators
3. **Query**: Type natural language questions in the chat interface
4. **View Citations**: Expand citations to see source documents
5. **Manage Files**: Use the file manager to view or delete indexed files

## Configuration

Edit `config.py` to customize:
- Model selection
- Embedding models
- Vector database settings
- File storage paths

## Notes

- First run will download embedding models (may take time)
- Processing large files may take several minutes
- Ensure sufficient disk space for uploaded files and vector indices

### Cross-Modal Search

The system currently uses separate embedding spaces:
- **Text**: 384D embeddings (MiniLM-L6-v2) for documents and queries
- **Images**: 512D embeddings (CLIP ViT-B/32) for image content

**Current Capabilities:**
- ✅ Text-to-text semantic search
- ✅ Image indexing with OCR text (searchable via text queries)
- ⚠️ Direct text-to-image semantic search requires CLIP text encoder (future enhancement)

Images are indexed with their OCR-extracted text, making them searchable through text queries that match the extracted content.

