# Setup Instructions

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running
3. **Tesseract OCR** installed (for image text extraction)

## Step-by-Step Setup

### 1. Create Virtual Environment

```powershell
# In PowerShell (run as Administrator or use bypass)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

### 2. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install Ollama

1. Download Ollama from: https://ollama.ai
2. Install and start Ollama service
3. Pull required models:

```powershell
ollama pull phi3:mini
ollama pull gemma2:2b
ollama pull qwen2.5:1.5b
```

**Note:** These models are large (several GB each). Ensure you have sufficient disk space.

### 4. Install Tesseract OCR

1. Download Tesseract for Windows from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. If installed elsewhere, update `TESSERACT_CMD` in `config.py`

### 5. Verify Installation

```powershell
# Check Ollama is running
ollama list

# Check Python packages
pip list | findstr "ollama chromadb streamlit"
```

### 6. Run the Application

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## First Run Notes

- **First run will download embedding models** (MiniLM-L6-v2 and CLIP) - this may take several minutes
- **Whisper model** will be downloaded on first audio file processing
- Ensure you have **stable internet connection** for initial model downloads

## Troubleshooting

### Ollama Connection Error

If you see "Connection refused" errors:
1. Ensure Ollama is running: `ollama serve`
2. Check `OLLAMA_BASE_URL` in `config.py` matches your Ollama instance

### Tesseract Not Found

If OCR fails:
1. Verify Tesseract installation path
2. Update `TESSERACT_CMD` in `config.py` with correct path
3. Add Tesseract to system PATH

### Out of Memory Errors

For low-end devices:
1. Use smaller Whisper model: Set `WHISPER_MODEL = "base"` in `config.py`
2. Reduce `TOP_K_RETRIEVAL` in `config.py`
3. Process files one at a time
4. Close other applications to free RAM

### Model Download Issues

If models fail to download:
1. Check internet connection
2. For Hugging Face models, you may need to login: `huggingface-cli login`
3. For Ollama models, ensure sufficient disk space

## Configuration

Edit `config.py` to customize:
- Model selection
- File size limits
- Processing parameters
- Storage paths

## System Requirements

### Minimum (Dev. 1 - Intel Core 5 120U)
- 16GB RAM
- 20GB free disk space
- Internet connection for initial setup

### Recommended (Dev. 2 - AMD Ryzen 5 5600H + RTX 3050)
- 16GB RAM
- 4GB VRAM
- 30GB free disk space
- Internet connection for initial setup

## Performance Tips

1. **Use smaller models** for faster inference
2. **Process files in batches** rather than all at once
3. **Clear vector database** periodically if it grows too large
4. **Use SSD** for better I/O performance
5. **Close unnecessary applications** to free system resources

