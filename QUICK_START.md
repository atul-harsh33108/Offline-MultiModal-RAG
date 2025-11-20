# Quick Start Guide

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Ollama installed and running
- [ ] Tesseract OCR installed (for image text extraction)
- [ ] Virtual environment created

## Setup Steps

### 1. Open PowerShell and Set Execution Policy

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

### 2. Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Ollama Models

```powershell
ollama pull phi3:mini
ollama pull gemma2:2b
ollama pull qwen2.5:1.5b
```

**Note:** These models are large (several GB each). Ensure sufficient disk space.

### 5. Verify Tesseract Installation

Update `TESSERACT_CMD` in `config.py` if Tesseract is not in the default location:
- Default: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 6. Run the Application

**Option A: Using the batch script**
```powershell
.\start.bat
```

**Option B: Manual start**
```powershell
.\venv\Scripts\activate
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## First Use

1. **Initialize Pipeline**: Click "Initialize/Update Pipeline" in the sidebar
2. **Upload Files**: Use the upload areas to add PDF, DOCX, Images, or Audio files
3. **Process Files**: Click "Process All Files" and wait for completion
4. **Start Chatting**: Type your questions in the chat interface

## Important Notes

### Cross-Modal Search Limitation

Currently, the system uses:
- **Text embeddings (384D)** for text documents and queries
- **Image embeddings (512D)** for images

For true text-to-image search, you would need CLIP's text encoder. The current implementation:
- ✅ Searches text documents with text queries
- ✅ Indexes images with their OCR text content (searchable via text)
- ⚠️ Direct text-to-image semantic search requires CLIP text encoder (future enhancement)

### Performance Tips

1. **First Run**: Will download embedding models (~500MB total) - be patient
2. **Large Files**: Process files one at a time for better stability
3. **Memory**: Close other applications to free RAM
4. **GPU**: If available, will be used automatically for image embeddings

### Troubleshooting

**Ollama Connection Error**
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` in `config.py`

**Out of Memory**
- Use smaller Whisper model: Set `WHISPER_MODEL = "base"` in `config.py`
- Process fewer files at once
- Reduce `TOP_K_RETRIEVAL` in `config.py`

**Tesseract Not Found**
- Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Update `TESSERACT_CMD` in `config.py` with correct path

## Next Steps

- Review `SETUP_INSTRUCTIONS.md` for detailed setup
- Check `README.md` for architecture and features
- Customize `config.py` for your needs

