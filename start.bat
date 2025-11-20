@echo off
REM Quick start script for Multimodal RAG System
REM Run this after setting up the environment

echo Starting Multimodal RAG System...
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please create it first:
    echo python -m venv venv
    echo .\venv\Scripts\activate
    pause
    exit /b 1
)

REM Check if Ollama is running
echo Checking Ollama connection...
ollama list >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama may not be running. Please start Ollama first.
    echo Download from: https://ollama.ai
    pause
)

REM Run Streamlit app
echo Starting Streamlit application...
streamlit run app.py

pause

