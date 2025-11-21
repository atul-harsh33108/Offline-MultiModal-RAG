"""
Setup script for Multimodal RAG System
"""
from setuptools import setup, find_packages

setup(
    name="multimodal-rag",
    version="1.0.0",
    description="Multimodal Retrieval-Augmented Generation System",
    author="Atul Harsh",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "langchain-core>=0.1.10",
        "ollama>=0.1.6",
        "chromadb>=0.4.22",
        "pypdf>=4.0.1",
        "python-docx>=1.1.0",
        "librosa>=0.10.2",
        "Pillow>=10.2.0",
        "opencv-python>=4.9.0.80",
        "pytesseract>=0.3.10",
        "openai-whisper>=20231117",
        "sentence-transformers>=2.3.1",
        "transformers>=4.36.2",
        "torch>=2.1.2",
        "torchvision>=0.16.2",
        "streamlit>=1.29.0",
        "gradio>=4.8.0",
        "numpy>=1.26.2",
        "pydantic>=2.5.3",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.1",
        "aiofiles>=23.2.1",
    ],
    python_requires=">=3.10",
)

