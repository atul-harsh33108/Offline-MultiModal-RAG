"""
Streamlit UI for Multimodal RAG System
"""
import streamlit as st
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
import time

from rag import RAGPipeline
from config import (
    OLLAMA_MODELS, DEFAULT_LLM,
    SUPPORTED_DOCUMENT_FORMATS, SUPPORTED_IMAGE_FORMATS, SUPPORTED_AUDIO_FORMATS,
    CHAT_HISTORY_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


def initialize_rag_pipeline(model_name: str = DEFAULT_LLM):
    """Initialize or reinitialize RAG pipeline"""
    try:
        if st.session_state.rag_pipeline is None:
            with st.spinner("Initializing RAG pipeline..."):
                st.session_state.rag_pipeline = RAGPipeline(llm_model=model_name)
        else:
            st.session_state.rag_pipeline.set_llm_model(model_name)
        return True
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        logger.error(f"Error initializing RAG pipeline: {e}")
        return False


def save_chat_history():
    """Save current chat history to file"""
    try:
        if not st.session_state.chat_history:
            st.warning("No chat history to save")
            return False
        
        # Create chat history directory if it doesn't exist
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"
        filepath = CHAT_HISTORY_DIR / filename
        
        # Save chat history
        chat_data = {
            "chat_id": st.session_state.current_chat_id or timestamp,
            "timestamp": timestamp,
            "model": st.session_state.rag_pipeline.llm.model_name if st.session_state.rag_pipeline else DEFAULT_LLM,
            "messages": st.session_state.chat_history
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        st.success(f"Chat history saved to {filename}")
        logger.info(f"Chat history saved to {filepath}")
        return True
        
    except Exception as e:
        st.error(f"Error saving chat history: {e}")
        logger.error(f"Error saving chat history: {e}")
        return False


def load_chat_history(filepath: Path):
    """Load chat history from file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
        
        st.session_state.chat_history = chat_data.get("messages", [])
        st.session_state.current_chat_id = chat_data.get("chat_id")
        
        st.success(f"Chat history loaded from {filepath.name}")
        return True
        
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        logger.error(f"Error loading chat history: {e}")
        return False


def process_uploaded_file(uploaded_file, file_type: str):
    """Process an uploaded file"""
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Update processing status
        st.session_state.processing_status = {
            "file_name": uploaded_file.name,
            "status": "processing",
            "stage": "uploading"
        }
        
        # Ingest file
        result = st.session_state.rag_pipeline.ingest_file(str(temp_path), file_type)
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        # Update status
        if result.get("success"):
            st.session_state.processing_status = {
                "file_name": uploaded_file.name,
                "status": "completed",
                "chunks_added": result.get("chunks_added", 0)
            }
            # Refresh indexed files
            st.session_state.indexed_files = st.session_state.rag_pipeline.get_indexed_files()
            return True
        else:
            st.session_state.processing_status = {
                "file_name": uploaded_file.name,
                "status": "error",
                "error": result.get("error", "Unknown error")
            }
            return False
            
    except Exception as e:
        st.session_state.processing_status = {
            "file_name": uploaded_file.name,
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Error processing file: {e}")
        return False


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # LLM Model Selection
        st.subheader("LLM Model")
        selected_model = st.selectbox(
            "Select LLM Model",
            options=list(OLLAMA_MODELS.keys()),
            index=list(OLLAMA_MODELS.keys()).index(DEFAULT_LLM) if DEFAULT_LLM in OLLAMA_MODELS else 0,
            help="Choose the language model for generating responses"
        )
        
        # Initialize RAG pipeline
        if st.button("Initialize/Update Pipeline", type="primary"):
            if initialize_rag_pipeline(selected_model):
                st.success("Pipeline initialized successfully!")
                st.session_state.indexed_files = st.session_state.rag_pipeline.get_indexed_files()
        
        st.divider()
        
        # File Management
        st.subheader("📁 File Management")
        
        if st.button("Refresh File List"):
            if st.session_state.rag_pipeline:
                st.session_state.indexed_files = st.session_state.rag_pipeline.get_indexed_files()
                st.rerun()
        
        # Show indexed files
        if st.session_state.indexed_files:
            st.write(f"**Indexed Files ({len(st.session_state.indexed_files)}):**")
            
            for file_info in st.session_state.indexed_files:
                with st.expander(f"📄 {file_info.get('file_name', 'Unknown')}"):
                    st.write(f"**Type:** {file_info.get('file_type', 'unknown')}")
                    st.write(f"**Size:** {file_info.get('size_mb', 0)} MB")
                    st.write(f"**Source:** {file_info.get('source', 'Unknown')}")
                    
                    if st.button(f"Delete {file_info.get('file_name')}", key=f"delete_{file_info.get('file_name')}"):
                        if st.session_state.rag_pipeline:
                            result = st.session_state.rag_pipeline.delete_file(file_info.get('source'))
                            if result.get("success"):
                                st.success(f"Deleted {result.get('vectors_deleted', 0)} vectors")
                                st.session_state.indexed_files = st.session_state.rag_pipeline.get_indexed_files()
                                st.rerun()
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.info("No files indexed yet")
        
        st.divider()
        
        # Chat Management
        st.subheader("💬 Chat Management")
        
        if st.button("🆕 New Chat"):
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.rerun()
        
        if st.button("💾 Save Chat"):
            save_chat_history()
        
        # Load chat history
        st.write("**Load Chat History:**")
        chat_files = list(CHAT_HISTORY_DIR.glob("*.json")) if CHAT_HISTORY_DIR.exists() else []
        if chat_files:
            selected_chat = st.selectbox(
                "Select chat to load",
                options=[f.name for f in chat_files],
                key="chat_selector"
            )
            if st.button("Load Selected Chat"):
                load_chat_history(CHAT_HISTORY_DIR / selected_chat)
                st.rerun()
        else:
            st.info("No saved chats found")
    
    # Main content area
    st.title("🤖 Multimodal RAG System")
    st.markdown("**Offline Retrieval-Augmented Generation with Multimodal Support**")
    
    # Check if pipeline is initialized
    if st.session_state.rag_pipeline is None:
        st.warning("⚠️ Please initialize the RAG pipeline from the sidebar first!")
        st.info("Click 'Initialize/Update Pipeline' in the sidebar to get started.")
        return
    
    # File Upload Section
    st.header("📤 Upload Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📄 Documents")
        doc_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
    
    with col2:
        st.subheader("🖼️ Images")
        image_files = st.file_uploader(
            "Upload image files",
            type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"],
            accept_multiple_files=True,
            key="image_uploader"
        )
    
    with col3:
        st.subheader("🎤 Audio")
        audio_files = st.file_uploader(
            "Upload audio files",
            type=["mp3", "wav", "m4a", "flac", "ogg"],
            accept_multiple_files=True,
            key="audio_uploader"
        )
    
    # Process uploaded files
    if doc_files or image_files or audio_files:
        if st.button("🚀 Process All Files", type="primary"):
            all_files = []
            
            for file in (doc_files or []):
                file_ext = Path(file.name).suffix.lower()
                if file_ext == ".pdf":
                    all_files.append(("pdf", file))
                elif file_ext == ".docx":
                    all_files.append(("docx", file))
            
            for file in (image_files or []):
                all_files.append(("image", file))
            
            for file in (audio_files or []):
                all_files.append(("audio", file))
            
            # Process files with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (file_type, file) in enumerate(all_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(all_files)})...")
                process_uploaded_file(file, file_type)
                progress_bar.progress((i + 1) / len(all_files))
            
            status_text.text("✅ All files processed!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
    
    # Processing Status
    if st.session_state.processing_status:
        status = st.session_state.processing_status
        if status["status"] == "processing":
            st.info(f"⏳ Processing: {status['file_name']} - {status.get('stage', 'processing')}")
        elif status["status"] == "completed":
            st.success(f"✅ Completed: {status['file_name']} - {status.get('chunks_added', 0)} chunks added")
        elif status["status"] == "error":
            st.error(f"❌ Error processing {status['file_name']}: {status.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Chat Interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Display citations if available
                    if "citations" in message and message["citations"]:
                        with st.expander("📚 Citations"):
                            for citation in message["citations"]:
                                st.write(f"**[{citation['number']}]** {citation['file_name']}")
                                st.caption(f"Type: {citation['file_type']} | Modality: {citation['modality']}")
    
    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query RAG system
                    result = st.session_state.rag_pipeline.query(user_query)
                    
                    # Display response
                    response_text = result["response"]
                    st.write(response_text)
                    
                    # Display citations
                    if result.get("citations"):
                        with st.expander("📚 Sources & Citations"):
                            for citation in result["citations"]:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**[{citation['number']}]** {citation['file_name']}")
                                    st.caption(f"Type: {citation['file_type']} | Modality: {citation['modality']}")
                                
                                with col2:
                                    if citation.get("image_path") and Path(citation["image_path"]).exists():
                                        st.image(citation["image_path"], width=100)
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "citations": result.get("citations", []),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    })


if __name__ == "__main__":
    main()

