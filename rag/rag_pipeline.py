"""
RAG pipeline for multimodal retrieval and generation
"""
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from loaders import PDFLoader, DOCXLoader, ImageLoader, AudioLoader
from embeddings import TextEmbedder, ImageEmbedder
from vectorstore import ChromaVectorStore
from llm import OllamaLLM
from utils.file_manager import FileManager
from config import (
    DOCUMENTS_DIR, IMAGES_DIR, AUDIO_DIR,
    TOP_K_RETRIEVAL, DEFAULT_LLM
)

logger = logging.getLogger(__name__)


def sanitize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata to be compatible with ChromaDB
    ChromaDB only accepts: str, int, float, bool, None, SparseVector
    Converts lists/dicts to JSON strings
    
    Args:
        metadata: Original metadata dictionary
        
    Returns:
        Sanitized metadata dictionary
    """
    sanitized = {}
    for key, value in metadata.items():
        # Skip None values
        if value is None:
            continue
        
        # Handle scalar types (str, int, float, bool)
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        
        # Convert lists and dicts to JSON strings
        elif isinstance(value, (list, dict)):
            try:
                sanitized[key] = json.dumps(value)
            except (TypeError, ValueError):
                # If JSON serialization fails, skip the field
                logger.warning(f"Could not serialize metadata field '{key}', skipping")
                continue
        
        # Convert other types to strings
        else:
            sanitized[key] = str(value)
    
    return sanitized


class RAGPipeline:
    """Main RAG pipeline for multimodal document processing and querying"""
    
    def __init__(self, llm_model: str = DEFAULT_LLM):
        # Initialize components
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.vector_store = ChromaVectorStore()
        self.llm = OllamaLLM(model_name=llm_model)
        self.file_manager = FileManager()
        
        # Initialize loaders
        self.pdf_loader = PDFLoader()
        self.docx_loader = DOCXLoader()
        self.image_loader = ImageLoader()
        self.audio_loader = AudioLoader()
    
    def ingest_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Ingest a file and add it to the vector store
        
        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, docx, image, audio)
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            file_path = Path(file_path)
            
            # Load file based on type
            if file_type == "pdf":
                data = self.pdf_loader.load(str(file_path))
            elif file_type == "docx":
                data = self.docx_loader.load(str(file_path))
            elif file_type == "image":
                data = self.image_loader.load(str(file_path))
            elif file_type == "audio":
                data = self.audio_loader.load(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Generate embeddings and add to vector store
            texts = []
            embeddings = []
            metadatas = []
            ids = []
            
            # Process text chunks
            for i, chunk in enumerate(data["chunks"]):
                chunk_text = chunk["text"]
                if not chunk_text.strip():
                    continue
                
                # Generate text embedding
                embedding = self.text_embedder.embed(chunk_text)[0]
                
                texts.append(chunk_text)
                embeddings.append(embedding)
                
                # Create metadata
                metadata = data["metadata"].copy()
                metadata.update({
                    "chunk_index": i,
                    "chunk_start": chunk.get("start_index", 0),
                    "chunk_length": chunk.get("length", len(chunk_text)),
                    "modality": "text"
                })
                # Sanitize metadata for ChromaDB compatibility
                sanitized_metadata = sanitize_metadata_for_chromadb(metadata)
                metadatas.append(sanitized_metadata)
                
                ids.append(f"{file_path.stem}_chunk_{i}")
            
            # Process image embedding if it's an image file
            if file_type == "image" and "image_path" in data:
                try:
                    image_embedding = self.image_embedder.embed(data["image_path"])[0]
                    
                    # Add image embedding to separate collection
                    image_text = f"[IMAGE] {data['metadata']['file_name']}"
                    image_metadata = data["metadata"].copy()
                    image_metadata.update({
                        "chunk_index": -1,  # Special index for image
                        "modality": "image",
                        "image_path": data["image_path"]
                    })
                    # Sanitize metadata for ChromaDB compatibility
                    sanitized_image_metadata = sanitize_metadata_for_chromadb(image_metadata)
                    
                    self.vector_store.add_documents(
                        texts=[image_text],
                        embeddings=[image_embedding],
                        metadatas=[sanitized_image_metadata],
                        ids=[f"{file_path.stem}_image"],
                        modality="image"
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate image embedding: {e}")
            
            # Add text chunks to vector store
            if texts:
                self.vector_store.add_documents(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                    modality="text"
                )
            
            # Copy file to storage directory
            stored_path = self.file_manager.store_file(file_path, file_type)
            
            return {
                "success": True,
                "file_name": file_path.name,
                "file_type": file_type,
                "chunks_added": len(texts),
                "stored_path": str(stored_path),
                "metadata": data["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_name": Path(file_path).name
            }
    
    def query(
        self,
        query_text: str,
        n_results: int = TOP_K_RETRIEVAL,
        include_images: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query_text: Natural language query
            n_results: Number of results to retrieve
            include_images: Whether to include image results
            
        Returns:
            Dictionary with query results, context, and response
        """
        try:
            # Generate query embedding (text embedding for text queries)
            query_embedding = self.text_embedder.embed(query_text)[0]
            
            # Search vector store (searches both text and image collections)
            # Note: For cross-modal search, we'd need to use CLIP's text encoder
            # For now, we search text collection with text embedding
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results * 2,  # Get more to filter
                modality=None  # Search both collections
            )
            
            # Filter and format results
            text_results = []
            image_results = []
            
            for result in search_results:
                metadata = result.get("metadata", {})
                modality = metadata.get("modality", "text")
                
                if modality == "image" and include_images:
                    image_results.append(result)
                elif modality == "text":
                    text_results.append(result)
            
            # Combine results (prioritize text, then images)
            all_results = text_results[:n_results] + image_results[:2]
            
            # Build context for LLM
            context_parts = []
            citations = []
            
            for i, result in enumerate(all_results[:n_results], 1):
                doc_text = result.get("document", "")
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown")
                file_name = metadata.get("file_name", "Unknown")
                
                if metadata.get("modality") == "image":
                    context_parts.append(f"[Image {i}]: {file_name}")
                else:
                    context_parts.append(f"[{i}] {doc_text}")
                
                citations.append({
                    "number": i,
                    "source": source,
                    "file_name": file_name,
                    "file_type": metadata.get("file_type", "unknown"),
                    "modality": metadata.get("modality", "text"),
                    "chunk_index": metadata.get("chunk_index", -1),
                    "image_path": metadata.get("image_path")
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate response using LLM
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Always cite your sources using the citation numbers [1], [2], etc. 
If the context doesn't contain relevant information, say so clearly."""
            
            response = self.llm.generate(
                prompt=query_text,
                context=context,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=512
            )
            
            return {
                "query": query_text,
                "response": response,
                "citations": citations,
                "retrieved_documents": all_results[:n_results],
                "context_used": context
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            raise
    
    def stream_query(
        self,
        query_text: str,
        n_results: int = TOP_K_RETRIEVAL,
        include_images: bool = True
    ):
        """
        Stream query response
        
        Args:
            query_text: Natural language query
            n_results: Number of results to retrieve
            include_images: Whether to include image results
            
        Yields:
            Response chunks and final results
        """
        try:
            # Generate query embedding
            query_embedding = self.text_embedder.embed(query_text)[0]
            
            # Search vector store
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results * 2
            )
            
            # Filter and format results
            text_results = []
            image_results = []
            
            for result in search_results:
                metadata = result.get("metadata", {})
                modality = metadata.get("modality", "text")
                
                if modality == "image" and include_images:
                    image_results.append(result)
                elif modality == "text":
                    text_results.append(result)
            
            all_results = text_results[:n_results] + image_results[:2]
            
            # Build context
            context_parts = []
            citations = []
            
            for i, result in enumerate(all_results[:n_results], 1):
                doc_text = result.get("document", "")
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown")
                file_name = metadata.get("file_name", "Unknown")
                
                if metadata.get("modality") == "image":
                    context_parts.append(f"[Image {i}]: {file_name}")
                else:
                    context_parts.append(f"[{i}] {doc_text}")
                
                citations.append({
                    "number": i,
                    "source": source,
                    "file_name": file_name,
                    "file_type": metadata.get("file_type", "unknown"),
                    "modality": metadata.get("modality", "text"),
                    "chunk_index": metadata.get("chunk_index", -1),
                    "image_path": metadata.get("image_path")
                })
            
            context = "\n\n".join(context_parts)
            
            # Stream response
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Always cite your sources using the citation numbers [1], [2], etc. 
If the context doesn't contain relevant information, say so clearly."""
            
            full_response = ""
            for chunk in self.llm.stream_generate(
                prompt=query_text,
                context=context,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=512
            ):
                full_response += chunk
                yield {
                    "type": "chunk",
                    "content": chunk
                }
            
            # Yield final results
            yield {
                "type": "complete",
                "response": full_response,
                "citations": citations,
                "retrieved_documents": all_results[:n_results]
            }
            
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def delete_file(self, file_source: str) -> Dict[str, Any]:
        """
        Delete a file and its associated vectors
        
        Args:
            file_source: Source file path to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Delete from vector store
            deleted_count = self.vector_store.delete_documents(file_source)
            
            # Delete from file storage
            file_deleted = self.file_manager.delete_file(file_source)
            
            return {
                "success": True,
                "vectors_deleted": deleted_count,
                "file_deleted": file_deleted
            }
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_indexed_files(self) -> List[Dict[str, Any]]:
        """
        Get list of all indexed files
        
        Returns:
            List of file information dictionaries
        """
        sources = self.vector_store.get_all_sources()
        files = []
        
        for source in sources:
            file_info = self.file_manager.get_file_info(source)
            if file_info:
                files.append(file_info)
        
        return files
    
    def set_llm_model(self, model_name: str):
        """Change the LLM model"""
        self.llm.set_model(model_name)

