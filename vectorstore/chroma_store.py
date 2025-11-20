"""
ChromaDB vector store for multimodal documents
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from pathlib import Path
from config import VECTOR_DB_DIR, VECTOR_DB_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-based vector store for multimodal documents"""
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = None
        self.text_collection = None
        self.image_collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collections"""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(VECTOR_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create separate collections for text and images
            # Text collection uses 384D embeddings (MiniLM)
            self.text_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_text",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Image collection uses 512D embeddings (CLIP)
            self.image_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_image",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized: {self.collection_name} (text + image collections)")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        modality: str = "text"
    ):
        """
        Add documents to the vector store
        
        Args:
            texts: List of text content
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
            modality: "text" or "image" to determine which collection to use
        """
        try:
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]
            
            # Ensure all lists have the same length
            assert len(texts) == len(embeddings) == len(metadatas) == len(ids), \
                "All input lists must have the same length"
            
            # Select collection based on modality
            collection = self.text_collection if modality == "text" else self.image_collection
            
            # Add to appropriate collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} {modality} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        modality: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            modality: "text", "image", or None (searches compatible collections)
            
        Returns:
            List of search results with documents, metadata, and distances
        """
        try:
            where = filter_metadata if filter_metadata else None
            all_results = []
            embedding_dim = len(query_embedding)
            
            # Text embeddings are 384D (MiniLM), Image embeddings are 512D (CLIP)
            # Only search collections with matching dimensions
            
            # Search text collection if modality is text or None, and embedding is 384D
            if modality in [None, "text"] and embedding_dim == 384:
                try:
                    text_results = self.text_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=where
                    )
                    
                    if text_results["ids"] and len(text_results["ids"][0]) > 0:
                        for i in range(len(text_results["ids"][0])):
                            all_results.append({
                                "id": text_results["ids"][0][i],
                                "document": text_results["documents"][0][i],
                                "metadata": text_results["metadatas"][0][i],
                                "distance": text_results["distances"][0][i] if "distances" in text_results else None,
                                "modality": "text"
                            })
                except Exception as e:
                    logger.warning(f"Error searching text collection: {e}")
            
            # Search image collection if modality is image or None, and embedding is 512D
            if modality in [None, "image"] and embedding_dim == 512:
                try:
                    image_results = self.image_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=where
                    )
                    
                    if image_results["ids"] and len(image_results["ids"][0]) > 0:
                        for i in range(len(image_results["ids"][0])):
                            all_results.append({
                                "id": image_results["ids"][0][i],
                                "document": image_results["documents"][0][i],
                                "metadata": image_results["metadatas"][0][i],
                                "distance": image_results["distances"][0][i] if "distances" in image_results else None,
                                "modality": "image"
                            })
                except Exception as e:
                    logger.warning(f"Error searching image collection: {e}")
            
            # Sort by distance and return top n_results
            all_results.sort(key=lambda x: x.get("distance", float("inf")))
            return all_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete_documents(self, file_source: str):
        """
        Delete all documents associated with a file source
        
        Args:
            file_source: Source file path to delete
        """
        try:
            total_deleted = 0
            
            # Delete from text collection
            try:
                text_results = self.text_collection.get(
                    where={"source": file_source}
                )
                if text_results["ids"]:
                    self.text_collection.delete(ids=text_results["ids"])
                    total_deleted += len(text_results["ids"])
            except Exception as e:
                logger.warning(f"Error deleting from text collection: {e}")
            
            # Delete from image collection
            try:
                image_results = self.image_collection.get(
                    where={"source": file_source}
                )
                if image_results["ids"]:
                    self.image_collection.delete(ids=image_results["ids"])
                    total_deleted += len(image_results["ids"])
            except Exception as e:
                logger.warning(f"Error deleting from image collection: {e}")
            
            if total_deleted > 0:
                logger.info(f"Deleted {total_deleted} documents for source: {file_source}")
            else:
                logger.warning(f"No documents found for source: {file_source}")
            
            return total_deleted
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def get_all_sources(self) -> List[str]:
        """
        Get list of all unique source files
        
        Returns:
            List of source file paths
        """
        try:
            sources = set()
            
            # Get sources from text collection
            try:
                text_results = self.text_collection.get()
                if text_results["metadatas"]:
                    for metadata in text_results["metadatas"]:
                        if "source" in metadata:
                            sources.add(metadata["source"])
            except Exception as e:
                logger.warning(f"Error getting text sources: {e}")
            
            # Get sources from image collection
            try:
                image_results = self.image_collection.get()
                if image_results["metadatas"]:
                    for metadata in image_results["metadatas"]:
                        if "source" in metadata:
                            sources.add(metadata["source"])
            except Exception as e:
                logger.warning(f"Error getting image sources: {e}")
            
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        try:
            text_count = self.text_collection.count() if self.text_collection else 0
            image_count = self.image_collection.count() if self.image_collection else 0
            return text_count + image_count
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

