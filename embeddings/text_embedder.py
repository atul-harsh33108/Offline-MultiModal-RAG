"""
Text embedding model using sentence-transformers
"""
import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import TEXT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generate embeddings for text content"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            try:
                logger.info(f"Loading text embedding model: {TEXT_EMBEDDING_MODEL}")
                self.model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
                logger.info("Text embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading text embedding model: {e}")
                raise
    
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        if self.model is None:
            self._load_model()
        
        try:
            # Handle single string input
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            
            # Convert to list format
            if len(embeddings.shape) == 1:
                return embeddings.tolist()
            else:
                return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self.model is None:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()

