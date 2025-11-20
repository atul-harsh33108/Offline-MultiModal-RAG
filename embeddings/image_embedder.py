"""
Image embedding model using CLIP
"""
import logging
from typing import List, Union
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from config import IMAGE_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """Generate embeddings for images using CLIP"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Lazy load the CLIP model"""
        if self.model is None:
            try:
                logger.info(f"Loading image embedding model: {IMAGE_EMBEDDING_MODEL}")
                self.model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
                self.processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Image embedding model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Error loading image embedding model: {e}")
                raise
    
    def embed(self, image_paths: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for image(s)
        
        Args:
            image_paths: Single image path or list of image paths
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        if self.model is None:
            self._load_model()
        
        try:
            # Handle single path input
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            # Load and process images
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            
            # Process images
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to CPU and list format
            embeddings = image_features.cpu().numpy()
            
            if len(embeddings.shape) == 1:
                return embeddings.tolist()
            else:
                return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"Error generating image embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self.model is None:
            self._load_model()
        return self.model.config.projection_dim

