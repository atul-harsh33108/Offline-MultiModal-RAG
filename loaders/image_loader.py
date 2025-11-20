"""
Image loader with OCR text extraction
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import cv2
import numpy as np
from config import TESSERACT_CMD

# Set Tesseract command path if provided
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logger = logging.getLogger(__name__)


class ImageLoader:
    """Load images and extract text using OCR"""
    
    def __init__(self):
        self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load image file and extract text using OCR
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            # Load image
            image = Image.open(file_path)
            image_array = np.array(image)
            
            # Convert to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Apply thresholding for better OCR
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text using Tesseract
            try:
                ocr_text = pytesseract.image_to_string(thresh, lang='eng')
            except Exception as e:
                logger.warning(f"OCR failed, trying without preprocessing: {e}")
                ocr_text = pytesseract.image_to_string(image, lang='eng')
            
            # Get image metadata
            width, height = image.size
            format_name = image.format or "Unknown"
            
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "image",
                "image_format": format_name,
                "dimensions": {"width": width, "height": height},
                "has_text": bool(ocr_text.strip())
            }
            
            # Extract additional OCR data
            try:
                ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
                word_count = len([w for w in ocr_data.get('text', []) if w.strip()])
                metadata["word_count"] = word_count
            except:
                pass
            
            chunks = []
            if ocr_text.strip():
                chunks = self._chunk_text(ocr_text)
            
            return {
                "content": ocr_text,
                "metadata": metadata,
                "chunks": chunks,
                "image_path": str(file_path)  # Store path for image embedding
            }
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Full text content
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks with metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_index": len(" ".join([c["text"] for c in chunks])) if chunks else 0,
                    "length": len(chunk_text)
                })
                
                # Overlap: keep last N words
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in overlap_words)
            
            current_chunk.append(word)
            current_length += word_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_index": len(" ".join([c["text"] for c in chunks])),
                "length": len(chunk_text)
            })
        
        return chunks

