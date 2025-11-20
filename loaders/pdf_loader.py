"""
PDF document loader with text extraction
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import pypdf

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load and extract text from PDF files"""
    
    def __init__(self):
        self.supported_formats = [".pdf"]
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load PDF file and extract text
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            text_content = []
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "pdf",
                "pages": []
            }
            
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(text)
                            metadata["pages"].append({
                                "page_number": page_num,
                                "text_length": len(text)
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
                
                metadata["total_pages"] = total_pages
                metadata["author"] = pdf_reader.metadata.get("/Author", "")
                metadata["title"] = pdf_reader.metadata.get("/Title", "")
            
            full_text = "\n\n".join(text_content)
            
            return {
                "content": full_text,
                "metadata": metadata,
                "chunks": self._chunk_text(full_text)
            }
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
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
                # Calculate start_index based on previous chunks (before appending current)
                start_index = len(" ".join([c["text"] for c in chunks])) if chunks else 0
                chunks.append({
                    "text": chunk_text,
                    "start_index": start_index,
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

