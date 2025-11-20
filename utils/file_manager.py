"""
File management utilities for storing and organizing uploaded files
"""
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from config import DOCUMENTS_DIR, IMAGES_DIR, AUDIO_DIR

logger = logging.getLogger(__name__)


class FileManager:
    """Manage file storage and retrieval"""
    
    def __init__(self):
        self.documents_dir = DOCUMENTS_DIR
        self.images_dir = IMAGES_DIR
        self.audio_dir = AUDIO_DIR
        
        # Ensure directories exist
        for dir_path in [self.documents_dir, self.images_dir, self.audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def store_file(self, file_path: Path, file_type: str) -> Path:
        """
        Store a file in the appropriate directory
        
        Args:
            file_path: Source file path
            file_type: Type of file (pdf, docx, image, audio)
            
        Returns:
            Path to stored file
        """
        try:
            # Determine target directory
            if file_type in ["pdf", "docx"]:
                target_dir = self.documents_dir
            elif file_type == "image":
                target_dir = self.images_dir
            elif file_type == "audio":
                target_dir = self.audio_dir
            else:
                target_dir = self.documents_dir
            
            # Copy file to target directory
            target_path = target_dir / file_path.name
            
            # Handle duplicate names
            counter = 1
            while target_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(file_path, target_path)
            logger.info(f"Stored file: {target_path}")
            
            return target_path
            
        except Exception as e:
            logger.error(f"Error storing file: {e}")
            raise
    
    def delete_file(self, file_source: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            file_source: Source file path
            
        Returns:
            True if file was deleted, False otherwise
        """
        try:
            file_path = Path(file_source)
            
            # Check if file exists in any storage directory
            for storage_dir in [self.documents_dir, self.images_dir, self.audio_dir]:
                potential_path = storage_dir / file_path.name
                if potential_path.exists():
                    potential_path.unlink()
                    logger.info(f"Deleted file: {potential_path}")
                    return True
            
            # Also check if the source path itself exists
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            
            logger.warning(f"File not found for deletion: {file_source}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def get_file_info(self, file_source: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a stored file
        
        Args:
            file_source: Source file path
            
        Returns:
            Dictionary with file information or None
        """
        try:
            file_path = Path(file_source)
            
            # Check all storage directories
            for storage_dir in [self.documents_dir, self.images_dir, self.audio_dir]:
                potential_path = storage_dir / file_path.name
                if potential_path.exists():
                    stat = potential_path.stat()
                    return {
                        "source": str(potential_path),
                        "file_name": potential_path.name,
                        "file_type": potential_path.suffix[1:].lower(),
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "storage_dir": str(storage_dir)
                    }
            
            # Check if source path exists
            if file_path.exists():
                stat = file_path.stat()
                return {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix[1:].lower(),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "storage_dir": "original"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return None

