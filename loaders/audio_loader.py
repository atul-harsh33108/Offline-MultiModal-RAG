"""
Audio loader with Whisper speech-to-text conversion
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import whisper
import librosa
from config import WHISPER_MODEL

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load audio files and convert to text using Whisper"""
    
    def __init__(self):
        self.supported_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
        self.whisper_model = None
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            try:
                logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load audio file and convert to text using Whisper
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with transcribed content and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio file
            logger.info(f"Loading audio file: {file_path}")
            audio, sr = librosa.load(str(file_path), sr=16000)  # Whisper uses 16kHz
            
            # Transcribe using Whisper
            logger.info("Transcribing audio...")
            result = self.whisper_model.transcribe(audio, language="en")
            
            # Extract full text
            full_text = result["text"]
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment.get("no_speech_prob", 0)
                })
            
            # Get audio metadata
            duration = len(audio) / sr  # Duration in seconds
            
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "audio",
                "duration_seconds": duration,
                "sample_rate": sr,
                "segments_count": len(segments),
                "language": result.get("language", "en")
            }
            
            chunks = []
            if full_text.strip():
                chunks = self._chunk_text(full_text)
            
            return {
                "content": full_text,
                "metadata": metadata,
                "chunks": chunks,
                "segments": segments  # Store segments for citation
            }
            
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
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

