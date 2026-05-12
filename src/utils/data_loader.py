# Data loading and processing utilities
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import config
from src.utils.logger import get_logger

logger = get_logger()


class DataLoader:
    """Load and process medical documents."""
    
    def __init__(self):
        """Initialize data loader."""
        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []
    
    def load_documents(self, directory: Path = None) -> List[Dict]:
        """Load all documents from directory.
        
        Args:
            directory: Path to documents directory. If None, uses config.DATA_DIR
        
        Returns:
            List of document dictionaries with metadata
        """
        if directory is None:
            directory = config.DATA_DIR
        
        directory = Path(directory)
        logger.info(f"Loading documents from: {directory}")
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        self.documents = []
        
        # Recursively load documents
        for file_path in directory.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty document: {file_path}")
                    continue
                
                # Extract relative path for document ID
                doc_id = str(file_path.relative_to(config.DATA_DIR))
                
                doc = {
                    "id": doc_id,
                    "path": str(file_path),
                    "content": content,
                    "word_count": len(content.split()),
                    "disease": self._extract_disease_folder(file_path)
                }
                
                self.documents.append(doc)
                logger.debug(f"Loaded: {doc_id} ({doc['word_count']} words)")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(self.documents)} documents")
        return self.documents
    
    @staticmethod
    def _extract_disease_folder(file_path: Path) -> str:
        """Extract disease folder from file path."""
        parts = file_path.parts
        if len(parts) >= 2:
            return parts[-2]  # Parent folder name (disease)
        return "unknown"
    
    def chunk_documents(
        self, 
        documents: List[Dict] = None,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict]:
        """Split documents into overlapping chunks.
        
        Args:
            documents: List of documents. If None, uses loaded documents
            chunk_size: Words per chunk. If None, uses config
            overlap: Word overlap. If None, uses config
        
        Returns:
            List of chunk dictionaries
        """
        if documents is None:
            documents = self.documents
        
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        
        if overlap is None:
            overlap = config.CHUNK_OVERLAP
        
        logger.info(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={overlap})")
        
        self.chunks = []
        chunk_id = 0
        
        for doc in documents:
            content = doc["content"]
            
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            current_chunk = []
            current_word_count = 0
            
            for sentence in sentences:
                words_in_sentence = len(sentence.split())
                
                # If adding this sentence would exceed chunk size and we have content
                if current_word_count + words_in_sentence > chunk_size and current_chunk:
                    # Save chunk
                    chunk_text = " ".join(current_chunk)
                    chunk = {
                        "id": f"{doc['id']}_chunk_{chunk_id}",
                        "doc_id": doc["id"],
                        "content": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "disease": doc["disease"],
                        "chunk_index": chunk_id
                    }
                    self.chunks.append(chunk)
                    chunk_id += 1
                    
                    # Keep last N words for overlap
                    overlap_words = overlap
                    words = chunk_text.split()
                    current_chunk = words[-overlap_words:] if len(words) > overlap_words else words
                    current_word_count = len(current_chunk)
                
                current_chunk.append(sentence)
                current_word_count += words_in_sentence
            
            # Don't forget last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = {
                    "id": f"{doc['id']}_chunk_{chunk_id}",
                    "doc_id": doc["id"],
                    "content": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "disease": doc["disease"],
                    "chunk_index": chunk_id
                }
                self.chunks.append(chunk)
                chunk_id += 1
        
        logger.info(f"Created {len(self.chunks)} chunks from documents")
        return self.chunks
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        """Get document by ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document dictionary or None if not found
        """
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded data."""
        total_words = sum(doc.get("word_count", 0) for doc in self.documents)
        total_chunks = len(self.chunks)
        
        return {
            "total_documents": len(self.documents),
            "total_words": total_words,
            "total_chunks": total_chunks,
            "avg_words_per_doc": total_words / len(self.documents) if self.documents else 0,
            "avg_words_per_chunk": total_words / total_chunks if total_chunks > 0 else 0
        }
    
    def validate_documents(self) -> Tuple[bool, List[str]]:
        """Validate documents for issues.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.documents:
            errors.append("No documents loaded")
            return False, errors
        
        for doc in self.documents:
            if not doc.get("content"):
                errors.append(f"Empty content in {doc['id']}")
            
            if doc.get("word_count", 0) < 10:
                errors.append(f"Document {doc['id']} has very few words ({doc.get('word_count')})")
        
        is_valid = len(errors) == 0
        return is_valid, errors


# Convenience function
def load_medical_documents() -> List[Dict]:
    """Load all medical documents and return chunks.
    
    Returns:
        List of document chunks ready for embedding
    """
    loader = DataLoader()
    docs = loader.load_documents()
    chunks = loader.chunk_documents(docs)
    
    stats = loader.get_stats()
    logger.info(f"Data stats: {stats}")
    
    return chunks
