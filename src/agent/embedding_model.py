# Embedding model interface
import numpy as np
from typing import List, Optional
import config
from src.utils.logger import get_logger

logger = get_logger()


class EmbeddingModel:
    """Interface for embedding model (bge-small-en-v1.5)."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model.
        
        Args:
            model_name: Name of model. If None, uses config.EMBEDDING_MODEL
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.embedding_dim = config.EMBEDDING_DIMENSION
        self._load_model()
    
    def _load_model(self):
        """Load embedding model from HuggingFace."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Import here to handle cases where transformers not installed yet
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(
                self.model_name,
                device=config.DEVICE,
                cache_folder=str(config.CACHE_DIR)
            )
            
            logger.info(f"✓ Embedding model loaded successfully")
            
        except ImportError:
            logger.error("sentence_transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector (1D numpy array)
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if isinstance(text, str):
            text = text.strip()
            if not text:
                logger.warning("Empty text provided for embedding")
                return np.zeros(self.embedding_dim, dtype=np.float32)
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing. If None, uses config.BATCH_SIZE
        
        Returns:
            Embedding matrix (2D numpy array, shape: [len(texts), embedding_dim])
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        # Clean texts
        texts = [str(t).strip() for t in texts if t]
        
        if not texts:
            logger.warning("No valid texts to embed")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        logger.debug(f"Embedding batch of {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        return embeddings.astype(np.float32)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Similarity score (0-1)
        """
        # Normalize embeddings (already done in encode, but ensure)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
