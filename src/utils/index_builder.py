# FAISS Index builder utility
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import config
from src.utils.logger import get_logger
from src.utils.data_loader import DataLoader
from src.agent.embedding_model import EmbeddingModel

logger = get_logger()


class FAISSIndexBuilder:
    """Build and manage FAISS index for semantic search."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize index builder."""
        self.loader = DataLoader()
        self.embedding_model = EmbeddingModel()
        self.index = None
        self.metadata = {}
        self.data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    
    def build_index(self) -> bool:
        """Build FAISS index from raw documents.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting FAISS index build process")
        logger.log_separator()
        
        try:
            import faiss
            
            # Step 1: Load documents
            documents = self.loader.load_documents(self.data_dir)
            if not documents:
                logger.error("No documents found")
                return False
            
            # Step 2: Chunk documents
            chunks = self.loader.chunk_documents(documents)
            if not chunks:
                logger.error("Failed to create chunks")
                return False
            
            logger.info(f"Processing {len(chunks)} chunks...")
            
            # Step 3: Embed all chunks
            chunk_texts = [chunk["content"] for chunk in chunks]
            logger.info("Embedding all chunks (this may take a minute)...")
            
            embeddings = self.embedding_model.embed_batch(chunk_texts)
            logger.info(f"✓ Embedded {len(embeddings)} chunks")
            
            # Step 4: Create FAISS index
            logger.info("Creating FAISS index...")
            dimension = embeddings.shape[1]
            
            # Use simple IndexFlatL2 for small datasets (fast and accurate)
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            logger.info(f"✓ FAISS index created ({self.index.ntotal} vectors)")
            
            # Step 5: Create metadata mapping
            logger.info("Creating metadata...")
            self.metadata = {}
            for i, chunk in enumerate(chunks):
                self.metadata[str(i)] = {
                    "id": chunk["id"],
                    "doc_id": chunk["doc_id"],
                    "content": chunk["content"],
                    "word_count": chunk["word_count"],
                    "disease": chunk["disease"],
                    "chunk_index": chunk["chunk_index"]
                }
            
            # Step 6: Save index and metadata
            self._save_index_and_metadata()
            
            logger.log_separator()
            logger.info("✓ FAISS index build completed successfully!")
            
            # Print statistics
            stats = self.loader.get_stats()
            logger.info(f"Index Statistics:")
            logger.info(f"  - Total documents: {stats['total_documents']}")
            logger.info(f"  - Total chunks: {stats['total_chunks']}")
            logger.info(f"  - Total words: {stats['total_words']}")
            logger.info(f"  - Avg words/document: {stats['avg_words_per_doc']:.0f}")
            logger.info(f"  - Avg words/chunk: {stats['avg_words_per_chunk']:.0f}")
            
            return True
            
        except ImportError as e:
            logger.error(f"FAISS not installed: {str(e)}")
            logger.info("Run: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            return False
    
    def _save_index_and_metadata(self):
        """Save FAISS index and metadata to disk."""
        try:
            import faiss
            
            config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save index
            index_path = config.FAISS_INDEX_DIR / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            logger.debug(f"Saved FAISS index to {index_path}")
            
            # Save metadata
            metadata_path = config.FAISS_INDEX_DIR / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def validate_index(self) -> bool:
        """Validate that index is correct.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            index_path = config.FAISS_INDEX_DIR / "index.faiss"
            metadata_path = config.FAISS_INDEX_DIR / "metadata.json"
            
            if not index_path.exists():
                logger.error(f"Index file not found: {index_path}")
                return False
            
            if not metadata_path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            
            import faiss
            index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if index.ntotal != len(metadata):
                logger.error(f"Index size ({index.ntotal}) doesn't match metadata ({len(metadata)})")
                return False
            
            logger.info(f"✓ Index validation passed ({index.ntotal} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Error validating index: {str(e)}")
            return False


def main():
    """Main entry point for index builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index for Medical RAG")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--validate", action="store_true", help="Only validate existing index")
    
    args = parser.parse_args()
    
    builder = FAISSIndexBuilder()
    
    if args.validate:
        success = builder.validate_index()
        exit(0 if success else 1)
    
    # Check if index exists and rebuild not forced
    index_path = config.FAISS_INDEX_DIR / "index.faiss"
    if index_path.exists() and not args.rebuild:
        logger.info("Index already exists. Use --rebuild to force rebuild.")
        builder.validate_index()
        exit(0)
    
    # Build index
    start_time = time.time()
    success = builder.build_index()
    elapsed = time.time() - start_time
    
    if success:
        logger.info(f"Index build completed in {elapsed:.1f}s")
        exit(0)
    else:
        logger.error("Index build failed")
        exit(1)


if __name__ == "__main__":
    main()
