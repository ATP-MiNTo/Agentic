# Core RAG Agent implementation
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import config
from src.utils.logger import get_logger
from src.utils.prompt_templates import (
    SYSTEM_PROMPT, REASONING_PROMPT, GENERATION_PROMPT,
    format_documents_for_context
)
from src.agent.embedding_model import EmbeddingModel
from src.agent.llm_interface import LLMInterface

logger = get_logger()


class MedicalRAGAgent:
    """Medical Information RAG Agent."""
    
    def __init__(self, faiss_index=None, use_faiss: bool = True):
        """Initialize RAG agent.
        
        Args:
            faiss_index: FAISS index object. If None, will attempt to load from disk
            use_faiss: Whether to use FAISS for retrieval
        """
        logger.info("Initializing Medical RAG Agent...")
        
        self.embedding_model = EmbeddingModel()
        self.llm = LLMInterface()
        self.faiss_index = faiss_index
        self.use_faiss = use_faiss
        self.metadata = {}  # Maps FAISS indices to document metadata
        
        if use_faiss and faiss_index is None:
            self._load_faiss_index()
        
        logger.info("✓ RAG Agent initialized successfully")
    
    def _load_faiss_index(self):
        """Load FAISS index from disk."""
        try:
            import faiss
            import json
            
            index_path = config.FAISS_INDEX_DIR / "index.faiss"
            metadata_path = config.FAISS_INDEX_DIR / "metadata.json"
            
            if not index_path.exists():
                logger.error(f"FAISS index not found at {index_path}")
                logger.info("Run: python src/utils/index_builder.py")
                raise FileNotFoundError(f"FAISS index not found")
            
            logger.debug(f"Loading FAISS index from {index_path}")
            self.faiss_index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"✓ FAISS index loaded ({len(self.metadata)} documents)")
            
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[str], List[float], List[str]]:
        """Retrieve relevant documents using semantic search.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve. If None, uses config
        
        Returns:
            Tuple of (document_contents, similarity_scores, document_ids)
        """
        if top_k is None:
            top_k = config.TOP_K
        min_score = getattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
        
        logger.debug(f"Retrieving top {top_k} documents for query")
        start_time = time.time()
        
        try:
            # Embed query
            query_embedding = self.embedding_model.embed_text(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Extract results
            documents = []
            scores = []
            doc_ids = []
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                # Convert distance to similarity (cosine distance -> similarity)
                # For L2 distance: similarity ≈ 1 / (1 + distance)
                similarity = 1 / (1 + distance)
                
                metadata_idx = str(idx)
                if metadata_idx in self.metadata:
                    doc_id = self.metadata[metadata_idx]["id"]
                    content = self.metadata[metadata_idx]["content"]

                    if similarity < min_score:
                        logger.debug(
                            f"Skipping {doc_id} below score threshold ({similarity:.4f} < {min_score:.4f})"
                        )
                        continue
                    
                    documents.append(content)
                    scores.append(similarity)
                    doc_ids.append(doc_id)
                    
                    logger.debug(f"Retrieved: {doc_id} (score: {similarity:.4f})")
            
            elapsed = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed:.2f}s ({len(documents)} docs)")
            logger.log_retrieval(doc_ids, scores)
            
            return documents, scores, doc_ids
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def reason(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str]
    ) -> str:
        """Agent reasoning step: analyze retrieved documents.
        
        Args:
            query: Original query
            documents: Retrieved documents
            doc_ids: Document IDs for reference
        
        Returns:
            Reasoning text
        """
        logger.debug("Starting reasoning step")
        
        # Format documents for reasoning
        context = ""
        for doc_id, doc_content in zip(doc_ids, documents):
            context += f"[From: {doc_id}]\n{doc_content[:300]}...\n\n"
        
        # Create reasoning prompt
        prompt = REASONING_PROMPT.format(
            query=query,
            documents=context
        )
        
        try:
            reasoning = self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.5
            )
            
            logger.log_reasoning(reasoning)
            return reasoning
            
        except Exception as e:
            logger.error(f"Error during reasoning: {str(e)}")
            # Return fallback reasoning
            fallback = f"Analysis of {len(documents)} retrieved documents shows relevant information about: {', '.join(doc_ids[:3])}"
            return fallback
    
    def generate_response(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        scores: List[float],
        reasoning: str
    ) -> str:
        """Generate final response using LLM.
        
        Args:
            query: Original query
            documents: Retrieved documents
            doc_ids: Document IDs
            scores: Similarity scores
            reasoning: Agent reasoning
        
        Returns:
            Final response with citations
        """
        logger.debug("Starting response generation")
        start_time = time.time()
        
        # Format context with citations
        context_parts = []
        for doc_id, doc_content, score in zip(doc_ids, documents, scores):
            context_parts.append(f"[Source: {doc_id} | Confidence: {score:.1%}]\n{doc_content}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Create generation prompt
        prompt = GENERATION_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            query=query,
            context=context,
            reasoning=reasoning
        )
        
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Response generated in {elapsed:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
    
    def chat(
        self,
        query: str,
        top_k: int = None,
        return_details: bool = False,
        conversation_context: str = ""
    ) -> Dict:
        """Main chat method: process query and return response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_details: Whether to return detailed execution info
            conversation_context: Optional prior chat context to resolve follow-up questions
        
        Returns:
            Dictionary with response and optional details
        """
        logger.log_query(query)
        start_time = time.time()
        
        try:
            # Step 1: Retrieval
            retrieval_query = query
            if conversation_context:
                retrieval_query = (
                    f"{query}\n\n"
                    f"Conversation context from the same chat:\n{conversation_context}"
                )

            documents, scores, doc_ids = self.retrieve(retrieval_query, top_k)
            
            if not documents:
                logger.warning("No documents retrieved")
                return {
                    "response": "I couldn't find sufficiently relevant medical information to answer your question safely.",
                    "success": False
                }
            
            # Step 2: Reasoning
            reasoning = self.reason(query, documents, doc_ids)
            
            # Step 3: Generation
            response = self.generate_response(query, documents, doc_ids, scores, reasoning)
            
            # Log final response
            logger.log_response(response, tokens=len(response.split()))
            
            elapsed = time.time() - start_time
            logger.log_execution_time("Total query", elapsed)
            
            result = {
                "response": response,
                "success": True,
                "execution_time": elapsed
            }
            
            # Add details if requested
            if return_details:
                result.update({
                    "retrieved_documents": [
                        {"id": doc_id, "score": score, "preview": doc[:200]}
                        for doc_id, doc, score in zip(doc_ids, documents, scores)
                    ],
                    "reasoning": reasoning,
                    "query": query,
                    "conversation_context": conversation_context
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "response": f"An error occurred: {str(e)}",
                "success": False,
                "error": str(e)
            }
