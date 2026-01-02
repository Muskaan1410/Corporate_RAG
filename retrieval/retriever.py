"""Retriever for semantic search over vector store."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from embedding import Embedder
from storage import VectorStore
from .query_rewriter import QueryRewriter


class Retriever:
    """Retriever for semantic search over vector store."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        query_rewriter: Optional[QueryRewriter] = None
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance with stored embeddings
            embedder: Embedder instance for query embedding
            query_rewriter: Optional QueryRewriter for query rephrasing
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_rewriter = query_rewriter
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query string
            k: Number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0 for cosine)
            
        Returns:
            List of chunk dictionaries with added 'similarity_score' key
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Filter by minimum score and format results
        retrieved_chunks = []
        for chunk, score in results:
            if score >= min_score:
                # Create a copy of chunk with similarity score
                chunk_with_score = chunk.copy()
                chunk_with_score['similarity_score'] = score
                retrieved_chunks.append(chunk_with_score)
        
        return retrieved_chunks
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant chunks with similarity scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results
    
    def retrieve_with_rephrasing(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        num_variations: int = 2,
        k_per_query: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using query rephrasing.
        
        Generates multiple query variations, retrieves with each, and merges results.
        
        Args:
            query: Search query string
            k: Final number of results to return
            min_score: Minimum similarity score threshold
            num_variations: Number of query variations to generate
            k_per_query: Number of results to retrieve per query variation
            
        Returns:
            List of chunk dictionaries with similarity scores (merged and deduplicated)
        """
        if self.query_rewriter is None:
            # Fallback to regular retrieval if no rewriter
            return self.retrieve(query, k=k, min_score=min_score)
        
        # Generate query variations
        query_variations = self.query_rewriter.rephrase(query, num_variations=num_variations)
        
        # Retrieve with each variation
        all_results = {}  # Use dict to deduplicate by chunk content
        
        for q in query_variations:
            chunks = self.retrieve(q, k=k_per_query, min_score=min_score)
            
            for chunk in chunks:
                # Use content as key for deduplication
                content_key = chunk.get('content', '')[:100]  # First 100 chars as key
                
                if content_key not in all_results:
                    all_results[content_key] = chunk
                else:
                    # Keep the chunk with higher similarity score
                    existing_score = all_results[content_key].get('similarity_score', 0)
                    new_score = chunk.get('similarity_score', 0)
                    if new_score > existing_score:
                        all_results[content_key] = chunk
        
        # Convert to list and sort by similarity score
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Return top k
        return merged_results[:k]

