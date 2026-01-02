"""Text embedding generation using sentence-transformers."""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with a sentence-transformer model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
                       Default: "all-MiniLM-L6-v2" (384 dimensions, fast)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            
        Returns:
            Numpy array of shape (num_chunks, embedding_dim)
        """
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple text strings.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings

