"""Text splitting with overlap for RAG system."""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """Splits text into chunks with configurable overlap."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters (default: 1000)
            chunk_overlap: Number of characters to overlap between chunks (default: 200)
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraphs, then sentences, then words
            length_function=len
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single document into chunks.
        
        Args:
            document: Dictionary with 'content' and 'metadata' keys
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        text = document['content']
        metadata = document.get('metadata', {})
        
        # Split the text
        chunks = self.splitter.split_text(text)
        
        # Create chunk documents with metadata
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'content': chunk,
                'metadata': {
                    **metadata,  # Preserve original metadata
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            chunk_documents.append(chunk_doc)
        
        return chunk_documents
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunk dictionaries from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks

