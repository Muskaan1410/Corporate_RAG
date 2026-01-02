"""Document loader for extracting text from PDF and DOCX files."""

import os
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from docx import Document


class DocumentLoader:
    """Loads and extracts text from PDF and DOCX files."""
    
    def __init__(self):
        """Initialize the document loader."""
        self.supported_formats = {'.pdf', '.docx'}
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single document and extract its text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with 'content' (text) and 'metadata' keys
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {self.supported_formats}")
        
        # Extract text based on file type
        if file_ext == '.pdf':
            return self._load_pdf(file_path)
        elif file_ext == '.docx':
            return self._load_docx(file_path)
    
    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file."""
        content_pages = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    content_pages.append(text)
        
        # Join all pages with double newline
        full_content = '\n\n'.join(content_pages)
        
        return {
            'content': full_content,
            'metadata': {
                'file_path': str(file_path),
                'file_name': os.path.basename(file_path),
                'file_type': 'pdf',
                'num_pages': len(content_pages)
            }
        }
    
    def _load_docx(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        content_paragraphs = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:  # Only add non-empty paragraphs
                content_paragraphs.append(text)
        
        # Join all paragraphs with double newline
        full_content = '\n\n'.join(content_paragraphs)
        
        return {
            'content': full_content,
            'metadata': {
                'file_path': str(file_path),
                'file_name': os.path.basename(file_path),
                'file_type': 'docx'
            }
        }
    
    def load_documents(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple documents from file paths and/or directories.
        
        Args:
            paths: List of file paths and/or directory paths to load
            
        Returns:
            List of document dictionaries
        """
        # Collect all file paths (expand directories)
        file_paths = []
        
        for path_str in paths:
            path = Path(path_str)
            
            if not path.exists():
                print(f"[ERROR] Path not found: {path_str}")
                continue
            
            if path.is_dir():
                # Find all supported files in directory
                for file_path in path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                        file_paths.append(str(file_path))
            elif path.is_file():
                # Add file if it's a supported format
                if path.suffix.lower() in self.supported_formats:
                    file_paths.append(str(path))
                else:
                    print(f"[ERROR] Unsupported format: {path_str}")
        
        # Load all collected files
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
                print(f"[OK] Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[ERROR] Error loading {file_path}: {e}")
        
        return documents

