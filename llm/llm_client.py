"""LLM client for generating responses using LLaMA via Ollama."""

from typing import List, Dict, Any, Optional
import ollama


class LLMClient:
    """Client for interacting with LLaMA LLM via Ollama."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize LLM client with Ollama.
        
        Args:
            model: Model name to use (e.g., "llama3.2", "llama3.1", "llama2")
            base_url: Ollama server URL (default: localhost)
        
        Note:
            Make sure Ollama is running and the model is pulled:
            - Install Ollama: https://ollama.ai
            - Pull model: ollama pull llama3.2
        """
        self.model = model
        self.base_url = base_url
        # Set Ollama host if custom URL
        if base_url != "http://localhost:11434":
            import os
            os.environ['OLLAMA_HOST'] = base_url
        
        # Verify connection (non-blocking - just a warning)
        try:
            models = ollama.list()
            if isinstance(models, dict) and 'models' in models:
                model_names = [m.get('name', '') for m in models.get('models', [])]
                if model not in model_names:
                    print(f"Warning: Model '{model}' not found. Available models: {model_names}")
                    print(f"To install: ollama pull {model}")
        except Exception:
            # Silently continue - connection will be tested when generating
            pass
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            
        Returns:
            Generated response text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Error generating response: {e}")
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_context_chunks: int = 5
    ) -> str:
        """
        Generate response using RAG context.
        
        Args:
            query: User query
            context_chunks: List of retrieved context chunks
            system_prompt: Optional system prompt
            max_context_chunks: Maximum number of chunks to include
            
        Returns:
            Generated response text
        """
        # Limit context chunks
        limited_chunks = context_chunks[:max_context_chunks]
        
        # Build context from chunks
        context_text = "\n\n".join([
            f"[Document {i+1} - {chunk.get('metadata', {}).get('file_name', 'Unknown')}]:\n{chunk['content']}"
            for i, chunk in enumerate(limited_chunks)
        ])
        
        # Build RAG prompt with more directive system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer questions based ONLY on the provided context documents. "
                "Be confident and direct in your answers. If the context mentions something, "
                "assume it's accurate and answer accordingly. Extract relevant information and provide a clear, "
                "structured answer. Only say 'not enough information' if the context truly doesn't address "
                "the question at all. Use the information from the context to give a comprehensive answer."
            )
        
        rag_prompt = f"""Use the following context documents to answer the question. 
Be direct and confident. Extract the relevant information and provide a clear answer.

Context:
{context_text}

Question: {query}

Answer based on the context:"""
        
        return self.generate(rag_prompt, system_prompt=system_prompt)

