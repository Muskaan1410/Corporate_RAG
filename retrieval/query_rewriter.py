"""Query rephrasing using LLM for better retrieval."""

from typing import List
from llm import LLMClient


class QueryRewriter:
    """Rewrites queries using LLM to improve retrieval."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize query rewriter.
        
        Args:
            llm_client: LLMClient instance for generating query variations
        """
        self.llm_client = llm_client
    
    def rephrase(
        self,
        query: str,
        num_variations: int = 2
    ) -> List[str]:
        """
        Generate query variations using LLM.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate (default: 2)
            
        Returns:
            List of queries including original + variations
        """
        if num_variations == 0:
            return [query]
        
        # Create prompt for query rephrasing
        prompt = f"""Generate {num_variations} different ways to ask this question for document search. 
Each variation should use different words but have the same meaning. 
Focus on terms that might appear in formal documents.

Original query: {query}

Generate {num_variations} variations (one per line, no numbering):"""
        
        try:
            # Generate variations
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that rephrases questions for better document search.",
                max_tokens=200,
                temperature=0.7
            )
            
            # Parse variations (split by newlines, clean up)
            variations = [
                line.strip()
                for line in response.strip().split('\n')
                if line.strip() and len(line.strip()) > 10  # Filter out empty/short lines
            ]
            
            # Limit to requested number
            variations = variations[:num_variations]
            
            # Combine with original query
            all_queries = [query] + variations
            
            return all_queries
            
        except Exception as e:
            # If rephrasing fails, just return original query
            print(f"Warning: Query rephrasing failed: {e}. Using original query.")
            return [query]

