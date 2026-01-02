"""Test script for complete RAG pipeline (Retrieval + LLM)."""

from storage import VectorStore
from embedding import Embedder
from retrieval import Retriever, QueryRewriter
from llm import LLMClient

# Fix Unicode encoding for Windows console
def safe_print(text):
    """Print text safely handling Unicode characters."""
    try:
        # Try UTF-8 first
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII with error handling
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

# Load saved vector store
print("Loading vector store...")
vector_store = VectorStore(embedding_dim=384)
vector_store.load("vector_store")
print(f"[OK] Loaded {vector_store.get_stats()['num_vectors']} vectors")

# Initialize components
print("\nInitializing components...")
embedder = Embedder(model_name="all-MiniLM-L6-v2")

# Initialize LLM (make sure Ollama is running and model is pulled)
print("Initializing LLM client...")
try:
    llm_client = LLMClient(model="llama3.2")
    print("[OK] LLM client ready")
except Exception as e:
    print(f"[ERROR] LLM initialization failed: {e}")
    print("Make sure Ollama is running: ollama serve")
    print("And model is pulled: ollama pull llama3.2")
    exit(1)

# Initialize query rewriter and retriever
print("Setting up query rephrasing...")
query_rewriter = QueryRewriter(llm_client)
retriever = Retriever(vector_store, embedder, query_rewriter=query_rewriter)
print("[OK] Retriever with query rephrasing ready")

# Test queries - covering both PMAY and PMJAY documents
test_queries = [
    # PMAY queries
    "What is PMAY?",
    "What are the eligibility criteria for PMAY?",
    "What are the different verticals under PMAY-U 2.0?",
    
    # PMJAY queries
    "What is PMJAY?",
    "How does PMJAY work?",
    "When was PMJAY launched?"
]

print("\n" + "=" * 60)
print("RAG Pipeline Test: Query -> Retrieve -> Generate Answer")
print("=" * 60)

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("-" * 60)
    
    # Step 1: Retrieve relevant chunks with query rephrasing
    print("\n[Step 1] Retrieving relevant chunks (with query rephrasing)...")
    retrieved_chunks = retriever.retrieve_with_rephrasing(
        query=query,
        k=3,
        min_score=0.0,
        num_variations=2,
        k_per_query=3
    )
    print(f"Retrieved {len(retrieved_chunks)} chunks (merged from multiple query variations)")
    
    if retrieved_chunks:
        print("\nTop retrieved chunk:")
        top_chunk = retrieved_chunks[0]
        print(f"  Score: {top_chunk.get('similarity_score', 0):.4f}")
        print(f"  Source: {top_chunk.get('metadata', {}).get('file_name', 'Unknown')}")
        preview = top_chunk.get('content', '')[:200]
        safe_print(f"  Preview: {preview}...")
    
    # Step 2: Generate answer using LLM
    print("\n[Step 2] Generating answer with LLM...")
    try:
        answer = llm_client.generate_with_context(
            query=query,
            context_chunks=retrieved_chunks,
            max_context_chunks=3
        )
        
        print("\nGenerated Answer:")
        print("-" * 60)
        safe_print(answer)
        print("-" * 60)
        
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")

print("\n" + "=" * 60)
print("RAG pipeline test complete!")
print("=" * 60)

