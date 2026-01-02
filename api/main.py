"""FastAPI application for RAG system."""

from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from storage import VectorStore
from embedding import Embedder
from retrieval import Retriever, QueryRewriter
from llm import LLMClient
from api.models import QueryRequest, QueryResponse

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0"
)

# Enable CORS (allow web clients to access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store initialized components
vector_store: VectorStore = None
embedder: Embedder = None
llm_client: LLMClient = None
retriever: Retriever = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components when server starts."""
    global vector_store, embedder, llm_client, retriever
    
    print("Initializing RAG components...")
    
    try:
        # Load vector store
        print("Loading vector store...")
        vector_store = VectorStore(embedding_dim=384)
        vector_store.load("vector_store")
        print(f"[OK] Loaded {vector_store.get_stats()['num_vectors']} vectors")
        
        # Initialize embedder
        print("Loading embedding model...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        
        # Initialize LLM client
        print("Initializing LLM client...")
        try:
            llm_client = LLMClient(model="llama3.2")
            print("[OK] LLM client ready")
        except Exception as e:
            print(f"[WARNING] LLM initialization failed: {e}")
            print("LLM features will not be available")
            llm_client = None
        
        # Initialize query rewriter and retriever
        if llm_client:
            print("Setting up query rephrasing...")
            query_rewriter = QueryRewriter(llm_client)
            retriever = Retriever(vector_store, embedder, query_rewriter=query_rewriter)
        else:
            print("Setting up retriever (without query rephrasing)...")
            retriever = Retriever(vector_store, embedder)
        
        print("[OK] All components initialized successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize components: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    - **query**: The question to ask
    - **k**: Number of chunks to retrieve (default: 3)
    - **num_variations**: Number of query variations for rephrasing (default: 2)
    - **min_score**: Minimum similarity score threshold (default: 0.0)
    """
    global retriever, llm_client
    
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not available")
    
    try:
        # Step 1: Retrieve relevant chunks
        if request.num_variations > 0 and retriever.query_rewriter:
            # Use query rephrasing
            retrieved_chunks = retriever.retrieve_with_rephrasing(
                query=request.query,
                k=request.k,
                min_score=request.min_score,
                num_variations=request.num_variations,
                k_per_query=request.k
            )
        else:
            # Use basic retrieval
            retrieved_chunks = retriever.retrieve(
                query=request.query,
                k=request.k,
                min_score=request.min_score
            )
        
        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found for the query"
            )
        
        # Step 2: Generate answer using LLM
        answer = llm_client.generate_with_context(
            query=request.query,
            context_chunks=retrieved_chunks,
            max_context_chunks=request.k
        )
        
        # Step 3: Return only the answer
        return QueryResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

