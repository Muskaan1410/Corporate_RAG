# Corporate RAG System

A production-ready, modular Retrieval-Augmented Generation (RAG) system built in Python. This system enables intelligent document Q&A by combining semantic search with large language models to provide accurate, context-aware answers from your document corpus.

## Overview

This RAG system implements a complete pipeline for document ingestion, processing, and intelligent querying. It's designed for corporate environments where you need to quickly find answers from large collections of documents (PDFs, DOCX files) without manually searching through them.

### Architecture

The system follows a modular architecture with clear separation of concerns:

1. **Document Ingestion** → Extract text from PDF and DOCX files
2. **Text Chunking** → Split documents into manageable chunks with overlap
3. **Embedding Generation** → Convert text chunks into vector embeddings
4. **Vector Storage** → Store embeddings in FAISS for fast similarity search
5. **Query Processing** → Rephrase queries using LLM for better retrieval
6. **Semantic Retrieval** → Find most relevant document chunks
7. **Answer Generation** → Use LLM to generate answers from retrieved context

## Use Cases

### 1. **Corporate Knowledge Base Q&A**
- **Problem**: Employees spend hours searching through policy documents, guidelines, and manuals
- **Solution**: Ask questions in natural language and get instant, accurate answers
- **Example**: "What are the eligibility criteria for PMAY?" → Get precise answer from government policy documents

### 2. **Customer Support Automation**
- **Problem**: Support teams repeatedly answer the same questions from documentation
- **Solution**: Automated Q&A system that references product manuals, FAQs, and support docs
- **Example**: "How do I reset my password?" → Answer pulled from user documentation

### 3. **Legal & Compliance Document Search**
- **Problem**: Legal teams need to quickly find relevant clauses, regulations, or precedents
- **Solution**: Semantic search across legal documents, contracts, and compliance materials
- **Example**: "What are the data retention requirements?" → Relevant sections from compliance docs

### 4. **Research & Academic Document Analysis**
- **Problem**: Researchers need to extract information from large collections of papers and reports
- **Solution**: Query research documents to find relevant findings, methodologies, or conclusions
- **Example**: "What are the key findings on Alzheimer's disease?" → Summarized insights from research papers

### 5. **Internal Documentation Portal**
- **Problem**: Onboarding new employees requires extensive documentation review
- **Solution**: Interactive Q&A system for company policies, procedures, and best practices
- **Example**: "What is the process for requesting time off?" → Step-by-step answer from HR docs

## Key Features

- **Multi-Format Support**: PDF and DOCX document ingestion
- **Intelligent Chunking**: Configurable chunk size (1500 chars) with overlap (300 chars) for context preservation
- **State-of-the-Art Embeddings**: Sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **Efficient Vector Storage**: FAISS-based storage with cosine similarity for fast retrieval
- **Query Enhancement**: LLM-powered query rephrasing for improved retrieval accuracy
- **Semantic Search**: Advanced retrieval with result merging and deduplication
- **LLM Integration**: LLaMA 3.2 via Ollama for context-aware answer generation
- **RESTful API**: FastAPI-based web service for easy integration
- **Modular Design**: Clean, maintainable codebase with separate modules for each component

## Project Structure

```
.
├── ingestion/          # Document loading and text extraction
│   ├── document_loader.py
│   └── __init__.py
├── chunking/           # Text splitting with overlap
│   ├── text_splitter.py
│   └── __init__.py
├── embedding/          # Vector embedding generation
│   ├── embedder.py
│   └── __init__.py
├── storage/            # FAISS vector storage
│   ├── vector_store.py
│   └── __init__.py
├── retrieval/          # Semantic search and query rephrasing
│   ├── retriever.py
│   ├── query_rewriter.py
│   └── __init__.py
├── llm/                # LLM integration (Ollama)
│   ├── llm_client.py
│   └── __init__.py
├── api/                # FastAPI web service
│   ├── main.py
│   ├── models.py
│   └── __init__.py
├── test_pipeline.py    # Build vector store from documents
├── test_rag.py         # Complete RAG pipeline test
├── test_api.py         # API endpoint testing
├── run_api.py          # Start FastAPI server
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama (for LLM)

1. Install Ollama from: https://ollama.ai
2. Start the server: `ollama serve`
3. Pull the LLaMA model: `ollama pull llama3.2`
4. See `SETUP_LLM.md` for detailed instructions

### 3. Build Vector Store

Place your documents (PDF/DOCX) in a `data/` folder, then run:

```bash
python test_pipeline.py
```

This will:
- Load all documents from the `data/` folder
- Chunk the text
- Generate embeddings
- Save the vector store to disk

### 4. Test RAG Pipeline

```bash
python test_rag.py
```

This tests the complete pipeline: retrieval → LLM answer generation

## Usage

### Building the Index

```python
from ingestion import DocumentLoader
from chunking import TextSplitter
from embedding import Embedder
from storage import VectorStore

# Load documents
loader = DocumentLoader()
docs = loader.load_documents(["data/"])

# Chunk
splitter = TextSplitter(chunk_size=1500, chunk_overlap=300)
chunks = splitter.split_documents(docs)

# Embed
embedder = Embedder()
embeddings = embedder.embed_chunks(chunks)

# Store
vector_store = VectorStore(embedding_dim=384, similarity="cosine")
vector_store.add_vectors(embeddings, chunks)
vector_store.save("vector_store")
```

### Querying

```python
from storage import VectorStore
from embedding import Embedder
from retrieval import Retriever, QueryRewriter
from llm import LLMClient

# Load vector store
vector_store = VectorStore(embedding_dim=384)
vector_store.load("vector_store")

# Initialize components
embedder = Embedder()
llm_client = LLMClient(model="llama3.2")
query_rewriter = QueryRewriter(llm_client)
retriever = Retriever(vector_store, embedder, query_rewriter=query_rewriter)

# Query
query = "What is PMAY?"
chunks = retriever.retrieve_with_rephrasing(query, k=3)
answer = llm_client.generate_with_context(query, chunks)
print(answer)
```

### Using the API

Start the FastAPI server:

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

**Query Endpoint:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is PMAY?", "k": 3}'
```

**Response:**
```json
{
  "answer": "PMAY stands for Pradhan Mantri Awas Yojana..."
}
```

See `API_USAGE.md` for detailed API documentation.

## Configuration

- **Chunk Size**: 1500 characters (configurable in `TextSplitter`)
- **Chunk Overlap**: 300 characters
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM Model**: llama3.2 (via Ollama)
- **Query Variations**: 2 variations per query (for better retrieval)
- **Retrieval**: Top 3 chunks by default

## Technology Stack

- **Python 3.8+**
- **FastAPI**: Web framework for API
- **FAISS**: Vector similarity search
- **Sentence-Transformers**: Text embeddings
- **Ollama**: Local LLM inference
- **LangChain**: Text splitting utilities
- **Pydantic**: Data validation

## License

This project is open source and available for corporate and personal use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
