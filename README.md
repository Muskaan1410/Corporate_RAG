# Corporate RAG System

A modular Retrieval-Augmented Generation (RAG) system built in Python.

## Features

- **Document Ingestion**: PDF and DOCX support
- **Text Chunking**: Intelligent splitting with overlap (1500 chars, 300 overlap)
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **Vector Storage**: FAISS-based with cosine similarity
- **Query Rephrasing**: LLM-powered query variations for better retrieval
- **Retrieval**: Semantic search with result merging
- **LLM Integration**: LLaMA via Ollama for answer generation

## Project Structure

```
.
├── ingestion/          # Document ingestion
├── chunking/           # Text chunking
├── embedding/          # Embedding generation
├── storage/            # Vector storage (FAISS)
├── retrieval/          # Retrieval with query rephrasing
├── llm/                # LLM integration (Ollama)
├── test_pipeline.py    # Build vector store from documents
├── test_rag.py         # Complete RAG pipeline test
└── requirements.txt    # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Ollama (for LLM):
   - Install from: https://ollama.ai
   - Start server: `ollama serve`
   - Pull model: `ollama pull llama3.2`
   - See `SETUP_LLM.md` for details

3. Build vector store:
```bash
python test_pipeline.py
```

4. Test RAG pipeline:
```bash
python test_rag.py
```

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

# Load
vector_store = VectorStore(embedding_dim=384)
vector_store.load("vector_store")

# Initialize
embedder = Embedder()
llm_client = LLMClient(model="llama3.2")
query_rewriter = QueryRewriter(llm_client)
retriever = Retriever(vector_store, embedder, query_rewriter=query_rewriter)

# Query
query = "What is PMAY?"
chunks = retriever.retrieve_with_rephrasing(query, k=3)
answer = llm_client.generate_with_context(query, chunks)
```

## Configuration

- **Chunk Size**: 1500 characters (configurable in `TextSplitter`)
- **Chunk Overlap**: 300 characters
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM Model**: llama3.2 (via Ollama)
- **Query Variations**: 2 variations per query

