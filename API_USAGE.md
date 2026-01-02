# API Usage Guide

## Starting the Server

```bash
python run_api.py
```

The server will start at: `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly.

## Endpoints

### 1. Root Endpoint
```
GET http://localhost:8000/
```
Returns basic API information.

### 2. Health Check
```
GET http://localhost:8000/health
```
Check if the API and RAG system are ready.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_loaded": true,
  "num_vectors": 203,
  "llm_ready": true
}
```

### 3. Get Statistics
```
GET http://localhost:8000/stats
```
Get statistics about the vector store and models.

**Response:**
```json
{
  "vector_store_stats": {
    "num_vectors": 203,
    "embedding_dim": 384,
    "similarity_metric": "cosine"
  },
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "llm_model": "llama3.2"
}
```

### 4. Query Endpoint (Main)
```
POST http://localhost:8000/query
```

**Request Body:**
```json
{
  "query": "What is PMAY?",
  "k": 3,
  "num_variations": 2,
  "min_score": 0.0
}
```

**Parameters:**
- `query` (required): The question to ask
- `k` (optional, default: 3): Number of chunks to retrieve
- `num_variations` (optional, default: 2): Number of query variations for rephrasing
- `min_score` (optional, default: 0.0): Minimum similarity score (0.0 to 1.0)

**Response:**
```json
{
  "answer": "PMAY stands for Pradhan Mantri Awas Yojana...",
  "chunks": [
    {
      "content": "Chunk text...",
      "score": 0.75,
      "source": "document.pdf",
      "chunk_index": 0,
      "total_chunks": 10
    }
  ],
  "sources": ["document1.pdf", "document2.pdf"],
  "query_time": 1.23,
  "num_chunks_retrieved": 3
}
```

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is PMAY?"}'
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is PMAY?"}
)

data = response.json()
print(data["answer"])
```

### Using the test script

```bash
python test_api.py
```

### Using the Swagger UI

1. Start the server: `python run_api.py`
2. Open browser: http://localhost:8000/docs
3. Click on `/query` endpoint
4. Click "Try it out"
5. Enter your query and click "Execute"

## Example Queries

```bash
# Simple query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is PMAY?"}'

# Query with custom parameters
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the eligibility criteria?",
       "k": 5,
       "num_variations": 3,
       "min_score": 0.5
     }'
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: No chunks found
- `503`: Service unavailable (components not initialized)

Error response format:
```json
{
  "detail": "Error message here"
}
```

