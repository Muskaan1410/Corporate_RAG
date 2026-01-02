"""Complete test script for RAG pipeline (ingestion -> chunking -> embedding -> storage)."""

from ingestion import DocumentLoader
from chunking import TextSplitter
from embedding import Embedder
from storage import VectorStore


def test_pipeline(docs=None, chunks=None, embeddings=None):
    """
    Test the complete pipeline up to vector storage.
    
    Args:
        docs: Optional pre-loaded documents (skip ingestion if provided)
        chunks: Optional pre-created chunks (skip chunking if provided)
        embeddings: Optional pre-generated embeddings (skip embedding if provided)
    
    Returns:
        Tuple of (docs, chunks, embeddings, vector_store) for reuse
    """
    
    print("=" * 60)
    print("RAG Pipeline Test: Ingestion -> Chunking -> Embedding -> Storage")
    print("=" * 60)
    
    # Step 1: Document Ingestion (skip if docs provided)
    if docs is None:
        print("\n[STEP 1] Document Ingestion")
        print("-" * 60)
        loader = DocumentLoader()
        
        try:
            docs = loader.load_documents(["data/"])
            print(f"\n[OK] Successfully loaded {len(docs)} documents")
            
            # Show document stats
            for i, doc in enumerate(docs, 1):
                content_len = len(doc['content'])
                metadata = doc['metadata']
                print(f"\n  Document {i}:")
                print(f"    File: {metadata['file_name']}")
                print(f"    Type: {metadata['file_type']}")
                print(f"    Content length: {content_len:,} characters")
                if 'num_pages' in metadata:
                    print(f"    Pages: {metadata['num_pages']}")
                print(f"    Preview: {doc['content'][:150]}...")
            
            if len(docs) == 0:
                print("[ERROR] No documents found in data/ folder")
                return None, None, None
            
        except Exception as e:
            print(f"[ERROR] Error in ingestion: {e}")
            return None, None, None
    else:
        print("\n[STEP 1] Document Ingestion (using provided documents)")
        print("-" * 60)
        print(f"[OK] Using {len(docs)} pre-loaded documents")
    
    # Step 2: Chunking (skip if chunks provided, use docs from step 1)
    if chunks is None:
        print("\n\n[STEP 2] Text Chunking")
        print("-" * 60)
        splitter = TextSplitter(chunk_size=1500, chunk_overlap=300)
        
        try:
            chunks = splitter.split_documents(docs)
            print(f"\n[OK] Successfully created {len(chunks)} chunks from {len(docs)} documents")
            
            # Show chunk stats
            chunk_lengths = [len(chunk['content']) for chunk in chunks]
            print(f"\n  Chunk Statistics:")
            print(f"    Average length: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
            print(f"    Min length: {min(chunk_lengths)} characters")
            print(f"    Max length: {max(chunk_lengths)} characters")
            
            print(f"\n  First 3 chunks preview:")
            for i in range(min(3, len(chunks))):
                chunk = chunks[i]
                print(f"\n    Chunk {i+1} (index {chunk['metadata']['chunk_index']}):")
                print(f"      Length: {len(chunk['content'])} chars")
                print(f"      From: {chunk['metadata']['file_name']}")
                print(f"      Preview: {chunk['content'][:100]}...")
            
            if len(chunks) == 0:
                print("[ERROR] No chunks created")
                return docs, None, None
            
        except Exception as e:
            print(f"[ERROR] Error in chunking: {e}")
            return docs, None, None
    else:
        print("\n\n[STEP 2] Text Chunking (using provided chunks)")
        print("-" * 60)
        print(f"[OK] Using {len(chunks)} pre-created chunks")
    
    # Step 3: Embeddings (use chunks from step 2)
    if embeddings is None:
        print("\n\n[STEP 3] Embedding Generation")
        print("-" * 60)
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        
        try:
            print("\n  Generating embeddings (this may take a moment)...")
            embeddings = embedder.embed_chunks(chunks)
            
            print(f"\n[OK] Successfully generated embeddings")
            print(f"\n  Embedding Statistics:")
            print(f"    Shape: {embeddings.shape}")
            print(f"    Number of embeddings: {embeddings.shape[0]}")
            print(f"    Embedding dimension: {embeddings.shape[1]}")
            print(f"    First embedding (first 5 values): {embeddings[0][:5]}")
            print(f"    Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            
        except Exception as e:
            print(f"[ERROR] Error in embedding: {e}")
            return docs, chunks, None, None
    else:
        print("\n\n[STEP 3] Embedding Generation (using provided embeddings)")
        print("-" * 60)
        print(f"[OK] Using pre-generated embeddings: {embeddings.shape}")
    
    # Step 4: Vector Storage (use embeddings and chunks from previous steps)
    print("\n\n[STEP 4] Vector Storage")
    print("-" * 60)
    
    try:
        # Initialize vector store
        vector_store = VectorStore(
            embedding_dim=embeddings.shape[1],
            similarity="cosine"
        )
        
        # Add vectors and chunks
        print("\n  Adding vectors to store...")
        vector_store.add_vectors(embeddings, chunks)
        
        print(f"\n[OK] Successfully stored vectors")
        stats = vector_store.get_stats()
        print(f"\n  Vector Store Statistics:")
        print(f"    Number of vectors: {stats['num_vectors']}")
        print(f"    Embedding dimension: {stats['embedding_dim']}")
        print(f"    Similarity metric: {stats['similarity_metric']}")
        
        # Test search
        print(f"\n  Testing search functionality...")
        test_query_embedding = embeddings[0]  # Use first embedding as test query
        results = vector_store.search(test_query_embedding, k=3)
        print(f"    Found {len(results)} results for test query")
        print(f"    Top result similarity: {results[0][1]:.4f}")
        
        # Save to disk
        print(f"\n  Saving vector store to disk...")
        save_path = "vector_store"
        vector_store.save(save_path)
        print(f"    [OK] Saved to: {save_path}.index, {save_path}.chunks, {save_path}.meta")
        print(f"    - Embeddings: {save_path}.index")
        print(f"    - Chunks with metadata: {save_path}.chunks")
        print(f"    - Store metadata: {save_path}.meta")
        
    except Exception as e:
        print(f"[ERROR] Error in vector storage: {e}")
        return docs, chunks, embeddings, None
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Pipeline Test Summary")
    print("=" * 60)
    print(f"[OK] Documents loaded: {len(docs)}")
    print(f"[OK] Chunks created: {len(chunks)}")
    print(f"[OK] Embeddings generated: {embeddings.shape[0]}")
    print(f"[OK] Embedding dimension: {embeddings.shape[1]}")
    print(f"[OK] Vectors stored: {stats['num_vectors']}")
    print("\nAll steps completed successfully!")
    print("=" * 60)
    
    return docs, chunks, embeddings, vector_store


if __name__ == "__main__":
    # Run full pipeline
    docs, chunks, embeddings, vector_store = test_pipeline()
    
    # Example: Reuse results for further testing
    # if docs and chunks and embeddings is not None and vector_store is not None:
    #     print("\n\nYou can now use docs, chunks, embeddings, and vector_store variables")
    #     print("for further testing or pass them to test_pipeline() to skip steps:")
    #     print("  test_pipeline(docs=docs)  # Skip ingestion")
    #     print("  test_pipeline(docs=docs, chunks=chunks)  # Skip ingestion and chunking")
    #     print("  test_pipeline(docs=docs, chunks=chunks, embeddings=embeddings)  # Skip to storage")

