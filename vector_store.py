"""Hybrid vector store combining semantic and keyword search."""
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import os
import ollama

from pdf_processor import DocumentChunk
from config import VECTOR_STORE_DIR, OLLAMA_HOST


class HybridVectorStore:
    """
    Combines semantic search (embeddings) with keyword search (BM25).

    This addresses the weakness of pure semantic search which can miss
    exact keyword matches or domain-specific terminology.
    """

    def __init__(self, collection_name: str = "manual_docs"):
        self.collection_name = collection_name

        # Initialize ChromaDB for semantic search
        self.chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Configure Ollama client
        os.environ['OLLAMA_HOST'] = OLLAMA_HOST
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)

        # Use Ollama for embeddings (no SSL issues!)
        self.embedding_model = "nomic-embed-text"
        print(f"Using Ollama embeddings: {self.embedding_model}")
        print(f"Ollama host: {OLLAMA_HOST}")

        # Ensure model is available
        try:
            self.ollama_client.pull(self.embedding_model)
            print("Embedding model ready")
        except Exception as e:
            print(f"Note: {e}")

        # BM25 will be initialized when documents are added
        self.bm25 = None
        self.chunks = []  # Store chunks for BM25
        self.chunk_texts = []  # Store tokenized texts for BM25

    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store."""
        if not chunks:
            return

        print(f"Adding {len(chunks)} chunks to vector store...")

        # Prepare data for ChromaDB
        texts = [chunk.text for chunk in chunks]
        ids = [f"{c.source}_{c.page_number}_{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            }
            for chunk in chunks
        ]

        # Generate embeddings with Ollama
        print("Generating embeddings with Ollama...")
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(texts)}")
            response = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            embeddings.append(response['embedding'])
        embeddings = np.array(embeddings)
        print(f"  Generated {len(embeddings)} embeddings")

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        # Initialize BM25 for keyword search
        print("Initializing BM25 index...")
        self.chunks = chunks
        self.chunk_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.chunk_texts)

        print(f"Vector store ready with {len(chunks)} chunks")

    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Perform semantic search using embeddings."""
        response = self.ollama_client.embeddings(
            model=self.embedding_model,
            prompt=query
        )
        query_embedding = response['embedding']

        # Validate embedding
        if not query_embedding or len(query_embedding) == 0:
            raise ValueError(f"Received empty embedding from Ollama for query: '{query}'")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Convert results back to DocumentChunks with scores
        matched_chunks = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Convert distance to similarity (1 - cosine_distance)
            similarity = 1 - distance

            chunk = DocumentChunk(
                text=doc,
                source=metadata['source'],
                page_number=metadata['page_number'],
                chunk_index=metadata['chunk_index'],
                metadata={k: v for k, v in metadata.items()
                          if k not in ['source', 'page_number', 'chunk_index']}
            )
            matched_chunks.append((chunk, similarity))

        return matched_chunks

    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Perform keyword search using BM25."""
        if self.bm25 is None:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        # Normalize scores to 0-1 range
        max_score = scores[top_k_idx[0]] if len(top_k_idx) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0

        matched_chunks = []
        for idx in top_k_idx:
            if scores[idx] > 0:  # Only include non-zero scores
                normalized_score = scores[idx] / max_score
                matched_chunks.append((self.chunks[idx], normalized_score))

        return matched_chunks

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Tuple[DocumentChunk, float, Dict]]:
        """
        Perform hybrid search combining semantic and keyword approaches.

        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search (default 0.7)
            keyword_weight: Weight for keyword search (default 0.3)

        Returns:
            List of (chunk, combined_score, score_breakdown) tuples
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, k=k*2)
        keyword_results = self.keyword_search(query, k=k*2)

        # Create score dictionaries
        semantic_scores = {id(chunk): score for chunk,
                           score in semantic_results}
        keyword_scores = {id(chunk): score for chunk, score in keyword_results}

        # Combine all unique chunks
        all_chunks = {}
        for chunk, _ in semantic_results + keyword_results:
            chunk_id = id(chunk)
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = chunk

        # Calculate combined scores
        scored_chunks = []
        for chunk_id, chunk in all_chunks.items():
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            keyword_score = keyword_scores.get(chunk_id, 0.0)

            combined_score = (
                semantic_weight * semantic_score +
                keyword_weight * keyword_score
            )

            score_breakdown = {
                'semantic': semantic_score,
                'keyword': keyword_score,
                'combined': combined_score
            }

            scored_chunks.append((chunk, combined_score, score_breakdown))

        # Sort by combined score and return top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:k]

    def clear(self):
        """Clear the vector store."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.chunks = []
        self.chunk_texts = []


if __name__ == "__main__":
    # Test the vector store
    from config import DATA_DIR
    from pdf_processor import PDFProcessor

    processor = PDFProcessor()
    chunks = processor.process_directory(DATA_DIR)

    store = HybridVectorStore()

    # recalc
    # store.add_documents(chunks)

    # Test searches
    test_query = "What is the rent amount?"
    print(f"\nTesting hybrid search for: '{test_query}'")
    results = store.hybrid_search(test_query, k=5)

    print(len(results))

    for i, (c, score, breakdown) in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {score:.3f}) ---")
        print(f"Source: {c.source}, Page: {c.page_number}")
        s = breakdown['semantic']
        k = breakdown['keyword']
        print(f"Semantic: {s:.3f}, Keyword: {k:.3f}")
        print(f"Text: {c.text}...")
