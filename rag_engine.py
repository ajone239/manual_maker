"""Progressive RAG engine with pluggable retrieval strategies."""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from vector_store import HybridVectorStore
from llm_interface import LLMInterface
from pdf_processor import DocumentChunk
from config import INITIAL_RETRIEVAL_K, RAG_LLM_TEMPERATURE, RAG_MAX_TOKENS

# Fallback query templates when LLM fails
FALLBACK_QUERY_TEMPLATES = [
    "{original_query} additional details",
    "{original_query} specific requirements",
    "{original_query} technical specifications"
]


@dataclass
class RetrievalResult:
    """Result from a RAG retrieval operation."""
    chunks: List[DocumentChunk]
    context: str
    iterations: int
    metadata: Dict


class RetrievalStrategy(ABC):
    """Abstract base class for different RAG retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        vector_store: HybridVectorStore,
        llm: LLMInterface,
        max_iterations: int = 5,
        k: int = INITIAL_RETRIEVAL_K
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for the given query.

        Args:
            query: User's question or prompt
            vector_store: Hybrid vector store for searching
            llm: LLM interface for query analysis/reformulation
            max_iterations: Maximum number of retrieval rounds
            k: Number of chunks to retrieve per search

        Returns:
            RetrievalResult with chunks, assembled context, and metadata
        """
        pass


class IterativeRefinementStrategy(RetrievalStrategy):
    """
    Iterative refinement strategy for progressive RAG.

    Process:
    1. Initial search with user query
    2. LLM analyzes retrieved chunks and identifies information gaps
    3. Generate refined queries for missing information
    4. Repeat until max_iterations or LLM determines sufficient coverage
    5. Assemble final context from all retrieved chunks
    """

    def retrieve(
        self,
        query: str,
        vector_store: HybridVectorStore,
        llm: LLMInterface,
        max_iterations: int = 5,
        k: int = INITIAL_RETRIEVAL_K
    ) -> RetrievalResult:
        """Perform iterative refinement retrieval."""

        m = max_iterations
        print(f"\n🔍 Starting iter retrieval max {m} rounds ...")
        print(f"Original query: {query}\n")

        all_chunks = []
        seen_chunk_ids = set()
        iteration_log = []

        for iteration in range(max_iterations):
            print(f"--- Round {iteration + 1}/{max_iterations} ---")

            # Determine search query for this iteration
            if iteration == 0:
                search_query = query
                print(f"Searching with original query...")
            else:
                # Ask LLM what information is missing
                refined_query, is_error = self._generate_refinement_query(
                    original_query=query,
                    retrieved_chunks=all_chunks,
                    llm=llm,
                    retry_count=0
                )

                # Handle errors with retry and fallback
                if is_error:
                    if iteration < max_iterations - 1:
                        # Retry with simplified prompt
                        print("⚠️  Retrying with reduced prompt...")
                        refined_query, is_error_retry = self._generate_refinement_query(
                            original_query=query,
                            retrieved_chunks=all_chunks,
                            llm=llm,
                            retry_count=1
                        )

                        # If retry still fails, use fallback template
                        if is_error_retry:
                            print("⚠️  Using fallback query template")
                            template_idx = min(
                                iteration - 1, len(FALLBACK_QUERY_TEMPLATES) - 1)
                            search_query = FALLBACK_QUERY_TEMPLATES[template_idx].format(
                                original_query=" ".join(query.split()[:5])
                            )
                            print(f"Fallback query: {search_query}")
                        else:
                            search_query = refined_query
                    else:
                        # Last iteration and error - stop
                        print("⚠️  LLM error on final iteration")
                        break
                else:
                    search_query = refined_query

                if search_query is None:
                    print("✓ LLM determined information is sufficient")
                    break

                print(f"Refined query: {search_query}")

            # Perform hybrid search
            results = vector_store.hybrid_search(search_query, k=k)

            # Track new chunks (avoid duplicates)
            new_chunks = 0
            for chunk, score, breakdown in results:
                s = chunk.source
                n = chunk.page_number
                i = chunk.chunk_index
                chunk_id = f"{s}_{n}_{i}"
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_chunks.append(chunk)
                    new_chunks += 1

            print(f"Retrieved {len(results)} results, {new_chunks} new chunks")

            iteration_log.append({
                'iteration': iteration + 1,
                'query': search_query,
                'new_chunks': new_chunks,
                'total_chunks': len(all_chunks)
            })

            # If no new chunks, stop iterating
            if new_chunks == 0:
                print("✓ No new information found")
                break

        # Assemble final context
        context = self._assemble_context(all_chunks)

        lac = len(all_chunks)
        lic = len(iteration_log)
        print(f"\n✓ Retrieval complete: {lac} total chunks: {lic} rounds")

        return RetrievalResult(
            chunks=all_chunks,
            context=context,
            iterations=len(iteration_log),
            metadata={
                'iteration_log': iteration_log,
                'strategy': 'iterative_refinement'
            }
        )

    def _is_error_response(self, response: str) -> bool:
        """Check if response indicates an error (empty or too short)."""
        if not response or not response.strip():
            return True
        if len(response.strip()) < 3:
            return True
        return False

    def _is_stop_signal(self, response: str) -> bool:
        """Check if response is an intentional stop signal."""
        if not response or not response.strip():
            return False

        response_upper = response.strip().upper()
        first_word = response_upper.split()[0] if response_upper else ""

        # Explicit stop signals
        if first_word in ['NONE', 'STOP', 'SUFFICIENT', 'DONE', 'COMPLETE']:
            return True

        # Implicit stop phrases
        response_lower = response.lower()
        stop_phrases = ['information is sufficient', 'enough information',
                        'no additional', 'fully answered', 'complete answer']
        return any(phrase in response_lower for phrase in stop_phrases)

    def _generate_refinement_query(
        self,
        original_query: str,
        retrieved_chunks: List[DocumentChunk],
        llm: LLMInterface,
        retry_count: int = 0
    ) -> Tuple[Optional[str], bool]:
        """
        Ask LLM to analyze retrieved chunks and generate refined query.

        Returns: (refined_query, is_error)
            - refined_query: None if information sufficient, str if more needed
            - is_error: True if LLM failed, False if intentional stop or success
        """
        # Simpler prompt that's easier for local models to follow
        # Don't ask for "NONE" - just ask what's missing
        chunk_summary = "\n".join([
            f"{c.text[:150]}"
            for c in retrieved_chunks[:2]  # Just 2 chunks to keep it focused
        ])

        analysis_prompt = f"""Based on the retrieved information below, what additional information would help answer this question: "{original_query}"

Retrieved info:
{chunk_summary}

If you need more information, write a simple search query (~10 words). If this is enough information, respond with: STOP

Your response:"""

        response = llm.generate(
            prompt=analysis_prompt,
            system_prompt="You are a helpful assistant.",  # Simpler, less restrictive
            temperature=RAG_LLM_TEMPERATURE,  # Increased to 0.7 from 0.3
            max_tokens=RAG_MAX_TOKENS        # Increased to 150 from 100
        )

        response = response.strip()
        print(f"LLM refinement: '{response if response else '(empty)'}'")

        # Check for errors
        if self._is_error_response(response):
            print("⚠️  LLM returned error/empty response")
            return None, True  # is_error=True

        # Check for intentional stop
        if self._is_stop_signal(response):
            print("✓ Information deemed sufficient")
            return None, False  # is_error=False

        # Clean up response for search query
        for prefix in ['SEARCH:', 'Search:', 'search:', 'Query:', 'query:', 'Need:', 'need:']:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        if len(response) > 100:
            response = response[:100]

        # Return the query with is_error=False (successful generation)
        if response:
            return response, False
        else:
            return None, True

    def _assemble_context(self, chunks: List[DocumentChunk]) -> str:
        """Assemble chunks into a coherent context string."""
        if not chunks:
            return ""

        # Group chunks by source document
        chunks_by_source = {}
        for chunk in chunks:
            if chunk.source not in chunks_by_source:
                chunks_by_source[chunk.source] = []
            chunks_by_source[chunk.source].append(chunk)

        # Build context with source attribution
        context_parts = []
        for source, source_chunks in chunks_by_source.items():
            # Sort by page number and chunk index
            source_chunks.sort(key=lambda c: (c.page_number, c.chunk_index))

            context_parts.append(f"=== From {source} ===\n")
            for chunk in source_chunks:
                context_parts.append(
                    f"[Page {chunk.page_number}]\n{chunk.text}\n"
                )

        return "\n".join(context_parts)


class RAGEngine:
    """
    Main RAG orchestrator supporting multiple retrieval strategies.

    This class coordinates the retrieval process but delegates the actual
    retrieval logic to pluggable strategies.
    """

    def __init__(
        self,
        vector_store: HybridVectorStore,
        llm: LLMInterface,
        strategy: Optional[RetrievalStrategy] = None
    ):
        """
        Initialize RAG engine.

        Args:
            vector_store: Hybrid vector store for document search
            llm: LLM interface for query analysis and generation
            strategy: Retrieval strategy to use (defaults to IterativeRefinementStrategy)
        """
        self.vector_store = vector_store
        self.llm = llm
        self.strategy = strategy or IterativeRefinementStrategy()

    def retrieve(
        self,
        query: str,
        max_iterations: int = 5,
        k: int = INITIAL_RETRIEVAL_K
    ) -> RetrievalResult:
        """
        Retrieve relevant context for the given query.

        Args:
            query: User's question or prompt
            max_iterations: Maximum retrieval iterations (strategy-dependent)
            k: Number of chunks to retrieve per search

        Returns:
            RetrievalResult with chunks, context, and metadata
        """
        return self.strategy.retrieve(
            query=query,
            vector_store=self.vector_store,
            llm=self.llm,
            max_iterations=max_iterations,
            k=k
        )

    def set_strategy(self, strategy: RetrievalStrategy):
        """Change the retrieval strategy."""
        self.strategy = strategy
        print(f"Switched to strategy: {strategy.__class__.__name__}")


if __name__ == "__main__":
    # Test the RAG engine
    from config import DATA_DIR
    from pdf_processor import PDFProcessor

    print("Testing RAG Engine...\n")

    # Setup
    processor = PDFProcessor()
    chunks = processor.process_directory(DATA_DIR)

    store = HybridVectorStore()
    # Uncomment to rebuild index:
    # store.add_documents(chunks)

    llm = LLMInterface()

    # Create RAG engine with default strategy
    rag_engine = RAGEngine(
        vector_store=store,
        llm=llm,
        strategy=IterativeRefinementStrategy()
    )

    # Test retrieval
    test_query = "What are the lease payment terms and when is rent due?"
    result = rag_engine.retrieve(
        query=test_query,
        max_iterations=3,
        k=5
    )

    print("\n" + "="*60)
    print("RETRIEVAL SUMMARY")
    print("="*60)
    print(f"Query: {test_query}")
    print(f"Iterations: {result.iterations}")
    print(f"Total chunks: {len(result.chunks)}")
    print(f"\nContext preview (first 500 chars):")
    print(result.context[:500] + "...")
