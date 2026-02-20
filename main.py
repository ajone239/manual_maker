"""Manual Maker - Progressive RAG system for PDF documentation generation."""
import sys
import argparse
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR
from pdf_processor import PDFProcessor
from vector_store import HybridVectorStore
from llm_interface import LLMInterface
from rag_engine import RAGEngine, IterativeRefinementStrategy
from markdown_generator import MarkdownGenerator


def main():
    """Main CLI orchestration."""
    parser = argparse.ArgumentParser(
        description="Manual Maker - Generate documentation from PDF manuals using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Create a quick-start guide for lease payments"
  python main.py "Summarize the maintenance responsibilities"
  python main.py --rebuild "What are the security deposit terms?"
  python main.py --iterations 3 "Explain the pet policy"

Note: If no prompt is provided, interactive mode will start.
        """
    )

    parser.add_argument(
        'prompt',
        nargs='?',
        help='Documentation generation prompt (e.g., "Create a summary of payment terms")'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild vector store from PDFs (use if PDFs have changed)'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Maximum retrieval iterations (default: 5)'
    )

    parser.add_argument(
        '--chunks',
        type=int,
        default=10,
        help='Number of chunks to retrieve per iteration (default: 10)'
    )

    args = parser.parse_args()

    print("="*60)
    print("MANUAL MAKER - Progressive RAG Documentation Generator")
    print("="*60)

    # Get prompt (from args or interactive)
    if args.prompt:
        prompt = args.prompt
    else:
        print("\nNo prompt provided. Enter your documentation request:")
        prompt = input("> ").strip()
        if not prompt:
            print("Error: Prompt cannot be empty")
            sys.exit(1)

    print(f"\nPrompt: {prompt}")
    print(f"Max iterations: {args.iterations}")
    print(f"Chunks per search: {args.chunks}")

    # Step 1: Initialize components
    print("\n" + "="*60)
    print("STEP 1: Initializing components")
    print("="*60)

    processor = PDFProcessor()
    store = HybridVectorStore()
    llm = LLMInterface()

    # Step 2: Build/load vector store
    print("\n" + "="*60)
    print("STEP 2: Loading document index")
    print("="*60)

    # Check if we need to rebuild the index
    if args.rebuild or store.collection.count() == 0:
        print(f"Building vector store from {DATA_DIR}...")
        chunks = processor.process_directory(DATA_DIR)
        store.add_documents(chunks)
    else:
        chunk_count = store.collection.count()
        print(f"✓ Vector store loaded ({chunk_count} chunks indexed)")

    # Step 3: Progressive retrieval
    print("\n" + "="*60)
    print("STEP 3: Progressive retrieval")
    print("="*60)

    rag_engine = RAGEngine(
        vector_store=store,
        llm=llm,
        strategy=IterativeRefinementStrategy()
    )

    retrieval_result = rag_engine.retrieve(
        query=prompt,
        max_iterations=args.iterations,
        k=args.chunks
    )

    # Step 4: Generate markdown
    print("\n" + "="*60)
    print("STEP 4: Generating documentation")
    print("="*60)

    generator = MarkdownGenerator(llm)
    output_path = generator.generate(
        prompt=prompt,
        retrieval_result=retrieval_result
    )

    # Success summary
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"✓ Retrieved {len(retrieval_result.chunks)} relevant chunks")
    print(f"✓ Used {retrieval_result.iterations} retrieval iterations")
    print(f"✓ Generated documentation: {output_path}")
    print("\nView your documentation:")
    print(f"  cat {output_path}")
    print(f"  open {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
