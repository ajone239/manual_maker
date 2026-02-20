"""Markdown document generator using LLM with RAG context."""
from pathlib import Path
from datetime import datetime
from typing import Optional
import re

from llm_interface import LLMInterface
from rag_engine import RetrievalResult
from config import OUTPUT_DIR


class MarkdownGenerator:
    """
    Generates markdown documentation from RAG context.

    Takes a user prompt and retrieved context, then uses an LLM to generate
    well-structured markdown documentation.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initialize markdown generator.

        Args:
            llm: LLM interface for text generation
        """
        self.llm = llm

    def generate(
        self,
        prompt: str,
        retrieval_result: RetrievalResult,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate markdown documentation.

        Args:
            prompt: User's original request (e.g., "Create a quick-start guide for lease payments")
            retrieval_result: Result from RAG engine with context
            output_filename: Optional custom filename (auto-generated if not provided)

        Returns:
            Path to the generated markdown file
        """
        print("\n📝 Generating markdown documentation...")

        # Build generation prompt
        generation_prompt = self._build_generation_prompt(
            user_prompt=prompt,
            context=retrieval_result.context
        )

        # Generate markdown content
        print("Calling LLM for generation...")
        markdown_content = self.llm.generate(
            prompt=generation_prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.7,
            max_tokens=4096
        )

        # Clean up the generated content
        markdown_content = self._clean_markdown(markdown_content)

        # Add metadata header
        markdown_with_metadata = self._add_metadata(
            content=markdown_content,
            prompt=prompt,
            retrieval_result=retrieval_result
        )

        # Determine output filename
        if output_filename is None:
            output_filename = self._generate_filename(prompt)

        output_path = OUTPUT_DIR / output_filename

        # Save to file
        output_path.write_text(markdown_with_metadata, encoding='utf-8')

        print(f"✓ Generated: {output_path}")
        print(f"  Size: {len(markdown_with_metadata)} characters")

        return output_path

    def _get_system_prompt(self) -> str:
        """System prompt for the LLM."""
        return """You are a technical documentation writer. Your task is to create clear,
well-structured markdown documentation based on source material provided to you.

Follow these guidelines:
- Use proper markdown formatting (headings, lists, code blocks, tables as appropriate)
- Structure content logically with clear sections
- Be concise but comprehensive
- Cite page numbers when referencing specific information
- Use professional, clear language
- Include relevant examples or details from the source material
- Do not invent information not present in the source material"""

    def _build_generation_prompt(self, user_prompt: str, context: str) -> str:
        """Build the prompt for content generation."""
        return f"""Based on the source material below, {user_prompt}

SOURCE MATERIAL:
{context}

---

Generate the requested documentation in markdown format. Structure it appropriately
with headings, sections, and formatting. Reference page numbers when citing specific
information from the source material."""

    def _clean_markdown(self, content: str) -> str:
        """Clean up generated markdown content."""
        # Remove any leading/trailing whitespace
        content = content.strip()

        # Ensure proper spacing around headings
        content = re.sub(r'\n(#{1,6} )', r'\n\n\1', content)
        content = re.sub(r'(#{1,6} .+)\n', r'\1\n\n', content)

        # Remove excessive blank lines (more than 2)
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content

    def _add_metadata(
        self,
        content: str,
        prompt: str,
        retrieval_result: RetrievalResult
    ) -> str:
        """Add metadata header to the document."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract unique sources
        sources = set(chunk.source for chunk in retrieval_result.chunks)
        sources_list = "\n".join(f"- {source}" for source in sorted(sources))

        metadata = f"""---
Generated: {timestamp}
Prompt: {prompt}
Sources: {len(sources)} document(s)
Chunks Retrieved: {len(retrieval_result.chunks)}
Retrieval Iterations: {retrieval_result.iterations}
---

<!--
Source Documents:
{sources_list}
-->

"""

        return metadata + content

    def _generate_filename(self, prompt: str) -> str:
        """Generate a filename from the prompt."""
        # Extract key words from prompt
        words = re.findall(r'\w+', prompt.lower())

        # Remove common words
        stopwords = {'create', 'generate', 'make', 'write', 'a', 'an', 'the',
                     'for', 'about', 'on', 'guide', 'documentation', 'doc'}
        key_words = [w for w in words if w not in stopwords][:4]

        # Build filename
        if key_words:
            base_name = '_'.join(key_words)
        else:
            base_name = 'output'

        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.md"

        return filename


if __name__ == "__main__":
    # Test the markdown generator
    from config import DATA_DIR
    from pdf_processor import PDFProcessor
    from vector_store import HybridVectorStore
    from rag_engine import RAGEngine, IterativeRefinementStrategy

    print("Testing Markdown Generator...\n")

    # Setup pipeline
    processor = PDFProcessor()
    chunks = processor.process_directory(DATA_DIR)

    store = HybridVectorStore()
    # Uncomment to rebuild:
    # store.add_documents(chunks)

    llm = LLMInterface()

    rag_engine = RAGEngine(
        vector_store=store,
        llm=llm,
        strategy=IterativeRefinementStrategy()
    )

    # Test retrieval
    test_prompt = "create a summary of the lease payment terms"
    print(f"Test prompt: {test_prompt}\n")

    result = rag_engine.retrieve(
        query=test_prompt,
        max_iterations=3,
        k=5
    )

    # Generate markdown
    generator = MarkdownGenerator(llm)
    output_path = generator.generate(
        prompt=test_prompt,
        retrieval_result=result
    )

    print(f"\n✓ Test complete! Check output: {output_path}")
