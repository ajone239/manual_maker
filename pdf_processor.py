"""PDF processing with smart chunking to preserve context."""
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import re

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    """A chunk of text with metadata."""
    text: str
    source: str
    page_number: int
    chunk_index: int
    metadata: Dict


class PDFProcessor:
    """Extract and chunk PDF documents intelligently."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract text from PDF, preserving page structure."""
        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            # Clean up text
            text = self._clean_text(text)
            pages.append({
                'page_number': page_num,
                'text': text,
                'source': pdf_path.name
            })

        doc.close()
        return pages

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove page numbers/headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 3]
        return '\n'.join(cleaned_lines)

    def chunk_text(self, text: str, source: str, page_number: int) -> List[DocumentChunk]:
        """
        Chunk text with overlap to preserve context.

        Uses sentence-aware chunking to avoid breaking mid-sentence.
        """
        chunks = []

        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    source=source,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        'char_count': len(current_chunk),
                        'sentence_count': len(re.findall(r'[.!?]', current_chunk))
                    }
                ))

                # Start new chunk with overlap
                # Get last N characters for overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source=source,
                page_number=page_number,
                chunk_index=chunk_index,
                metadata={
                    'char_count': len(current_chunk),
                    'sentence_count': len(re.findall(r'[.!?]', current_chunk))
                }
            ))

        return chunks

    def process_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a PDF into chunks."""
        print(f"Processing {pdf_path.name}...")
        pages = self.extract_text_from_pdf(pdf_path)

        all_chunks = []
        for page in pages:
            chunks = self.chunk_text(
                text=page['text'],
                source=page['source'],
                page_number=page['page_number']
            )
            all_chunks.extend(chunks)

        print(f"  Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks

    def process_directory(self, directory: Path) -> List[DocumentChunk]:
        """Process all PDFs in a directory."""
        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory}")

        print(f"Found {len(pdf_files)} PDF files")

        all_chunks = []
        for pdf_path in pdf_files:
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)

        print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} documents")
        return all_chunks


if __name__ == "__main__":
    # Test the processor
    from config import DATA_DIR

    processor = PDFProcessor()
    chunks = processor.process_directory(DATA_DIR)

    # Show sample
    print("\nSample chunk:")
    print(f"Source: {chunks[0].source}")
    print(f"Page: {chunks[0].page_number}")
    print(f"Text preview: {chunks[0].text[:200]}...")
