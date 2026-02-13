# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Manual Maker is a progressive RAG (Retrieval-Augmented Generation) system that processes PDF manuals and generates new documentation (markdown/PDF/text) based on user prompts. The system addresses common RAG weaknesses through hybrid search and iterative retrieval.

**Primary Use Case:** Process AFIS 6 instruction manuals in `data/` to generate quick-start guides, summaries, or custom documentation based on user prompts.

## Architecture

The system follows a modular pipeline:

```
PDFs (data/)
  → PDF Processor (chunking with overlap)
  → Vector Store (hybrid: semantic + keyword search)
  → Progressive RAG Engine (iterative retrieval)
  → LLM Interface (Claude API or Ollama)
  → Markdown Output (output/)
```

### Key Design Decisions

1. **Hybrid Search**: Combines semantic search (embeddings) with keyword search (BM25) to address weak retrieval. Default weights: 70% semantic, 30% keyword. This prevents missing exact technical terms while maintaining semantic understanding.

2. **Local-First**: Entire system runs locally using Ollama for both embeddings (`nomic-embed-text`) and text generation. No external API calls required (Claude API is optional).

3. **SSL Workaround**: Corporate SSL issues are handled via environment variables in `config.py`. The system disables SSL verification when `DISABLE_SSL_VERIFY=true` is set.

4. **Progressive RAG**: Implements iterative retrieval where the RAG engine orchestrates multiple rounds of retrieval based on LLM analysis (not yet fully implemented - pending task #5).

## Component Responsibilities

### Vector Store vs RAG Engine Boundary

**`vector_store.py`** (search primitive):
- Stateless search operations
- Single-shot retrieval for a given query
- Returns ranked chunks with scores
- Manages ChromaDB and BM25 indices

**`rag_engine.py`** (orchestrator - pending):
- Multi-round retrieval strategy
- Query reformulation and expansion
- Re-ranking based on relevance to original prompt
- Context assembly for LLM
- Stateful iteration tracking

### LLM Interface Abstraction

**`llm_interface.py`** provides unified interface for:
- **Claude API** (anthropic package): Requires `ANTHROPIC_API_KEY`
- **Ollama** (local models): Requires Ollama service running at `OLLAMA_HOST`
- Auto-detection: Prefers Ollama (local, free) over Claude (API, paid)

## Development Setup

### Prerequisites

1. **Install Ollama**: https://ollama.ai/download

2. **Pull required models**:
```bash
ollama pull nomic-embed-text  # For embeddings
ollama pull llama2            # For text generation (or any preferred model)
```

3. **Create environment file**:
```bash
cp .env.example .env
```

4. **Install Python dependencies**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Edit `.env` to configure:

- `LLM_PROVIDER`: "ollama" (default) or "anthropic"
- `OLLAMA_MODEL`: Local model name (default: "llama2")
- `OLLAMA_HOST`: Ollama service URL (default: "http://localhost:11434")
- `ANTHROPIC_API_KEY`: Required only if using Claude API
- `DISABLE_SSL_VERIFY`: Set to "true" for corporate environments with SSL issues

## Running Components

### Test PDF Processing
```bash
python pdf_processor.py
```
Processes all PDFs in `data/`, shows chunking results with overlap preservation.

### Test Vector Store
```bash
python vector_store.py
```
Builds hybrid search index and runs sample queries. Shows semantic vs keyword score breakdown.

### Test LLM Interface
```bash
python llm_interface.py
```
Tests LLM connectivity and displays active provider info.

## Important Technical Notes

### SSL Certificate Issues

**Problem**: Corporate environments may have broken SSL chains that prevent downloading models.

**Solution**: Set `DISABLE_SSL_VERIFY=true` in `.env`. The system:
1. Sets SSL env vars (`CURL_CA_BUNDLE`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`) to empty
2. Disables Python SSL verification
3. Suppresses urllib3 warnings

**Impact**: All embeddings now use Ollama (`nomic-embed-text`) instead of sentence-transformers to avoid HuggingFace SSL issues entirely.

### Embeddings Strategy

- **No external dependencies**: Uses Ollama's `nomic-embed-text` model for all embeddings
- **Consistency**: Both document and query embeddings use same model
- **Performance**: Embeddings generated on-demand (not batched) - progress shown every 10 chunks

### ChromaDB Persistence

- Vector store persists to `vector_store/` directory
- No external services required
- Reloading vector store is fast (no re-embedding needed)
- Clear store with `vector_store.clear()` method

## Project Status

**Completed Components:**
- ✅ PDF processing with smart chunking
- ✅ Hybrid vector store (semantic + keyword)
- ✅ LLM interface abstraction (Claude/Ollama)

**Pending Components:**
- ⏳ Progressive RAG engine (task #5)
- ⏳ Main CLI orchestration (task #6)
- ⏳ Markdown output generation

## Modifying Components

### Adjusting Chunking
Edit `config.py`:
- `CHUNK_SIZE`: Characters per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap to preserve context (default: 200)

### Tuning Hybrid Search
In `vector_store.hybrid_search()`:
- `semantic_weight`: Emphasis on meaning (default: 0.7)
- `keyword_weight`: Emphasis on exact matches (default: 0.3)

For technical docs with precise terminology, increase keyword weight:
```python
results = store.hybrid_search(query, semantic_weight=0.5, keyword_weight=0.5)
```

### Adding New LLM Providers
Extend `llm_interface.py`:
1. Create new class inheriting from `BaseLLM`
2. Implement `generate()` and `is_available()` methods
3. Add to `LLMInterface._select_backend()`
