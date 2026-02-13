# Manual Maker - Local RAG System

A fully local RAG system that processes PDF manuals and generates custom documentation using hybrid search and local LLMs.

## Features

- 🔒 **Fully offline** - Runs completely local after initial model download
- 🔍 **Hybrid search** - Combines semantic embeddings (Ollama) with keyword matching (BM25)
- 🏢 **Corporate-friendly** - Handles broken SSL environments
- 📦 **Lightweight** - Uses efficient models (all-minilm ~25MB for embeddings)

## How It Works (All Local)

```
PDFs → Text Chunks → Embeddings (Ollama) → Vector Store (ChromaDB + BM25) → Search
```

After downloading models once, everything runs offline:
- ✅ PDF processing (PyMuPDF)
- ✅ Embeddings generation (Ollama: all-minilm)
- ✅ Vector storage (ChromaDB local DB)
- ✅ Keyword search (BM25 algorithm)
- ✅ LLM generation (Ollama: your choice of model)

## Setup

1. **Install Ollama**: https://ollama.ai/download

2. **Pull models**:
```bash
ollama pull all-minilm    # Embeddings (~25MB)
ollama pull llama2        # LLM for generation
```

3. **Install dependencies**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. **Configure**:
```bash
cp .env.example .env
# Edit .env with your Ollama host (especially for WSL2)
```

## Usage

```bash
# Test PDF processing
python pdf_processor.py

# Test vector store with hybrid search
python vector_store.py

# Test LLM connection
python llm_interface.py
```

## Configuration

**`.env` settings:**
```bash
LLM_PROVIDER=ollama
OLLAMA_HOST=http://172.18.48.1:11434  # WSL2 (use localhost for native)
OLLAMA_MODEL=llama2
DISABLE_SSL_VERIFY=true  # If corporate SSL issues
```

**Tuning hybrid search** (`config.py`):
- `CHUNK_SIZE=1000` - Characters per chunk
- `CHUNK_OVERLAP=200` - Overlap for context preservation
- Hybrid weights: 70% semantic, 30% keyword (adjustable in code)

## Troubleshooting

**WSL2 Ollama connection**: Use Windows host IP `http://172.18.48.1:11434`
**SSL errors**: Set `DISABLE_SSL_VERIFY=true` in `.env`
**Model not found**: Run `ollama pull <model-name>`

## TODO

- [ ] Progressive RAG engine with iterative retrieval
- [ ] Query reformulation and re-ranking
- [ ] Main CLI orchestration
- [ ] Markdown output generation
- [ ] End-to-end pipeline from prompt to document
