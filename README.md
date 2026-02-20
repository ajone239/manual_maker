# Manual Maker - Progressive RAG System

A fully local progressive RAG system that processes PDF manuals and generates custom documentation using hybrid search, iterative retrieval, and local LLMs.

## Features

- 🔒 **Fully offline** - Runs completely local after initial model download
- 🔍 **Hybrid search** - Combines semantic embeddings (70%) with keyword matching (30%)
- 🔄 **Progressive RAG** - Iterative retrieval with query refinement
- 🏢 **Corporate-friendly** - Handles broken SSL environments
- 📝 **End-to-end** - PDF input to markdown output in one command

## How It Works

```
PDFs → Chunks → Vector Store (ChromaDB + BM25)
                      ↓
User Prompt → Progressive RAG Engine (iterative search)
                      ↓
              LLM Generation → Markdown Output
```

Everything runs locally:
- ✅ PDF processing with smart chunking
- ✅ Embeddings (Ollama: nomic-embed-text)
- ✅ Hybrid search (semantic + keyword)
- ✅ Progressive retrieval (multi-round refinement)
- ✅ LLM generation (Ollama: your choice of model)
- ✅ Markdown documentation output

## Setup

1. **Install Ollama**: https://ollama.ai/download

2. **Pull models**:
```bash
ollama pull nomic-embed-text  # Embeddings
ollama pull llama2            # LLM for generation (or any other model)
```

3. **Install dependencies**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. **Configure** (optional):
```bash
cp .env.example .env
# Edit .env for custom settings (Ollama host, model, SSL)
```

## Usage

**Basic usage** (place PDFs in `data/` directory):

```bash
# Generate documentation from your prompt
python main.py "Create a quick-start guide for lease payments"

# Interactive mode (no arguments)
python main.py

# Custom iterations and chunks
python main.py --iterations 3 --chunks 5 "Summarize the maintenance policy"

# Rebuild index after adding new PDFs
python main.py --rebuild "What are the security deposit terms?"
```

**Test individual components**:

```bash
python pdf_processor.py    # Test PDF processing
python vector_store.py     # Test hybrid search
python llm_interface.py    # Test LLM connection
python rag_engine.py       # Test progressive retrieval
```

## Configuration

**`.env` settings** (optional):
```bash
LLM_PROVIDER=ollama                   # "ollama" or "anthropic"
OLLAMA_HOST=http://localhost:11434    # Use 172.18.48.1 for WSL2
OLLAMA_MODEL=llama2                   # Or any other Ollama model
DISABLE_SSL_VERIFY=true               # Set if corporate SSL issues
```

**Tuning in `config.py`**:
- `CHUNK_SIZE=1000` / `CHUNK_OVERLAP=200` - Text chunking
- `RAG_LLM_TEMPERATURE=0.7` - LLM creativity for query refinement
- Hybrid search weights: 70% semantic, 30% keyword (in `vector_store.py`)

## Troubleshooting

- **WSL2 Ollama connection**: Set `OLLAMA_HOST=http://172.18.48.1:11434` in `.env`
- **SSL errors**: Set `DISABLE_SSL_VERIFY=true` in `.env`
- **Model not found**: Run `ollama pull <model-name>`
- **Empty LLM responses**: Try a different model or increase `RAG_MAX_TOKENS` in `config.py`
- **Iterating only once**: System now includes retry logic and fallback queries

## Output

Generated markdown files are saved to `output/` with timestamps. Each file includes:
- Source attribution (which PDFs/pages were used)
- AI-generated documentation based on retrieved context
- Metadata about the retrieval process
