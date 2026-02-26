# RAG Test Project

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using Google Gemini for embeddings and LiteLLM/OpenRouter for generation.

## Project Structure

```
rag-test/
├── src/
│   ├── __init__.py         # Package exports
│   ├── pdf_processor.py   # PDF parsing and chunking
│   ├── embedding.py        # Gemini embedding generation
│   ├── vector_store.py     # FAISS vector storage
│   ├── storage.py          # SQLite metadata storage
│   ├── summarizer.py       # Text summarization (LiteLLM)
│   ├── retriever.py        # RAG retrieval and answer generation (LiteLLM)
│   └── benchmark.py        # Benchmark evaluation
├── data/
│   ├── pdfs/               # Input PDF files
│   ├── faiss_index/       # FAISS vector indices
│   ├── benchmarks/         # Benchmark question sets
│   └── summaries.db        # SQLite metadata database
├── config.py               # Configuration settings
├── main.py                 # CLI entry point
├── .env                    # Environment variables
└── requirements.txt       # Python dependencies
```

## Dependencies

- **google-genai**: Gemini API client (for embeddings)
- **litellm**: Unified API for OpenRouter and other LLM providers
- **faiss-cpu**: Vector similarity search
- **pypdf**: PDF text extraction
- **langchain-text-splitters**: Text chunking
- **python-dotenv**: Environment variable loading
- **numpy**: Numerical operations

## Configuration

Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here      # For embeddings
OPENROUTER_API_KEY=your_openrouter_api_key   # For generation
```

Key settings in `config.py`:
- `GEMINI_EMBEDDING_MODEL`: "gemini-embedding-001"
- `GENERATION_MODEL`: "openrouter/google/gemini-2.0-flash"
- `CHUNK_SIZE`: 1000
- `CHUNK_OVERLAP`: 200
- `TOP_K`: 4

## CLI Usage (main.py)

```bash
# Activate virtual environment
venv\Scripts\activate

# Ingest PDFs and build vector store
python main.py ingest

# Query the RAG system
python main.py query "Your question here"

# Query with verbose output
python main.py query "Your question" -v

# Run benchmarks
python main.py benchmark
python main.py benchmark -f data/benchmarks/test_set.json -o results.json
```

## Programmatic Usage

```python
from pathlib import Path
import config
from src.pdf_processor import process_pdfs
from src.embedding import get_embeddings_batch
from src.vector_store import VectorStore, SummaryVectorStore
from src.summarizer import summarize_chunks
from src.storage import SummaryStorage
from src.retriever import Retriever

# Process PDFs
pdfs_dir = config.PDFS_DIR
chunks = process_pdfs(pdfs_dir)

# Generate embeddings and create vector store
embeddings = get_embeddings_batch([c.text for c in chunks])
vector_store = VectorStore(dimension=embeddings.shape[1])
vector_store.add_chunks(chunks, embeddings)

# Store in FAISS
vector_store.save(config.FAISS_DIR)

# Initialize retriever
summary_storage = SummaryStorage()
summary_vector_store = SummaryVectorStore()
summary_vector_store.load(config.FAISS_DIR)

retriever = Retriever(vector_store, summary_vector_store, summary_storage)

# Query
result = retriever.query("What is the main topic of the document?")
print(result["answer"])
```

## Running Benchmarks

```python
from src.benchmark import load_benchmark, BenchmarkRunner

questions = load_benchmark(config.BENCHMARKS_DIR / "test_set.json")
runner = BenchmarkRunner(retriever)
results = runner.run_benchmark(questions)
```
