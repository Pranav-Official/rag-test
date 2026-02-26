import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
FAISS_DIR = DATA_DIR / "faiss_index"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
DB_PATH = DATA_DIR / "summaries.db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
GENERATION_MODEL = "openrouter/z-ai/glm-4.7-flash"

VOTING_LLMS = [
    "openrouter/mistralai/mistral-nemo",
    "openrouter/google/gemma-3-12b-it",
    "openrouter/qwen/qwen3-8b",
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter/deepseek/deepseek-chat",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 3
