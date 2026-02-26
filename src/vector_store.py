import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
import pickle
from typing import List, Tuple
import config


class VectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        
    def create_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        
    def add_chunks(self, chunks: List, embeddings: np.ndarray):
        if self.index is None:
            self.create_index()
            
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple]:
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() or load() first.")
        
        if len(self.chunks) == 0:
            raise ValueError("No chunks in vector store. Run ingest first.")
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.dimension}")
        
        k = min(k, len(self.chunks))
        
        try:
            query_2d = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_2d)
            distances, indices = self.index.search(query_2d, k)
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}")
        
        if distances is None or indices is None:
            return []
        
        if len(distances.shape) != 2 or len(indices.shape) != 2:
            raise ValueError(f"Unexpected FAISS result shape: distances={distances.shape}, indices={indices.shape}")
        
        if distances.shape[1] == 0:
            return []
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        return results
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
            
    def load(self, path: Path):
        index_file = path / "index.faiss"
        chunks_file = path / "chunks.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}. Run ingest first.")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}. Run ingest first.")
        
        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        
        self.dimension = self.index.d
        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. Run ingest to add vectors.")
            
    @property
    def total_chunks(self):
        return len(self.chunks)


class SummaryVectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.summaries = []
        
    def create_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        
    def add_summaries(self, summaries: List, embeddings: np.ndarray):
        if self.index is None:
            self.create_index()
            
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.summaries.extend(summaries)
        
    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple]:
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() or load() first.")
        
        if len(self.summaries) == 0:
            raise ValueError("No summaries in vector store. Run ingest first.")
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.dimension}")
        
        k = min(k, len(self.summaries))
        
        query_2d = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_2d)
        distances, indices = self.index.search(query_2d, k)
        
        if distances is None or indices is None:
            return []
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.summaries):
                results.append((self.summaries[idx], float(dist)))
        return results
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "summary_index.faiss"))
        with open(path / "summaries.pkl", "wb") as f:
            pickle.dump(self.summaries, f)
            
    def load(self, path: Path):
        index_file = path / "summary_index.faiss"
        summaries_file = path / "summaries.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Summary index file not found: {index_file}. Run ingest first.")
        if not summaries_file.exists():
            raise FileNotFoundError(f"Summaries file not found: {summaries_file}. Run ingest first.")
        
        self.index = faiss.read_index(str(index_file))
        with open(summaries_file, "rb") as f:
            self.summaries = pickle.load(f)
        
        self.dimension = self.index.d
        if self.index.ntotal == 0:
            raise ValueError("Summary FAISS index is empty. Run ingest to add vectors.")
            
    @property
    def total_summaries(self):
        return len(self.summaries)
