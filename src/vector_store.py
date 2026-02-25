import faiss
import numpy as np
import pickle
from pathlib import Path
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
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(dist)))
        return results
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
            
    def load(self, path: Path):
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.dimension = self.index.d
            
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
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append((self.summaries[idx], float(dist)))
        return results
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "summary_index.faiss"))
        with open(path / "summaries.pkl", "wb") as f:
            pickle.dump(self.summaries, f)
            
    def load(self, path: Path):
        self.index = faiss.read_index(str(path / "summary_index.faiss"))
        with open(path / "summaries.pkl", "rb") as f:
            self.summaries = pickle.load(f)
        self.dimension = self.index.d
            
    @property
    def total_summaries(self):
        return len(self.summaries)
