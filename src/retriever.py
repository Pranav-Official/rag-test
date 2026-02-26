import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
from litellm import completion
from typing import List, Tuple, Dict
import config
from src.vector_store import VectorStore, SummaryVectorStore
from src.embedding import get_query_embedding
from src.storage import SummaryStorage

litellm.suppress_debug_info = True

os.environ["OPENROUTER_API_KEY"] = config.OPENROUTER_API_KEY or ""
os.environ["OPENROUTER_API_BASE"] = config.OPENROUTER_API_BASE


class Retriever:
    def __init__(self, chunk_vector_store: VectorStore, summary_vector_store: SummaryVectorStore, summary_storage: SummaryStorage):
        self.chunk_vector_store = chunk_vector_store
        self.summary_vector_store = summary_vector_store
        self.summary_storage = summary_storage
        
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        k = top_k or config.TOP_K
        
        query_embedding = get_query_embedding(query)
        
        results = self.summary_vector_store.search(query_embedding, k=k)
        
        retrieved = []
        for summary_obj, score in results:
            chunk = self.summary_storage.get_chunk_for_summary(summary_obj.chunk_id)
            retrieved.append({
                "chunk_id": summary_obj.chunk_id,
                "text": chunk.text if chunk else "N/A",
                "source": summary_obj.source,
                "page": summary_obj.page,
                "score": score,
                "summary": summary_obj.summary
            })
            
        return retrieved
    
    def generate_answer(self, query: str, context: List[Dict]) -> str:
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        context_text = "\n\n".join([
            f"[Source {i+1}]: {item['summary']}"
            for i, item in enumerate(context)
        ])
        
        messages = [
            {
                "role": "system",
                "content": "Based on the following context, answer the question. If the context doesn't contain enough information to answer, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]
        
        response = completion(
            model=config.GENERATION_MODEL,
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def query(self, query: str) -> Dict:
        context = self.retrieve(query)
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "context": context
        }
