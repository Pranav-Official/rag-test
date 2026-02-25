from google import genai
from typing import List, Tuple, Dict
import config
from src.vector_store import VectorStore, SummaryVectorStore
from src.embedding import get_query_embedding
from src.storage import SummaryStorage


class Retriever:
    def __init__(self, chunk_vector_store: VectorStore, summary_vector_store: SummaryVectorStore, summary_storage: SummaryStorage):
        self.chunk_vector_store = chunk_vector_store
        self.summary_vector_store = summary_vector_store
        self.summary_storage = summary_storage
        self.client = None
        if config.GEMINI_API_KEY:
            self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        
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
        if not self.client:
            raise ValueError("GEMINI_API_KEY not set")
        
        context_text = "\n\n".join([
            f"[Source {i+1}]: {item['summary']}"
            for i, item in enumerate(context)
        ])
        
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        response = self.client.models.generate_content(
            model=config.GEMINI_GENERATION_MODEL,
            contents=prompt
        )
        
        return response.text
    
    def query(self, query: str) -> Dict:
        context = self.retrieve(query)
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "context": context
        }
