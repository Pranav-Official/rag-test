import json
from pathlib import Path
from typing import List, Dict, Any
import config
from src.retriever import Retriever
from src.vector_store import VectorStore
from src.storage import SummaryStorage


class BenchmarkQuestion:
    def __init__(self, question: str, expected_keywords: List[str] = None, 
                 expected_sources: List[str] = None, answer: str = None):
        self.question = question
        self.expected_keywords = expected_keywords or []
        self.expected_sources = expected_sources or []
        self.answer = answer


def load_benchmark(file_path: Path) -> List[BenchmarkQuestion]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        questions.append(BenchmarkQuestion(
            question=item.get("question", ""),
            expected_keywords=item.get("expected_keywords", []),
            expected_sources=item.get("expected_sources", []),
            answer=item.get("answer", "")
        ))
    return questions


def calculate_keyword_score(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0
    
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords)


def calculate_source_score(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    if not expected_sources:
        return 0.0
    
    expected_lower = [s.lower() for s in expected_sources]
    matches = sum(1 for src in retrieved_sources if src.lower() in expected_lower)
    return min(matches / len(expected_sources), 1.0)


def calculate_context_relevance(context: List[Dict]) -> float:
    if not context:
        return 0.0
    
    scores = [item.get("score", 0.0) for item in context]
    return sum(scores) / len(scores)


class BenchmarkRunner:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        
    def run_question(self, benchmark_q: BenchmarkQuestion) -> Dict[str, Any]:
        result = self.retriever.query(benchmark_q.question)
        
        retrieved_sources = list(set(item["source"] for item in result["context"]))
        
        keyword_score = calculate_keyword_score(
            result["answer"], 
            benchmark_q.expected_keywords
        )
        
        source_score = calculate_source_score(
            retrieved_sources,
            benchmark_q.expected_sources
        )
        
        relevance_score = calculate_context_relevance(result["context"])
        
        return {
            "question": benchmark_q.question,
            "answer": result["answer"],
            "context": result["context"],
            "retrieved_sources": retrieved_sources,
            "scores": {
                "keyword_match": keyword_score,
                "source_match": source_score,
                "context_relevance": relevance_score,
                "overall": (keyword_score * 0.4 + source_score * 0.3 + relevance_score * 0.3)
            }
        }
        
    def run_benchmark(self, questions: List[BenchmarkQuestion]) -> Dict[str, Any]:
        results = []
        
        for i, q in enumerate(questions):
            print(f"Running question {i+1}/{len(questions)}: {q.question[:50]}...")
            result = self.run_question(q)
            results.append(result)
            
        total_scores = {
            "keyword_match": sum(r["scores"]["keyword_match"] for r in results),
            "source_match": sum(r["scores"]["source_match"] for r in results),
            "context_relevance": sum(r["scores"]["context_relevance"] for r in results),
            "overall": sum(r["scores"]["overall"] for r in results)
        }
        
        num = len(results) if results else 1
        avg_scores = {k: v / num for k, v in total_scores.items()}
        
        return {
            "results": results,
            "summary": {
                "total_questions": len(results),
                "average_scores": avg_scores
            }
        }
