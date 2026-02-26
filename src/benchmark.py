import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import time
import logging
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

import litellm
litellm.suppress_debug_info = True

logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

import logging.handlers

from litellm import completion
import config
from src.vector_store import VectorStore, SummaryVectorStore
from src.storage import SummaryStorage
from src.embedding import get_query_embedding

os.environ["OPENROUTER_API_KEY"] = config.OPENROUTER_API_KEY or ""
os.environ["OPENROUTER_API_BASE"] = config.OPENROUTER_API_BASE

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DROP_PARAMS"] = "True"

LOG_DIR = config.BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkQuestion:
    def __init__(self, question: str, expected_keywords: List[str] = None,
                 expected_sources: List[str] = None, answer: str = None):
        self.question = question
        self.expected_keywords = expected_keywords or []
        self.expected_sources = expected_sources or []
        self.answer = answer


def load_benchmark(file_path: Path) -> List[BenchmarkQuestion]:
    logger.info(f"Loading benchmark from {file_path}")
    try:
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
        logger.info(f"Loaded {len(questions)} questions from benchmark")
        return questions
    except FileNotFoundError:
        logger.error(f"Benchmark file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in benchmark file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading benchmark: {e}")
        raise


def generate_with_llm(prompt: str, model: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    logger.debug(f"Generating with model: {model}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
        {"role": "user", "content": prompt}
    ]

    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            response = completion(model=model, messages=messages)
            content = response.choices[0].message.content
            logger.debug(f"Generated response ({len(content)} chars)")
            return content
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            if "429" in error_str or "rate_limit" in error_str.lower():
                delay = base_delay * (2 ** retry_count)
                logger.warning(f"Rate limited. Waiting {delay:.1f}s before retry ({retry_count+1}/{max_retries})")
                time.sleep(delay)
                retry_count += 1
            elif "API key" in error_str or "authentication" in error_str.lower():
                logger.error(f"API authentication error: {error_str}")
                raise
            else:
                logger.warning(f"LLM error (retry {retry_count+1}/{max_retries}): {error_str}")
                if retry_count < max_retries - 1:
                    time.sleep(base_delay)
                    retry_count += 1
                else:
                    raise

    logger.error(f"Failed to generate after {max_retries} retries: {last_error}")
    raise Exception(f"Failed to generate response after {max_retries} retries: {last_error}")


def llm_judge_vote(question: str, answer_simple: str, answer_summary: str, judge_model: str) -> str:
    logger.debug(f"Judge voting with model: {judge_model}")
    
    prompt = f"""You are an expert judge. Compare two answers to the same question and determine which one is BETTER.

Question: {question}

Answer A (Simple Embedding):
{answer_simple}

Answer B (Summary Embedding):
{answer_summary}

Which answer is better? Consider:
- Accuracy to the question
- Completeness of information
- Clarity and coherence

Respond with ONLY "A" if Answer A is better, "B" if Answer B is better, or "TIE" if they are equally good.
"""

    messages = [
        {"role": "system", "content": "You are an expert judge. Respond with only A, B, or TIE."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = completion(model=judge_model, messages=messages)
        vote = response.choices[0].message.content.strip().upper()
        
        if vote in ["A", "B", "TIE"]:
            logger.debug(f"Vote cast: {vote}")
            return vote
        
        logger.warning(f"Invalid vote '{vote}', defaulting to TIE")
        return "TIE"
    except Exception as e:
        logger.error(f"Judge error with model {judge_model}: {e}")
        return "TIE"


class BenchmarkRunner:
    def __init__(self):
        logger.info("Initializing BenchmarkRunner...")
        
        try:
            self.chunk_vector_store = VectorStore()
            self.chunk_vector_store.load(config.FAISS_DIR)
            logger.info(f"Loaded chunk vector store with {self.chunk_vector_store.total_chunks} chunks")
        except Exception as e:
            logger.error(f"Failed to load chunk vector store: {e}")
            raise

        try:
            self.summary_vector_store = SummaryVectorStore()
            self.summary_vector_store.load(config.FAISS_DIR)
            logger.info(f"Loaded summary vector store with {self.summary_vector_store.total_summaries} summaries")
        except Exception as e:
            logger.error(f"Failed to load summary vector store: {e}")
            raise

        try:
            self.storage = SummaryStorage()
            logger.info("Initialized summary storage")
        except Exception as e:
            logger.error(f"Failed to initialize summary storage: {e}")
            raise

    def retrieve_simple_embedding(self, query: str, top_k: int = 3) -> List[Dict]:
        logger.debug(f"Simple embedding retrieval for query: {query[:50]}...")
        
        try:
            logger.debug(f"Getting query embedding...")
            query_embedding = get_query_embedding(query)
            logger.debug(f"Query embedding shape: {query_embedding.shape}, dimension: {self.chunk_vector_store.dimension}, chunks: {self.chunk_vector_store.total_chunks}")
            
            results = self.chunk_vector_store.search(query_embedding, k=top_k)
            logger.debug(f"Retrieved {len(results)} chunks from simple embedding")

            retrieved = []
            for chunk, score in results:
                retrieved.append({
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": score
                })
            return retrieved
        except Exception as e:
            import traceback
            logger.error(f"Error in simple embedding retrieval: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def retrieve_summary_embedding(self, query: str, top_k: int = 3) -> List[Dict]:
        logger.debug(f"Summary embedding retrieval for query: {query[:50]}...")
        
        try:
            query_embedding = get_query_embedding(query)
            results = self.summary_vector_store.search(query_embedding, k=top_k)
            logger.debug(f"Retrieved {len(results)} summaries from summary embedding")

            retrieved = []
            for summary_obj, score in results:
                try:
                    chunk = self.storage.get_chunk_for_summary(summary_obj.chunk_id)
                except Exception as e:
                    logger.warning(f"Failed to get chunk for summary {summary_obj.chunk_id}: {e}")
                    chunk = None
                    
                retrieved.append({
                    "text": chunk.text if chunk else "N/A",
                    "source": summary_obj.source,
                    "page": summary_obj.page,
                    "score": score,
                    "summary": summary_obj.summary
                })
            return retrieved
        except Exception as e:
            logger.error(f"Error in summary embedding retrieval: {e}")
            raise

    def generate_answer(self, query: str, context: List[Dict], model: str) -> str:
        logger.debug(f"Generating answer with model: {model}")
        
        context_text = "\n\n".join([
            f"[Source {i+1}]: {item['text'][:500]}"
            for i, item in enumerate(context)
        ])

        prompt = f"""Based ONLY on the following context, answer the question. If the context doesn't contain enough information to answer, say so.

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            answer = generate_with_llm(prompt, model)
            logger.debug(f"Generated answer ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def run_question(self, benchmark_q: BenchmarkQuestion) -> Dict[str, Any]:
        question = benchmark_q.question
        logger.info(f"Running question: {question[:60]}...")

        try:
            logger.info("  Retrieving with simple embedding...")
            simple_context = self.retrieve_simple_embedding(question, top_k=config.TOP_K)
            logger.debug(f"Simple context: {len(simple_context)} chunks")

            logger.info("  Retrieving with summary embedding...")
            summary_context = self.retrieve_summary_embedding(question, top_k=config.TOP_K)
            logger.debug(f"Summary context: {len(summary_context)} chunks")

            logger.info(f"  Generating answers with {config.GENERATION_MODEL}...")
            
            try:
                simple_answer = self.generate_answer(question, simple_context, config.GENERATION_MODEL)
            except Exception as e:
                logger.error(f"    Failed to generate simple answer: {e}")
                simple_answer = "[ERROR]"
            
            try:
                summary_answer = self.generate_answer(question, summary_context, config.GENERATION_MODEL)
            except Exception as e:
                logger.error(f"    Failed to generate summary answer: {e}")
                summary_answer = "[ERROR]"
            
            logger.info(f"  Voting with {len(config.VOTING_LLMS)} LLMs...")
            voting_results = {"A": 0, "B": 0, "TIE": 0}

            for i, llm_model in enumerate(config.VOTING_LLMS):
                model_short = '/'.join(llm_model.split('/')[-2:])
                logger.info(f"    [{i+1}/{len(config.VOTING_LLMS)}] {model_short}")
                
                try:
                    vote = llm_judge_vote(question, simple_answer, summary_answer, llm_model)
                    voting_results[vote] += 1
                    logger.info(f"      Vote: {vote}")
                except Exception as e:
                    logger.error(f"    Voting failed: {e}")
                    voting_results["TIE"] += 1

            simple_wins = voting_results["A"]
            summary_wins = voting_results["B"]
            ties = voting_results["TIE"]

            if simple_wins > summary_wins:
                winner = "simple"
                winner_score = simple_wins / len(config.VOTING_LLMS)
            elif summary_wins > simple_wins:
                winner = "summary"
                winner_score = summary_wins / len(config.VOTING_LLMS)
            else:
                winner = "tie"
                winner_score = 0.5

            simple_answers = {config.GENERATION_MODEL: simple_answer}
            summary_answers = {config.GENERATION_MODEL: summary_answer}
            keyword_score_simple = self._calculate_keyword_match(list(simple_answers.values()), benchmark_q.expected_keywords)
            keyword_score_summary = self._calculate_keyword_match(list(summary_answers.values()), benchmark_q.expected_keywords)

            logger.info(f"  Results: Simple={simple_wins}, Summary={summary_wins}, Tie={ties}, Winner={winner}")

            return {
                "question": question,
                "simple_context": [{"text": c["text"][:200], "source": c["source"], "page": c["page"]} for c in simple_context],
                "summary_context": [{"text": c["text"][:200], "source": c["source"], "page": c["page"]} for c in summary_context],
                "simple_answers": simple_answers,
                "summary_answers": summary_answers,
                "voting_results": voting_results,
                "winner": winner,
                "winner_votes": max(simple_wins, summary_wins),
                "total_votes": len(config.VOTING_LLMS),
                "winner_score": winner_score,
                "keyword_score_simple": keyword_score_simple,
                "keyword_score_summary": keyword_score_summary,
                "keyword_winner": "simple" if keyword_score_simple > keyword_score_summary else "summary" if keyword_score_summary > keyword_score_simple else "tie"
            }
            
        except Exception as e:
            logger.error(f"Error running question: {e}")
            raise

    def _calculate_keyword_match(self, answers, expected_keywords):
        if not expected_keywords:
            return 0.0

        best_match = 0.0
        for answer in answers:
            if answer == "[ERROR]":
                continue
            answer_lower = answer.lower()
            matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            match_ratio = matches / len(expected_keywords)
            best_match = max(best_match, match_ratio)
        return best_match

    def run_benchmark(self, questions: List[BenchmarkQuestion]) -> Dict[str, Any]:
        logger.info(f"Starting benchmark with {len(questions)} questions")
        
        results = []
        start_time = time.time()

        for i, q in enumerate(questions):
            logger.info(f"\n{'='*60}")
            logger.info(f"Question {i+1}/{len(questions)}: {q.question[:60]}...")
            logger.info(f"{'='*60}")

            question_start = time.time()
            
            try:
                result = self.run_question(q)
                results.append(result)
                
                question_time = time.time() - question_start
                logger.info(f"  Question completed in {question_time:.1f}s")
                logger.info(f"    Simple wins: {result['voting_results']['A']}/{result['total_votes']}")
                logger.info(f"    Summary wins: {result['voting_results']['B']}/{result['total_votes']}")
                logger.info(f"    Winner: {result['winner']}")
                
            except Exception as e:
                logger.error(f"  Failed to process question: {e}")
                results.append({
                    "question": q.question,
                    "error": str(e),
                    "winner": "error"
                })

        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.1f}s")

        total_questions = len(results)
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        if failed_results:
            logger.warning(f"Failed questions: {len(failed_results)}/{total_questions}")

        valid_results = [r for r in successful_results if r.get("winner") not in [None, "error"]]
        
        if not valid_results:
            logger.error("No valid results to compute summary")
            return {
                "results": results,
                "summary": {
                    "total_questions": total_questions,
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                    "error": "No valid results"
                }
            }

        simple_embedding_wins = sum(1 for r in valid_results if r["winner"] == "simple")
        summary_embedding_wins = sum(1 for r in valid_results if r["winner"] == "summary")
        ties = sum(1 for r in valid_results if r["winner"] == "tie")

        avg_simple_keyword = sum(r["keyword_score_simple"] for r in valid_results) / len(valid_results)
        avg_summary_keyword = sum(r["keyword_score_summary"] for r in valid_results) / len(valid_results)
        avg_winner_score = sum(r["winner_score"] for r in valid_results) / len(valid_results)

        summary = {
            "total_questions": total_questions,
            "successful": len(successful_results),
            "failed": len(failed_results),
            "simple_embedding_wins": simple_embedding_wins,
            "summary_embedding_wins": summary_embedding_wins,
            "ties": ties,
            "simple_win_rate": simple_embedding_wins / len(valid_results),
            "summary_win_rate": summary_embedding_wins / len(valid_results),
            "tie_rate": ties / len(valid_results),
            "average_winner_score": avg_winner_score,
            "average_keyword_score_simple": avg_simple_keyword,
            "average_keyword_score_summary": avg_summary_keyword,
            "better_method": "simple_embedding" if simple_embedding_wins > summary_embedding_wins else "summary_embedding" if summary_embedding_wins > simple_embedding_wins else "tie",
            "total_time_seconds": total_time
        }

        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Questions: {total_questions}")
        logger.info(f"Successful: {len(successful_results)}, Failed: {len(failed_results)}")
        logger.info(f"Simple Embedding Wins: {simple_embedding_wins} ({summary['simple_win_rate']:.1%})")
        logger.info(f"Summary Embedding Wins: {summary_embedding_wins} ({summary['summary_win_rate']:.1%})")
        logger.info(f"Ties: {ties} ({summary['tie_rate']:.1%})")
        logger.info(f"Better Method: {summary['better_method']}")
        logger.info(f"Total Time: {total_time:.1f}s")

        return {
            "results": results,
            "summary": summary
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Benchmark System")
    parser.add_argument("-f", "--file", type=Path, help="Benchmark file path")
    parser.add_argument("-o", "--output", type=Path, help="Output results to file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    benchmark_file = args.file or config.BENCHMARKS_DIR / "bench1.json"

    logger.info(f"Starting benchmark - file: {benchmark_file}")
    
    if not benchmark_file.exists():
        logger.error(f"Benchmark file not found: {benchmark_file}")
        print(f"Error: Benchmark file not found: {benchmark_file}")
        sys.exit(1)

    questions = load_benchmark(benchmark_file)
    print(f"Loaded {len(questions)} questions")

    print(f"\nUsing {len(config.VOTING_LLMS)} LLMs for voting:")
    for llm in config.VOTING_LLMS:
        print(f"  - {llm}")
    print()

    try:
        runner = BenchmarkRunner()
        results = runner.run_benchmark(questions)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\nError: Benchmark failed - {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal Questions: {results['summary']['total_questions']}")
    print(f"Successful: {results['summary'].get('successful', 'N/A')}")
    print(f"Failed: {results['summary'].get('failed', 'N/A')}")
    
    if 'simple_embedding_wins' in results['summary']:
        print(f"\nLLM Voting Results:")
        print(f"  Simple Embedding Wins: {results['summary']['simple_embedding_wins']} ({results['summary']['simple_win_rate']:.1%})")
        print(f"  Summary Embedding Wins: {results['summary']['summary_embedding_wins']} ({results['summary']['summary_win_rate']:.1%})")
        print(f"  Ties: {results['summary']['ties']} ({results['summary']['tie_rate']:.1%})")
        print(f"\nKeyword Matching:")
        print(f"  Average Score (Simple): {results['summary']['average_keyword_score_simple']:.2f}")
        print(f"  Average Score (Summary): {results['summary']['average_keyword_score_summary']:.2f}")
        print(f"\nOverall Winner: {results['summary']['better_method'].replace('_', ' ').title()}")
        print(f"Average Winner Score: {results['summary']['average_winner_score']:.2f}")
        print(f"Total Time: {results['summary'].get('total_time_seconds', 'N/A'):.1f}s")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
