#!/usr/bin/env python
import sys
import argparse
from pathlib import Path
import numpy as np

import config
from src.pdf_processor import process_pdfs
from src.embedding import get_embeddings_batch
from src.vector_store import VectorStore, SummaryVectorStore
from src.summarizer import summarize_chunk, summarize_chunks
from src.storage import SummaryStorage, SummaryData
from src.retriever import Retriever
from src.benchmark import load_benchmark, BenchmarkRunner


def cmd_ingest(args):
    print("=" * 50)
    print("INGEST: PDF Processing & Vector Store Building")
    print("=" * 50)
    
    pdfs_dir = config.PDFS_DIR
    if not pdfs_dir.exists():
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created PDF directory: {pdfs_dir}")
        print("Please add PDF files and run again.")
        return
    
    pdfs = list(pdfs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdfs_dir}")
        return
    
    print(f"\nFound {len(pdfs)} PDF(s)")
    
    storage = SummaryStorage()
    existing_chunk_ids = storage.get_all_chunk_ids()
    
    print("\n[1/5] Extracting text and chunking...")
    all_chunks = process_pdfs(pdfs_dir)
    
    new_chunks = [c for c in all_chunks if c.chunk_id not in existing_chunk_ids]
    skipped = len(all_chunks) - len(new_chunks)
    
    print(f"Total chunks: {len(all_chunks)} (new: {len(new_chunks)}, existing: {skipped})")
    
    if not new_chunks:
        print("\nAll chunks already exist. Loading existing indexes...")
    else:
        print(f"\n[2/5] Generating embeddings for {len(new_chunks)} new chunks...")
        texts = [c.text for c in new_chunks]
        chunk_embeddings = get_embeddings_batch(texts)
        print(f"Chunk embeddings shape: {chunk_embeddings.shape}")
        
        print("\n[3/5] Building chunk FAISS index...")
        chunk_vector_store = VectorStore(dimension=chunk_embeddings.shape[1])
        chunk_vector_store.add_chunks(new_chunks, chunk_embeddings)
        
        if config.FAISS_DIR.exists() and (config.FAISS_DIR / "index.faiss").exists():
            print("  Loading and merging with existing chunk FAISS index...")
            import pickle
            with open(config.FAISS_DIR / "chunks.pkl", "rb") as f:
                existing_chunks = pickle.load(f)
            import faiss
            existing_index = faiss.read_index(str(config.FAISS_DIR / "index.faiss"))
            existing_chunks.extend(new_chunks)
            faiss.normalize_L2(chunk_embeddings)
            existing_index.add(chunk_embeddings)
            faiss.write_index(existing_index, str(config.FAISS_DIR / "index.faiss"))
            with open(config.FAISS_DIR / "chunks.pkl", "wb") as f:
                pickle.dump(existing_chunks, f)
            print(f"Saved chunk FAISS index with {len(existing_chunks)} chunks")
        else:
            chunk_vector_store.save(config.FAISS_DIR)
            print(f"Saved chunk FAISS index with {len(new_chunks)} chunks")
        
        print(f"\n[4/5] Generating and storing summaries for {len(new_chunks)} new chunks...")
        
        for i, chunk in enumerate(new_chunks):
            storage.add_chunk(chunk.chunk_id, chunk.text, chunk.source, chunk.page)
            if i % 10 == 0:
                print(f"  Processing chunk {i+1}/{len(new_chunks)}...")
            summary = summarize_chunk(chunk.text)
            storage.add_summary(
                chunk.chunk_id, 
                summary,
                source=chunk.source,
                page=chunk.page
            )
        
        print(f"Stored {storage.count()} summaries in SQLite")
        
        print(f"\n[5/5] Generating summary embeddings for {len(new_chunks)} new summaries...")
        all_summaries = storage.get_all_summaries()
        summary_texts = [s.summary for s in all_summaries]
        summary_embeddings = get_embeddings_batch(summary_texts)
        print(f"Summary embeddings shape: {summary_embeddings.shape}")
        
        print("\nBuilding summary FAISS index...")
        summary_vector_store = SummaryVectorStore(dimension=summary_embeddings.shape[1])
        summary_vector_store.add_summaries(all_summaries, summary_embeddings)
        summary_vector_store.save(config.FAISS_DIR)
        print(f"Saved summary FAISS index with {summary_vector_store.total_summaries} summaries")
    
    print("\n" + "=" * 50)
    print("INGEST COMPLETE!")
    print("=" * 50)


def cmd_query(args):
    if not args.question:
        print("Error: Please provide a question")
        return
    
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set")
        return
    
    print(f"Loading vector stores from {config.FAISS_DIR}...")
    chunk_vector_store = VectorStore()
    chunk_vector_store.load(config.FAISS_DIR)
    
    summary_vector_store = SummaryVectorStore()
    summary_vector_store.load(config.FAISS_DIR)
    
    storage = SummaryStorage()
    
    retriever = Retriever(chunk_vector_store, summary_vector_store, storage)
    
    print(f"\nQuery: {args.question}\n")
    result = retriever.query(args.question)
    
    print("ANSWER:")
    print("-" * 40)
    print(result["answer"])
    print("-" * 40)
    
    if args.verbose:
        print("\nRETRIEVED CONTEXT:")
        for i, ctx in enumerate(result["context"], 1):
            print(f"\n[{i}] Source: {ctx['source']} (page {ctx['page']})")
            print(f"    Score: {ctx['score']:.3f}")
            print(f"    Summary: {ctx['summary'][:200]}...")


def cmd_benchmark(args):
    from src.benchmark import BenchmarkRunner, load_benchmark
    import logging
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    benchmark_file = args.benchmark_file or config.BENCHMARKS_DIR / "bench1.json"
    
    if not benchmark_file.exists():
        print(f"Benchmark file not found: {benchmark_file}")
        print("Create a JSON file with questions like:")
        print("""[
  {
    "question": "What is the main topic?",
    "expected_keywords": ["topic", "concept"],
    "expected_sources": ["document.pdf"]
  }
]""")
        return
    
    print(f"Loading benchmark from {benchmark_file}...")
    questions = load_benchmark(benchmark_file)
    print(f"Loaded {len(questions)} questions")
    
    print(f"\nUsing {len(config.VOTING_LLMS)} LLMs for voting:")
    for llm in config.VOTING_LLMS:
        print(f"  - {llm}")
    print()
    
    runner = BenchmarkRunner()
    results = runner.run_benchmark(questions)
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
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
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs and build vector store")
    
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", nargs="?", help="Question to ask")
    query_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark tests")
    benchmark_parser.add_argument("-f", "--file", dest="benchmark_file", type=Path, help="Benchmark file path")
    benchmark_parser.add_argument("-o", "--output", type=Path, help="Output results to file")
    benchmark_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
