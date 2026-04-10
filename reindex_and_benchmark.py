import argparse
import statistics
import time
from pathlib import Path
from dotenv import load_dotenv
from core.gpu_runtime import configure_gpu_environment

load_dotenv(override=True)
configure_gpu_environment()

from rag_pipeline import (
    append_document_to_vector_store,
    ask_question_with_sources,
    compute_metrics,
    reset_vector_store,
    vector_store_exists,
)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm"}
DEFAULT_QUERIES = [
    "What is the main topic of the uploaded documents?",
    "List the key conclusions or takeaways.",
    "What methods or approaches are discussed?",
    "What timeline, dates, or milestones are mentioned?",
]


def _load_queries_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    queries = [line for line in lines if line and not line.startswith("#")]
    if not queries:
        raise ValueError(f"No usable queries found in file: {path}")
    return queries


def _iter_supported_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def reindex_documents(docs_dir: Path, namespace: str | None) -> dict:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}")

    reset_vector_store(namespace=namespace)

    processed = 0
    skipped = 0
    total_added = 0
    total_seen = 0
    details = []

    for file_path in _iter_supported_files(docs_dir):
        result = append_document_to_vector_store(str(file_path), namespace=namespace)
        processed += 1
        total_added += int(result.get("added_chunks", 0))
        total_seen += int(result.get("total_chunks", 0))
        if result.get("skipped"):
            skipped += 1
        details.append(
            {
                "file": file_path.name,
                "total_chunks": int(result.get("total_chunks", 0)),
                "added_chunks": int(result.get("added_chunks", 0)),
                "duplicate_chunks": int(result.get("duplicate_chunks", 0)),
                "chunk_size": (result.get("chunking") or {}).get("chunk_size"),
                "chunk_overlap": (result.get("chunking") or {}).get("chunk_overlap"),
            }
        )

    return {
        "processed_files": processed,
        "skipped_files": skipped,
        "total_chunks": total_seen,
        "added_chunks": total_added,
        "details": details,
    }


def benchmark_queries(queries: list[str], namespace: str | None, k_per_source: int) -> dict:
    if not vector_store_exists(namespace=namespace):
        raise RuntimeError("Vector store does not exist. Re-index first.")

    rows = []
    durations = []

    for q in queries:
        start = time.perf_counter()
        result = ask_question_with_sources(q, k_per_source=k_per_source, namespace=namespace)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        durations.append(elapsed_ms)

        answer = (result.get("answer") or "").strip()
        sources = result.get("sources") or []
        rows.append(
            {
                "question": q,
                "latency_ms": round(elapsed_ms, 2),
                "answer_len": len(answer),
                "sources": len(sources),
                "error": result.get("error"),
            }
        )

    summary = {
        "count": len(durations),
        "p50_ms": round(statistics.median(durations), 2) if durations else 0.0,
        "avg_ms": round(statistics.mean(durations), 2) if durations else 0.0,
        "max_ms": round(max(durations), 2) if durations else 0.0,
    }

    return {"summary": summary, "rows": rows}


def _print_benchmark(label: str, bench: dict) -> None:
    print(
        f"{label}: n={bench['summary']['count']} p50={bench['summary']['p50_ms']}ms "
        f"avg={bench['summary']['avg_ms']}ms max={bench['summary']['max_ms']}ms"
    )
    for row in bench["rows"]:
        err = row["error"] or "-"
        print(
            f"  - latency={row['latency_ms']}ms sources={row['sources']} answer_len={row['answer_len']} "
            f"error={err} q={row['question']}"
        )


def _print_comparison(before: dict, after: dict) -> None:
    before_s = before["summary"]
    after_s = after["summary"]

    def _delta(key: str) -> float:
        return round(float(after_s.get(key, 0.0)) - float(before_s.get(key, 0.0)), 2)

    print("Comparison (after - before):")
    print(f"  - p50_ms_delta: {_delta('p50_ms')}")
    print(f"  - avg_ms_delta: {_delta('avg_ms')}")
    print(f"  - max_ms_delta: {_delta('max_ms')}")

    before_by_q = {row["question"]: row for row in before.get("rows", [])}
    after_by_q = {row["question"]: row for row in after.get("rows", [])}
    common_questions = [q for q in after_by_q if q in before_by_q]
    if common_questions:
        print("Per-question latency delta (ms):")
        for question in common_questions:
            before_ms = float(before_by_q[question].get("latency_ms", 0.0))
            after_ms = float(after_by_q[question].get("latency_ms", 0.0))
            print(f"  - {round(after_ms - before_ms, 2)} : {question}")


def main():
    parser = argparse.ArgumentParser(description="Reset/re-index documents and run quick QA benchmark.")
    parser.add_argument("--docs-dir", default="uploaded_docs", help="Directory containing source documents.")
    parser.add_argument("--namespace", default=None, help="Namespace to re-index/benchmark.")
    parser.add_argument("--k-per-source", type=int, default=8, help="Retriever depth per source for benchmark runs.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Benchmark question (can be passed multiple times).",
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Text file with one benchmark question per line. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="Only run benchmark against existing index.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run benchmark before and after re-index, then print deltas.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if args.queries:
        queries = args.queries
    elif args.queries_file:
        queries = _load_queries_from_file(Path(args.queries_file))
    else:
        queries = DEFAULT_QUERIES

    before_bench = None
    if args.compare and vector_store_exists(namespace=args.namespace):
        print("[0/2] Running baseline benchmark (current index)...")
        before_bench = benchmark_queries(queries=queries, namespace=args.namespace, k_per_source=args.k_per_source)
        _print_benchmark("Baseline summary", before_bench)
    elif args.compare:
        print("[0/2] Baseline skipped: no existing index to compare.")

    if not args.skip_reindex:
        print("[1/2] Resetting and re-indexing documents...")
        reindex_stats = reindex_documents(docs_dir=docs_dir, namespace=args.namespace)
        print(
            f"Re-index complete: files={reindex_stats['processed_files']} skipped={reindex_stats['skipped_files']} "
            f"chunks_seen={reindex_stats['total_chunks']} chunks_added={reindex_stats['added_chunks']}"
        )
        for row in reindex_stats["details"]:
            print(
                f"  - {row['file']}: total={row['total_chunks']} added={row['added_chunks']} "
                f"dup={row['duplicate_chunks']} chunk={row['chunk_size']} overlap={row['chunk_overlap']}"
            )
    else:
        print("[1/2] Skipping re-index step.")

    print("[2/2] Running benchmark queries...")
    bench = benchmark_queries(queries=queries, namespace=args.namespace, k_per_source=args.k_per_source)
    _print_benchmark("Benchmark summary", bench)

    if args.compare and before_bench is not None:
        _print_comparison(before_bench, bench)

    metrics = compute_metrics(namespace=args.namespace)
    if metrics:
        print("Additional RAG metrics:")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")
    else:
        print("Additional RAG metrics unavailable (no active vector store).")


if __name__ == "__main__":
    main()