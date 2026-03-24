import os
import time
import importlib
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.stress
def test_concurrency_benchmark_ask_question(monkeypatch, tmp_path):
    if os.getenv("RUN_STRESS_TESTS", "false").strip().lower() != "true":
        pytest.skip("RUN_STRESS_TESTS is not enabled")

    fake = types.ModuleType("rag_pipeline")
    fake.ask_question = lambda question, namespace=None: "ok"
    fake.ask_question_with_sources = lambda question, k_per_source=None, namespace=None, strict_evidence=False: {
        "answer": "ok",
        "sources": [{"source": "doc1"}],
    }
    fake.define_technical_term = lambda term, namespace=None: "definition"
    fake.extract_key_insights = lambda namespace=None: "insights"
    fake.one_line_summary = lambda namespace=None: "summary"
    fake.vector_store_exists = lambda namespace=None: True
    fake.compute_metrics = lambda namespace=None: {"context_precision": 1.0}
    fake.append_document_to_vector_store = lambda *args, **kwargs: {"source": "uploaded", "added_chunks": 1}
    fake.reset_vector_store = lambda namespace=None: None
    fake.generate_insights_by_document = lambda namespace=None: [{"source": "doc1", "summary": "s", "key_insights": "k"}]

    monkeypatch.setenv("CONTEXTA_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("AUTH_MODE", "legacy")
    monkeypatch.setenv("CLIENT_NAMESPACE_MAP", '{"test-client-key":"team-a"}')

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    client = TestClient(main.app)

    total_requests = int(os.getenv("STRESS_TOTAL_REQUESTS", "120"))
    concurrency = int(os.getenv("STRESS_CONCURRENCY", "20"))
    max_p95_ms = float(os.getenv("STRESS_MAX_P95_MS", "1200"))
    min_success_rate = float(os.getenv("STRESS_MIN_SUCCESS_RATE", "0.95"))

    latencies = []
    status_codes = []

    def _send_one(idx: int):
        t0 = time.perf_counter()
        resp = client.post(
            "/ask-question",
            headers={"X-Client-Key": "test-client-key"},
            json={"question": f"What is this? #{idx}"},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return resp.status_code, elapsed_ms

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_one, i) for i in range(total_requests)]
        for fut in as_completed(futures):
            status, elapsed_ms = fut.result()
            status_codes.append(status)
            latencies.append(elapsed_ms)

    latencies_sorted = sorted(latencies)
    p95_index = max(0, int(0.95 * len(latencies_sorted)) - 1)
    p95_ms = latencies_sorted[p95_index]

    success_count = sum(1 for sc in status_codes if sc == 200)
    success_rate = success_count / len(status_codes)

    assert success_rate >= min_success_rate, (
        f"Success rate below threshold: {success_rate:.3f} < {min_success_rate:.3f}"
    )
    assert p95_ms <= max_p95_ms, (
        f"p95 latency above threshold: {p95_ms:.2f}ms > {max_p95_ms:.2f}ms"
    )

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)
