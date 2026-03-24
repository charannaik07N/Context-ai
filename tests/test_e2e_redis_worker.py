import importlib
import io
import os
import sys
import time
import types
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.e2e
def test_e2e_real_redis_queue_job_lifecycle(monkeypatch, tmp_path):
    if os.getenv("RUN_E2E_REDIS_TESTS", "false").strip().lower() != "true":
        pytest.skip("RUN_E2E_REDIS_TESTS is not enabled")

    redis_url = (os.getenv("REDIS_URL") or "").strip()
    if not redis_url:
        pytest.skip("REDIS_URL is required for Redis e2e tests")

    redis = pytest.importorskip("redis")
    rq = pytest.importorskip("rq")

    state = {
        "vector_exists": True,
    }

    fake = types.ModuleType("rag_pipeline")

    def append_document_to_vector_store(file_path, chunk_size=None, chunk_overlap=None, batch_size=32, namespace=None):
        return {
            "source": "uploaded",
            "added_chunks": 1,
            "namespace": namespace,
        }

    fake.ask_question = lambda question, namespace=None: "ok"
    fake.ask_question_with_sources = lambda question, k_per_source=None, namespace=None, strict_evidence=False: {
        "answer": "ok",
        "sources": [{"source": "doc1"}],
    }
    fake.define_technical_term = lambda term, namespace=None: "definition"
    fake.extract_key_insights = lambda namespace=None: "insights"
    fake.one_line_summary = lambda namespace=None: "summary"
    fake.vector_store_exists = lambda namespace=None: state["vector_exists"]
    fake.compute_metrics = lambda namespace=None: {"context_precision": 1.0}
    fake.append_document_to_vector_store = append_document_to_vector_store
    fake.reset_vector_store = lambda namespace=None: None
    fake.generate_insights_by_document = lambda namespace=None: [{"source": "doc1", "summary": "s", "key_insights": "k"}]

    queue_name = f"contexta-e2e-{uuid.uuid4().hex[:8]}"

    monkeypatch.setenv("CONTEXTA_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("AUTH_MODE", "legacy")
    monkeypatch.setenv("CLIENT_NAMESPACE_MAP", '{"test-client-key":"team-a"}')
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "rq")
    monkeypatch.setenv("TASK_QUEUE_REQUIRED", "true")
    monkeypatch.setenv("TASK_QUEUE_NAME", queue_name)
    monkeypatch.setenv("TASK_QUEUE_RESULT_TTL_SECONDS", "600")
    monkeypatch.setenv("TASK_QUEUE_FAILURE_TTL_SECONDS", "600")
    monkeypatch.setenv("TASK_QUEUE_JOB_TTL_SECONDS", "600")
    monkeypatch.setenv("REDIS_URL", redis_url)

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    conn = redis.Redis.from_url(redis_url)
    conn.ping()

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    client = TestClient(main.app)
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}

    submit = client.post(
        "/upload-paper?async_mode=true",
        headers={"X-Client-Key": "test-client-key"},
        files=files,
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    queue = rq.Queue(name=queue_name, connection=conn)
    worker = rq.SimpleWorker([queue], connection=conn)
    worker.work(burst=True, with_scheduler=False)

    deadline = time.time() + 10
    status_payload = None
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}", headers={"X-Client-Key": "test-client-key"})
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload.get("status") in {"completed", "failed", "dead-letter"}:
            break
        time.sleep(0.2)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["namespace"] == "team-a"

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)
