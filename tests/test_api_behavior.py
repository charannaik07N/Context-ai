import importlib
import io
import sys
import time
import types
import json
import base64
import hmac
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_client(monkeypatch, tmp_path):
    state = {
        "vector_exists": True,
        "append_calls": [],
        "ask_calls": [],
        "reset_calls": [],
        "define_calls": [],
        "insights_calls": [],
        "metrics_calls": [],
    }

    fake = types.ModuleType("rag_pipeline")

    def append_document_to_vector_store(file_path, chunk_size=None, chunk_overlap=None, batch_size=32, namespace=None):
        state["append_calls"].append(
            {
                "file_path": file_path,
                "batch_size": batch_size,
                "namespace": namespace,
            }
        )
        return {"source": "uploaded", "added_chunks": 1}

    def ask_question_with_sources(question, k_per_source=None, namespace=None, strict_evidence=False):
        state["ask_calls"].append(
            {
                "question": question,
                "k_per_source": k_per_source,
                "namespace": namespace,
            }
        )
        return {"answer": "ok", "sources": [{"source": "doc1"}]}

    def vector_store_exists(namespace=None):
        return state["vector_exists"]

    def reset_vector_store(namespace=None):
        state["reset_calls"].append(namespace)

    def define_technical_term(term, namespace=None):
        state["define_calls"].append({"term": term, "namespace": namespace})
        return "definition"

    def generate_insights_by_document(namespace=None):
        state["insights_calls"].append(namespace)
        return [{"source": "doc1", "summary": "s", "key_insights": "k"}]

    def compute_metrics(namespace=None):
        state["metrics_calls"].append(namespace)
        return {"context_precision": 1.0}

    fake.ask_question = lambda question, namespace=None: "ok"
    fake.ask_question_with_sources = ask_question_with_sources
    fake.define_technical_term = define_technical_term
    fake.extract_key_insights = lambda namespace=None: "insights"
    fake.one_line_summary = lambda namespace=None: "summary"
    fake.vector_store_exists = vector_store_exists
    fake.compute_metrics = compute_metrics
    fake.append_document_to_vector_store = append_document_to_vector_store
    fake.reset_vector_store = reset_vector_store
    fake.generate_insights_by_document = generate_insights_by_document

    monkeypatch.setenv("CONTEXTA_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("CLIENT_NAMESPACE_MAP", '{"test-client-key":"team-a","key-n1":"n1"}')
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "local")

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    client = TestClient(main.app)
    yield client, state

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_upload_rejects_unsupported_file_extension(app_client):
    client, _ = app_client
    files = {"file": ("malware.exe", io.BytesIO(b"x"), "application/octet-stream")}

    response = client.post("/upload-paper", headers={"X-Client-Key": "test-client-key"}, files=files)

    assert response.status_code == 415
    assert "Unsupported file type" in response.text


def test_upload_accepts_txt_and_propagates_namespace(app_client):
    client, state = app_client
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}

    response = client.post(
        "/upload-paper",
        headers={"X-Client-Key": "test-client-key"},
        files=files,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["namespace"] == "team-a"
    assert state["append_calls"][-1]["namespace"] == "team-a"


def test_ask_question_returns_404_when_index_missing(app_client):
    client, state = app_client
    state["vector_exists"] = False

    response = client.post(
        "/ask-question",
        headers={"X-Client-Key": "test-client-key"},
        json={"question": "What is this?"},
    )

    assert response.status_code == 404

def test_ask_question_success_passes_namespace(app_client):
    client, state = app_client
    state["vector_exists"] = True

    response = client.post(
        "/ask-question",
        headers={"X-Client-Key": "test-client-key"},
        json={"question": "What is this?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["namespace"] == "team-a"
    assert state["ask_calls"][-1]["namespace"] == "team-a"


def test_reset_index_is_namespace_scoped(app_client):
    client, state = app_client

    response = client.delete("/reset-index", headers={"X-Client-Key": "key-n1"})

    assert response.status_code == 200
    assert state["reset_calls"][-1] == "n1"


def test_missing_client_key_is_rejected_when_map_is_enabled(app_client):
    client, _ = app_client

    response = client.post("/ask-question", json={"question": "What is this?"})

    assert response.status_code == 401


def test_rate_limit_returns_429_when_exceeded(app_client, monkeypatch):
    client, _ = app_client
    import main as app_main

    monkeypatch.setattr(app_main, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(app_main, "RATE_LIMIT_WINDOW_SECONDS", 60)
    monkeypatch.setattr(app_main, "RATE_LIMIT_ASK_MAX", 1)
    with app_main._rate_limiter._local_lock:
        app_main._rate_limiter._local_buckets.clear()

    first = client.post(
        "/ask-question",
        headers={"X-Client-Key": "test-client-key"},
        json={"question": "What is this?"},
    )
    second = client.post(
        "/ask-question",
        headers={"X-Client-Key": "test-client-key"},
        json={"question": "What is this?"},
    )

    assert first.status_code == 200
    assert second.status_code == 429


def test_async_upload_returns_job_and_status_is_visible(app_client):
    client, _ = app_client
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}

    submit = client.post(
        "/upload-paper?async_mode=true",
        headers={"X-Client-Key": "test-client-key"},
        files=files,
    )

    assert submit.status_code == 200
    payload = submit.json()
    assert payload["status"] in {"queued", "running", "completed"}
    job_id = payload["job_id"]

    status = client.get(f"/jobs/{job_id}", headers={"X-Client-Key": "test-client-key"})
    assert status.status_code == 200
    assert status.json()["namespace"] == "team-a"


def test_async_metrics_returns_job(app_client):
    client, state = app_client
    state["vector_exists"] = True

    submit = client.get(
        "/metrics?async_mode=true",
        headers={"X-Client-Key": "test-client-key"},
    )

    assert submit.status_code == 200
    payload = submit.json()
    assert payload["status"] in {"queued", "running", "completed"}
    job_id = payload["job_id"]

    status = client.get(f"/jobs/{job_id}", headers={"X-Client-Key": "test-client-key"})
    assert status.status_code == 200
    assert status.json()["kind"] == "metrics"


def test_observability_metrics_endpoint_exposes_prometheus_format(app_client):
    client, _ = app_client

    response = client.get("/observability/metrics")

    assert response.status_code == 200
    assert "contexta_http_requests_total" in response.text


def test_observability_status_endpoint_returns_backend_readiness(app_client):
    client, _ = app_client

    response = client.get("/observability/status")

    assert response.status_code == 200
    body = response.json()
    assert "tracing" in body
    assert "queue" in body
    assert "rate_limit" in body


def test_request_id_is_returned_in_response_headers(app_client):
    client, _ = app_client

    response = client.post(
        "/ask-question",
        headers={"X-Client-Key": "test-client-key", "X-Request-ID": "req-123"},
        json={"question": "What is this?"},
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "req-123"


def test_async_upload_fails_when_external_queue_required_but_unavailable(monkeypatch, tmp_path):
    state = {
        "vector_exists": True,
    }

    fake = types.ModuleType("rag_pipeline")

    def append_document_to_vector_store(file_path, chunk_size=None, chunk_overlap=None, batch_size=32, namespace=None):
        return {"source": "uploaded", "added_chunks": 1}

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

    monkeypatch.setenv("CONTEXTA_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("CLIENT_NAMESPACE_MAP", '{"test-client-key":"team-a"}')
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "rq")
    monkeypatch.setenv("TASK_QUEUE_REQUIRED", "true")
    monkeypatch.delenv("REDIS_URL", raising=False)

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    client = TestClient(main.app)
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}

    response = client.post(
        "/upload-paper?async_mode=true",
        headers={"X-Client-Key": "test-client-key"},
        files=files,
    )

    assert response.status_code == 503
    assert "external queue" in response.text.lower()

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_local_job_retries_then_moves_to_dead_letter(app_client, monkeypatch):
    _, _state = app_client
    import main as app_main

    monkeypatch.setattr(app_main, "LOCAL_JOB_MAX_RETRIES", 2)
    monkeypatch.setattr(app_main, "LOCAL_JOB_RETRY_BACKOFF_SECONDS", 0.0)

    def _always_fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(app_main, "append_document_to_vector_store", _always_fail)

    job_id = app_main._create_job(namespace="team-a", kind="upload-paper")
    app_main._run_single_upload_job(
        job_id=job_id,
        namespace="team-a",
        filename="bad.txt",
        file_path="dummy",
    )

    with app_main._jobs_lock:
        job = app_main._jobs[job_id]

    assert job["status"] == "dead-letter"
    assert job["dead_letter"] is True
    assert job["attempts"] == 2


def test_local_job_retention_prunes_stale_records(app_client, monkeypatch):
    _, _state = app_client
    import main as app_main

    monkeypatch.setattr(app_main, "JOB_RETENTION_SECONDS", 60)
    job_old = app_main._create_job(namespace="team-a", kind="metrics")
    job_new = app_main._create_job(namespace="team-a", kind="metrics")

    with app_main._jobs_lock:
        app_main._jobs[job_old]["updated_at"] = time.time() - 120

    app_main._prune_local_jobs()

    with app_main._jobs_lock:
        assert job_old not in app_main._jobs
        assert job_new in app_main._jobs


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _make_hs256_jwt(payload: dict, key: str, kid: str = "k1") -> str:
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    h_enc = _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    p_enc = _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    msg = f"{h_enc}.{p_enc}".encode("ascii")
    sig = hmac.new(key.encode("utf-8"), msg, hashlib.sha256).digest()
    s_enc = _b64url(sig)
    return f"{h_enc}.{p_enc}.{s_enc}"


def test_jwt_rbac_denies_reader_for_reset(monkeypatch, tmp_path):
    now = int(time.time())

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
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_REQUIRED", "true")
    monkeypatch.setenv("JWT_AUDIENCE", "contexta-api")
    monkeypatch.setenv("JWT_ISSUER", "contexta-auth")
    monkeypatch.setenv("JWT_SIGNING_KEYS_JSON", '{"k1":"secret-one-32-chars-minimum-key-a","k2":"secret-two-32-chars-minimum-key-b"}')

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    token = _make_hs256_jwt(
        {
            "sub": "user-1",
            "namespace": "team-a",
            "roles": ["reader"],
            "iat": now,
            "nbf": now,
            "exp": now + 300,
            "iss": "contexta-auth",
            "aud": "contexta-api",
            "jti": "t-1",
        },
        key="secret-one-32-chars-minimum-key-a",
        kid="k1",
    )

    client = TestClient(main.app)
    resp = client.delete(
        "/reset-index",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 403
    assert "insufficient role" in resp.text.lower()

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_jwt_key_rotation_accepts_new_kid(monkeypatch, tmp_path):
    now = int(time.time())

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
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_REQUIRED", "true")
    monkeypatch.setenv("JWT_AUDIENCE", "contexta-api")
    monkeypatch.setenv("JWT_ISSUER", "contexta-auth")
    monkeypatch.setenv("JWT_SIGNING_KEYS_JSON", '{"k1":"secret-one-32-chars-minimum-key-a","k2":"secret-two-32-chars-minimum-key-b"}')

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    rotated_token = _make_hs256_jwt(
        {
            "sub": "user-2",
            "namespace": "team-a",
            "roles": ["reader"],
            "iat": now,
            "nbf": now,
            "exp": now + 300,
            "iss": "contexta-auth",
            "aud": "contexta-api",
            "jti": "t-2",
        },
        key="secret-two-32-chars-minimum-key-b",
        kid="k2",
    )

    client = TestClient(main.app)
    resp = client.post(
        "/ask-question",
        headers={"Authorization": f"Bearer {rotated_token}"},
        json={"question": "What is this?"},
    )

    assert resp.status_code == 200
    assert resp.json()["namespace"] == "team-a"

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_jwt_tenant_mismatch_blocked_by_deployment_tenant(monkeypatch, tmp_path):
    now = int(time.time())

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
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_REQUIRED", "true")
    monkeypatch.setenv("JWT_AUDIENCE", "contexta-api")
    monkeypatch.setenv("JWT_ISSUER", "contexta-auth")
    monkeypatch.setenv("JWT_SIGNING_KEYS_JSON", '{"k1":"secret-one-32-chars-minimum-key-a"}')
    monkeypatch.setenv("DEPLOYMENT_TENANT_ID", "tenant-a")

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    token = _make_hs256_jwt(
        {
            "sub": "user-3",
            "tenant_id": "tenant-b",
            "namespace": "team-a",
            "roles": ["reader"],
            "iat": now,
            "nbf": now,
            "exp": now + 300,
            "iss": "contexta-auth",
            "aud": "contexta-api",
            "jti": "t-3",
        },
        key="secret-one-32-chars-minimum-key-a",
        kid="k1",
    )

    client = TestClient(main.app)
    resp = client.post(
        "/ask-question",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What is this?"},
    )

    assert resp.status_code == 403
    assert "tenant" in resp.text.lower()

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_legacy_tenant_mismatch_blocked_by_deployment_tenant(monkeypatch, tmp_path):
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
    monkeypatch.setenv("CLIENT_NAMESPACE_MAP", '{"test-client-key":"tenant-b"}')
    monkeypatch.setenv("DEPLOYMENT_TENANT_ID", "tenant-a")

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    client = TestClient(main.app)
    resp = client.post(
        "/ask-question",
        headers={"X-Client-Key": "test-client-key"},
        json={"question": "What is this?"},
    )

    assert resp.status_code == 403
    assert "tenant" in resp.text.lower()

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_refresh_endpoint_issues_new_tokens(monkeypatch, tmp_path):
    fake = types.ModuleType("rag_pipeline")
    fake.ask_question = lambda question, namespace=None: "ok"
    fake.ask_question_with_sources = lambda question, k_per_source=None, namespace=None, strict_evidence=False: {"answer": "ok", "sources": [{"source": "doc1"}]}
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
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_REQUIRED", "true")
    monkeypatch.setenv("JWT_ENABLE_REFRESH_SERVICE", "true")
    monkeypatch.setenv("JWT_AUDIENCE", "contexta-api")
    monkeypatch.setenv("JWT_SIGNING_KEYS_JSON", '{"k1":"secret-one-32-chars-minimum-key-a"}')
    monkeypatch.setenv("JWT_ACTIVE_KID", "k1")
    monkeypatch.setenv("JWT_REFRESH_SIGNING_KEY", "refresh-secret-32-chars-min-key-123")

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    tokens = main.AUTH_MANAGER.issue_tokens(
        subject="user-refresh",
        tenant_id="tenant-a",
        namespace="team-a",
        roles=["reader"],
    )

    client = TestClient(main.app)
    resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)


def test_revoked_access_token_is_rejected(monkeypatch, tmp_path):
    fake = types.ModuleType("rag_pipeline")
    fake.ask_question = lambda question, namespace=None: "ok"
    fake.ask_question_with_sources = lambda question, k_per_source=None, namespace=None, strict_evidence=False: {"answer": "ok", "sources": [{"source": "doc1"}]}
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
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_REQUIRED", "true")
    monkeypatch.setenv("JWT_ENABLE_REFRESH_SERVICE", "true")
    monkeypatch.setenv("JWT_AUDIENCE", "contexta-api")
    monkeypatch.setenv("JWT_SIGNING_KEYS_JSON", '{"k1":"secret-one-32-chars-minimum-key-a"}')
    monkeypatch.setenv("JWT_ACTIVE_KID", "k1")

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.modules["rag_pipeline"] = fake
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    importlib.reload(main)

    tokens = main.AUTH_MANAGER.issue_tokens(
        subject="user-revoke",
        tenant_id="tenant-a",
        namespace="team-a",
        roles=["reader"],
    )

    client = TestClient(main.app)
    revoke_resp = client.post("/auth/revoke", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert revoke_resp.status_code == 200

    ask_resp = client.post(
        "/ask-question",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
        json={"question": "What is this?"},
    )
    assert ask_resp.status_code in {401, 403}

    sys.modules.pop("main", None)
    sys.modules.pop("rag_pipeline", None)
