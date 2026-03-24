from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import gc
import re
import uuid
import logging
import json
import hmac
import hashlib
import time
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Request, BackgroundTasks, Query
from threading import Lock
from contextlib import nullcontext
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from fastapi.responses import Response
from core.task_queue import OptionalRQQueue
from core.rate_limiter import HybridRateLimiter
from core.auth import AuthManager, require_roles
from core.tracing import initialize_tracing, get_tracer, get_tracing_state
from worker_tasks import process_single_upload_task, process_batch_upload_task, compute_metrics_task

load_dotenv(override=True)

from rag_pipeline import (
    ask_question,
    ask_question_with_sources,
    define_technical_term,
    extract_key_insights,
    one_line_summary,
    vector_store_exists,
    compute_metrics,
    append_document_to_vector_store,
    reset_vector_store,
    generate_insights_by_document,
)

# ================= STARTUP VALIDATION =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
OLLAMA_ENABLED = (os.getenv("OLLAMA_ENABLED", "false").strip().lower() == "true")
if not GROQ_API_KEY and not (LLM_PROVIDER in {"ollama", "extractive"} or OLLAMA_ENABLED):
    logging.warning(
        "GROQ_API_KEY is not set. Running in fallback mode (extractive answers only)."
    )

app = FastAPI()

TRACING_STATE = initialize_tracing()

# ================= OBSERVABILITY =================
METRICS_REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "contexta_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=METRICS_REGISTRY,
)
REQUEST_DURATION_SECONDS = Histogram(
    "contexta_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    registry=METRICS_REGISTRY,
)
REQUEST_EXCEPTIONS_TOTAL = Counter(
    "contexta_http_request_exceptions_total",
    "Total request exceptions",
    ["method", "path", "exception"],
    registry=METRICS_REGISTRY,
)
INPROGRESS_REQUESTS = Gauge(
    "contexta_http_requests_inprogress",
    "Requests currently being processed",
    registry=METRICS_REGISTRY,
)
NAMESPACE_REQUESTS_TOTAL = Counter(
    "contexta_namespace_requests_total",
    "Total requests per resolved namespace",
    ["namespace"],
    registry=METRICS_REGISTRY,
)
logger = logging.getLogger("contexta.api")


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Capture request metrics, tracing, and structured logs for diagnostics."""
    request_id = (request.headers.get("X-Request-ID") or uuid.uuid4().hex)
    request.state.request_id = request_id

    method = request.method
    path = request.url.path
    start = perf_counter()
    INPROGRESS_REQUESTS.inc()
    tracer = get_tracer()
    span_name = f"http.{method.lower()} {path}"
    span_ctx = tracer.start_as_current_span(span_name) if tracer is not None else nullcontext()

    with span_ctx as span:
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            REQUEST_EXCEPTIONS_TOTAL.labels(method=method, path=path, exception=exc.__class__.__name__).inc()
            if span is not None:
                span.set_attribute("error", True)
                span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            duration = perf_counter() - start
            REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
            REQUESTS_TOTAL.labels(method=method, path=path, status=str(status_code)).inc()
            namespace = getattr(request.state, "namespace", "unresolved")
            NAMESPACE_REQUESTS_TOTAL.labels(namespace=namespace).inc()
            INPROGRESS_REQUESTS.dec()

            if span is not None:
                span.set_attribute("http.method", method)
                span.set_attribute("http.route", path)
                span.set_attribute("http.status_code", int(status_code))
                span.set_attribute("contexta.request_id", request_id)
                span.set_attribute("contexta.namespace", namespace)

            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status": status_code,
                        "duration_ms": round(duration * 1000, 2),
                        "namespace": namespace,
                    },
                    ensure_ascii=True,
                )
            )

    response.headers["X-Request-ID"] = request_id
    return response

# ================= CORS =================
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= STORAGE =================
STORAGE_ROOT = os.getenv("CONTEXTA_STORAGE_ROOT", str(Path(__file__).resolve().parent))
UPLOAD_DIR = os.getenv("CONTEXTA_UPLOAD_DIR", os.path.join(STORAGE_ROOT, "uploaded_docs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

BATCH_SIZE = 32
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_FILES_PER_UPLOAD = 10
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm"}
ASYNC_BY_DEFAULT = (os.getenv("ASYNC_BY_DEFAULT", "false").strip().lower() == "true")
RATE_LIMIT_ENABLED = (os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() == "true")
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_DEFAULT_MAX = int(os.getenv("RATE_LIMIT_DEFAULT_MAX", "120"))
RATE_LIMIT_UPLOAD_MAX = int(os.getenv("RATE_LIMIT_UPLOAD_MAX", "20"))
RATE_LIMIT_UPLOAD_BATCH_MAX = int(os.getenv("RATE_LIMIT_UPLOAD_BATCH_MAX", "8"))
RATE_LIMIT_ASK_MAX = int(os.getenv("RATE_LIMIT_ASK_MAX", "120"))
RATE_LIMIT_DEFINE_MAX = int(os.getenv("RATE_LIMIT_DEFINE_MAX", "60"))
RATE_LIMIT_INSIGHTS_MAX = int(os.getenv("RATE_LIMIT_INSIGHTS_MAX", "30"))
RATE_LIMIT_RESET_MAX = int(os.getenv("RATE_LIMIT_RESET_MAX", "5"))
RATE_LIMIT_METRICS_MAX = int(os.getenv("RATE_LIMIT_METRICS_MAX", "10"))
RATE_LIMIT_JOB_STATUS_MAX = int(os.getenv("RATE_LIMIT_JOB_STATUS_MAX", "240"))
STRICT_EVIDENCE_BY_DEFAULT = (os.getenv("STRICT_EVIDENCE_BY_DEFAULT", "false").strip().lower() == "true")
LOCAL_JOB_MAX_RETRIES = max(1, int(os.getenv("LOCAL_JOB_MAX_RETRIES", "2")))
LOCAL_JOB_RETRY_BACKOFF_SECONDS = max(0.0, float(os.getenv("LOCAL_JOB_RETRY_BACKOFF_SECONDS", "1")))
JOB_RETENTION_SECONDS = max(60, int(os.getenv("JOB_RETENTION_SECONDS", "86400")))
JOB_MAX_RECORDS = max(100, int(os.getenv("JOB_MAX_RECORDS", "10000")))

_jobs_lock = Lock()
_jobs: dict[str, dict] = {}
_external_queue = OptionalRQQueue()
_rate_limiter = HybridRateLimiter()


def _endpoint_limit(endpoint_name: str) -> int:
    if endpoint_name == "upload-paper":
        return RATE_LIMIT_UPLOAD_MAX
    if endpoint_name == "upload-papers":
        return RATE_LIMIT_UPLOAD_BATCH_MAX
    if endpoint_name == "ask-question":
        return RATE_LIMIT_ASK_MAX
    if endpoint_name == "define-term":
        return RATE_LIMIT_DEFINE_MAX
    if endpoint_name == "insights":
        return RATE_LIMIT_INSIGHTS_MAX
    if endpoint_name == "reset-index":
        return RATE_LIMIT_RESET_MAX
    if endpoint_name == "metrics":
        return RATE_LIMIT_METRICS_MAX
    if endpoint_name == "job-status":
        return RATE_LIMIT_JOB_STATUS_MAX
    return RATE_LIMIT_DEFAULT_MAX


def _rate_limit_key(request: Request, namespace: str, endpoint_name: str) -> str:
    """Create stable limiter key per identity + namespace + endpoint."""
    tenant_id = getattr(getattr(request, "state", object()), "tenant_id", "unknown")
    client_key = (request.headers.get("X-Client-Key") or "").strip()
    if client_key:
        identity = f"client:{client_key}"
    else:
        host = request.client.host if request.client else "unknown"
        agent = request.headers.get("user-agent", "")[:80]
        identity = f"anon:{host}:{agent}"
    return f"tenant:{tenant_id}|{identity}|ns:{namespace}|ep:{endpoint_name}"


def _enforce_rate_limit(request: Request, namespace: str, endpoint_name: str) -> None:
    """Sliding-window rate limiter to protect expensive/open endpoints."""
    if not RATE_LIMIT_ENABLED:
        return

    limit = max(1, _endpoint_limit(endpoint_name))
    window = max(1, RATE_LIMIT_WINDOW_SECONDS)
    key = _rate_limit_key(request, namespace, endpoint_name)
    try:
        allowed, retry_after = _rate_limiter.check(
            identity_key=key,
            limit=limit,
            window_seconds=window,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Rate limiter unavailable: {str(e)}",
        )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded for {endpoint_name}. "
                f"Allowed {limit} requests per {window} seconds. Retry in {retry_after}s."
            ),
        )


def _create_job(namespace: str, kind: str) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _prune_local_jobs_locked(time.time())
        _jobs[job_id] = {
            "job_id": job_id,
            "namespace": namespace,
            "kind": kind,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
            "attempts": 0,
            "max_attempts": LOCAL_JOB_MAX_RETRIES,
            "dead_letter": False,
        }
    return job_id


def _prune_local_jobs_locked(now_ts: float) -> None:
    """Prune local in-memory jobs by retention window and hard record cap."""
    expiry_cutoff = now_ts - JOB_RETENTION_SECONDS

    expired_ids = [
        job_id
        for job_id, job in _jobs.items()
        if float(job.get("updated_at", 0.0)) <= expiry_cutoff
    ]
    for job_id in expired_ids:
        _jobs.pop(job_id, None)

    if len(_jobs) <= JOB_MAX_RECORDS:
        return

    overflow = len(_jobs) - JOB_MAX_RECORDS
    oldest_ids = sorted(
        _jobs.keys(),
        key=lambda jid: float(_jobs[jid].get("updated_at", 0.0)),
    )[:overflow]
    for job_id in oldest_ids:
        _jobs.pop(job_id, None)


def _prune_local_jobs() -> None:
    with _jobs_lock:
        _prune_local_jobs_locked(time.time())


def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    result: dict | None = None,
    error: str | None = None,
    extra_fields: dict | None = None,
) -> None:
    with _jobs_lock:
        _prune_local_jobs_locked(time.time())
        job = _jobs.get(job_id)
        if not job:
            return
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        if extra_fields:
            job.update(extra_fields)
        job["updated_at"] = time.time()


def _run_job_with_retries(job_id: str, namespace: str, kind: str, work_func) -> None:
    """Execute local async jobs with bounded retries and dead-letter on terminal failure."""
    max_attempts = LOCAL_JOB_MAX_RETRIES
    last_error = "Unknown job failure"

    for attempt in range(1, max_attempts + 1):
        _update_job(
            job_id,
            status="running",
            extra_fields={
                "attempts": attempt,
                "max_attempts": max_attempts,
            },
        )
        try:
            result = work_func()
            _update_job(
                job_id,
                status="completed",
                result=result,
                error=None,
                extra_fields={
                    "dead_letter": False,
                },
            )
            return
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts:
                _update_job(
                    job_id,
                    status="retrying",
                    error=f"Attempt {attempt}/{max_attempts} failed: {last_error}",
                    extra_fields={"attempts": attempt},
                )
                if LOCAL_JOB_RETRY_BACKOFF_SECONDS > 0:
                    time.sleep(LOCAL_JOB_RETRY_BACKOFF_SECONDS)
                continue

            _update_job(
                job_id,
                status="dead-letter",
                error=f"All {max_attempts} attempts failed: {last_error}",
                extra_fields={
                    "attempts": attempt,
                    "dead_letter": True,
                    "failed_at": time.time(),
                },
            )
            logging.error(
                f"Job moved to dead-letter. job_id={job_id} kind={kind} namespace={namespace} error={last_error}"
            )
            return


def _run_single_upload_job(job_id: str, namespace: str, filename: str, file_path: str) -> None:
    try:
        def _work():
            ingest_stats = append_document_to_vector_store(
                file_path=file_path,
                batch_size=BATCH_SIZE,
                namespace=namespace,
            )
            return {
                "message": "Document processed successfully.",
                "filename": filename,
                "namespace": namespace,
                "ingestion": ingest_stats,
            }

        _run_job_with_retries(job_id, namespace, "upload-paper", _work)
    finally:
        gc.collect()


def _run_batch_upload_job(job_id: str, namespace: str, file_records: list[dict]) -> None:
    def _work():
        processed = []
        failed = []
        for record in file_records:
            try:
                ingest_stats = append_document_to_vector_store(
                    file_path=record["file_path"],
                    batch_size=BATCH_SIZE,
                    namespace=namespace,
                )
                processed.append({
                    "filename": record["filename"],
                    "ingestion": ingest_stats,
                })
            except Exception as e:
                failed.append({
                    "filename": record["filename"],
                    "error": f"Failed to process the document. Ensure it is a valid supported file (.pdf, .docx, .txt, .html, .htm). ({str(e)})",
                })
            finally:
                gc.collect()

        if not processed:
            raise RuntimeError(
                json.dumps(
                    {
                        "message": "No files were processed successfully.",
                        "failed": failed,
                    }
                )
            )

        return {
            "message": "Batch upload completed.",
            "namespace": namespace,
            "processed_count": len(processed),
            "failed_count": len(failed),
            "processed": processed,
            "failed": failed,
        }

    _run_job_with_retries(job_id, namespace, "upload-papers", _work)


def _run_metrics_job(job_id: str, namespace: str) -> None:
    def _work():
        result = compute_metrics(namespace=namespace)
        if result is None:
            raise RuntimeError("Failed to compute metrics.")
        return result

    _run_job_with_retries(job_id, namespace, "metrics", _work)


def _enqueue_or_fallback(
    *,
    background_tasks: BackgroundTasks,
    namespace: str,
    kind: str,
    local_func,
    local_args: tuple,
    rq_func,
    rq_kwargs: dict,
) -> str:
    """Enqueue to external queue when available, else use in-process background jobs."""
    if _external_queue.required and not _external_queue.enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Async processing requires an external queue, but it is unavailable. "
                "Ensure Redis is reachable and an RQ worker is running."
            ),
        )

    if _external_queue.enabled:
        try:
            return _external_queue.enqueue(
                rq_func,
                namespace=namespace,
                kind=kind,
                **rq_kwargs,
            )
        except Exception as e:
            if _external_queue.required:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Failed to enqueue async job to external queue. "
                        "Retry after queue service recovers."
                    ),
                )
            logging.warning(f"External queue enqueue failed for {kind}; falling back to local background task: {e}")

    job_id = _create_job(namespace=namespace, kind=kind)
    background_tasks.add_task(local_func, job_id, *local_args)
    return job_id


def _sanitize_namespace(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw).strip()).strip("-_").lower()
    return safe[:64] if safe else ""


def _load_client_namespace_map() -> dict[str, str]:
    """Load API-key to namespace mapping from env JSON."""
    raw = (os.getenv("CLIENT_NAMESPACE_MAP") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("CLIENT_NAMESPACE_MAP is invalid JSON. Ignoring map and using anonymous identity binding.")
        return {}
    if not isinstance(parsed, dict):
        logging.warning("CLIENT_NAMESPACE_MAP must be a JSON object. Ignoring map.")
        return {}

    cleaned: dict[str, str] = {}
    for key, namespace in parsed.items():
        token = str(key).strip()
        ns = _sanitize_namespace(str(namespace))
        if token and ns:
            cleaned[token] = ns
    return cleaned


CLIENT_NAMESPACE_MAP = _load_client_namespace_map()
NAMESPACE_SIGNING_KEY = (os.getenv("NAMESPACE_SIGNING_KEY") or "contexta-default-signing-key").strip()
AUTH_MANAGER = AuthManager(
    namespace_signing_key=NAMESPACE_SIGNING_KEY,
    client_namespace_map=CLIENT_NAMESPACE_MAP,
)

ROLE_POLICY = {
    "upload-paper": ["writer", "admin"],
    "upload-papers": ["writer", "admin"],
    "ask-question": ["reader", "writer", "admin"],
    "define-term": ["reader", "writer", "admin"],
    "insights": ["reader", "writer", "admin"],
    "metrics": ["reader", "writer", "admin"],
    "job-status": ["reader", "writer", "admin"],
    "reset-index": ["admin"],
}


def _resolve_namespace(request: Request) -> str:
    """Resolve namespace using JWT/legacy hybrid auth and store context on request state."""
    try:
        ctx = AUTH_MANAGER.authenticate(request)
    except PermissionError as e:
        msg = str(e).strip().lower()
        status = 403
        if "required" in msg or "missing" in msg or "empty" in msg:
            status = 401
        raise HTTPException(status_code=status, detail=str(e))

    request.state.namespace = ctx.namespace
    request.state.tenant_id = ctx.tenant_id
    request.state.auth_context = ctx
    return ctx.namespace


def _require_roles(request: Request, endpoint_name: str) -> None:
    allowed = ROLE_POLICY.get(endpoint_name, [])
    ctx = getattr(request.state, "auth_context", None)
    if not ctx:
        return
    if not require_roles(ctx, allowed):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Insufficient role for {endpoint_name}. "
                f"Required one of: {', '.join(allowed)}"
            ),
        )

# ================= REQUEST MODELS =================
class QuestionRequest(BaseModel):
    question: str

class TermRequest(BaseModel):
    term: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str

# ================= ROUTES =================

@app.post("/auth/refresh")
def refresh_access_token(body: RefreshTokenRequest):
    try:
        return AUTH_MANAGER.refresh_tokens(body.refresh_token.strip())
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


@app.post("/auth/revoke")
def revoke_current_access_token(request: Request):
    _resolve_namespace(request)
    ctx = getattr(request.state, "auth_context", None)
    if not ctx or not ctx.token_jti:
        raise HTTPException(status_code=400, detail="No revocable JWT context found on request.")

    AUTH_MANAGER.revoke_jti(ctx.token_jti, ctx.token_exp)
    return {
        "message": "Token revoked.",
        "tenant_id": getattr(ctx, "tenant_id", None),
        "namespace": getattr(ctx, "namespace", None),
        "jti": ctx.token_jti,
    }


@app.get("/index-status")
def get_index_status(request: Request):
    """Return whether a searchable vector index exists for the caller namespace."""
    namespace = _resolve_namespace(request)
    return {
        "namespace": namespace,
        "ready": bool(vector_store_exists(namespace=namespace)),
    }

async def _read_and_save_file(file: UploadFile, namespace: str) -> tuple[str, str]:
    """Validate uploaded document and save it to disk. Returns (safe_filename, saved_path)."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file received.")

    safe_filename = Path(file.filename).name
    ext = Path(safe_filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {safe_filename}. "
                "Allowed types are .pdf, .docx, .txt, .html, .htm"
            ),
        )

    # File size check (read up to limit + 1 byte to detect oversized files)
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({file.filename}). "
                f"Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
            ),
        )

    namespace_dir = os.path.join(UPLOAD_DIR, namespace)
    os.makedirs(namespace_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
    saved_path = os.path.join(namespace_dir, unique_name)

    try:
        with open(saved_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file {safe_filename}: {str(e)}")

    return safe_filename, saved_path

@app.post("/upload-paper")
async def upload_paper(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_mode: bool = Query(default=ASYNC_BY_DEFAULT),
):
    namespace = _resolve_namespace(request)
    _require_roles(request, "upload-paper")
    _enforce_rate_limit(request, namespace, "upload-paper")
    filename, file_path = await _read_and_save_file(file, namespace)

    if async_mode:
        job_id = _enqueue_or_fallback(
            background_tasks=background_tasks,
            namespace=namespace,
            kind="upload-paper",
            local_func=_run_single_upload_job,
            local_args=(namespace, filename, file_path),
            rq_func=process_single_upload_task,
            rq_kwargs={
                "namespace": namespace,
                "filename": filename,
                "file_path": file_path,
                "batch_size": BATCH_SIZE,
            },
        )
        return {
            "message": "Upload accepted for background processing.",
            "job_id": job_id,
            "namespace": namespace,
            "status": "queued",
        }

    # 2. Process and append into persistent index with chunk-hash deduplication
    try:
        ingest_stats = append_document_to_vector_store(
            file_path=file_path,
            batch_size=BATCH_SIZE,
            namespace=namespace,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the document. Ensure it is a valid supported file (.pdf, .docx, .txt, .html, .htm).")
    finally:
        gc.collect()

    return {
        "message": "Document processed successfully.",
        "filename": filename,
        "namespace": namespace,
        "ingestion": ingest_stats,
    }


@app.post("/upload-papers")
async def upload_papers(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    async_mode: bool = Query(default=ASYNC_BY_DEFAULT),
):
    """Upload and ingest multiple documents in one request."""
    namespace = _resolve_namespace(request)
    _require_roles(request, "upload-papers")
    _enforce_rate_limit(request, namespace, "upload-papers")
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")

    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_FILES_PER_UPLOAD} files per request.",
        )

    processed = []
    failed = []
    file_records = []

    for file in files:
        try:
            filename, file_path = await _read_and_save_file(file, namespace)
            if async_mode:
                file_records.append({"filename": filename, "file_path": file_path})
                continue

            ingest_stats = append_document_to_vector_store(
                file_path=file_path,
                batch_size=BATCH_SIZE,
                namespace=namespace,
            )
            processed.append({
                "filename": filename,
                "ingestion": ingest_stats,
            })
        except HTTPException as e:
            failed.append({
                "filename": Path(file.filename).name if file and file.filename else "unknown",
                "error": e.detail,
            })
        except Exception as e:
            failed.append({
                "filename": Path(file.filename).name if file and file.filename else "unknown",
                "error": f"Failed to process the document. Ensure it is a valid supported file (.pdf, .docx, .txt, .html, .htm). ({str(e)})",
            })
        finally:
            gc.collect()

    if async_mode:
        if not file_records:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "No files were accepted for background processing.",
                    "failed": failed,
                },
            )
        job_id = _enqueue_or_fallback(
            background_tasks=background_tasks,
            namespace=namespace,
            kind="upload-papers",
            local_func=_run_batch_upload_job,
            local_args=(namespace, file_records),
            rq_func=process_batch_upload_task,
            rq_kwargs={
                "namespace": namespace,
                "file_records": file_records,
                "batch_size": BATCH_SIZE,
            },
        )
        return {
            "message": "Batch upload accepted for background processing.",
            "job_id": job_id,
            "namespace": namespace,
            "status": "queued",
            "accepted_count": len(file_records),
            "failed_precheck_count": len(failed),
            "failed_precheck": failed,
        }

    if not processed:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "No files were processed successfully.",
                "failed": failed,
            },
        )

    return {
        "message": "Batch upload completed.",
        "namespace": namespace,
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
    }


@app.post("/ask-question")
def query_paper(
    request: Request,
    body: QuestionRequest,
    strict_evidence: bool = Query(default=STRICT_EVIDENCE_BY_DEFAULT),
):
    namespace = _resolve_namespace(request)
    _require_roles(request, "ask-question")
    _enforce_rate_limit(request, namespace, "ask-question")
    question = body.question.strip() if body.question else ""
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required and cannot be empty.")

    if not vector_store_exists(namespace=namespace):
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a supported file first.")

    result = ask_question_with_sources(
        question,
        namespace=namespace,
        strict_evidence=strict_evidence,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Detect error strings returned by the pipeline and surface them properly
    if answer and answer.startswith("Error while processing"):
        raise HTTPException(status_code=500, detail=answer)

    return {"question": question, "answer": answer, "sources": sources, "namespace": namespace}


@app.post("/define-term")
def define_term(request: Request, body: TermRequest):
    namespace = _resolve_namespace(request)
    _require_roles(request, "define-term")
    _enforce_rate_limit(request, namespace, "define-term")
    term = body.term.strip() if body.term else ""
    if not term:
        raise HTTPException(status_code=400, detail="'term' field is required and cannot be empty.")

    if not vector_store_exists(namespace=namespace):
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a supported file first.")

    definition = define_technical_term(term, namespace=namespace)
    return {"term": term, "definition": definition, "namespace": namespace}


@app.get("/insights")
def get_insights(request: Request):
    namespace = _resolve_namespace(request)
    _require_roles(request, "insights")
    _enforce_rate_limit(request, namespace, "insights")
    if not vector_store_exists(namespace=namespace):
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a supported file first.")

    docs = generate_insights_by_document(namespace=namespace)
    if not docs:
        raise HTTPException(status_code=500, detail="No document insights could be generated.")

    return {
        "namespace": namespace,
        "documents": docs,
    }


@app.delete("/reset-index")
def reset_index(request: Request):
    """Wipe the FAISS index and metadata so a fresh set of documents can be uploaded."""
    namespace = _resolve_namespace(request)
    _require_roles(request, "reset-index")
    _enforce_rate_limit(request, namespace, "reset-index")
    try:
        reset_vector_store(namespace=namespace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset index: {str(e)}")
    return {"message": "Index reset successfully. Please re-upload your documents.", "namespace": namespace}


@app.get("/metrics")
def get_rag_metrics(
    request: Request,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=ASYNC_BY_DEFAULT),
):
    """Compute 9 RAG evaluation metrics against the currently uploaded document.
    Runs 2 sample queries — expect ~30-60 seconds."""
    namespace = _resolve_namespace(request)
    _require_roles(request, "metrics")
    _enforce_rate_limit(request, namespace, "metrics")
    if not vector_store_exists(namespace=namespace):
        raise HTTPException(
            status_code=404,
            detail="No document uploaded yet. Please upload a supported file first."
        )

    if async_mode:
        job_id = _enqueue_or_fallback(
            background_tasks=background_tasks,
            namespace=namespace,
            kind="metrics",
            local_func=_run_metrics_job,
            local_args=(namespace,),
            rq_func=compute_metrics_task,
            rq_kwargs={
                "namespace": namespace,
            },
        )
        return {
            "message": "Metrics computation accepted for background processing.",
            "job_id": job_id,
            "namespace": namespace,
            "status": "queued",
        }

    try:
        result = compute_metrics(namespace=namespace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {str(e)}")

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to compute metrics.")

    return result


@app.get("/jobs/{job_id}")
def get_job_status(request: Request, job_id: str):
    namespace = _resolve_namespace(request)
    _require_roles(request, "job-status")
    _enforce_rate_limit(request, namespace, "job-status")
    _prune_local_jobs()

    if _external_queue.enabled:
        try:
            external_job = _external_queue.get(job_id)
        except Exception:
            external_job = None
        if external_job:
            if external_job.get("namespace") != namespace:
                raise HTTPException(status_code=403, detail="Job does not belong to this namespace.")
            status_map = {
                "queued": "queued",
                "started": "running",
                "finished": "completed",
                "failed": "failed",
                "deferred": "queued",
                "scheduled": "queued",
                "stopped": "failed",
                "canceled": "failed",
            }
            external_job["status"] = status_map.get(external_job.get("status", "queued"), external_job.get("status", "queued"))
            return external_job

    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.get("namespace") != namespace:
            raise HTTPException(status_code=403, detail="Job does not belong to this namespace.")
        return {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "namespace": job["namespace"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "result": job["result"],
            "error": job["error"],
            "attempts": job.get("attempts", 0),
            "max_attempts": job.get("max_attempts", LOCAL_JOB_MAX_RETRIES),
            "dead_letter": bool(job.get("dead_letter", False)),
        }


@app.get("/observability/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint for production dashboards and alerts."""
    return Response(content=generate_latest(METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/observability/status")
def observability_status():
    """Report observability backend readiness for tracing, queue, and rate-limit backends."""
    tracing = get_tracing_state()
    return {
        "tracing": {
            "enabled": tracing.enabled,
            "exporter": tracing.exporter,
            "service_name": tracing.service_name,
            "endpoint": tracing.endpoint,
        },
        "queue": {
            "enabled": _external_queue.enabled,
            "required": _external_queue.required,
            "backend": _external_queue.backend,
        },
        "rate_limit": {
            "redis_enabled": _rate_limiter.redis_enabled,
            "backend": _rate_limiter.backend,
            "required": _rate_limiter.required,
        },
        "tenant_isolation": {
            "deployment_tenant_id": AUTH_MANAGER.deployment_tenant_id,
            "jwt_tenant_claim": AUTH_MANAGER.jwt_tenant_claim,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
