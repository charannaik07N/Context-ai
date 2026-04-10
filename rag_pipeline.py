import os
import time
import json
import hashlib
import gc
import copy
import logging
import re
import io
import warnings
import tempfile
import uuid
import subprocess
import urllib.request
import numpy as np
from contextlib import contextmanager
from contextlib import redirect_stdout, redirect_stderr
from threading import Lock
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, BSHTMLLoader
from langchain_community.vectorstores import FAISS
try:
    from langchain_community.vectorstores import Qdrant as LCQdrant
except Exception:  # pragma: no cover - optional dependency
    LCQdrant = None
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional dependency/runtime
    try:
        from langchain_community.chat_models import ChatOllama
    except Exception:  # pragma: no cover - optional dependency/runtime
        ChatOllama = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency/runtime
    ChatGoogleGenerativeAI = None
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from core.faiss_integrity import verify_faiss_integrity, write_faiss_integrity_manifest

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency
    Redis = None

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None

# Load environment variables
load_dotenv(override=True)

# âœ… Centralized Configuration (Single source of truth)
# Use absolute paths to fix "paths mismatch" errors
BASE_DIR = Path(__file__).resolve().parent
STORAGE_ROOT = Path(os.getenv("CONTEXTA_STORAGE_ROOT", str(BASE_DIR))).resolve()
DB_FAISS_PATH = os.path.join(STORAGE_ROOT, "vectorstore", "db_faiss")
INDEX_META_PATH = os.path.join(STORAGE_ROOT, "vector_store_meta.json")
NAMESPACED_STORE_ROOT = os.path.join(STORAGE_ROOT, "vectorstore", "namespaces")
UPLOADS_ROOT = Path(os.getenv("CONTEXTA_UPLOAD_DIR", os.path.join(STORAGE_ROOT, "uploaded_docs"))).resolve()
DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "default")
# Read model name from env so it is configurable without code changes
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "120"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
OLLAMA_ENABLED = (os.getenv("OLLAMA_ENABLED", "false").strip().lower() == "true")
OLLAMA_BASE_URL = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "llama3.2:3b").strip()
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
LOCAL_LLM_TIMEOUT_SECONDS = int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120"))
OLLAMA_KEEP_ALIVE = (os.getenv("OLLAMA_KEEP_ALIVE") or "30m").strip()
OLLAMA_NUM_CTX = max(256, int(os.getenv("OLLAMA_NUM_CTX", "1024")))
OLLAMA_NUM_PREDICT = max(32, int(os.getenv("OLLAMA_NUM_PREDICT", "80")))
OLLAMA_NUM_THREAD = max(0, int(os.getenv("OLLAMA_NUM_THREAD", "0")))
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))
OLLAMA_GPU_ONLY = (os.getenv("OLLAMA_GPU_ONLY", "false").strip().lower() == "true")
OLLAMA_REQUIRED = (os.getenv("OLLAMA_REQUIRED", "false").strip().lower() == "true") or LLM_PROVIDER == "ollama"
GEMINI_API_KEY = ((os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip())
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "256"))
GEMINI_REQUIRED = (os.getenv("GEMINI_REQUIRED", "false").strip().lower() == "true") or LLM_PROVIDER == "gemini"
HYBRID_REQUIRE_BOTH = (os.getenv("HYBRID_REQUIRE_BOTH", "false").strip().lower() == "true")
HYBRID_PREFERRED_PROVIDER = (os.getenv("HYBRID_PREFERRED_PROVIDER") or "ollama").strip().lower()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
INDEX_INTEGRITY_KEY = (os.getenv("INDEX_INTEGRITY_KEY") or "").strip()
VECTOR_BACKEND = (os.getenv("VECTOR_BACKEND", "faiss") or "faiss").strip().lower()
QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip() or None
QDRANT_PREFER_GRPC = (os.getenv("QDRANT_PREFER_GRPC", "false").strip().lower() == "true")
QDRANT_COLLECTION_PREFIX = (os.getenv("QDRANT_COLLECTION_PREFIX") or "contexta").strip()
GPU_ENABLED = (os.getenv("GPU_ENABLED", "false").strip().lower() == "true")
GPU_DEVICE_ID = (os.getenv("GPU_DEVICE_ID") or "0").strip()
EMBEDDING_DEVICE = (os.getenv("EMBEDDING_DEVICE") or "auto").strip().lower()
RERANKER_DEVICE = (os.getenv("RERANKER_DEVICE") or "auto").strip().lower()
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2").strip()
RERANKER_MODEL = (os.getenv("RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
RERANKER_ENABLED = (os.getenv("RERANKER_ENABLED", "true").strip().lower() == "true")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
RERANKER_BATCH_SIZE = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "24")))
FAST_QUERY_MODE = (os.getenv("FAST_QUERY_MODE", "false").strip().lower() == "true")
FAST_QUERY_TERM_THRESHOLD = max(3, int(os.getenv("FAST_QUERY_TERM_THRESHOLD", "8")))
MAX_CONTEXT_DOCS = max(2, int(os.getenv("MAX_CONTEXT_DOCS", "6")))
RETRIEVAL_THRESHOLD_ENABLED = (os.getenv("RETRIEVAL_THRESHOLD_ENABLED", "true").strip().lower() == "true")
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.75"))
STRUCTURED_FIELD_FALLBACK_ENABLED = (os.getenv("STRUCTURED_FIELD_FALLBACK_ENABLED", "true").strip().lower() == "true")

# Keep runtime logs clean by muting non-actionable model-load chatter.
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Suppress repeated unauthenticated HF warnings when token is intentionally absent.
warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub.*",
)

_embedding_model = None
_reranker = None
_llm = None
_ollama_llm = None
_gemini_llm = None

QUERY_CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "180"))
QUERY_CACHE_MAX_ENTRIES = int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "200"))
QUERY_CACHE_BACKEND = os.getenv("QUERY_CACHE_BACKEND", "filesystem").strip().lower()
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "24"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.25"))
MIN_GROUNDEDNESS_RATIO = float(os.getenv("MIN_GROUNDEDNESS_RATIO", "0.70"))
MIN_SENTENCE_GROUNDEDNESS = float(os.getenv("MIN_SENTENCE_GROUNDEDNESS", "0.35"))
DEFAULT_K_PER_SOURCE = int(os.getenv("DEFAULT_K_PER_SOURCE", "8"))
VECTOR_SCORE_DIRECTION = (os.getenv("VECTOR_SCORE_DIRECTION", "auto") or "auto").strip().lower()
FORCE_CHUNK_SIZE = int(os.getenv("FORCE_CHUNK_SIZE", "0"))
FORCE_CHUNK_OVERLAP = int(os.getenv("FORCE_CHUNK_OVERLAP", "0"))
STORE_LOCK_TIMEOUT_SECONDS = float(os.getenv("STORE_LOCK_TIMEOUT_SECONDS", "30"))
STORE_LOCK_STALE_SECONDS = float(os.getenv("STORE_LOCK_STALE_SECONDS", "300"))
STORE_LOCK_BACKEND = os.getenv("STORE_LOCK_BACKEND", "auto").strip().lower()
STORE_LOCK_REDIS_REQUIRED = (os.getenv("STORE_LOCK_REDIS_REQUIRED", "false").strip().lower() == "true")
STORE_LOCK_REDIS_KEY_PREFIX = (os.getenv("STORE_LOCK_REDIS_KEY_PREFIX") or "contexta:storelock").strip()
RETRIEVAL_PDF_ONLY = (os.getenv("RETRIEVAL_PDF_ONLY", "true").strip().lower() == "true")
DIRECT_QA_OVERRIDE_ENABLED = (os.getenv("DIRECT_QA_OVERRIDE_ENABLED", "false").strip().lower() == "true")
DATASET_VALIDATION_ENABLED = (os.getenv("DATASET_VALIDATION_ENABLED", "false").strip().lower() == "true")
DATASET_SIMILARITY_THRESHOLD = float(os.getenv("DATASET_SIMILARITY_THRESHOLD", "0.80"))
INSIGHTS_MAX_SOURCES = max(1, int(os.getenv("INSIGHTS_MAX_SOURCES", "3")))
INSIGHTS_ALLOW_PARTIAL_HYBRID = (os.getenv("INSIGHTS_ALLOW_PARTIAL_HYBRID", "true").strip().lower() == "true")
DATASET_REFERENCE_PATH = Path(
    os.getenv(
        "DATASET_REFERENCE_PATH",
        str(BASE_DIR / "training_data" / "limitations_qa_finetune.jsonl"),
    )
).resolve()
QA_OVERRIDES_PATH = DATASET_REFERENCE_PATH
ANSWER_NOT_FOUND_TEXT = (os.getenv("ANSWER_NOT_FOUND_TEXT") or "Not found in document").strip()
_query_result_cache: dict[tuple[str, str, int, str], tuple[float, dict]] = {}
_query_cache_lock = Lock()
_store_lock_guard = Lock()
_store_lock_counts: dict[str, int] = {}
_qdrant_client_singleton = None
_qa_overrides_cache: dict[str, dict[str, str]] = {}
_qa_overrides_mtime: float | None = None
_dataset_context_cache: list[str] = []
_dataset_context_embeddings: np.ndarray | None = None
_dataset_context_mtime: float | None = None

SYSTEM_PROMPT_TEMPLATE = """You are a highly accurate document question-answering system.

STRICT RULES:
1. Answer ONLY using the provided context.
2. Do NOT use prior knowledge.
3. If the answer is not explicitly present in the context, respond EXACTLY with:
    \"Not found in document\"
4. Do NOT guess, assume, or hallucinate.
5. Keep answers concise, factual, and precise.
6. If multiple sources exist, prioritize the most relevant and recent.
7. If the context is insufficient or unclear, say you don't know.

ANSWER FORMAT:
- Direct answer only
- No explanations unless explicitly asked
- No extra text
- No assumptions

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:"""

USER_PROMPT_TEMPLATE = """Answer the question using ONLY the given context.

Context:
{context}

Question:
{question}

Rules:
- Only use the context
- If answer not found -> say \"Not found in document\"
- Do not explain extra

Answer:"""

STRICT_GROUNDED_PROMPT_TEMPLATE = """You must behave as a strict retrieval-based QA system.

CRITICAL:
- If the exact answer is NOT in the context -> DO NOT answer.
- DO NOT infer, summarize beyond text, or guess.
- DO NOT complete partial information.

If answer is missing -> respond EXACTLY:
\"Not found in document\"

Never break this rule.

Context:
{context}

Question:
{question}

Answer:"""


def _detect_cuda_available() -> bool:
    """Best-effort CUDA check that avoids hard dependency on torch at import time."""
    if not GPU_ENABLED:
        return False
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_runtime_device(preferred: str = "auto") -> str:
    """Resolve model device from env + runtime capability."""
    pref = (preferred or "auto").strip().lower()
    if pref in {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
        if pref.startswith("cuda") and not _detect_cuda_available():
            return "cpu"
        return pref

    if _detect_cuda_available():
        # Respect configured GPU index when set.
        return f"cuda:{GPU_DEVICE_ID}" if GPU_DEVICE_ID else "cuda"
    return "cpu"


def _init_store_lock_redis_client():
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    if STORE_LOCK_BACKEND == "local":
        return None
    if not redis_url:
        return None
    if Redis is None:
        return None
    try:
        client = Redis.from_url(redis_url)
        client.ping()
        return client
    except Exception:
        return None


_store_lock_redis_client = _init_store_lock_redis_client()


def _can_use_redis_store_lock() -> bool:
    return _store_lock_redis_client is not None


def _redis_store_lock_key(namespace: str | None = None) -> str:
    ns = _normalize_namespace(namespace)
    return f"{STORE_LOCK_REDIS_KEY_PREFIX}:{ns}"


def _release_redis_store_lock(key: str, token: str) -> None:
    if not _store_lock_redis_client:
        return
    release_script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""
    _store_lock_redis_client.eval(release_script, 1, key, token)

GROUNDING_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "without", "about",
    "into", "over", "under", "between", "through", "during", "before", "after",
    "this", "that", "these", "those", "it", "its", "they", "their", "them", "we",
    "our", "you", "your", "i", "me", "my", "he", "she", "his", "her", "also",
    "can", "could", "should", "would", "may", "might", "must", "will", "shall",
    "not", "no", "yes", "there", "here", "such", "which", "who", "whom", "whose",
}


def _normalize_query(question: str) -> str:
    """Normalize user query so equivalent text maps to a single cache key."""
    return " ".join((question or "").split()).strip().lower()


def _is_allowed_retrieval_source_name(source_name: str) -> bool:
    """Return True when the source is eligible for retrieval context."""
    name = Path(str(source_name or "")).name.lower().strip()
    if not name:
        return False

    # Never retrieve from local QA seed/training artifacts.
    if name in {"limitations_qa_seed.txt", "limitations_qa_finetune.jsonl"}:
        return False

    if RETRIEVAL_PDF_ONLY and not name.endswith(".pdf"):
        return False

    return True


def _doc_is_allowed_for_retrieval(doc: Document) -> bool:
    metadata = doc.metadata or {}
    source_name = str(metadata.get("source", "")).strip()
    return _is_allowed_retrieval_source_name(source_name)


def _same_source_name(a: str, b: str) -> bool:
    """Compare source names using basename semantics."""
    return Path(str(a or "")).name.lower().strip() == Path(str(b or "")).name.lower().strip()


def _load_qa_overrides() -> dict[str, dict[str, str]]:
    """Load optional exact-match QA overrides from JSONL for deterministic answers."""
    global _qa_overrides_cache, _qa_overrides_mtime

    try:
        if not QA_OVERRIDES_PATH.exists():
            _qa_overrides_cache = {}
            _qa_overrides_mtime = None
            return _qa_overrides_cache

        mtime = QA_OVERRIDES_PATH.stat().st_mtime
        if _qa_overrides_mtime is not None and _qa_overrides_mtime == mtime:
            return _qa_overrides_cache

        overrides: dict[str, dict[str, str]] = {}
        with open(QA_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except Exception:
                    continue

                question = str(row.get("question", "")).strip()
                answer = str(row.get("answer", row.get("output", ""))).strip()
                context = str(row.get("context", "")).strip()

                if not question:
                    input_text = str(row.get("input", "")).strip()
                    if input_text:
                        q_match = re.search(r"Question\s*:\s*(.+)$", input_text, re.IGNORECASE)
                        if q_match:
                            question = q_match.group(1).strip()

                        c_match = re.search(r"Context\s*:\s*(.*?)\s*Question\s*:", input_text, re.IGNORECASE | re.DOTALL)
                        if c_match and not context:
                            context = " ".join(c_match.group(1).split()).strip()

                if not question or not answer:
                    continue

                overrides[_normalize_query(question)] = {
                    "answer": answer,
                    "context": context,
                }

        _qa_overrides_cache = overrides
        _qa_overrides_mtime = mtime
        return _qa_overrides_cache
    except Exception:
        return _qa_overrides_cache


def _lookup_qa_override(question: str) -> dict | None:
    """Return deterministic answer when question exactly matches override dataset."""
    if not DIRECT_QA_OVERRIDE_ENABLED:
        return None

    normalized = _normalize_query(question)
    if not normalized:
        return None

    entry = _load_qa_overrides().get(normalized)
    if not entry:
        return None

    answer = str(entry.get("answer", "")).strip()
    if not answer:
        return None

    snippet = " ".join((entry.get("context") or answer).split())[:220]
    source_name = "limitations_qa_seed.txt"
    return {
        "answer": answer,
        "sources": [
            {
                "source": source_name,
                "file": source_name,
                "snippet": snippet,
            }
        ],
    }


def _extract_dataset_context_from_row(row: dict) -> str:
    """Extract canonical context text from a dataset row."""
    context = " ".join(str(row.get("context", "")).split()).strip()
    if context:
        return context

    input_text = str(row.get("input", "")).strip()
    if not input_text:
        return ""

    m = re.search(
        r"Context\s*:\s*(.*?)\s*Question\s*:",
        input_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    return " ".join(m.group(1).split()).strip()


def _normalize_embedding_matrix(values: list[list[float]] | np.ndarray) -> np.ndarray:
    """Convert embeddings to row-normalized float32 matrix for cosine similarity."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _load_dataset_reference_context_matrix() -> tuple[list[str], np.ndarray | None]:
    """Load dataset reference contexts and precomputed embedding matrix."""
    global _dataset_context_cache, _dataset_context_embeddings, _dataset_context_mtime

    if not DATASET_VALIDATION_ENABLED:
        return [], None

    try:
        if not DATASET_REFERENCE_PATH.exists():
            _dataset_context_cache = []
            _dataset_context_embeddings = None
            _dataset_context_mtime = None
            return [], None

        mtime = DATASET_REFERENCE_PATH.stat().st_mtime
        if _dataset_context_mtime is not None and _dataset_context_mtime == mtime:
            return _dataset_context_cache, _dataset_context_embeddings

        contexts: list[str] = []
        seen = set()
        with open(DATASET_REFERENCE_PATH, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                context = _extract_dataset_context_from_row(row)
                if not context:
                    continue
                key = context.lower()
                if key in seen:
                    continue
                seen.add(key)
                contexts.append(context)

        if not contexts:
            _dataset_context_cache = []
            _dataset_context_embeddings = None
            _dataset_context_mtime = mtime
            return [], None

        embeddings = get_embedding_model().embed_documents(contexts)
        matrix = _normalize_embedding_matrix(embeddings)

        _dataset_context_cache = contexts
        _dataset_context_embeddings = matrix
        _dataset_context_mtime = mtime
        return contexts, matrix
    except Exception:
        return _dataset_context_cache, _dataset_context_embeddings


def _filter_docs_by_dataset_similarity(docs: list[Document]) -> list[Document]:
    """Keep only retrieved docs whose semantic similarity to dataset contexts exceeds threshold."""
    if not DATASET_VALIDATION_ENABLED:
        return docs
    if not docs:
        return []

    _, context_matrix = _load_dataset_reference_context_matrix()
    if context_matrix is None or context_matrix.size == 0:
        return []

    valid_docs: list[Document] = []
    texts: list[str] = []
    for doc in docs:
        text = " ".join((doc.page_content or "").split()).strip()
        if not text:
            continue
        valid_docs.append(doc)
        texts.append(text)

    if not texts:
        return []

    doc_matrix = _normalize_embedding_matrix(get_embedding_model().embed_documents(texts))
    similarity = np.matmul(doc_matrix, context_matrix.T)

    kept: list[Document] = []
    for idx, doc in enumerate(valid_docs):
        best = float(np.max(similarity[idx])) if similarity.shape[1] > 0 else 0.0
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["dataset_similarity"] = best
        if best >= DATASET_SIMILARITY_THRESHOLD:
            kept.append(doc)

    kept.sort(key=lambda d: float((d.metadata or {}).get("dataset_similarity", 0.0)), reverse=True)
    return kept


def _normalize_namespace(namespace: str | None) -> str:
    """Normalize user/session namespace into a safe directory key."""
    raw = (namespace or DEFAULT_NAMESPACE).strip().lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")
    return safe[:64] if safe else "default"


def _db_faiss_path(namespace: str | None = None) -> str:
    """Return namespace-scoped FAISS directory path."""
    ns = _normalize_namespace(namespace)
    if ns == "default":
        return DB_FAISS_PATH
    return os.path.join(NAMESPACED_STORE_ROOT, ns, "db_faiss")


def _index_meta_path(namespace: str | None = None) -> str:
    """Return namespace-scoped metadata file path."""
    ns = _normalize_namespace(namespace)
    if ns == "default":
        return INDEX_META_PATH
    return os.path.join(NAMESPACED_STORE_ROOT, ns, "vector_store_meta.json")


def _store_lock_path(namespace: str | None = None) -> Path:
    """Return lock-file path for namespace store operations."""
    db_path = Path(_db_faiss_path(namespace))
    return db_path.parent / ".store.lock"


def _query_cache_file_path(namespace: str | None = None) -> Path:
    """Return namespace-scoped cache file path for distributed cache backend."""
    ns = _normalize_namespace(namespace)
    if ns == "default":
        return Path(STORAGE_ROOT) / "query_cache.json"
    return Path(NAMESPACED_STORE_ROOT) / ns / "query_cache.json"


@contextmanager
def _acquire_store_lock(namespace: str | None = None):
    """Cross-process lock for namespace store operations (reentrant in-process)."""
    lock_path = _store_lock_path(namespace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(lock_path)

    with _store_lock_guard:
        current = _store_lock_counts.get(lock_key, 0)
        if current > 0:
            _store_lock_counts[lock_key] = current + 1
            reentrant = True
        else:
            reentrant = False

    use_redis_lock = _can_use_redis_store_lock() and STORE_LOCK_BACKEND in {"auto", "redis"}
    if STORE_LOCK_REDIS_REQUIRED and not use_redis_lock:
        raise RuntimeError("STORE_LOCK_REDIS_REQUIRED=true but Redis lock backend is unavailable.")

    redis_token = None
    redis_key = None

    if not reentrant and use_redis_lock:
        deadline = time.time() + max(1.0, STORE_LOCK_TIMEOUT_SECONDS)
        redis_key = _redis_store_lock_key(namespace)
        redis_token = uuid.uuid4().hex
        ttl_seconds = max(2, int(STORE_LOCK_STALE_SECONDS))
        while True:
            acquired = bool(_store_lock_redis_client.set(redis_key, redis_token, nx=True, ex=ttl_seconds))
            if acquired:
                with _store_lock_guard:
                    _store_lock_counts[lock_key] = 1
                break
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for redis store lock: {redis_key}")
            time.sleep(0.05)

    elif not reentrant:
        deadline = time.time() + max(1.0, STORE_LOCK_TIMEOUT_SECONDS)
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} ts={time.time()}\n")
                break
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > max(1.0, STORE_LOCK_STALE_SECONDS):
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue

                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for store lock: {lock_path}"
                    )
                time.sleep(0.05)

        with _store_lock_guard:
            _store_lock_counts[lock_key] = 1

    try:
        yield
    finally:
        remove_file = False
        with _store_lock_guard:
            remaining = _store_lock_counts.get(lock_key, 0)
            if remaining <= 1:
                _store_lock_counts.pop(lock_key, None)
                remove_file = not reentrant
            else:
                _store_lock_counts[lock_key] = remaining - 1

        if remove_file:
            if use_redis_lock and redis_key and redis_token:
                try:
                    _release_redis_store_lock(redis_key, redis_token)
                except Exception:
                    pass
            else:
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass


def _document_version_key(namespace: str | None = None) -> str:
    """Create a stable version key that changes whenever indexed documents change."""
    meta = _load_index_meta(namespace=namespace)
    payload = {
        "sources": sorted(str(s) for s in meta.get("sources", [])),
        "file_hashes": sorted(
            (str(k), str(v)) for k, v in meta.get("file_hashes", {}).items()
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prune_query_cache(now_ts: float) -> None:
    """Drop expired entries and enforce max cache size."""
    expired_keys = [
        key for key, (expires_at, _) in _query_result_cache.items() if expires_at <= now_ts
    ]
    for key in expired_keys:
        _query_result_cache.pop(key, None)

    if len(_query_result_cache) <= QUERY_CACHE_MAX_ENTRIES:
        return

    # Evict entries closest to expiry first.
    oldest_first = sorted(_query_result_cache.items(), key=lambda kv: kv[1][0])
    overflow = len(_query_result_cache) - QUERY_CACHE_MAX_ENTRIES
    for key, _ in oldest_first[:overflow]:
        _query_result_cache.pop(key, None)


def _prune_file_cache_entries(entries: dict[str, dict], now_ts: float) -> tuple[dict[str, dict], bool]:
    """Prune expired entries and enforce max cache size for file-backed cache."""
    cleaned: dict[str, dict] = {}
    changed = False

    for key, entry in (entries or {}).items():
        try:
            expires_at = float(entry.get("expires_at", 0))
        except Exception:
            changed = True
            continue
        if expires_at <= now_ts:
            changed = True
            continue
        payload = entry.get("payload")
        cleaned[str(key)] = {"expires_at": expires_at, "payload": payload}

    if len(cleaned) <= QUERY_CACHE_MAX_ENTRIES:
        return cleaned, changed

    ordered = sorted(cleaned.items(), key=lambda kv: kv[1]["expires_at"])
    keep = ordered[-QUERY_CACHE_MAX_ENTRIES:]
    trimmed = dict(keep)
    return trimmed, True


def _cache_key_to_string(cache_key: tuple[str, str, int, str]) -> str:
    """Serialize cache key into stable string for file-backed cache."""
    return json.dumps(list(cache_key), ensure_ascii=True, separators=(",", ":"))


def _read_file_cache(namespace: str | None = None) -> dict[str, dict]:
    """Read file-backed cache map from disk."""
    cache_path = _query_cache_file_path(namespace)
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_file_cache(entries: dict[str, dict], namespace: str | None = None) -> None:
    """Atomically persist file-backed cache map to disk."""
    cache_path = _query_cache_file_path(namespace)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, ensure_ascii=True, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(cache_path.parent),
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(cache_path)


def _cache_get(cache_key: tuple[str, str, int, str]) -> dict | None:
    """Read from cache if entry is still valid."""
    now_ts = time.time()

    if QUERY_CACHE_BACKEND == "filesystem":
        namespace = cache_key[3]
        key_str = _cache_key_to_string(cache_key)
        with _acquire_store_lock(namespace):
            entries = _read_file_cache(namespace)
            entries, changed = _prune_file_cache_entries(entries, now_ts)
            entry = entries.get(key_str)
            if changed:
                _write_file_cache(entries, namespace)
            if not entry:
                return None
            return copy.deepcopy(entry.get("payload"))

    with _query_cache_lock:
        _prune_query_cache(now_ts)
        entry = _query_result_cache.get(cache_key)
        if not entry:
            return None
        expires_at, payload = entry
        if expires_at <= now_ts:
            _query_result_cache.pop(cache_key, None)
            return None
        return copy.deepcopy(payload)


def _cache_set(cache_key: tuple[str, str, int, str], payload: dict) -> None:
    """Store successful query result in cache for short TTL."""
    now_ts = time.time()

    if QUERY_CACHE_BACKEND == "filesystem":
        namespace = cache_key[3]
        key_str = _cache_key_to_string(cache_key)
        with _acquire_store_lock(namespace):
            entries = _read_file_cache(namespace)
            entries, _ = _prune_file_cache_entries(entries, now_ts)
            entries[key_str] = {
                "expires_at": now_ts + max(1, QUERY_CACHE_TTL_SECONDS),
                "payload": copy.deepcopy(payload),
            }
            entries, _ = _prune_file_cache_entries(entries, now_ts)
            _write_file_cache(entries, namespace)
        return

    with _query_cache_lock:
        _prune_query_cache(now_ts)
        _query_result_cache[cache_key] = (
            now_ts + max(1, QUERY_CACHE_TTL_SECONDS),
            copy.deepcopy(payload),
        )


def _clear_query_cache(namespace: str | None = None) -> None:
    """Clear all cached query results."""
    if QUERY_CACHE_BACKEND == "filesystem":
        with _acquire_store_lock(namespace):
            cache_path = _query_cache_file_path(namespace)
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass

    with _query_cache_lock:
        if namespace is None:
            _query_result_cache.clear()
            return

        ns = _normalize_namespace(namespace)
        keys_to_drop = [key for key in _query_result_cache if key[3] == ns]
        for key in keys_to_drop:
            _query_result_cache.pop(key, None)


def _heading_like_ratio(text: str) -> float:
    """Estimate how often lines look like section headings."""
    if not text:
        return 0.0

    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    if not lines:
        return 0.0

    heading_like = 0
    for line in lines:
        if len(line) > 90:
            continue
        if line[-1:] in {".", ",", ";", ":"}:
            continue
        if line.isupper() or line.istitle():
            heading_like += 1
            continue
        if line[:1].isdigit() and any(ch.isalpha() for ch in line):
            heading_like += 1

    return heading_like / max(1, len(lines))


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MIN_ADAPTIVE_CHUNK_TOKENS = 300
MAX_ADAPTIVE_CHUNK_TOKENS = 800


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate for splitter sizing without model-specific tokenizers."""
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


def _default_overlap_for_chunk_size(chunk_size: int) -> int:
    """Use ~18% overlap with sane floors/ceilings for context continuity."""
    overlap = int(round(chunk_size * 0.18))
    overlap = max(60, overlap)
    overlap = min(180, overlap)
    return min(overlap, max(1, chunk_size - 40))


def _normalize_chunk_params(chunk_size: int, chunk_overlap: int | None = None) -> tuple[int, int]:
    """Clamp chunk params to adaptive bounds and ensure valid overlap."""
    size = max(MIN_ADAPTIVE_CHUNK_TOKENS, min(MAX_ADAPTIVE_CHUNK_TOKENS, int(chunk_size)))
    if chunk_overlap is None:
        overlap = _default_overlap_for_chunk_size(size)
    else:
        overlap = int(chunk_overlap)
    overlap = max(40, overlap)
    overlap = min(overlap, max(1, size - 40))
    return size, overlap


def _choose_chunking_strategy(file_path: str, docs: list[Document]) -> tuple[int, int]:
    """
    Pick chunk size/overlap adaptively based on document type, page density,
    and heading-like structure density.
    """
    if FORCE_CHUNK_SIZE > 0:
        forced_overlap = FORCE_CHUNK_OVERLAP if FORCE_CHUNK_OVERLAP > 0 else None
        return _normalize_chunk_params(FORCE_CHUNK_SIZE, forced_overlap)

    suffix = Path(file_path).suffix.lower()
    content = [d.page_content or "" for d in docs]
    total_chars = sum(len(c) for c in content)
    units = max(1, len(content))
    avg_chars_per_unit = total_chars / units

    file_hint = Path(file_path).name.lower()
    merged_text = "\n".join(content[:40])
    merged_lower = merged_text.lower()
    heading_ratio = _heading_like_ratio(merged_text)

    is_offer_or_letter = any(k in file_hint for k in {"offer", "letter", "agreement", "contract"})
    is_research_like = any(
        k in merged_lower
        for k in {"abstract", "introduction", "methodology", "results", "conclusion", "references"}
    )

    # Letter-style documents are short and precise; smaller chunks reduce noise.
    if is_offer_or_letter:
        return _normalize_chunk_params(320, 64)

    # Research/report documents benefit from larger semantic windows.
    if is_research_like:
        if avg_chars_per_unit >= 4200:
            return _normalize_chunk_params(780, 140)
        return _normalize_chunk_params(700, 126)

    # PDF defaults tuned by density and heading signals.
    if avg_chars_per_unit >= 4500 and heading_ratio < 0.06:
        return _normalize_chunk_params(800, 150)
    if heading_ratio >= 0.12:
        return _normalize_chunk_params(520, 100)
    if avg_chars_per_unit <= 1700:
        return _normalize_chunk_params(420, 80)
    return _normalize_chunk_params(640, 120)


def get_embedding_model():
    """Lazy-load embeddings model so startup/reset does not incur model load time."""
    global _embedding_model
    if _embedding_model is None:
        model_kwargs = {"device": _resolve_runtime_device(EMBEDDING_DEVICE)}
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN

        # Some third-party loaders print noisy, non-actionable startup reports to stdout/stderr.
        # Keep startup logs clean while still allowing real exceptions to propagate.
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
            )
    return _embedding_model


def get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder reranker so model weights are only fetched once."""
    global _reranker
    if _reranker is None:
        reranker_kwargs = {
            "max_length": RERANKER_MAX_LENGTH,
            "device": _resolve_runtime_device(RERANKER_DEVICE),
        }
        if HF_TOKEN:
            reranker_kwargs["token"] = HF_TOKEN

        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            _reranker = CrossEncoder(
                RERANKER_MODEL, **reranker_kwargs
            )
    return _reranker


def _rerank_with_scores(question: str, docs: list) -> list[tuple[float, Document]]:
    """Re-score candidate docs with the cross-encoder and keep scores."""
    if not docs:
        return []
    if not RERANKER_ENABLED:
        return [(float(len(docs) - i), doc) for i, doc in enumerate(docs)]
    try:
        reranker = get_reranker()
        pairs = [(question, doc.page_content) for doc in docs]
        try:
            scores = reranker.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
        except TypeError:
            scores = reranker.predict(pairs)
        return sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    except Exception as e:
        print(f"Reranker failed, returning original order: {e}")
        return [(float(len(docs) - i), doc) for i, doc in enumerate(docs)]


def _rerank(question: str, docs: list, top_n: int = 6) -> list:
    """
    Re-score candidate docs with the cross-encoder and return the top_n by score.
    Falls back to returning docs as-is if reranking fails.
    """
    ranked = _rerank_with_scores(question, docs)
    return [doc for _, doc in ranked[:top_n]]


def _vector_score_lower_is_better() -> bool:
    """Return score direction for backend similarity scores."""
    if VECTOR_SCORE_DIRECTION in {"lower", "distance", "l2"}:
        return True
    if VECTOR_SCORE_DIRECTION in {"higher", "similarity", "cosine"}:
        return False
    # Auto mode: FAISS returns distances (lower better), Qdrant similarity usually higher better.
    return VECTOR_BACKEND == "faiss"


def _score_is_better(new_score: float, prev_score: float) -> bool:
    if _vector_score_lower_is_better():
        return new_score < prev_score
    return new_score > prev_score


def _sort_pairs_by_score(pairs: list[tuple[float, Document]]) -> list[tuple[float, Document]]:
    return sorted(pairs, key=lambda x: x[0], reverse=not _vector_score_lower_is_better())


def vector_store_exists(namespace: str | None = None):
    """Check if vector store exists for active backend."""
    if VECTOR_BACKEND == "qdrant":
        if QdrantClient is None or not QDRANT_URL:
            return False
        try:
            client = _get_qdrant_client()
            name = _qdrant_collection_name(namespace)
            if hasattr(client, "collection_exists"):
                return bool(client.collection_exists(name))
            client.get_collection(name)
            return True
        except Exception:
            return False

    db_path = Path(_db_faiss_path(namespace))
    return db_path.exists() and (db_path / "index.faiss").exists() and (db_path / "index.pkl").exists()


def _qdrant_collection_name(namespace: str | None = None) -> str:
    ns = _normalize_namespace(namespace)
    return f"{QDRANT_COLLECTION_PREFIX}_{ns}".replace("-", "_")


def _get_qdrant_client():
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        if QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed.")
        if not QDRANT_URL:
            raise RuntimeError("QDRANT_URL must be set when VECTOR_BACKEND=qdrant.")
        _qdrant_client_singleton = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=QDRANT_PREFER_GRPC)
    return _qdrant_client_singleton


def _load_vectorstore(namespace: str | None = None, *, create_if_missing: bool = False, seed_docs: list[Document] | None = None):
    if VECTOR_BACKEND == "qdrant":
        if LCQdrant is None:
            raise RuntimeError("LangChain Qdrant vectorstore adapter is unavailable.")
        collection_name = _qdrant_collection_name(namespace)
        if create_if_missing and seed_docs:
            return LCQdrant.from_documents(
                seed_docs,
                get_embedding_model(),
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=QDRANT_PREFER_GRPC,
                collection_name=collection_name,
            )
        return LCQdrant(
            client=_get_qdrant_client(),
            collection_name=collection_name,
            embeddings=get_embedding_model(),
        )

    return _load_faiss_store_secure(namespace=namespace)


def _load_faiss_store_secure(namespace: str | None = None) -> FAISS:
    """Load FAISS only after signed artifact integrity verification succeeds."""
    db_path = _db_faiss_path(namespace)
    with _acquire_store_lock(namespace):
        verify_faiss_integrity(db_path, INDEX_INTEGRITY_KEY)
        return FAISS.load_local(
            db_path,
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )


def _file_hash(path: str) -> str:
    """Return SHA-256 hex digest of the raw file bytes for file-level dedup."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_index_meta(namespace: str | None = None) -> dict:
    """Load persisted ingestion metadata used for deduplication."""
    meta_path = Path(_index_meta_path(namespace))
    if not meta_path.exists():
        return {"chunk_hashes": [], "sources": [], "file_hashes": {}}

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        chunk_hashes = data.get("chunk_hashes", [])
        if not isinstance(chunk_hashes, list):
            chunk_hashes = []
        sources = data.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        file_hashes = data.get("file_hashes", {})
        if not isinstance(file_hashes, dict):
            file_hashes = {}
        return {
            "chunk_hashes": [str(h) for h in chunk_hashes],
            "sources": [str(s) for s in sources],
            "file_hashes": {str(k): str(v) for k, v in file_hashes.items()},
        }
    except Exception:
        return {"chunk_hashes": [], "sources": [], "file_hashes": {}}


def _save_index_meta(meta: dict, namespace: str | None = None) -> None:
    """Persist ingestion metadata used for deduplication."""
    meta_path = Path(_index_meta_path(namespace))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(meta, ensure_ascii=True, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(meta_path.parent),
        prefix=f".{meta_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(meta_path)


def _stable_chunk_hash(text: str) -> str:
    """Hash normalized chunk text so duplicate chunks can be skipped across uploads."""
    normalized = " ".join((text or "").split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_documents(file_path: str) -> list[Document]:
    """Load supported document content into LangChain Document objects."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()
    if suffix == ".docx":
        return Docx2txtLoader(file_path).load()
    if suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8", autodetect_encoding=True).load()
    if suffix in {".html", ".htm"}:
        return BSHTMLLoader(file_path).load()

    raise ValueError("Unsupported file type. Allowed: .pdf, .docx, .txt, .html, .htm")


def append_document_to_vector_store(
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int = 32,
    namespace: str | None = None,
) -> dict:
    """
        Append a supported document into the existing FAISS index with two-layer deduplication:
      1. File hash  â€” identical file â†’ skip loading/parsing/embedding entirely.
      2. Chunk hash â€” changed/new file â†’ only embed chunks not already in the index.
    Returns ingestion stats.
    """
    with _acquire_store_lock(namespace):
        source_name = Path(file_path).name
        meta = _load_index_meta(namespace=namespace)

        # â”€â”€ Layer 1: file-level hash check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        incoming_file_hash = _file_hash(file_path)
        known_file_hashes: dict = meta.get("file_hashes", {})
        if known_file_hashes.get(source_name) == incoming_file_hash:
            # Byte-for-byte identical file already indexed â€” skip everything.
            return {
                "source": source_name,
                "total_chunks": 0,
                "added_chunks": 0,
                "duplicate_chunks": 0,
                "skipped": True,
                "reason": "Identical file already indexed (file hash match).",
            }

        # â”€â”€ Layer 2: load, split, chunk-hash dedup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pages = _load_documents(file_path)
        if not pages:
            raise ValueError("Document is empty or could not be read.")

        adaptive_chunk_size, adaptive_chunk_overlap = _choose_chunking_strategy(file_path, pages)
        requested_size = chunk_size if chunk_size is not None else adaptive_chunk_size
        requested_overlap = chunk_overlap if chunk_overlap is not None else adaptive_chunk_overlap
        final_chunk_size, final_chunk_overlap = _normalize_chunk_params(requested_size, requested_overlap)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=_estimate_tokens,
        )

        known_hashes = set(meta.get("chunk_hashes", []))

        docs_to_add = []
        total_chunks = 0
        duplicate_chunks = 0

        for page in pages:
            chunks = splitter.split_documents([page])
            for chunk in chunks:
                total_chunks += 1
                chunk_hash = _stable_chunk_hash(chunk.page_content)
                if chunk_hash in known_hashes:
                    duplicate_chunks += 1
                    continue

                chunk.metadata["source"] = source_name
                chunk.metadata["chunk_hash"] = chunk_hash
                docs_to_add.append(chunk)
                known_hashes.add(chunk_hash)

        added_chunks = len(docs_to_add)
        if added_chunks == 0:
            # All chunks already present â€” still update file hash so next upload of
            # the same file is short-circuited at layer 1.
            known_file_hashes[source_name] = incoming_file_hash
            meta["file_hashes"] = known_file_hashes
            _save_index_meta(meta, namespace=namespace)
            _clear_query_cache(namespace=namespace)
            return {
                "source": source_name,
                "total_chunks": total_chunks,
                "added_chunks": 0,
                "duplicate_chunks": duplicate_chunks,
            }

        vectorstore = None
        use_qdrant = (VECTOR_BACKEND == "qdrant")
        db_path = _db_faiss_path(namespace)
        if vector_store_exists(namespace=namespace):
            vectorstore = _load_vectorstore(namespace=namespace)

        for i in range(0, len(docs_to_add), batch_size):
            batch_docs = docs_to_add[i:i + batch_size]
            if vectorstore is None:
                if use_qdrant:
                    vectorstore = _load_vectorstore(namespace=namespace, create_if_missing=True, seed_docs=batch_docs)
                else:
                    vectorstore = FAISS.from_documents(batch_docs, get_embedding_model())
            else:
                vectorstore.add_documents(batch_docs)

        if vectorstore is None:
            raise RuntimeError("Failed to create or update vector store.")

        if not use_qdrant:
            vectorstore.save_local(db_path)
            write_faiss_integrity_manifest(db_path, INDEX_INTEGRITY_KEY)

        # Track source names and update the file hash so future identical uploads
        # are caught at layer 1 without loading or embedding anything.
        known_sources = meta.get("sources", [])
        if source_name not in known_sources:
            known_sources = known_sources + [source_name]

        known_file_hashes[source_name] = incoming_file_hash

        _save_index_meta({
            "chunk_hashes": sorted(known_hashes),
            "sources": known_sources,
            "file_hashes": known_file_hashes,
        }, namespace=namespace)
        _clear_query_cache(namespace=namespace)

        return {
            "source": source_name,
            "total_chunks": total_chunks,
            "added_chunks": added_chunks,
            "duplicate_chunks": duplicate_chunks,
            "chunking": {
                "chunk_size": final_chunk_size,
                "chunk_overlap": final_chunk_overlap,
                "adaptive": chunk_size is None and chunk_overlap is None,
            },
        }

def process_pdf(pdf_path, namespace: str | None = None):
    """Backward-compatible ingest API: append to index with deduplication."""
    try:
        append_document_to_vector_store(pdf_path, namespace=namespace)
        return True
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise


def append_pdf_to_vector_store(pdf_path: str, chunk_size: int | None = None, chunk_overlap: int | None = None, batch_size: int = 32, namespace: str | None = None) -> dict:
    """Backward-compatible alias; now supports non-PDF docs via generic implementation."""
    return append_document_to_vector_store(pdf_path, chunk_size, chunk_overlap, batch_size, namespace)

def load_retriever(namespace: str | None = None):
    """Load the FAISS vector store and return as retriever with safety checks"""
    try:
        # CRITICAL: Check if vector store exists before attempting to load
        if not vector_store_exists(namespace=namespace):
            raise FileNotFoundError(
                "No PDF has been processed yet. Please upload and process a PDF first."
            )
        
        vectorstore = _load_vectorstore(namespace=namespace)
        # Use higher k so chunks from multiple documents all surface
        return vectorstore.as_retriever(search_kwargs={"k": 10})
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        print(f"Error loading retriever: {str(e)}")
        raise


def _build_multi_source_docs(question: str, k_per_source: int = 6, namespace: str | None = None) -> list[Document]:
    """
    Retrieve top-k chunks per source document so every document is guaranteed
    representation in the context, regardless of overall similarity ranking.
    Falls back to a global search when only one source is tracked.
    """
    vectorstore = _load_vectorstore(namespace=namespace)
    sources = [
        s
        for s in _load_index_meta(namespace=namespace).get("sources", [])
        if _is_allowed_retrieval_source_name(str(s))
    ]

    seen_best: dict[str, tuple[float, Document]] = {}
    q_terms = _content_words(question)
    short_query_fast_mode = FAST_QUERY_MODE and len(q_terms) <= FAST_QUERY_TERM_THRESHOLD and not _is_count_question(question)
    vector_only_mode = short_query_fast_mode or (not RERANKER_ENABLED)
    if vector_only_mode:
        global_k = max(10, min(MAX_RERANK_CANDIDATES, 24))
    else:
        global_k = max(12, min(MAX_RERANK_CANDIDATES * 4, 64))

    # First pass: global candidate retrieval prioritizes pure relevance.
    global_pairs = vectorstore.similarity_search_with_score(question, k=global_k)
    for doc, score in global_pairs:
        if not _doc_is_allowed_for_retrieval(doc):
            continue
        uid = doc.metadata.get("chunk_hash", doc.page_content[:80])
        best = seen_best.get(uid)
        if best is None or _score_is_better(float(score), float(best[0])):
            doc.metadata["vector_score"] = float(score)
            seen_best[uid] = (float(score), doc)

    # Second pass: optional light per-source fallback so smaller docs are not starved.
    if len(sources) > 1 and not vector_only_mode:
        per_source_k = max(1, min(2, k_per_source))
        for source in sources:
            try:
                pairs = vectorstore.similarity_search_with_score(
                    question,
                    k=per_source_k,
                    filter={"source": source},
                )
            except Exception:
                pairs = []
            for doc, score in pairs:
                uid = doc.metadata.get("chunk_hash", doc.page_content[:80])
                best = seen_best.get(uid)
                if best is None or _score_is_better(float(score), float(best[0])):
                    doc.metadata["vector_score"] = float(score)
                    seen_best[uid] = (float(score), doc)

    candidates = [
        doc
        for _, doc in _sort_pairs_by_score(list(seen_best.values()))[
            : (min(MAX_RERANK_CANDIDATES, 10) if vector_only_mode else MAX_RERANK_CANDIDATES)
        ]
    ]
    if vector_only_mode:
        # For short factoid queries, vector ranking is usually sufficient and much faster.
        ranked = [(float(len(candidates) - i), doc) for i, doc in enumerate(candidates)]
    else:
        ranked = _rerank_with_scores(question, candidates)

    if not ranked:
        return []

    # Intent-aware lexical boost helps short factual questions (e.g., location, salary, date)
    # prioritize chunks with exact domain terms from the PDF.
    boosted_terms = set(q_terms)
    if {"location", "locate", "located", "address", "office", "company"} & q_terms:
        boosted_terms |= {"address", "located", "location", "office", "city", "state", "headquarters"}
    if {"salary", "ctc", "pay", "compensation", "stipend", "package"} & q_terms:
        boosted_terms |= {"salary", "ctc", "compensation", "package", "pay", "stipend", "lpa"}
    if {"joining", "join", "start", "date", "doj"} & q_terms:
        boosted_terms |= {"joining", "join", "start", "date", "doj", "commencement"}

    rescored: list[tuple[float, float, Document]] = []
    for rerank_score, doc in ranked:
        d_words = _content_words(doc.page_content or "")
        lexical = len(boosted_terms & d_words) / max(1, len(boosted_terms)) if boosted_terms else 0.0
        doc.metadata["lexical_score"] = float(lexical)
        rescored.append((lexical, float(rerank_score), doc))

    rescored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    max_docs = min(MAX_CONTEXT_DOCS, max(2 if vector_only_mode else 3, k_per_source))
    kept = [doc for lexical, score, doc in rescored if score >= RERANK_MIN_SCORE or lexical >= 0.10][:max_docs]

    # Ensure a non-empty context even for difficult queries without forcing many weak chunks.
    if not kept:
        kept = [doc for _, _, doc in rescored[: min(2, len(rescored))]]

    for score, doc in ranked:
        doc.metadata["rerank_score"] = float(score)

    return kept


def _recommended_k_per_source(question: str) -> int:
    """Pick retrieval depth based on query complexity and configured default."""
    q_words = _content_words(question)
    if len(q_words) >= 10:
        return max(DEFAULT_K_PER_SOURCE, 10)
    if len(q_words) <= 4:
        # Keep short factual queries (e.g., FAISS limitations) from missing sparse chunks.
        return max(10, DEFAULT_K_PER_SOURCE)
    return max(6, DEFAULT_K_PER_SOURCE)


def _content_words(text: str) -> set[str]:
    """Tokenize text and keep informational words for grounding checks."""
    tokens = {
        t
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", (text or "").lower())
        if len(t) > 2 and t not in GROUNDING_STOPWORDS
    }
    return tokens


def _grounding_ratio(answer: str, context: str) -> float:
    """Fraction of answer content words that are present in retrieved context."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = _content_words(context)
    return len(answer_words & context_words) / len(answer_words)


def _sentence_grounding_ratio(sentence: str, context_words: set[str]) -> float:
    """Grounding score for a single sentence against context terms."""
    words = _content_words(sentence)
    if not words:
        return 1.0
    return len(words & context_words) / len(words)


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter that preserves simple formatting."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if parts:
        return parts

    # Fallback for OCR-heavy text that may miss sentence punctuation.
    line_parts = [s.strip(" -\t") for s in re.split(r"[\n\r;]+", (text or "").strip()) if s.strip()]
    if line_parts:
        return line_parts

    fallback = (text or "").strip()
    return [fallback] if fallback else []


COUNT_QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "at", "by", "with",
    "is", "are", "was", "were", "be", "been", "being", "there", "this", "that",
    "give", "me", "please", "tell", "show", "related", "about", "from", "document",
    "count", "number", "total", "many", "much", "all",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _parse_number_token(value: str | None) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace(",", "")
    if not token:
        return None
    if token.isdigit():
        try:
            return int(token)
        except Exception:
            return None
    return NUMBER_WORDS.get(token)


def _extract_count_target(question: str) -> str:
    """Extract likely count target phrase from noisy user questions."""
    q = (question or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", q)
    if not tokens:
        return ""

    starts: list[int] = []
    for i in range(max(0, len(tokens) - 1)):
        if tokens[i] == "how" and tokens[i + 1] == "many":
            starts.append(i + 2)
    for i in range(max(0, len(tokens) - 1)):
        if tokens[i] in {"number", "count", "total"} and tokens[i + 1] == "of":
            starts.append(i + 2)

    if not starts and "count" in tokens:
        starts.append(tokens.index("count") + 1)

    if not starts:
        return ""

    best_terms: list[str] = []
    for start in starts:
        tail = tokens[start:start + 8]
        filtered = [t for t in tail if len(t) > 2 and t not in COUNT_QUERY_STOPWORDS]
        if filtered and len(filtered) > len(best_terms):
            best_terms = filtered

    return " ".join(best_terms[:4]).strip()


def _build_count_query_candidates(question: str) -> list[str]:
    """Generate resilient retrieval queries for count-style prompts."""
    normalized = " ".join((question or "").split()).strip()
    target = _extract_count_target(question)
    candidates = [normalized]
    if target:
        candidates.extend([
            f"how many {target}",
            f"number of {target}",
            f"total {target}",
            target,
        ])

    # Generic fallback for very noisy prompts.
    candidates.append("count total")

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        c = candidate.strip()
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _docstore_count_context(vectorstore, target: str, max_docs: int = 200) -> tuple[str, list[Document]]:
    """Build context directly from in-memory docstore when retrieval misses count evidence."""
    raw_docs = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
    if not isinstance(raw_docs, dict):
        return "", []

    target_terms = [t for t in _content_words(target) if t]
    selected: list[Document] = []
    fallback: list[Document] = []

    for value in raw_docs.values():
        content = getattr(value, "page_content", "")
        metadata = getattr(value, "metadata", {}) or {}
        if not _is_allowed_retrieval_source_name(str(metadata.get("source", ""))):
            continue
        if not content:
            continue

        doc = Document(page_content=content, metadata=metadata)
        fallback.append(doc)

        lowered = content.lower()
        if target_terms and any(term in lowered for term in target_terms):
            selected.append(doc)

    docs = (selected or fallback)[:max_docs]
    return _docs_to_context(docs), docs


def _estimate_limitation_count_from_docstore(vectorstore) -> tuple[int | None, list[Document], str]:
    """Estimate limitation row count for table-style PDFs without explicit numeric totals."""
    raw_docs = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
    if not isinstance(raw_docs, dict) or not raw_docs:
        return None, [], ""

    docs: list[Document] = []
    all_text_parts: list[str] = []
    for value in raw_docs.values():
        content = (getattr(value, "page_content", "") or "").strip()
        if not content:
            continue
        metadata = getattr(value, "metadata", {}) or {}
        if not _is_allowed_retrieval_source_name(str(metadata.get("source", ""))):
            continue
        docs.append(value if isinstance(value, Document) else Document(page_content=content, metadata=metadata))
        all_text_parts.append(content)

    if not all_text_parts:
        return None, [], ""

    merged_text = "\n".join(all_text_parts)
    merged_lower = merged_text.lower()
    if "limita" not in merged_lower:
        return None, docs, merged_text

    action_prefixes = (
        "use ",
        "add ",
        "keep ",
        "cache ",
        "move ",
        "batch ",
        "run ",
        "periodically ",
    )

    # In limitation/fix tables, each limitation row typically has exactly one actionable fix line.
    action_lines: list[str] = []
    seen = set()
    for raw_line in merged_text.splitlines():
        line = " ".join(raw_line.strip().split())
        if len(line) < 10:
            continue
        # OCR/table wraps can produce continuation lines; count only primary action lines.
        if not line[:1].isupper():
            continue
        line_lower = line.lower()
        if not line_lower.startswith(action_prefixes):
            continue
        if line_lower in seen:
            continue
        seen.add(line_lower)
        action_lines.append(line)

    if len(action_lines) >= 2:
        return len(action_lines), docs, merged_text
    return None, docs, merged_text


def _is_count_question(question: str) -> bool:
    """Detect requests asking for counts/totals/how-many style answers."""
    q = (question or "").lower()
    return any(token in q for token in [
        "how many", "number of", "count", "total number", "total count", "no. of", "no of", "how much"
    ])


def _is_smalltalk_question(question: str) -> bool:
    """Detect greeting/chitchat prompts that should not hit strict document retrieval."""
    q = _normalize_query(question)
    if not q:
        return False

    direct_greetings = {
        "hi", "hii", "hiii", "hello", "hey", "hey there", "yo", "sup",
        "good morning", "good afternoon", "good evening",
    }
    if q in direct_greetings:
        return True

    return any(
        q.startswith(prefix)
        for prefix in (
            "hi ", "hello ", "hey ", "how are you", "who are you", "what can you do",
        )
    )


def _smalltalk_response() -> str:
    return (
        "Hi! I can answer questions from your uploaded document. "
        "Ask something like: What are the key points, main limitations, or summary?"
    )


def _finalize_formatted_answer(question: str, answer: str, context: str) -> str:
    """Preserve structure for insights prompts (bullets, one-line summaries)."""
    text = (answer or "").strip()
    if not text:
        return _extractive_fallback_answer(question, context) if context else ANSWER_NOT_FOUND_TEXT

    q_norm = _normalize_query(question)

    # Keep exactly one concise line for one-line summary prompts.
    if "one-line summary" in q_norm:
        sentences = _split_sentences(text)
        if sentences:
            return re.sub(r"\s+", " ", sentences[0]).strip()[:220].rstrip()
        return re.sub(r"\s+", " ", text).strip()[:220].rstrip()

    # Preserve bullet formatting for insights output.
    if "key insights" in q_norm or "bullet" in q_norm:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullet_lines = [
            ln for ln in lines
            if re.match(r"^(?:[-*•]|\d+[\).])\s+", ln)
        ]
        if bullet_lines:
            return "\n".join(bullet_lines[:5])

        flattened = re.sub(r"\s+", " ", text).strip()
        sentences = _split_sentences(flattened)
        if not sentences:
            return flattened[:520].rstrip()

        bullets: list[str] = []
        for sent in sentences[:5]:
            clean_sent = sent.strip()
            if not clean_sent:
                continue
            bullets.append(f"- {clean_sent}")
        return "\n".join(bullets) if bullets else flattened[:520].rstrip()

    return text


def _is_page_count_question(question: str) -> bool:
    q = (question or "").lower()
    return _is_count_question(question) and any(token in q for token in ["page", "pages"])


def _extract_first_number(text: str) -> str | None:
    """Return the first integer-like token from text (e.g. 12, 1,234)."""
    if not text:
        return None
    match = re.search(r"\b\d{1,3}(?:,\d{3})*\b|\b\d+\b", text)
    if not match:
        return None
    return match.group(0)


def _extract_count_value(answer: str, context: str) -> str | None:
    """Infer a count value from explicit totals or enumerated lists."""
    combined_answer = answer or ""
    combined_context = context or ""
    number_token_pattern = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"

    total_patterns = [
        rf"(?:there are|there is|count is|total(?:\s+of)?|number of [a-z\s]+ is)\s+{number_token_pattern}",
        rf"(?:count|total|number)\s*[:=]\s*{number_token_pattern}",
        rf"{number_token_pattern}\s+(?:[a-z-]+\s+){{0,3}}(?:items|points|steps|limitations|issues|findings|pages)\b",
    ]
    for pattern in total_patterns:
        m = re.search(pattern, combined_answer, re.IGNORECASE)
        if m:
            parsed = _parse_number_token(m.group(1))
            if parsed is not None and parsed > 0:
                return str(parsed)

    # For structured lists like "1) ... 2) ... 3) ...", use the largest index.
    enum_values = re.findall(r"(?:^|\s)(\d{1,3})\s*[\).:-]", combined_answer + "\n" + combined_context)
    if enum_values:
        try:
            nums = [int(v) for v in enum_values]
            if len(nums) >= 2:
                return str(max(nums))
        except Exception:
            pass

    fallback = _extract_first_number(combined_answer) or _extract_first_number(combined_context)
    if fallback is None:
        return None
    # Avoid false count from zero-indexed metadata tags (e.g., page=0)
    try:
        if int(fallback.replace(",", "")) <= 0:
            return None
    except Exception:
        pass
    return fallback


def _format_count_answer(question: str, count: str, detail: str | None = None) -> str:
    """Render count answers in natural, question-framed style."""
    q = (question or "").lower()
    if "page" in q:
        base = f"There are {count} pages in the document."
    elif "step" in q:
        base = f"There are {count} steps."
    elif "limitation" in q:
        base = f"There are {count} limitations."
    else:
        base = f"There are {count} relevant items."

    if detail:
        return f"{base} {detail}."
    return base


def _count_pages_from_faiss(namespace: str | None = None) -> int | None:
    """Estimate total pages from FAISS chunk metadata (max page index + 1)."""
    if VECTOR_BACKEND != "faiss":
        return None
    try:
        vectorstore = _load_vectorstore(namespace=namespace)
        raw_docs = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
        if not isinstance(raw_docs, dict):
            return None
        page_values: list[int] = []
        for doc in raw_docs.values():
            metadata = getattr(doc, "metadata", {}) or {}
            page = metadata.get("page")
            if page is None:
                continue
            try:
                p = int(page)
            except Exception:
                continue
            if p >= 0:
                page_values.append(p)
        if not page_values:
            return None
        return max(page_values) + 1
    except Exception:
        return None


def _fast_count_response(question: str, namespace: str | None = None) -> dict | None:
    """Fast deterministic path for count questions to reduce latency."""
    if not _is_count_question(question):
        return None

    count_value: str | None = None
    used_structured_estimate = False
    selected_docs: list[Document] = []
    selected_context = ""
    if _is_page_count_question(question):
        pages = _count_pages_from_faiss(namespace=namespace)
        if pages is not None and pages > 0:
            count_value = str(pages)

    try:
        vectorstore = _load_vectorstore(namespace=namespace)
        query_candidates = _build_count_query_candidates(question)
        docs: list[Document] = []
        for idx, candidate in enumerate(query_candidates):
            k = 6 if idx == 0 else 14
            docs = vectorstore.similarity_search(candidate, k=k)
            context = _docs_to_context(docs) if docs else ""
            if count_value is None:
                count_value = _extract_count_value("", context)
            if count_value:
                selected_docs = docs
                selected_context = context
                break

        if not count_value:
            target = _extract_count_target(question)
            context, fallback_docs = _docstore_count_context(vectorstore, target)
            fallback_count = _extract_count_value("", context)
            if fallback_count:
                count_value = fallback_count
                selected_docs = fallback_docs
                selected_context = context

        # Fallback for table-style limitation documents that list items but omit explicit totals.
        if not count_value and "limitation" in (question or "").lower():
            estimated, limitation_docs, limitation_context = _estimate_limitation_count_from_docstore(vectorstore)
            if estimated is not None and estimated > 0:
                count_value = str(estimated)
                used_structured_estimate = True
                selected_docs = limitation_docs
                selected_context = limitation_context
    except Exception:
        selected_docs = []
        selected_context = ""

    if not count_value:
        return None

    detail = None if used_structured_estimate else _supporting_detail(question, selected_context, selected_context)
    answer = _format_count_answer(question, count_value, detail)
    return {
        "answer": answer,
        "sources": _serialize_sources(selected_docs, max_items=1),
    }


def _supporting_detail(question: str, answer: str, context: str) -> str | None:
    """Pick one short relevant sentence to accompany count answers."""
    q_words = _content_words(question)
    candidates = _split_sentences(answer) + _split_sentences(context)
    if not candidates:
        return None

    ranked = sorted(
        (c.strip() for c in candidates if c.strip()),
        key=lambda s: len(q_words & _content_words(s)),
        reverse=True,
    )
    for sent in ranked:
        # Skip plain count-only lines.
        if re.fullmatch(r"(?i)(?:the\s+)?(?:count|total|number)\s*(?:is|:)\s*\d+\.?", sent):
            continue
        if "source=" in sent.lower() or sent.strip().startswith("["):
            continue
        trimmed = re.sub(r"\s+", " ", sent).strip()
        if trimmed:
            return trimmed[:180].rstrip(" .,;:!?")
    return None


def _postprocess_answer(question: str, answer: str, context: str) -> str:
    """Keep responses short/relevant and make count answers explicit."""
    cleaned = re.sub(r"\s+", " ", (answer or "")).strip()
    if not cleaned:
        return ANSWER_NOT_FOUND_TEXT

    if _is_count_question(question):
        number = _extract_count_value(cleaned, context)
        if number:
            detail = _supporting_detail(question, cleaned, context)
            return _format_count_answer(question, number, detail)
        return ANSWER_NOT_FOUND_TEXT

    sentences = _split_sentences(cleaned)
    if not sentences:
        return cleaned[:320]

    # Keep normal answers focused but not over-compressed.
    q_words = _content_words(question)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: len(q_words & _content_words(item[1])),
        reverse=True,
    )
    take_n = 3 if _is_summary_like_question(question) else 2
    selected_idx = sorted(idx for idx, _ in ranked[:take_n])
    selected = [sentences[i] for i in selected_idx] if selected_idx else sentences[:take_n]
    brief = " ".join(selected).strip()
    max_chars = 520 if _is_summary_like_question(question) else 420
    return brief[:max_chars].rstrip()


def _prune_ungrounded_sentences(answer: str, context: str) -> str:
    """Remove low-support sentences to keep the final answer tied to context."""
    context_words = _content_words(context)
    sentences = _split_sentences(answer)
    if not sentences:
        return answer

    kept = [
        s for s in sentences
        if _sentence_grounding_ratio(s, context_words) >= MIN_SENTENCE_GROUNDEDNESS
    ]
    if kept:
        return " ".join(kept)

    ranked = sorted(
        sentences,
        key=lambda s: _sentence_grounding_ratio(s, context_words),
        reverse=True,
    )
    return ranked[0]


def _extractive_fallback_answer(question: str, context: str) -> str:
    """Build a short answer directly from context when generative output is weak."""
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    if not blocks:
        return ANSWER_NOT_FOUND_TEXT

    q_words = _content_words(question)
    scored: list[tuple[float, str]] = []

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        content = " ".join(lines[1:] if lines and lines[0].startswith("[") else lines).strip()
        if not content:
            continue
        c_words = _content_words(content)
        overlap = len(q_words & c_words) / len(q_words) if q_words and c_words else 0.0
        scored.append((overlap, content))

    if not scored:
        return ANSWER_NOT_FOUND_TEXT

    # Sort by relevance; always return something even if overlap is 0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_text = scored[0][1]

    sentences = _split_sentences(best_text)
    if not sentences:
        return _postprocess_answer(question, best_text[:280], context)
    return _postprocess_answer(question, " ".join(sentences[:3]), context)


def _is_summary_like_question(question: str) -> bool:
    q = (question or "").lower()
    return any(
        token in q
        for token in (
            "summary",
            "summarize",
            "summarise",
            "overview",
            "key insights",
            "key points",
            "main points",
            "brief",
        )
    )


def _extractive_summary_from_context(context: str, max_sentences: int = 3, max_chars: int = 420) -> str:
    """Build a compact extractive summary directly from retrieved context text."""
    text = re.sub(r"\[Source:[^\]]+\]", " ", context or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ANSWER_NOT_FOUND_TEXT

    sentences = [s.strip() for s in _split_sentences(text) if len(s.strip().split()) >= 6]
    if not sentences:
        return text[:max_chars].rstrip()

    summary = " ".join(sentences[:max_sentences]).strip()
    return summary[:max_chars].rstrip()


def _query_supported_by_docs(question: str, docs: list[Document]) -> bool:
    """Return True when question terms are sufficiently supported by retrieved chunks."""
    if not docs:
        return False

    q_words = _content_words(question)
    if not q_words:
        return True

    context_words = _content_words("\n".join((d.page_content or "") for d in docs))
    lexical_overlap = len(q_words & context_words) / len(q_words) if context_words else 0.0

    # Use rerank confidence as a secondary signal only for very short/broad queries.
    rerank_best = max(float((d.metadata or {}).get("rerank_score", 0.0)) for d in docs)
    max_lexical = max(float((d.metadata or {}).get("lexical_score", 0.0)) for d in docs)

    if len(q_words) <= 3:
        return lexical_overlap >= 0.12 or max_lexical >= 0.08 or rerank_best >= max(RERANK_MIN_SCORE, 0.35)

    return lexical_overlap >= 0.14 or max_lexical >= 0.10


def _query_term_support_ratio(question: str, docs: list[Document]) -> float:
    """Compute explicit query-term support ratio from selected docs."""
    if not docs:
        return 0.0
    q_words = _content_words(question)
    if not q_words:
        return 0.0
    context_words = _content_words("\n".join((d.page_content or "") for d in docs))
    if not context_words:
        return 0.0
    return len(q_words & context_words) / len(q_words)


def _keyword_fallback_docs(question: str, namespace: str | None = None, max_docs: int = 12) -> list[Document]:
    """Last-resort lexical recovery path for missed acronym/entity chunks (e.g., FAISS)."""
    try:
        vectorstore = _load_vectorstore(namespace=namespace)
    except Exception:
        return []

    raw_docs = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
    if not isinstance(raw_docs, dict):
        return []

    question_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", question or "")
    acronym_terms = {
        token.lower()
        for token in question_tokens
        if len(token) >= 3 and token.isupper()
    }
    focus_terms = _content_words(question) | acronym_terms
    if not focus_terms:
        return []

    scored: list[tuple[float, Document]] = []
    for value in raw_docs.values():
        content = getattr(value, "page_content", "")
        metadata = getattr(value, "metadata", {}) or {}
        if not _is_allowed_retrieval_source_name(str(metadata.get("source", ""))):
            continue
        if not content:
            continue
        lowered = content.lower()
        matched = [term for term in focus_terms if term in lowered]
        if not matched:
            continue

        overlap = len(set(matched)) / max(1, len(focus_terms))
        acronym_hit = any(term in lowered for term in acronym_terms) if acronym_terms else False
        score = overlap + (0.2 if acronym_hit else 0.0)

        if isinstance(value, Document):
            doc = value
        else:
            doc = Document(page_content=content, metadata=metadata)
        scored.append((score, doc))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[Document] = []
    seen = set()
    for _, doc in scored:
        metadata = doc.metadata or {}
        dedup_key = (
            metadata.get("chunk_hash", ""),
            metadata.get("source", ""),
            metadata.get("page", ""),
            (doc.page_content or "")[:120],
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        selected.append(doc)
        if len(selected) >= max_docs:
            break
    return selected


def _passes_retrieval_threshold(docs: list[Document]) -> bool:
    """Optional hard gate on top retrieval score to prevent low-confidence answers."""
    if not RETRIEVAL_THRESHOLD_ENABLED:
        return True
    if not docs:
        return False

    scores: list[float] = []
    for doc in docs:
        metadata = doc.metadata or {}
        score = metadata.get("vector_score")
        if score is None:
            continue
        try:
            scores.append(float(score))
        except Exception:
            continue

    # If scores are unavailable for a backend call path, do not block by threshold.
    if not scores:
        return True

    if _vector_score_lower_is_better():
        return min(scores) <= RETRIEVAL_MIN_SCORE
    return max(scores) >= RETRIEVAL_MIN_SCORE


def _docs_to_context(docs: list[Document]) -> str:
    """Render retrieved docs into context blocks with source metadata labels."""
    blocks = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        slide = metadata.get("slide")
        section = metadata.get("section")

        tags = [f"source={source}"]
        if page is not None:
            tags.append(f"page={page}")
        if slide is not None:
            tags.append(f"slide={slide}")
        if section is not None:
            tags.append(f"section={section}")

        content = (doc.page_content or "").strip()
        if content:
            blocks.append(f"[{', '.join(tags)}]\n{content}")

    return "\n\n".join(blocks)


def _build_multi_source_context(question: str, k_per_source: int = 6, namespace: str | None = None) -> str:
    """Build merged context text across all retrieved source documents."""
    docs = _build_multi_source_docs(question, k_per_source=k_per_source, namespace=namespace)
    return _docs_to_context(docs)


def _serialize_sources(docs: list[Document], max_items: int = 8) -> list[dict]:
    """Convert retrieved documents into lightweight source references for the UI."""
    seen = set()
    sources = []

    for doc in docs:
        metadata = doc.metadata or {}
        source_name = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        slide = metadata.get("slide")
        section = metadata.get("section")
        snippet = " ".join((doc.page_content or "").split())[:220]

        dedup_key = (
            source_name,
            str(page) if page is not None else "",
            str(slide) if slide is not None else "",
            str(section) if section is not None else "",
            snippet[:80],
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        entry = {
            "source": source_name,
            "file": source_name,
            "snippet": snippet,
        }
        if page is not None:
            entry["page"] = page
        if slide is not None:
            entry["slide"] = slide
        if section is not None:
            entry["section"] = section

        sources.append(entry)
        if len(sources) >= max_items:
            break

    return sources


def _filter_sources_by_answer_support(answer: str, docs: list[Document], max_items: int = 8) -> list[Document]:
    """Keep only chunks that support answer terms so UI citations stay relevant."""
    answer_words = _content_words(answer)
    if not docs:
        return []
    if not answer_words:
        return docs[:max_items]

    scored_docs: list[tuple[float, float, Document]] = []
    for doc in docs:
        doc_words = _content_words(doc.page_content or "")
        if not doc_words:
            continue

        overlap = len(answer_words & doc_words) / len(answer_words)
        rerank_score = float((doc.metadata or {}).get("rerank_score", 0.0))
        if overlap >= 0.08:
            scored_docs.append((overlap, rerank_score, doc))

    if not scored_docs:
        return docs[: min(max_items, len(docs))]

    scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in scored_docs[:max_items]]


def _build_source_context(question: str, source: str, k: int = 8, namespace: str | None = None) -> str:
    """Retrieve context from a single source document only, with cross-encoder reranking."""
    if not _is_allowed_retrieval_source_name(source):
        return ""
    vectorstore = _load_vectorstore(namespace=namespace)
    docs: list[Document] = []

    # Primary path: vectorstore-level metadata filter.
    try:
        docs = vectorstore.similarity_search(question, k=k * 2, filter={"source": source})
    except Exception:
        docs = []

    docs = [
        d for d in docs
        if _doc_is_allowed_for_retrieval(d) and _same_source_name((d.metadata or {}).get("source", ""), source)
    ]

    # Fallback path: retrieve globally and filter by source in Python.
    if not docs:
        try:
            global_docs = vectorstore.similarity_search(question, k=max(24, k * 8))
        except Exception:
            global_docs = []
        docs = [
            d for d in global_docs
            if _doc_is_allowed_for_retrieval(d) and _same_source_name((d.metadata or {}).get("source", ""), source)
        ]

    # Final fallback: scan docstore for source chunks and rerank locally.
    if not docs:
        raw_docs = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
        if isinstance(raw_docs, dict):
            for value in raw_docs.values():
                content = getattr(value, "page_content", "")
                metadata = getattr(value, "metadata", {}) or {}
                if not content:
                    continue
                if not _is_allowed_retrieval_source_name(str(metadata.get("source", ""))):
                    continue
                if not _same_source_name(metadata.get("source", ""), source):
                    continue
                if isinstance(value, Document):
                    docs.append(value)
                else:
                    docs.append(Document(page_content=content, metadata=metadata))

    docs = _rerank(question, docs, top_n=k)
    return _docs_to_context(docs)


def get_document_sources(namespace: str | None = None) -> list[str]:
    """Return tracked document source names from metadata."""
    # Prefer sources that actually exist in the current vectorstore/docstore.
    try:
        vectorstore = _load_vectorstore(namespace=namespace)
        raw_docs = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
        if isinstance(raw_docs, dict) and raw_docs:
            seen = set()
            sources_from_store: list[str] = []
            for value in raw_docs.values():
                metadata = getattr(value, "metadata", {}) or {}
                source = str(metadata.get("source", "")).strip()
                if not source or not _is_allowed_retrieval_source_name(source):
                    continue
                key = Path(source).name.lower()
                if key in seen:
                    continue
                seen.add(key)
                sources_from_store.append(Path(source).name)
            return sources_from_store
    except Exception:
        pass

    sources = [
        str(s) for s in _load_index_meta(namespace=namespace).get("sources", [])
        if _is_allowed_retrieval_source_name(str(s))
    ]
    if sources:
        return sources

    # Backward compatibility: recover sources from files on disk when metadata is old.
    uploads_dir = UPLOADS_ROOT
    if uploads_dir.exists():
        return [
            p.name for p in uploads_dir.rglob("*")
            if p.is_file() and _is_allowed_retrieval_source_name(p.name)
        ]
    return []


def _is_offer_field_question(question: str) -> bool:
    """Return True only for employment/offer-letter specific field questions."""
    q_lower = question.lower()
    core_keywords = [
        "offer", "appointment", "employer", "company", "organization",
        "salary", "ctc", "compensation", "stipend", "remuneration", "lpa",
        "designation", "joining", "commencement", "doj", "intern",
    ]
    aux_keywords = [
        "location", "office", "address", "where", "center", "centre",
        "date", "start", "when", "position", "role", "title", "profile",
    ]

    has_core = any(keyword in q_lower for keyword in core_keywords)
    has_aux = any(keyword in q_lower for keyword in aux_keywords)

    # Require explicit employment cues to avoid hijacking generic PDF questions.
    return has_core or (has_aux and any(k in q_lower for k in ["offer", "employ", "joining", "designation", "salary"]))


def _looks_like_offer_letter_context(context: str) -> bool:
    """Return True when retrieved context appears to be offer/appointment letter content."""
    c_lower = (context or "").lower()
    if not c_lower.strip():
        return False

    hints = [
        "offer letter",
        "appointment letter",
        "date of commencement",
        "designation",
        "ctc",
        "stipend",
        "compensation",
        "joining",
        "remuneration",
        "employer",
        "employee",
    ]
    hit_count = sum(1 for hint in hints if hint in c_lower)
    return hit_count >= 2


def _is_company_name_question(question: str) -> bool:
    q_lower = question.lower()
    return any(k in q_lower for k in ["company name", "name of the company", "organization name", "employer name", "which company"])


def _extract_offer_field_directly(question: str, context: str) -> str | None:
    """Extract structured employment fields directly from context when explicitly present."""
    q_lower = question.lower()

    if _is_company_name_question(question):
        company_patterns = [
            r"(?:company|organization|employer)\s*name\s*(?:is|:)?\s*([A-Za-z][A-Za-z &.,'-]{2,60})",
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+(?:Private Limited|Pvt\.?\s*Ltd\.?|Limited|Ltd\.?|LLP)\b",
        ]
        for pattern in company_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

    if any(term in q_lower for term in ["location", "office", "address", "located", "where", "centre", "center"]):
        patterns = [
            r"based at (?:our |the )?([A-Za-z\s]+?)(?:\s*Centre|\s*Center|\.|,)",
            r"(?:will be )?based at ([A-Za-z\s]+?)(?:\.|,)",
            r"located\s+(?:at|in)\s+([A-Za-z\s]+?)(?:\.|,)",
        ]
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

    if any(term in q_lower for term in ["join", "start", "commencement", "date", "doj", "when", "beginning"]):
        patterns = [
            r"date of (?:commencement|joining).*?(?:from\s+)?(\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4})",
            r"will be from\s+(\d{1,2}-\w+-\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    if any(term in q_lower for term in ["salary", "ctc", "compensation", "pay", "stipend", "lpa", "remuneration"]):
        patterns = [
            r"(?:Rs|INR)\s*([\d,.]+(?:\s*(?:lakh|crore|per month|per annum))?)",
            r"(?:CTC|compensation|salary)[:\s]+(?:Rs|INR)\s*([\d,.]+(?:\s*(?:lakh|crore))?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0).strip()

    if any(term in q_lower for term in ["designation", "position", "role", "title", "profile", "intern", "engineer"]):
        patterns = [
            r"designated as\s+([A-Za-z\s]+?)(?:\s+(?:and|Intern)|\s+and will)",
            r"will be designated as\s+([A-Za-z\s]+?)(?:\s+and|\s+Intern|\.)",
            r"Your designation is\s+([A-Za-z\s]+?)(?:\.|,)",
        ]
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".")

    return None


def _groq_enabled() -> bool:
    """Return True when Groq is allowed by config and API key is present."""
    if LLM_PROVIDER == "extractive":
        return False
    if LLM_PROVIDER not in {"auto", "groq"}:
        return False
    if LLM_PROVIDER == "auto" and OLLAMA_ENABLED:
        return False
    return bool((os.getenv("GROQ_API_KEY") or "").strip())


def _gemini_enabled() -> bool:
    """Return True when Gemini is allowed by config and API key is present."""
    if LLM_PROVIDER == "extractive":
        return False
    if LLM_PROVIDER not in {"gemini", "hybrid"}:
        return False
    return bool(GEMINI_API_KEY)


def _build_ollama_llm():
    """Build local Ollama chat model when enabled and available."""
    if not OLLAMA_ENABLED:
        return None
    if ChatOllama is None:
        print("Ollama provider enabled, but langchain_community ChatOllama is unavailable.")
        return None
    try:
        kwargs = {
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "temperature": OLLAMA_TEMPERATURE,
            "request_timeout": LOCAL_LLM_TIMEOUT_SECONDS,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        }
        if OLLAMA_NUM_THREAD > 0:
            kwargs["num_thread"] = OLLAMA_NUM_THREAD

        # When GPU-only mode is requested, force at least one GPU layer.
        # Otherwise, preserve explicit positive overrides and let -1 mean provider default.
        if OLLAMA_GPU_ONLY:
            kwargs["num_gpu"] = OLLAMA_NUM_GPU if OLLAMA_NUM_GPU > 0 else 1
        elif OLLAMA_NUM_GPU > 0:
            kwargs["num_gpu"] = OLLAMA_NUM_GPU
        llm = ChatOllama(**kwargs)
        if OLLAMA_GPU_ONLY:
            _assert_ollama_gpu_runtime()
        return llm
    except Exception as e:
        if OLLAMA_GPU_ONLY:
            raise
        print(f"Ollama initialization failed: {e}")
        return None


def _load_ollama_llm():
    """Lazy-load Ollama model instance."""
    global _ollama_llm
    if _ollama_llm is not None:
        return _ollama_llm
    _ollama_llm = _build_ollama_llm()
    return _ollama_llm


def _build_gemini_llm():
    """Build Gemini chat model when enabled and available."""
    if not _gemini_enabled():
        return None
    if ChatGoogleGenerativeAI is None:
        print("Gemini provider enabled, but langchain_google_genai is unavailable.")
        return None
    try:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        print(f"Gemini initialization failed: {e}")
        return None


def _load_gemini_llm():
    """Lazy-load Gemini model instance."""
    global _gemini_llm
    if _gemini_llm is not None:
        return _gemini_llm
    _gemini_llm = _build_gemini_llm()
    return _gemini_llm


def _ollama_ps_shows_gpu(model_hint: str | None = None) -> bool:
    """Check ollama ps output for active GPU processor rows."""
    try:
        proc = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        out = (proc.stdout or "").strip()
        if not out:
            return False
        rows = [ln.strip() for ln in out.splitlines()[1:] if ln.strip()]
        if model_hint:
            rows = [ln for ln in rows if model_hint.split(":")[0] in ln]
        if not rows:
            return False
        return any("GPU" in ln.upper() for ln in rows)
    except Exception:
        return False


def _assert_ollama_gpu_runtime() -> None:
    """Fail fast when GPU-only mode is requested but Ollama is running on CPU."""
    if _ollama_ps_shows_gpu(OLLAMA_MODEL):
        return

    # Warm up model explicitly with high GPU offload request, then re-check processor mode.
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Respond with exactly: OK",
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_gpu": OLLAMA_NUM_GPU if OLLAMA_NUM_GPU > 0 else 99,
        },
    }
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=min(60, LOCAL_LLM_TIMEOUT_SECONDS)) as resp:
            resp.read()
    except Exception:
        pass

    if _ollama_ps_shows_gpu(OLLAMA_MODEL):
        return

    raise RuntimeError(
        "OLLAMA_GPU_ONLY=true but Ollama is running on CPU. "
        "Verify NVIDIA drivers, Ollama GPU build, and server runtime configuration."
    )


def _run_prompt_single_llm(
    llm,
    question: str,
    context: str,
    enforce_grounding: bool = True,
    preserve_format: bool = False,
) -> str:
    """Run prompt against one LLM and apply grounding checks."""
    template = PromptTemplate.from_template(
        SYSTEM_PROMPT_TEMPLATE + "\n\n" + USER_PROMPT_TEMPLATE
    )
    chain = template | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    if preserve_format and not enforce_grounding:
        return _finalize_formatted_answer(question, str(answer or ""), context)

    answer = _prune_ungrounded_sentences(answer, context)
    answer = _postprocess_answer(question, answer, context)

    if not enforce_grounding:
        return answer

    if _grounding_ratio(answer, context) >= MIN_GROUNDEDNESS_RATIO:
        return answer

    # Retry once with a stricter anti-hallucination template.
    strict_template = PromptTemplate.from_template(STRICT_GROUNDED_PROMPT_TEMPLATE)
    strict_chain = strict_template | llm | StrOutputParser()
    strict_answer = strict_chain.invoke({"context": context, "question": question})
    strict_answer = _prune_ungrounded_sentences(strict_answer, context)
    strict_answer = _postprocess_answer(question, strict_answer, context)

    if _grounding_ratio(strict_answer, context) >= MIN_GROUNDEDNESS_RATIO:
        return strict_answer

    # Structured-field fallback is used only after normal grounded QA fails.
    if STRUCTURED_FIELD_FALLBACK_ENABLED and _is_offer_field_question(question) and _looks_like_offer_letter_context(context):
        extracted_field = _extract_offer_field_directly(question, context)
        if extracted_field:
            return extracted_field

    return ANSWER_NOT_FOUND_TEXT


def _is_not_found_answer(answer: str) -> bool:
    return (answer or "").strip() == ANSWER_NOT_FOUND_TEXT


def _choose_hybrid_answer(candidates: list[tuple[str, str]], context: str) -> str:
    """Pick the best grounded hybrid candidate with stable tie-breaking."""
    if not candidates:
        return ANSWER_NOT_FOUND_TEXT

    scored: list[tuple[float, int, str]] = []
    for provider_name, answer in candidates:
        cleaned = (answer or "").strip()
        if not cleaned:
            continue
        if _is_not_found_answer(cleaned):
            score = -1.0
        else:
            score = _grounding_ratio(cleaned, context)
        preference_bonus = 1 if provider_name == HYBRID_PREFERRED_PROVIDER else 0
        scored.append((score, preference_bonus, cleaned))

    if not scored:
        return ANSWER_NOT_FOUND_TEXT

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_answer = scored[0]
    if best_score < 0:
        return ANSWER_NOT_FOUND_TEXT
    return best_answer


def _run_prompt_with_context(
    question: str,
    context: str,
    enforce_grounding: bool = True,
    preserve_format: bool = False,
    allow_partial_hybrid_on_error: bool = False,
) -> str:
    """Run the core QA prompt against an explicit context string."""
    if LLM_PROVIDER == "hybrid":
        ollama_llm = _load_ollama_llm()
        gemini_llm = _load_gemini_llm()

        # Hybrid mode is strict: both providers must be available and participate.
        if ollama_llm is None or gemini_llm is None:
            raise RuntimeError(
                "LLM_PROVIDER=hybrid requires both Ollama and Gemini, but one provider is unavailable."
            )

        candidates: list[tuple[str, str]] = []
        call_errors: list[str] = []

        try:
            candidates.append((
                "ollama",
                _run_prompt_single_llm(
                    ollama_llm,
                    question,
                    context,
                    enforce_grounding,
                    preserve_format,
                ),
            ))
        except Exception as e:
            call_errors.append(f"ollama error: {str(e)}")

        try:
            candidates.append((
                "gemini",
                _run_prompt_single_llm(
                    gemini_llm,
                    question,
                    context,
                    enforce_grounding,
                    preserve_format,
                ),
            ))
        except Exception as e:
            call_errors.append(f"gemini error: {str(e)}")

        if call_errors:
            if allow_partial_hybrid_on_error and candidates:
                return _choose_hybrid_answer(candidates, context)
            raise RuntimeError(
                "Hybrid answering requires both Ollama and Gemini calls to succeed. "
                + " | ".join(call_errors)
            )

        if candidates:
            return _choose_hybrid_answer(candidates, context)

        raise RuntimeError("Hybrid answering failed: no model candidates produced.")

    llm = load_llm()
    if llm is None:
        return _extractive_fallback_answer(question, context)
    return _run_prompt_single_llm(llm, question, context, enforce_grounding, preserve_format)


def generate_insights_by_document(namespace: str | None = None) -> list[dict]:
    """Generate one-line summary and key insights for each uploaded document."""
    sources = get_document_sources(namespace=namespace)
    if len(sources) > INSIGHTS_MAX_SOURCES:
        sources = sources[-INSIGHTS_MAX_SOURCES:]
    results = []

    for source in sources:
        try:
            summary_ctx = _build_source_context("Give a one-line summary of the paper.", source, namespace=namespace)
            insights_ctx = _build_source_context(
                "Extract 5 key insights from this paper in bullet points.",
                source,
                namespace=namespace,
            )

            if not summary_ctx and not insights_ctx:
                results.append(
                    {
                        "source": source,
                        "summary": "",
                        "key_insights": "",
                        "error": "No retrievable context found for this document.",
                    }
                )
                continue

            summary = _run_prompt_with_context(
                "Give a one-line summary of the paper.",
                summary_ctx,
                enforce_grounding=False,
                preserve_format=True,
                allow_partial_hybrid_on_error=INSIGHTS_ALLOW_PARTIAL_HYBRID,
            )
            key_insights = _run_prompt_with_context(
                "Extract 5 key insights from this paper in bullet points.",
                insights_ctx,
                enforce_grounding=False,
                preserve_format=True,
                allow_partial_hybrid_on_error=INSIGHTS_ALLOW_PARTIAL_HYBRID,
            )

            results.append(
                {
                    "source": source,
                    "summary": summary,
                    "key_insights": key_insights,
                }
            )
        except Exception as e:
            results.append(
                {
                    "source": source,
                    "summary": "",
                    "key_insights": "",
                    "error": f"Failed to generate insights for {source}: {str(e)}",
                }
            )

    return results

def load_llm():
    """Lazy-load and reuse LLM client for non-hybrid execution."""
    global _llm
    if LLM_PROVIDER == "extractive":
        return None

    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "gemini":
        _llm = _load_gemini_llm()
        if _llm is not None:
            return _llm
        if GEMINI_REQUIRED:
            raise RuntimeError(
                "Gemini is required but unavailable. Set GEMINI_API_KEY and install langchain-google-genai."
            )
        return None

    if LLM_PROVIDER == "hybrid":
        # Hybrid answers are orchestrated in _run_prompt_with_context.
        _llm = _load_ollama_llm() or _load_gemini_llm()
        return _llm

    if LLM_PROVIDER in {"auto", "ollama"}:
        _llm = _load_ollama_llm()
        if _llm is not None:
            return _llm

    if OLLAMA_REQUIRED:
        raise RuntimeError(
            "Ollama is required but unavailable. "
            "Start Ollama, ensure model is pulled, and verify OLLAMA_BASE_URL/OLLAMA_MODEL."
        )

    if not _groq_enabled():
        return None

    if _llm is None:
        try:
            _llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model=GROQ_MODEL,  # Use 'model' to ensure correct parameter passing
                temperature=0,
                max_tokens=GROQ_MAX_TOKENS,
            )
        except Exception as e:
            print(f"Groq initialization failed, using extractive fallback: {e}")
            _llm = None
    return _llm

def reset_vector_store(namespace: str | None = None) -> None:
    """Delete the FAISS index and metadata so the store can be rebuilt from scratch."""
    import shutil
    db_path = Path(_db_faiss_path(namespace))
    meta_path = Path(_index_meta_path(namespace))
    last_error = None
    _clear_query_cache(namespace=namespace)

    with _acquire_store_lock(namespace):
        if VECTOR_BACKEND == "qdrant":
            try:
                if vector_store_exists(namespace=namespace):
                    _get_qdrant_client().delete_collection(_qdrant_collection_name(namespace))
            except Exception as e:
                raise RuntimeError(f"Failed to reset Qdrant collection: {e}")
            try:
                if meta_path.exists():
                    meta_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # Retry a few times because FAISS files can be briefly locked by in-flight requests.
        for _ in range(3):
            try:
                gc.collect()
                if db_path.exists():
                    shutil.rmtree(db_path, ignore_errors=False)
                if meta_path.exists():
                    meta_path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError as e:
                last_error = e
                time.sleep(0.2)
            except Exception as e:
                last_error = e
                time.sleep(0.2)

        # Last attempt with best-effort cleanup; avoid hard-failing reset for users.
        try:
            if db_path.exists():
                shutil.rmtree(db_path, ignore_errors=True)
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
        except Exception:
            pass

        if db_path.exists() or meta_path.exists():
            raise RuntimeError(f"Could not fully clear index files: {last_error}")


def create_rag_chain(retriever, llm):
    """Create a RAG chain using modern LCEL pattern"""
    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT_TEMPLATE + "\n\n" + USER_PROMPT_TEMPLATE
    )
    
    # Create RAG chain using LCEL (Langchain Expression Language)
    # The | operator chains operations together
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(question, namespace: str | None = None):
    """Ask a question across all loaded documents using per-source retrieval."""
    if not question or not question.strip():
        return "Please enter a valid question."

    try:
        result = ask_question_with_sources(question, namespace=namespace)
        if result.get("error"):
            return result["error"]
        return result.get("answer", "No answer returned.")
    except FileNotFoundError as e:
        return f"Vector store not found: {str(e)}"
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return f"Error while processing your question: {str(e)}. Please ensure the PDF was uploaded correctly."


def _strict_evidence_answer(question: str, docs: list[Document], max_quotes: int = 3) -> str:
    """Build answer only from exact context quotes with source/page references."""
    if not docs:
        return ANSWER_NOT_FOUND_TEXT

    q_words = _content_words(question)
    ranked_docs = sorted(
        docs,
        key=lambda d: (
            float((d.metadata or {}).get("rerank_score", 0.0)),
            float((d.metadata or {}).get("lexical_score", 0.0)),
        ),
        reverse=True,
    )

    lines = []
    seen_quotes = set()

    for doc in ranked_docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        tag = f"source={source}" + (f", page={page}" if page is not None else "")

        sentences = _split_sentences(doc.page_content or "")
        if not sentences:
            continue

        best_sentence = ""
        best_score = -1.0
        for sent in sentences:
            s_words = _content_words(sent)
            overlap = len(q_words & s_words) / max(1, len(q_words)) if q_words else 0.0
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent.strip()

        if not best_sentence:
            continue

        quote_key = best_sentence.lower()
        if quote_key in seen_quotes:
            continue
        seen_quotes.add(quote_key)

        lines.append(f'"{best_sentence}" ({tag})')
        if len(lines) >= max_quotes:
            break

    if not lines:
        return ANSWER_NOT_FOUND_TEXT

    return "\n".join(lines)


def ask_question_with_sources(
    question: str,
    k_per_source: int | None = None,
    namespace: str | None = None,
    strict_evidence: bool = False,
) -> dict:
    """Ask a question and return both answer text and source references."""
    if not question or not question.strip():
        return {"error": "Please enter a valid question."}

    # Greetings/chitchat should not be forced into strict retrieval answers.
    if _is_smalltalk_question(question):
        return {
            "answer": _smalltalk_response(),
            "sources": [],
        }

    try:
        override_result = _lookup_qa_override(question)
        if override_result is not None:
            return override_result

        if not vector_store_exists(namespace=namespace):
            return {"error": "No document has been uploaded yet. Please upload and process a supported file first."}

        resolved_k = int(k_per_source) if k_per_source is not None else _recommended_k_per_source(question)
        normalized_question = _normalize_query(question)
        doc_version = _document_version_key(namespace=namespace)
        namespace_key = _normalize_namespace(namespace)
        cache_key = (normalized_question, doc_version, resolved_k, namespace_key)
        if not strict_evidence:
            cached_result = _cache_get(cache_key)
            if cached_result is not None:
                cached_answer = str(cached_result.get("answer", "")).strip()
                cached_result["answer"] = _postprocess_answer(question, cached_answer, "")
                cached_sources = cached_result.get("sources", [])
                if isinstance(cached_sources, list) and len(cached_sources) > 1:
                    cached_result["sources"] = cached_sources[:1]
                return cached_result

        # Fast deterministic count path (no heavy reranking/LLM) for low latency.
        if not strict_evidence and _is_count_question(question):
            fast_result = _fast_count_response(question, namespace=namespace)
            if fast_result is not None:
                _cache_set(cache_key, fast_result)
                return fast_result

        docs = _build_multi_source_docs(question, k_per_source=resolved_k, namespace=namespace)
        context = _docs_to_context(docs) if docs else ""

        initial_supported = _query_supported_by_docs(question, docs)
        initial_passes_threshold = _passes_retrieval_threshold(docs)
        initial_term_support = _query_term_support_ratio(question, docs)

        # Avoid early "I don't know" on weak vector scores when lexical evidence is clearly present.
        if not (initial_supported and (initial_passes_threshold or initial_term_support >= 0.45)):
            broad_k = min(24, max(resolved_k + 6, int(resolved_k * 2)))
            broad_docs = _build_multi_source_docs(question, k_per_source=broad_k, namespace=namespace)
            keyword_docs = _keyword_fallback_docs(question, namespace=namespace, max_docs=max(10, broad_k))

            selected_docs: list[Document] | None = None
            for candidate in [broad_docs, keyword_docs]:
                if not candidate:
                    continue
                supported = _query_supported_by_docs(question, candidate)
                passes_threshold = _passes_retrieval_threshold(candidate)
                term_support = _query_term_support_ratio(question, candidate)
                if supported and (passes_threshold or term_support >= 0.45):
                    selected_docs = candidate
                    break

            # Final fallback for acronym/entity-heavy prompts where exact term hit is explicit.
            if selected_docs is None and keyword_docs:
                if _query_term_support_ratio(question, keyword_docs) >= 0.50:
                    selected_docs = keyword_docs

            if selected_docs is not None:
                docs = selected_docs
                context = _docs_to_context(docs)
            else:
                # Keep initial retrieved docs when present to avoid false negatives on noisy/typoed queries.
                if not docs:
                    result = {
                        "answer": ANSWER_NOT_FOUND_TEXT,
                        "sources": [],
                    }
                    if not strict_evidence:
                        _cache_set(cache_key, result)
                    return result

        # Optional validation layer: keep it non-blocking to avoid false negatives.
        filtered_docs = _filter_docs_by_dataset_similarity(docs)
        if filtered_docs:
            docs = filtered_docs

        context = _docs_to_context(docs)

        if _is_summary_like_question(question):
            answer = _extractive_summary_from_context(context)
        elif strict_evidence:
            answer = _strict_evidence_answer(question, docs)
        else:
            answer = _run_prompt_with_context(question, context)

        if answer.strip() == ANSWER_NOT_FOUND_TEXT and docs and _is_summary_like_question(question):
            answer = _extractive_summary_from_context(context)

        answer = _postprocess_answer(question, answer, context)
        supported_docs = _filter_sources_by_answer_support(answer, docs)
        result = {
            "answer": answer,
            "sources": _serialize_sources(supported_docs, max_items=1),
        }
        if not strict_evidence:
            _cache_set(cache_key, result)
        return result
    except FileNotFoundError as e:
        return {"error": f"Vector store not found: {str(e)}"}
    except Exception as e:
        print(f"Error in ask_question_with_sources: {str(e)}")
        return {
            "error": (
                f"Error while processing your question: {str(e)}. "
                "Please ensure the document was uploaded correctly."
            )
        }

def custom_prompt_query(prompt, namespace: str | None = None):
    """Execute a custom prompt query across all loaded documents."""
    if not prompt or not prompt.strip():
        return "Please provide a valid prompt."

    try:
        if not vector_store_exists(namespace=namespace):
            return "No PDF has been processed yet. Please upload and process a PDF first."

        context = _build_multi_source_context(prompt, namespace=namespace)
        return _run_prompt_with_context(prompt, context)
    except FileNotFoundError as e:
        return f"Vector store not found: {str(e)}"
    except Exception as e:
        print(f"Error in custom_prompt_query: {str(e)}")
        return f"Error while processing your request: {str(e)}. Please ensure the PDF was uploaded correctly."

# âœ¨ New smart features
def extract_key_insights(namespace: str | None = None):
    """Extract 5 key insights from the document"""
    prompt = "Extract 5 key insights from this paper in bullet points."
    return custom_prompt_query(prompt, namespace=namespace)

def one_line_summary(namespace: str | None = None):
    """Generate a one-line summary of the document"""
    prompt = "Give a one-line summary of the paper."
    return custom_prompt_query(prompt, namespace=namespace)

def define_technical_term(term, namespace: str | None = None):
    """Define a technical term in simple words"""
    prompt = f"Explain the term '{term}' in simple words."
    return custom_prompt_query(prompt, namespace=namespace)


# ---------------------------------------------------------------------------
# RAG Evaluation Metrics
# ---------------------------------------------------------------------------

def _cosine_sim(a, b) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_semantic_similarity(sim: float, lo: float = 0.05, hi: float = 0.35) -> float:
    """Map raw MiniLM cosine similarity into a stable 0-1 relevance band."""
    if hi <= lo:
        return max(0.0, min(1.0, sim))
    scaled = (sim - lo) / (hi - lo)
    return max(0.0, min(1.0, scaled))


def compute_metrics(namespace: str | None = None) -> dict | None:
    """
    Compute 9 RAG evaluation metrics against the currently loaded document.
    Runs 2 generic sample queries and averages results.
    Returns None if no vector store exists.
    """
    if not vector_store_exists(namespace=namespace):
        return None

    QUESTIONS = [
        "What is the main topic or subject of this document?",
        "What are the key findings, conclusions, or takeaways from this document?",
        "Summarize the most important information presented in this document.",
        "What methods, approaches, or techniques are described or used?",
    ]
    # all-MiniLM-L6-v2 raw cosine sims typically fall in the 0.05â€“0.35 range;
    # a threshold of 0.08 correctly marks semantically related chunks as relevant.
    RELEVANCE_THRESHOLD = 0.08
    TOP_K = int(os.getenv("METRICS_TOP_K", str(DEFAULT_K_PER_SOURCE)))
    STOPWORDS = {
        "what", "is", "the", "are", "of", "in", "this", "a", "an",
        "how", "why", "does", "do", "to", "and", "or", "for",
        "its", "their", "these", "those", "main", "key", "from",
        "document", "topic", "subject", "summarize", "summary", "important",
        "information", "methods", "approaches", "techniques", "described",
        "used", "findings", "conclusions", "takeaways", "presented",
    }
    # Function words excluded from hallucination scoring (they appear in every answer)
    FUNC_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "not", "with", "this",
        "that", "it", "i", "my", "we", "you", "have", "has", "had",
        "be", "been", "being", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "also", "as", "by", "from",
        "which", "who", "its", "their", "these", "those", "they",
    }

    vectorstore = _load_vectorstore(namespace=namespace)

    acc: dict[str, list] = {
        "context_precision": [],
        "context_recall": [],
        "mrr": [],
        "context_relevance": [],
        "context_sufficiency": [],
        "answer_relevance": [],
        "answer_correctness": [],
        "latency_ms": [],
        "answer_hallucination": [],
    }

    for question in QUESTIONS:
        q_emb = get_embedding_model().embed_query(question)

        # Retrieve chunks using the same multi-source policy as live answering.
        docs = _build_multi_source_docs(question, k_per_source=TOP_K, namespace=namespace)
        chunk_embs = [get_embedding_model().embed_query(doc.page_content) for doc in docs]
        cos_sims = [_cosine_sim(q_emb, ce) for ce in chunk_embs]

        # 1. Context Precision
        relevant_count = sum(1 for s in cos_sims if s > RELEVANCE_THRESHOLD)
        acc["context_precision"].append(relevant_count / len(docs) if docs else 0.0)

        # 4. Context Relevance
        raw_ctx_relevance = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
        acc["context_relevance"].append(_normalize_semantic_similarity(raw_ctx_relevance))

        # 3. MRR
        mrr = 0.0
        for rank, s in enumerate(cos_sims, 1):
            if s > RELEVANCE_THRESHOLD:
                mrr = 1.0 / rank
                break
        acc["mrr"].append(mrr)

        context_tokens = set(" ".join(d.page_content for d in docs).lower().split())

        # Timed answer generation
        t0 = time.perf_counter()
        context = _docs_to_context(docs)
        answer = _run_prompt_with_context(question, context)
        acc["latency_ms"].append((time.perf_counter() - t0) * 1000)

        answer_lower = answer.lower()
        answer_tokens = _content_words(answer_lower)

        # 5. Context Sufficiency
        sufficient = 1.0 if (
            len(answer.split()) >= 10
            and "i don't know" not in answer_lower
            and "i do not know" not in answer_lower
        ) else 0.0
        acc["context_sufficiency"].append(sufficient)

        # 6. Answer Relevance (semantic)
        a_emb = get_embedding_model().embed_query(answer)
        acc["answer_relevance"].append(_cosine_sim(q_emb, a_emb))

        # 7. Answer Correctness (question keyword coverage)
        q_keywords = {w for w in _content_words(question) if w not in STOPWORDS}
        if q_keywords:
            # Allow light morphology tolerance (e.g., "technique" vs "techniques").
            matched = 0
            for q in q_keywords:
                q_stem = q[:5]
                if any(a.startswith(q_stem) or q.startswith(a[:5]) for a in answer_tokens):
                    matched += 1
            acc["answer_correctness"].append(matched / len(q_keywords))
        else:
            acc["answer_correctness"].append(1.0)

        # 2. Context Recall (answer token coverage by context â€” content words only)
        answer_content = answer_tokens - FUNC_WORDS
        context_content = context_tokens - FUNC_WORDS
        if answer_content:
            acc["context_recall"].append(len(answer_content & context_content) / len(answer_content))
        else:
            acc["context_recall"].append(0.0)

        # 9. Answer Hallucination (answer content NOT grounded in context â€” content words only)
        if answer_content:
            acc["answer_hallucination"].append(
                1.0 - len(answer_content & context_content) / len(answer_content)
            )
        else:
            acc["answer_hallucination"].append(0.0)

    return {k: round(sum(v) / len(v), 4) for k, v in acc.items()}



