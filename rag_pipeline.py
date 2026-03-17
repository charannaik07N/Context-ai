import os
import time
import json
import hashlib
import gc
import copy
import logging
import re
import numpy as np
from threading import Lock
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ✅ Centralized Configuration (Single source of truth)
# Use absolute paths to fix "paths mismatch" errors
BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
INDEX_META_PATH = os.path.join(BASE_DIR, "vector_store_meta.json")
# Read model name from env so it is configurable without code changes
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Keep runtime logs clean by muting non-actionable model-load chatter.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

_embedding_model = None
_reranker = None
_llm = None

QUERY_CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "180"))
QUERY_CACHE_MAX_ENTRIES = int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "200"))
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "24"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.25"))
MIN_GROUNDEDNESS_RATIO = float(os.getenv("MIN_GROUNDEDNESS_RATIO", "0.70"))
MIN_SENTENCE_GROUNDEDNESS = float(os.getenv("MIN_SENTENCE_GROUNDEDNESS", "0.35"))
_query_result_cache: dict[tuple[str, str, int], tuple[float, dict]] = {}
_query_cache_lock = Lock()

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


def _document_version_key() -> str:
    """Create a stable version key that changes whenever indexed documents change."""
    meta = _load_index_meta()
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


def _cache_get(cache_key: tuple[str, str, int]) -> dict | None:
    """Read from cache if entry is still valid."""
    now_ts = time.time()
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


def _cache_set(cache_key: tuple[str, str, int], payload: dict) -> None:
    """Store successful query result in cache for short TTL."""
    now_ts = time.time()
    with _query_cache_lock:
        _prune_query_cache(now_ts)
        _query_result_cache[cache_key] = (
            now_ts + max(1, QUERY_CACHE_TTL_SECONDS),
            copy.deepcopy(payload),
        )


def _clear_query_cache() -> None:
    """Clear all cached query results."""
    with _query_cache_lock:
        _query_result_cache.clear()


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


def _choose_chunking_strategy(file_path: str, docs: list[Document]) -> tuple[int, int]:
    """
    Pick chunk size/overlap adaptively based on document type, page density,
    and heading-like structure density.
    """
    suffix = Path(file_path).suffix.lower()
    content = [d.page_content or "" for d in docs]
    total_chars = sum(len(c) for c in content)
    units = max(1, len(content))
    avg_chars_per_unit = total_chars / units

    merged_text = "\n".join(content[:40])
    heading_ratio = _heading_like_ratio(merged_text)

    # PDF defaults tuned by density and heading signals.
    if avg_chars_per_unit >= 4500 and heading_ratio < 0.06:
        return 1100, 220
    if heading_ratio >= 0.12:
        return 650, 120
    if avg_chars_per_unit <= 1700:
        return 560, 110
    return 820, 150


def get_embedding_model():
    """Lazy-load embeddings model so startup/reset does not incur model load time."""
    global _embedding_model
    if _embedding_model is None:
        model_kwargs = {}
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN

        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
        )
    return _embedding_model


def get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder reranker so model weights are only fetched once."""
    global _reranker
    if _reranker is None:
        reranker_kwargs = {"max_length": 512}
        if HF_TOKEN:
            reranker_kwargs["token"] = HF_TOKEN

        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", **reranker_kwargs
        )
    return _reranker


def _rerank_with_scores(question: str, docs: list) -> list[tuple[float, Document]]:
    """Re-score candidate docs with the cross-encoder and keep scores."""
    if not docs:
        return []
    try:
        reranker = get_reranker()
        pairs = [(question, doc.page_content) for doc in docs]
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


def vector_store_exists():
    """Check if the FAISS vector store exists on disk"""
    db_path = Path(DB_FAISS_PATH)
    return db_path.exists() and (db_path / "index.faiss").exists() and (db_path / "index.pkl").exists()


def _file_hash(path: str) -> str:
    """Return SHA-256 hex digest of the raw file bytes for file-level dedup."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_index_meta() -> dict:
    """Load persisted ingestion metadata used for deduplication."""
    meta_path = Path(INDEX_META_PATH)
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


def _save_index_meta(meta: dict) -> None:
    """Persist ingestion metadata used for deduplication."""
    Path(INDEX_META_PATH).write_text(
        json.dumps(meta, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _stable_chunk_hash(text: str) -> str:
    """Hash normalized chunk text so duplicate chunks can be skipped across uploads."""
    normalized = " ".join((text or "").split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_documents(file_path: str) -> list[Document]:
    """Load PDF content into LangChain Document objects."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()

    raise ValueError("Unsupported file type. Allowed: .pdf")


def append_document_to_vector_store(
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int = 32,
) -> dict:
    """
    Append a PDF into the existing FAISS index with two-layer deduplication:
      1. File hash  — identical file → skip loading/parsing/embedding entirely.
      2. Chunk hash — changed/new file → only embed chunks not already in the index.
    Returns ingestion stats.
    """
    source_name = Path(file_path).name
    meta = _load_index_meta()

    # ── Layer 1: file-level hash check ────────────────────────────────────────
    incoming_file_hash = _file_hash(file_path)
    known_file_hashes: dict = meta.get("file_hashes", {})
    if known_file_hashes.get(source_name) == incoming_file_hash:
        # Byte-for-byte identical file already indexed — skip everything.
        return {
            "source": source_name,
            "total_chunks": 0,
            "added_chunks": 0,
            "duplicate_chunks": 0,
            "skipped": True,
            "reason": "Identical file already indexed (file hash match).",
        }

    # ── Layer 2: load, split, chunk-hash dedup ────────────────────────────────
    pages = _load_documents(file_path)
    if not pages:
        raise ValueError("Document is empty or could not be read.")

    adaptive_chunk_size, adaptive_chunk_overlap = _choose_chunking_strategy(file_path, pages)
    final_chunk_size = chunk_size if chunk_size is not None else adaptive_chunk_size
    final_chunk_overlap = chunk_overlap if chunk_overlap is not None else adaptive_chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
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
        # All chunks already present — still update file hash so next upload of
        # the same file is short-circuited at layer 1.
        known_file_hashes[source_name] = incoming_file_hash
        meta["file_hashes"] = known_file_hashes
        _save_index_meta(meta)
        _clear_query_cache()
        return {
            "source": source_name,
            "total_chunks": total_chunks,
            "added_chunks": 0,
            "duplicate_chunks": duplicate_chunks,
        }

    vectorstore = None
    if vector_store_exists():
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )

    for i in range(0, len(docs_to_add), batch_size):
        batch_docs = docs_to_add[i:i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch_docs, get_embedding_model())
        else:
            vectorstore.add_documents(batch_docs)

    if vectorstore is None:
        raise RuntimeError("Failed to create or update vector store.")

    vectorstore.save_local(DB_FAISS_PATH)

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
    })
    _clear_query_cache()

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

def process_pdf(pdf_path):
    """Backward-compatible ingest API: append to index with deduplication."""
    try:
        append_document_to_vector_store(pdf_path)
        return True
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise


def append_pdf_to_vector_store(pdf_path: str, chunk_size: int | None = None, chunk_overlap: int | None = None, batch_size: int = 32) -> dict:
    """Backward-compatible alias; now supports non-PDF docs via generic implementation."""
    return append_document_to_vector_store(pdf_path, chunk_size, chunk_overlap, batch_size)

def load_retriever():
    """Load the FAISS vector store and return as retriever with safety checks"""
    try:
        # CRITICAL: Check if vector store exists before attempting to load
        if not vector_store_exists():
            raise FileNotFoundError(
                "No PDF has been processed yet. Please upload and process a PDF first."
            )
        
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            get_embedding_model(), 
            allow_dangerous_deserialization=True
        )
        # Use higher k so chunks from multiple documents all surface
        return vectorstore.as_retriever(search_kwargs={"k": 10})
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        print(f"Error loading retriever: {str(e)}")
        raise


def _build_multi_source_docs(question: str, k_per_source: int = 6) -> list[Document]:
    """
    Retrieve top-k chunks per source document so every document is guaranteed
    representation in the context, regardless of overall similarity ranking.
    Falls back to a global search when only one source is tracked.
    """
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
    sources = _load_index_meta().get("sources", [])

    seen_best: dict[str, tuple[float, Document]] = {}
    global_k = max(16, min(MAX_RERANK_CANDIDATES * 4, 64))

    # First pass: global candidate retrieval prioritizes pure relevance.
    global_pairs = vectorstore.similarity_search_with_score(question, k=global_k)
    for doc, score in global_pairs:
        uid = doc.metadata.get("chunk_hash", doc.page_content[:80])
        best = seen_best.get(uid)
        if best is None or score < best[0]:
            seen_best[uid] = (score, doc)

    # Second pass: optional light per-source fallback so smaller docs are not starved.
    if len(sources) > 1:
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
                if best is None or score < best[0]:
                    seen_best[uid] = (score, doc)

    candidates = [doc for _, doc in sorted(seen_best.values(), key=lambda x: x[0])[:MAX_RERANK_CANDIDATES]]
    ranked = _rerank_with_scores(question, candidates)

    if not ranked:
        return []

    max_docs = min(8, max(3, k_per_source))
    kept = [doc for score, doc in ranked if score >= RERANK_MIN_SCORE][:max_docs]

    # Ensure a non-empty context even for difficult queries without forcing many weak chunks.
    if not kept:
        kept = [doc for _, doc in ranked[: min(2, len(ranked))]]

    for score, doc in ranked:
        doc.metadata["rerank_score"] = float(score)

    return kept


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
    fallback = (text or "").strip()
    return [fallback] if fallback else []


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
        return "I don't know based on the provided context."

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
        return "I don't know based on the provided context."

    # Sort by relevance; always return something even if overlap is 0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_text = scored[0][1]

    sentences = _split_sentences(best_text)
    if not sentences:
        return best_text[:280]
    return " ".join(sentences[:3])


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


def _build_multi_source_context(question: str, k_per_source: int = 6) -> str:
    """Build merged context text across all retrieved source documents."""
    docs = _build_multi_source_docs(question, k_per_source=k_per_source)
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


def _build_source_context(question: str, source: str, k: int = 8) -> str:
    """Retrieve context from a single source document only, with cross-encoder reranking."""
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
    # Fetch extra candidates so the reranker has room to reorder
    docs = vectorstore.similarity_search(question, k=k * 2, filter={"source": source})
    docs = _rerank(question, docs, top_n=k)
    return _docs_to_context(docs)


def get_document_sources() -> list[str]:
    """Return tracked document source names from metadata."""
    sources = _load_index_meta().get("sources", [])
    if sources:
        return sources

    # Backward compatibility: recover sources from files on disk when metadata is old.
    uploads_dir = BASE_DIR / "uploaded_docs"
    if uploads_dir.exists():
        return [p.name for p in uploads_dir.glob("*.pdf")]
    return []


def _run_prompt_with_context(question: str, context: str, enforce_grounding: bool = True) -> str:
    """Run the core QA prompt against an explicit context string."""
    llm = load_llm()
    template = PromptTemplate.from_template(
        """You have access to content from multiple documents. \
Answer using ONLY the context below. \
Do not use outside knowledge or guess. \
If the answer spans multiple documents, synthesize strictly from the provided context. \
If the answer is not in the context, reply exactly: I don't know based on the provided context.

Write a concise answer (3-6 sentences) that directly addresses the user's question terms.
Use wording from the context where possible, and avoid introducing new claims or entities.
Only include claims that can be supported by the context.
Do not add prefatory text.

Context:
{context}

Question:
{question}

Answer:"""
    )
    chain = template | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    answer = _prune_ungrounded_sentences(answer, context)

    if not enforce_grounding:
        return answer

    if _grounding_ratio(answer, context) >= MIN_GROUNDEDNESS_RATIO:
        return answer

    # Retry once with stricter extractive constraints.
    strict_template = PromptTemplate.from_template(
        """You are a strict grounded answerer.
Use ONLY facts explicitly stated in the context.
Do not infer, generalize, or add unstated information.
If context is insufficient, reply exactly: I don't know based on the provided context.

Requirements:
- 2-5 sentences.
- Reuse exact terms from the question.
- Keep wording close to context language.
- No speculation.

Context:
{context}

Question:
{question}

Answer:"""
    )
    strict_chain = strict_template | llm | StrOutputParser()
    strict_answer = strict_chain.invoke({"context": context, "question": question})
    strict_answer = _prune_ungrounded_sentences(strict_answer, context)

    if _grounding_ratio(strict_answer, context) >= MIN_GROUNDEDNESS_RATIO:
        return strict_answer

    return _extractive_fallback_answer(question, context)


def generate_insights_by_document() -> list[dict]:
    """Generate one-line summary and key insights for each uploaded document."""
    sources = get_document_sources()
    results = []

    for source in sources:
        try:
            summary_ctx = _build_source_context("Give a one-line summary of the paper.", source)
            insights_ctx = _build_source_context(
                "Extract 5 key insights from this paper in bullet points.",
                source,
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
            )
            key_insights = _run_prompt_with_context(
                "Extract 5 key insights from this paper in bullet points.",
                insights_ctx,
                enforce_grounding=False,
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
    """Lazy-load and reuse Groq client to reduce per-request latency."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=GROQ_MODEL,  # Use 'model' to ensure correct parameter passing
            temperature=0,
        )
    return _llm

def reset_vector_store() -> None:
    """Delete the FAISS index and metadata so the store can be rebuilt from scratch."""
    import shutil
    db_path = Path(DB_FAISS_PATH)
    meta_path = Path(INDEX_META_PATH)
    last_error = None
    _clear_query_cache()

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
        """You have access to content from multiple documents. \
Answer using ONLY the context below. \
Do not use outside knowledge or guess. \
If the answer spans multiple documents, synthesize strictly from the provided context. \
If the answer is not in the context, reply exactly: I don't know based on the provided context.

Write a concise answer (3-6 sentences) that directly addresses the user's question terms.
Only include claims that can be supported by the context.
Do not add prefatory text.

Context:
{context}

Question:
{question}

Answer:"""
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

def ask_question(question):
    """Ask a question across all loaded documents using per-source retrieval."""
    if not question or not question.strip():
        return "Please enter a valid question."

    try:
        result = ask_question_with_sources(question)
        if result.get("error"):
            return result["error"]
        return result.get("answer", "No answer returned.")
    except FileNotFoundError as e:
        return f"Vector store not found: {str(e)}"
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return f"Error while processing your question: {str(e)}. Please ensure the PDF was uploaded correctly."


def ask_question_with_sources(question: str, k_per_source: int = 6) -> dict:
    """Ask a question and return both answer text and source references."""
    if not question or not question.strip():
        return {"error": "Please enter a valid question."}

    try:
        if not vector_store_exists():
            return {"error": "No document has been uploaded yet. Please upload and process a supported file first."}

        normalized_question = _normalize_query(question)
        doc_version = _document_version_key()
        cache_key = (normalized_question, doc_version, int(k_per_source))
        cached_result = _cache_get(cache_key)
        if cached_result is not None:
            cached_sources = cached_result.get("sources", [])
            if isinstance(cached_sources, list) and len(cached_sources) > 1:
                cached_result["sources"] = cached_sources[:1]
            return cached_result

        docs = _build_multi_source_docs(question, k_per_source=k_per_source)
        context = _docs_to_context(docs)
        answer = _run_prompt_with_context(question, context)
        supported_docs = _filter_sources_by_answer_support(answer, docs)
        result = {
            "answer": answer,
            "sources": _serialize_sources(supported_docs, max_items=1),
        }
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

def custom_prompt_query(prompt):
    """Execute a custom prompt query across all loaded documents."""
    if not prompt or not prompt.strip():
        return "Please provide a valid prompt."

    try:
        if not vector_store_exists():
            return "No PDF has been processed yet. Please upload and process a PDF first."

        context = _build_multi_source_context(prompt)
        return _run_prompt_with_context(prompt, context)
    except FileNotFoundError as e:
        return f"Vector store not found: {str(e)}"
    except Exception as e:
        print(f"Error in custom_prompt_query: {str(e)}")
        return f"Error while processing your request: {str(e)}. Please ensure the PDF was uploaded correctly."

# ✨ New smart features
def extract_key_insights():
    """Extract 5 key insights from the document"""
    prompt = "Extract 5 key insights from this paper in bullet points."
    return custom_prompt_query(prompt)

def one_line_summary():
    """Generate a one-line summary of the document"""
    prompt = "Give a one-line summary of the paper."
    return custom_prompt_query(prompt)

def define_technical_term(term):
    """Define a technical term in simple words"""
    prompt = f"Explain the term '{term}' in simple words."
    return custom_prompt_query(prompt)


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


def compute_metrics() -> dict | None:
    """
    Compute 9 RAG evaluation metrics against the currently loaded document.
    Runs 2 generic sample queries and averages results.
    Returns None if no vector store exists.
    """
    if not vector_store_exists():
        return None

    QUESTIONS = [
        "What is the main topic or subject of this document?",
        "What are the key findings, conclusions, or takeaways from this document?",
        "Summarize the most important information presented in this document.",
        "What methods, approaches, or techniques are described or used?",
    ]
    # all-MiniLM-L6-v2 raw cosine sims typically fall in the 0.05–0.35 range;
    # a threshold of 0.08 correctly marks semantically related chunks as relevant.
    RELEVANCE_THRESHOLD = 0.08
    TOP_K = 6  # aligned with ask_question_with_sources default
    STOPWORDS = {
        "what", "is", "the", "are", "of", "in", "this", "a", "an",
        "how", "why", "does", "do", "to", "and", "or", "for",
        "its", "their", "these", "those", "main", "key", "from",
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

    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, get_embedding_model(), allow_dangerous_deserialization=True
    )

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
        docs = _build_multi_source_docs(question, k_per_source=TOP_K)
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
        answer_tokens = set(answer_lower.split())

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
        q_keywords = set(question.lower().split()) - STOPWORDS
        if q_keywords:
            acc["answer_correctness"].append(len(q_keywords & answer_tokens) / len(q_keywords))
        else:
            acc["answer_correctness"].append(1.0)

        # 2. Context Recall (answer token coverage by context — content words only)
        answer_content = answer_tokens - FUNC_WORDS
        context_content = context_tokens - FUNC_WORDS
        if answer_content:
            acc["context_recall"].append(len(answer_content & context_content) / len(answer_content))
        else:
            acc["context_recall"].append(0.0)

        # 9. Answer Hallucination (answer content NOT grounded in context — content words only)
        if answer_content:
            acc["answer_hallucination"].append(
                1.0 - len(answer_content & context_content) / len(answer_content)
            )
        else:
            acc["answer_hallucination"].append(0.0)

    return {k: round(sum(v) / len(v), 4) for k, v in acc.items()}
