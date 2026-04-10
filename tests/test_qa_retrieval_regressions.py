import importlib


class _Doc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class _VectorStoreStub:
    def __init__(self, docs_by_query, docstore_docs=None):
        self._docs_by_query = docs_by_query
        self.docstore = type("DocStore", (), {"_dict": docstore_docs or {}})()

    def similarity_search(self, question, k=6, filter=None):
        q = (question or "").lower()
        for key, docs in self._docs_by_query.items():
            if key in q:
                return docs[:k]
        return self._docs_by_query.get("default", [])[:k]


def test_score_direction_higher_is_better_for_similarity(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    monkeypatch.setattr(rp, "VECTOR_SCORE_DIRECTION", "similarity")

    assert rp._score_is_better(0.9, 0.5) is True
    assert rp._score_is_better(0.2, 0.5) is False


def test_score_direction_lower_is_better_for_distance(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    monkeypatch.setattr(rp, "VECTOR_SCORE_DIRECTION", "distance")

    assert rp._score_is_better(0.1, 0.4) is True
    assert rp._score_is_better(0.8, 0.4) is False


def test_ask_question_uses_broad_fallback_when_initial_docs_weak(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    monkeypatch.setattr(rp, "vector_store_exists", lambda namespace=None: True)
    monkeypatch.setattr(rp, "_recommended_k_per_source", lambda question: 6)
    monkeypatch.setattr(rp, "_cache_get", lambda cache_key: None)
    monkeypatch.setattr(rp, "_cache_set", lambda cache_key, payload: None)

    calls = {"count": 0}

    weak_docs = [_Doc("completely unrelated text", {"source": "doc-weak", "chunk_hash": "w1"})]
    strong_docs = [_Doc("authentication uses oauth tokens", {"source": "doc-strong", "chunk_hash": "s1"})]

    def _build_docs(question, k_per_source=6, namespace=None):
        calls["count"] += 1
        return weak_docs if calls["count"] == 1 else strong_docs

    monkeypatch.setattr(rp, "_build_multi_source_docs", _build_docs)
    monkeypatch.setattr(rp, "_run_prompt_with_context", lambda question, context, enforce_grounding=True: "Authentication uses OAuth tokens.")

    result = rp.ask_question_with_sources("How is authentication handled?", namespace="team-a")

    assert calls["count"] >= 2
    assert "OAuth" in result["answer"] or "oauth" in result["answer"].lower()
    assert isinstance(result.get("sources"), list)


def test_ask_question_returns_unknown_when_both_passes_weak(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    monkeypatch.setattr(rp, "vector_store_exists", lambda namespace=None: True)
    monkeypatch.setattr(rp, "_recommended_k_per_source", lambda question: 6)
    monkeypatch.setattr(rp, "_cache_get", lambda cache_key: None)
    monkeypatch.setattr(rp, "_cache_set", lambda cache_key, payload: None)

    weak_docs = [_Doc("unrelated content only", {"source": "doc-weak", "chunk_hash": "w2"})]

    monkeypatch.setattr(rp, "_build_multi_source_docs", lambda question, k_per_source=6, namespace=None: weak_docs)

    result = rp.ask_question_with_sources("What is the payment retry policy?", namespace="team-a")

    assert result["answer"] == "I don't know based on the provided documents."
    assert result["sources"] == []


def test_is_count_question_handles_noisy_prompt(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    assert rp._is_count_question("GIVE ME HOW MANY limitation are the give me count") is True


def test_extract_count_value_parses_number_words(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    result = rp._extract_count_value(
        "There are five limitations in this document.",
        "",
    )
    assert result == "5"


def test_fast_count_response_uses_fallback_queries_and_docstore(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    limitation_docs = [
        _Doc(
            "Performance limitations:\n1) Rebuilding index on each upload.\n2) Fixed chunk size.\n3) No reranker.\n4) CPU-only embeddings.\n5) No deduplication.",
            {"source": "limits_test.txt", "page": 0},
        )
    ]
    unrelated_docs = [_Doc("Unrelated architecture notes.", {"source": "other.txt", "page": 0})]

    vectorstore = _VectorStoreStub(
        docs_by_query={
            "default": unrelated_docs,
            "limitation": limitation_docs,
        },
        docstore_docs={
            "a": limitation_docs[0],
            "b": unrelated_docs[0],
        },
    )

    monkeypatch.setattr(rp, "_load_vectorstore", lambda namespace=None: vectorstore)

    response = rp._fast_count_response("GIVE ME HOW MANY limitation are the give me count", namespace="team-a")

    assert response is not None
    assert "5" in response["answer"]
    assert "limitations" in response["answer"].lower()


def test_ask_question_recovers_from_threshold_failure_via_keyword_fallback(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    monkeypatch.setattr(rp, "vector_store_exists", lambda namespace=None: True)
    monkeypatch.setattr(rp, "_recommended_k_per_source", lambda question: 10)
    monkeypatch.setattr(rp, "_cache_get", lambda cache_key: None)
    monkeypatch.setattr(rp, "_cache_set", lambda cache_key, payload: None)

    weak_docs = [_Doc("general overview without retrieval details", {"source": "doc-weak", "chunk_hash": "w1", "vector_score": 2.0})]
    faiss_docs = [_Doc("FAISS index growth without pruning can slow retrieval and increase memory usage.", {"source": "doc-faiss", "chunk_hash": "f1", "vector_score": 1.8})]

    monkeypatch.setattr(rp, "_build_multi_source_docs", lambda question, k_per_source=10, namespace=None: weak_docs)
    monkeypatch.setattr(rp, "_keyword_fallback_docs", lambda question, namespace=None, max_docs=12: faiss_docs)
    monkeypatch.setattr(
        rp,
        "_passes_retrieval_threshold",
        lambda docs: any("faiss" in (d.page_content or "").lower() for d in docs),
    )
    monkeypatch.setattr(
        rp,
        "_run_prompt_with_context",
        lambda question, context, enforce_grounding=True: "FAISS index growth without pruning can slow retrieval and increase memory usage.",
    )

    result = rp.ask_question_with_sources("What is the FAISS limitation?", namespace="team-a")

    assert "FAISS" in result["answer"]
    assert "retrieval" in result["answer"].lower()
    assert isinstance(result.get("sources"), list)


def test_recommended_k_for_short_query_is_at_least_ten(monkeypatch):
    import rag_pipeline
    rp = importlib.reload(rag_pipeline)

    k = rp._recommended_k_per_source("FAISS limitation?")
    assert k >= 10
