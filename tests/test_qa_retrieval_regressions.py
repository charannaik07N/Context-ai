import importlib


class _Doc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


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

    assert result["answer"] == "I don't know based on the provided context."
    assert result["sources"] == []
