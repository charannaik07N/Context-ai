import importlib
import tempfile
from pathlib import Path


def _reload_rag_pipeline(monkeypatch):
    import rag_pipeline
    module = importlib.reload(rag_pipeline)
    return module


def test_extractive_fallback_prefers_relevant_sentence(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)
    context = (
        "The server uses PostgreSQL for storage. "
        "Authentication is done with OAuth 2.0 tokens and rotating refresh tokens. "
        "Logs are exported to a central collector."
    )
    question = "How is authentication implemented?"

    answer = rp._extractive_fallback_answer(question, context)

    assert "OAuth 2.0" in answer


def test_prune_ungrounded_sentences_removes_hallucinated_lines(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)
    context = "Payments are processed by Stripe and reconciled daily."
    answer = "Payments are processed by Stripe. The system trains LLMs nightly."

    pruned = rp._prune_ungrounded_sentences(answer, context)

    assert "Stripe" in pruned
    assert "trains LLMs nightly" not in pruned


def test_load_documents_rejects_unsupported_file_type(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)

    with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
        try:
            rp._load_documents(tmp.name)
            assert False, "Expected ValueError for unsupported extension"
        except ValueError as e:
            assert "Unsupported file type" in str(e)


def test_groq_enabled_respects_provider_mode(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)

    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(rp, "LLM_PROVIDER", "extractive")
    assert rp._groq_enabled() is False

    monkeypatch.setattr(rp, "LLM_PROVIDER", "groq")
    assert rp._groq_enabled() is True


def test_offer_context_detector_is_false_for_general_technical_text(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)
    context = "FAISS index growth without pruning can increase memory usage and retrieval latency."

    assert rp._looks_like_offer_letter_context(context) is False


def test_offer_context_detector_is_true_for_offer_letter_text(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)
    context = (
        "Offer Letter\n"
        "Your designation is Frontend Developer Intern.\n"
        "Date of commencement will be from 11-Mar-2026.\n"
        "Monthly stipend and CTC are described below."
    )

    assert rp._looks_like_offer_letter_context(context) is True


def test_run_prompt_does_not_apply_offer_extractor_for_general_pdf(monkeypatch):
    rp = _reload_rag_pipeline(monkeypatch)

    monkeypatch.setattr(rp, "_extract_offer_field_directly", lambda q, c: "EXTRACTED_OFFER_FIELD")
    monkeypatch.setattr(rp, "load_llm", lambda: None)

    context = "FAISS index growth without pruning can increase memory usage and retrieval latency."
    answer = rp._run_prompt_with_context("Where is FAISS discussed?", context)

    assert "EXTRACTED_OFFER_FIELD" not in answer
