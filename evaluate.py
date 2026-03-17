"""
evaluate.py — RAG Pipeline Evaluation
======================================
Metrics computed (SQuAD-style, token-level):
  • Precision  = overlapping tokens / predicted tokens
  • Recall     = overlapping tokens / ground-truth tokens
  • F1         = harmonic mean of Precision and Recall
  • Exact Match (EM) = 1 if normalised strings match exactly, else 0

Clarity checks (no ground truth needed):
  • Answer is not empty
  • Answer is not an error string
  • Answer does not flatly refuse ("I don't know" with nothing else)
  • Answer meets a minimum length

Usage
-----
1. Make sure the backend has processed a PDF (upload via the app first, OR
   set PDF_PATH below to auto-process one before running tests).
2. Edit TEST_CASES with Q&A pairs relevant to YOUR uploaded document.
3. Run:
       python evaluate.py
"""

import os
import re
import string
import sys
import io
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

# Force UTF-8 output on Windows to avoid charmap errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

# ---------------------------------------------------------------------------
# ❶  Optional: auto-process a PDF before running tests
#     Leave as None if a PDF is already uploaded via the app.
# ---------------------------------------------------------------------------
PDF_PATH = None   # e.g. "uploaded_docs/my_paper.pdf"

# ---------------------------------------------------------------------------
# ❷  Edit these Q&A pairs to match YOUR uploaded document.
#     'expected' is the reference answer used for metric calculation.
#     Leave 'expected' as "" to run clarity-only checks for that question.
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "question": "What is the main contribution of this paper?",
        "expected": "",   # fill in after reading the paper
    },
    {
        "question": "What dataset was used in the experiments?",
        "expected": "",
    },
    {
        "question": "What method or approach is proposed?",
        "expected": "",
    },
    {
        "question": "What are the key findings or results?",
        "expected": "",
    },
    {
        "question": "Who are the authors of this paper?",
        "expected": "",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation and extra whitespace (SQuAD style)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


def squad_metrics(predicted: str, ground_truth: str):
    """
    Returns (precision, recall, f1, exact_match) for a single prediction.
    All values are floats in [0, 1].
    """
    pred_tokens = _tokens(predicted)
    gt_tokens   = _tokens(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0, 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter   = Counter(gt_tokens)

    overlap = sum((pred_counter & gt_counter).values())

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gt_tokens)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    em        = 1.0 if _normalise(predicted) == _normalise(ground_truth) else 0.0

    return precision, recall, f1, em


# ---------------------------------------------------------------------------
# Clarity checker  (no ground truth required)
# ---------------------------------------------------------------------------

_ERROR_PREFIXES = (
    "error while processing",
    "vector store not found",
    "no pdf has been processed",
    "please enter a valid",
    "please provide a valid",
)

_REFUSAL_PATTERNS = re.compile(
    r"^(i (don'?t|do not) know|i('m| am) not sure|i cannot answer)[.,!]?\s*$",
    re.IGNORECASE,
)

MIN_ANSWER_WORDS = 5


def clarity_check(answer: str) -> dict:
    """
    Returns a dict with individual flags and an overall 'passed' bool.
    """
    stripped = answer.strip()

    not_empty    = len(stripped) > 0
    not_error    = not any(stripped.lower().startswith(p) for p in _ERROR_PREFIXES)
    not_refusal  = not bool(_REFUSAL_PATTERNS.match(stripped))
    long_enough  = len(stripped.split()) >= MIN_ANSWER_WORDS

    passed = all([not_empty, not_error, not_refusal, long_enough])

    return {
        "not_empty":   not_empty,
        "not_error":   not_error,
        "not_refusal": not_refusal,
        "long_enough": long_enough,
        "passed":      passed,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

BAR   = "-" * 88
DBAR  = "=" * 88
TICK  = "PASS"
CROSS = "FAIL"


def _fmt_bool(val: bool) -> str:
    return f"\033[92m{TICK}\033[0m" if val else f"\033[91m{CROSS}\033[0m"


def _fmt_score(val: float) -> str:
    colour = "\033[92m" if val >= 0.5 else "\033[93m" if val >= 0.25 else "\033[91m"
    return f"{colour}{val:.3f}\033[0m"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation():
    from rag_pipeline import ask_question, process_pdf, vector_store_exists

    # Auto-process PDF if requested
    if PDF_PATH:
        pdf = Path(PDF_PATH)
        if not pdf.exists():
            print(f"\n[ERROR] PDF not found: {PDF_PATH}\n")
            sys.exit(1)
        print(f"\nProcessing PDF: {pdf.name} …")
        process_pdf(str(pdf))
        print("Done.\n")

    if not vector_store_exists():
        print("\n[ERROR] No vector store found.")
        print("Upload a PDF via the app (http://localhost:5173) OR set PDF_PATH in this script.\n")
        sys.exit(1)

    print(f"\n{DBAR}")
    print("  CONTEXTA-AI  —  RAG EVALUATION REPORT")
    print(DBAR)
    print(f"  Model : {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
    print(f"  Cases : {len(TEST_CASES)}")
    print(DBAR)

    results = []

    for idx, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        expected = case.get("expected", "").strip()

        print(f"\n[{idx}/{len(TEST_CASES)}] {question}")
        print(BAR)

        # Get answer from RAG
        predicted = ask_question(question)
        preview = predicted[:300] + ("..." if len(predicted) > 300 else "")
        print(f"  Answer   : {preview}")

        # Clarity
        clarity = clarity_check(predicted)
        print(f"\n  Clarity checks:")
        print(f"    Not empty   : {_fmt_bool(clarity['not_empty'])}")
        print(f"    Not error   : {_fmt_bool(clarity['not_error'])}")
        print(f"    Not refusal : {_fmt_bool(clarity['not_refusal'])}")
        print(f"    Long enough : {_fmt_bool(clarity['long_enough'])}  (≥{MIN_ANSWER_WORDS} words)")
        print(f"    Overall     : {_fmt_bool(clarity['passed'])}")

        row = {
            "question": question,
            "predicted": predicted,
            "expected": expected,
            "clarity": clarity,
            "precision": None,
            "recall": None,
            "f1": None,
            "em": None,
        }

        # SQuAD metrics (only when ground truth is provided)
        if expected:
            p, r, f1, em = squad_metrics(predicted, expected)
            row.update({"precision": p, "recall": r, "f1": f1, "em": em})
            print(f"\n  SQuAD-style metrics (vs. ground truth):")
            print(f"    Precision   : {_fmt_score(p)}")
            print(f"    Recall      : {_fmt_score(r)}")
            print(f"    F1          : {_fmt_score(f1)}")
            print(f"    Exact Match : {_fmt_bool(bool(em))}")
        else:
            print(f"\n  SQuAD metrics : \033[93mskipped\033[0m  (no 'expected' answer provided)")

        results.append(row)

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    print(f"\n{DBAR}")
    print("  SUMMARY")
    print(DBAR)

    clarity_passed = sum(1 for r in results if r["clarity"]["passed"])
    print(f"  Clarity passed   : {clarity_passed}/{len(results)}")

    metric_rows = [r for r in results if r["f1"] is not None]
    if metric_rows:
        avg_p  = sum(r["precision"] for r in metric_rows) / len(metric_rows)
        avg_r  = sum(r["recall"]    for r in metric_rows) / len(metric_rows)
        avg_f1 = sum(r["f1"]        for r in metric_rows) / len(metric_rows)
        avg_em = sum(r["em"]        for r in metric_rows) / len(metric_rows)

        print(f"\n  Avg Precision    : {_fmt_score(avg_p)}")
        print(f"  Avg Recall       : {_fmt_score(avg_r)}")
        print(f"  Avg F1           : {_fmt_score(avg_f1)}")
        print(f"  Avg Exact Match  : {_fmt_score(avg_em)}")

        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  {'Question':<35} {'P':>6} {'R':>6} {'F1':>6} │")
        print(f"  ├─────────────────────────────────────────────────┤")
        for r in metric_rows:
            q = r["question"][:35]
            print(f"  │  {q:<35} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} │")
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │  {'AVERAGE':<35} {avg_p:>6.3f} {avg_r:>6.3f} {avg_f1:>6.3f} │")
        print(f"  └─────────────────────────────────────────────────┘")
    else:
        print("\n  No ground-truth answers provided — only clarity was evaluated.")
        print("  To get Precision / Recall / F1, fill in the 'expected' field in TEST_CASES.")

    print(f"\n{DBAR}\n")


if __name__ == "__main__":
    run_evaluation()
