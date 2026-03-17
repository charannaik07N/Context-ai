from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import gc
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Please add it to your .env file before starting the server."
    )

app = FastAPI()

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
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

BATCH_SIZE = 32
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_FILES_PER_UPLOAD = 10
ALLOWED_EXTENSIONS = {".pdf"}

# ================= REQUEST MODELS =================
class QuestionRequest(BaseModel):
    question: str

class TermRequest(BaseModel):
    term: str

# ================= ROUTES =================

async def _read_and_save_file(file: UploadFile) -> tuple[str, str]:
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
                "Allowed types are .pdf"
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

    saved_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(saved_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file {safe_filename}: {str(e)}")

    return safe_filename, saved_path

@app.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    filename, file_path = await _read_and_save_file(file)

    # 2. Process and append into persistent index with chunk-hash deduplication
    try:
        ingest_stats = append_document_to_vector_store(
            file_path=file_path,
            batch_size=BATCH_SIZE,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the document. Ensure it is a valid .pdf file.")
    finally:
        gc.collect()

    return {
        "message": "Document processed successfully.",
        "filename": filename,
        "ingestion": ingest_stats,
    }


@app.post("/upload-papers")
async def upload_papers(files: list[UploadFile] = File(...)):
    """Upload and ingest multiple documents in one request."""
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")

    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_FILES_PER_UPLOAD} files per request.",
        )

    processed = []
    failed = []

    for file in files:
        try:
            filename, file_path = await _read_and_save_file(file)
            ingest_stats = append_document_to_vector_store(
                file_path=file_path,
                batch_size=BATCH_SIZE,
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
                "error": f"Failed to process the document. Ensure it is a valid .pdf file. ({str(e)})",
            })
        finally:
            gc.collect()

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
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
    }


@app.post("/ask-question")
def query_paper(body: QuestionRequest):
    question = body.question.strip() if body.question else ""
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required and cannot be empty.")

    if not vector_store_exists():
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a supported file first.")

    result = ask_question_with_sources(question)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Detect error strings returned by the pipeline and surface them properly
    if answer and answer.startswith("Error while processing"):
        raise HTTPException(status_code=500, detail=answer)

    return {"question": question, "answer": answer, "sources": sources}


@app.post("/define-term")
def define_term(body: TermRequest):
    term = body.term.strip() if body.term else ""
    if not term:
        raise HTTPException(status_code=400, detail="'term' field is required and cannot be empty.")

    if not vector_store_exists():
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a supported file first.")

    definition = define_technical_term(term)
    return {"term": term, "definition": definition}


@app.get("/insights")
def get_insights():
    if not vector_store_exists():
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a supported file first.")

    docs = generate_insights_by_document()
    if not docs:
        raise HTTPException(status_code=500, detail="No document insights could be generated.")

    return {
        "documents": docs,
    }


@app.delete("/reset-index")
def reset_index():
    """Wipe the FAISS index and metadata so a fresh set of documents can be uploaded."""
    try:
        reset_vector_store()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset index: {str(e)}")
    return {"message": "Index reset successfully. Please re-upload your documents."}


@app.get("/metrics")
def get_rag_metrics():
    """Compute 9 RAG evaluation metrics against the currently uploaded document.
    Runs 2 sample queries — expect ~30-60 seconds."""
    if not vector_store_exists():
        raise HTTPException(
            status_code=404,
            detail="No document uploaded yet. Please upload a supported file first."
        )
    try:
        result = compute_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {str(e)}")

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to compute metrics.")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
