# Contexta-AI

Contexta-AI is a document Q and A and insights platform built with FastAPI + React. You can upload documents, ask grounded questions, extract terminology, generate insights, and monitor runtime metrics.

Last updated: April 10, 2026.

## What This Project Uses

### Backend

- Python 3.10+
- FastAPI + Uvicorn
- LangChain
- FAISS vector store (default) with optional Qdrant backend
- Sentence Transformers embeddings
- Optional reranker (CrossEncoder)
- Ollama local LLM support (current default flow in this repo)
- Optional Groq support
- Prometheus metrics
- Optional Redis + RQ for async jobs and distributed rate limiting
- JWT + legacy auth modes with namespace isolation

### Frontend

- React 18
- Vite 5
- React Router
- Axios
- Framer Motion
- Tailwind CSS

### Key Python Dependencies

- `fastapi`, `uvicorn`, `python-dotenv`
- `langchain`, `langchain-community`, `langchain-ollama`, `langchain-groq`, `langchain-google-genai`
- `faiss-cpu`, `sentence-transformers`, `langchain-huggingface`
- `redis`, `rq`, `PyJWT`
- `prometheus-client`, `opentelemetry-*`

## Core Features

- Multi-format upload: `.pdf`, `.docx`, `.txt`, `.html`, `.htm`
- Namespace-aware retrieval and answers
- Grounded Q and A with source snippets
- Term definition and per-document insights
- Async ingestion and metrics jobs
- Observability endpoints for metrics and runtime status
- Optional GPU and remote DGX helper scripts

## Project Layout

```text
Contexta-AI/
  main.py
  rag_pipeline.py
  requirements.txt
  .env
  core/
    auth.py
    rate_limiter.py
    task_queue.py
    faiss_integrity.py
    tracing.py
  frontend/
    package.json
    vite.config.js
    src/
  tests/
  observability/prometheus/contexta-alerts.yml
  start_with_gpu.ps1
  start_on_dgx.ps1
  gpu_health_check.py
```

## Quick Start

### 1. Backend Setup

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` in repo root.

Recommended hybrid mode (Ollama 3B + Gemini):

```env
LLM_PROVIDER=hybrid
OLLAMA_ENABLED=true
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
# Optional: fail fast if either provider is unavailable
# HYBRID_REQUIRE_BOTH=true
```

Ollama-only mode is still supported by setting:

```env
LLM_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_REQUIRED=true
```

### 3. Start Ollama

```bash
ollama pull llama3.2:3b
ollama serve
```

### 4. Start Backend API

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL: `http://localhost:5173`

## Frontend API Proxy

Vite proxies `/api/*` to backend.

Default proxy target:

- `http://127.0.0.1:8000`

Optional frontend env:

- `VITE_API_PROXY_TARGET`
- `VITE_CLIENT_KEY`

If backend is not running, Vite will log `ECONNREFUSED` for API calls.

## API Endpoints

Auth:

- `POST /auth/refresh`
- `POST /auth/revoke`

Index and upload:

- `GET /index-status`
- `POST /upload-paper`
- `POST /upload-papers`
- `DELETE /reset-index`

Q and A:

- `POST /ask-question`
- `POST /define-term`
- `GET /insights`
- `GET /metrics`

Async jobs:

- `GET /jobs/{job_id}`

Observability:

- `GET /observability/metrics`
- `GET /observability/status`

## Auth and Namespace Model

The API resolves caller namespace through `AuthManager`.

Supported patterns:

- JWT mode (`AUTH_MODE=jwt`)
- Legacy key mapping (`CLIENT_NAMESPACE_MAP` + `X-Client-Key`)
- Hybrid mode (`AUTH_MODE=hybrid`)

Role checks are enforced for endpoint policy, especially in JWT mode.

## Async Jobs and Queue

By default, jobs can run locally in-process.

Optional distributed mode:

- Set `TASK_QUEUE_BACKEND=rq`
- Provide `REDIS_URL`
- Run an RQ worker for your queue name

Related settings:

- `TASK_QUEUE_REQUIRED`
- `TASK_QUEUE_NAME`
- `TASK_QUEUE_JOB_TIMEOUT_SECONDS`
- `TASK_QUEUE_RETRY_MAX`

## Performance Tuning

Current low-latency settings in this repo include:

- `RERANKER_ENABLED=false`
- `FAST_QUERY_MODE=true`
- `FAST_QUERY_TERM_THRESHOLD=12`
- `MAX_CONTEXT_DOCS=4`

Ollama speed knobs:

- `OLLAMA_NUM_CTX`
- `OLLAMA_NUM_PREDICT`
- `OLLAMA_NUM_THREAD`
- `OLLAMA_KEEP_ALIVE`
- `OLLAMA_NUM_GPU`

If answers are slow, confirm actual runtime mode:

```bash
ollama ps
```

Check `PROCESSOR` field for CPU vs GPU.

## GPU and DGX Helpers

Local GPU diagnostics:

```bash
python gpu_health_check.py
```

Windows helper script:

```powershell
.\start_with_gpu.ps1
```

Remote DGX helper script:

```powershell
.\start_on_dgx.ps1
```

## Testing

Run all default tests:

```bash
pytest
```

Markers configured:

- `e2e`
- `stress`

Example:

```bash
pytest -m "not e2e and not stress"
```

## Troubleshooting

### Frontend shows proxy errors

Symptom:

- `http proxy error ... ECONNREFUSED 127.0.0.1:8000`

Fix:

- Start backend on port 8000.
- Verify `frontend/vite.config.js` proxy target.

### Ollama answers fail

Check:

- `ollama serve` is running.
- Model is pulled (`ollama pull llama3.2:3b`).
- `OLLAMA_BASE_URL` is reachable.

### Wrong or verbose answers

Tune:

- lower `MAX_CONTEXT_DOCS`
- keep reranker off for speed
- keep question short and specific

## Security Note

Do not commit secrets from `.env`.

Rotate keys if credentials are ever exposed.

## License

MIT
