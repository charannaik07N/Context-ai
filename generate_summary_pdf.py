"""Generates a professional project summary PDF for Contexta AI."""

from fpdf import FPDF, XPos, YPos
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Contexta_AI_Project_Summary.pdf")

# Colour palette (RGB)
DARK_BG  = (15,  17,  23)
PANEL_BG = (22,  27,  34)
PRIMARY  = (59, 130, 246)
WHITE    = (243, 244, 246)
MUTED    = (156, 163, 175)
BORDER   = (48,  54,  61)
ACCENT   = (34, 197,  94)
RED      = (239,  68,  68)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def rgb(self, c):
        self.set_text_color(*c)

    def fill(self, x, y, w, h, c):
        self.set_fill_color(*c)
        self.rect(x, y, w, h, "F")

    def section_header(self, title):
        self.ln(4)
        bx, by = self.get_x(), self.get_y()
        self.fill(bx, by, 4, 8, PRIMARY)
        self.set_xy(bx + 7, by)
        self.set_font("Helvetica", "B", 13)
        self.rgb(PRIMARY)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BORDER)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def para(self, text):
        self.set_font("Helvetica", "", 10)
        self.rgb(WHITE)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=5):
        self.set_font("Helvetica", "", 10)
        x0 = self.l_margin + indent
        self.set_x(x0)
        self.rgb(PRIMARY)
        self.cell(5, 5.5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.rgb(WHITE)
        self.multi_cell(self.w - self.r_margin - self.get_x(), 5.5, text)

    def kv(self, key, value):
        KEY_W = 52
        self.set_font("Helvetica", "B", 10)
        self.rgb(PRIMARY)
        self.set_x(self.l_margin)
        self.cell(KEY_W, 6, key + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 10)
        self.rgb(WHITE)
        val_w = self.w - self.r_margin - self.l_margin - KEY_W
        self.multi_cell(val_w, 6, value)

    def header(self):
        self.fill(0, 0, self.w, 18, DARK_BG)
        self.set_font("Helvetica", "B", 11)
        self.rgb(PRIMARY)
        self.set_y(5)
        self.cell(0, 8, "  Contexta AI - Project Summary", new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
        self.set_font("Helvetica", "", 9)
        self.rgb(MUTED)
        self.cell(0, 8, "March 2026  ", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.fill(0, self.h - 14, self.w, 14, DARK_BG)
        self.set_font("Helvetica", "I", 8)
        self.rgb(MUTED)
        self.cell(0, 10, "Page " + str(self.page_no()) + "  |  Contexta AI", align="C")


def build():
    pdf = PDF()
    pdf.set_margins(18, 22, 18)
    W = 210 - 18 - 18  # usable page width (A4)

    # ======================================================================
    # PAGE 1 - Cover
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)
    pdf.fill(0, 70, pdf.w, 3, PRIMARY)

    pdf.set_y(85)
    pdf.set_font("Helvetica", "B", 34)
    pdf.rgb(WHITE)
    pdf.cell(0, 14, "Contexta AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", "", 14)
    pdf.rgb(PRIMARY)
    pdf.cell(0, 9, "RAG-Powered Research Paper Intelligence", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.rgb(MUTED)
    pdf.cell(0, 7, "Understand research papers faster with natural language queries", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 7, "and AI-driven summarization powered by LangChain + Groq.", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(14)
    pdf.set_draw_color(*BORDER)
    pdf.line(40, pdf.get_y(), pdf.w - 40, pdf.get_y())
    pdf.ln(14)

    stats = [
        ("Type",      "Full-Stack AI Web App"),
        ("LLM",       "Llama 3.3-70B (Groq)"),
        ("Vector DB", "FAISS (CPU)"),
        ("Stack",     "FastAPI + React 18"),
    ]
    cell_w = W / len(stats)
    x0 = pdf.l_margin
    y0 = pdf.get_y()
    for i, (label, val) in enumerate(stats):
        cx = x0 + i * cell_w
        pdf.fill(cx, y0, cell_w - 3, 22, PANEL_BG)
        pdf.set_xy(cx + 3, y0 + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.rgb(PRIMARY)
        pdf.cell(cell_w - 10, 5, label.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(cx + 3)
        pdf.set_font("Helvetica", "", 9)
        pdf.rgb(WHITE)
        pdf.cell(cell_w - 10, 6, val, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(30)

    badges = ["Open Source", "Python 3.x", "LangChain", "FAISS", "React 18", "Vite"]
    pdf.set_x(pdf.l_margin)
    for badge in badges:
        bw = pdf.get_string_width(badge) + 8
        bx = pdf.get_x()
        by = pdf.get_y()
        pdf.fill(bx, by, bw, 8, PANEL_BG)
        pdf.set_font("Helvetica", "", 8)
        pdf.rgb(PRIMARY)
        pdf.cell(bw, 8, badge, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(pdf.get_x() + 3)

    # ======================================================================
    # PAGE 2 - Overview & Features
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)

    pdf.section_header("1.  Project Overview")
    pdf.para(
        "Contexta AI is a GenAI-powered research assistant that enables students, researchers, "
        "and professionals to understand complex research papers faster through natural language "
        "interaction. It combines Retrieval-Augmented Generation (RAG) with state-of-the-art "
        "Large Language Models (LLMs) to deliver accurate, context-aware answers grounded "
        "directly in the uploaded document."
    )
    pdf.para(
        "The system follows a clean separation of concerns: a FastAPI Python backend handles "
        "all AI/ML workloads, while a modern React 18 + Vite frontend delivers an intuitive, "
        "dark-themed interface reminiscent of modern developer tools."
    )

    pdf.section_header("2.  Core Features")

    features = [
        ("PDF Upload & Processing",
         "Users upload research papers (up to 50 MB). The backend validates file type and size, "
         "extracts text page-by-page with PyPDFLoader, splits into overlapping chunks, embeds "
         "them with sentence-transformers, and persists the result in a local FAISS index."),
        ("Natural Language Q&A",
         "Powered by a LangChain LCEL RAG chain: the user question is embedded, top-k relevant "
         "document chunks are retrieved from FAISS, then injected into a structured prompt sent "
         "to Llama 3.3-70B via Groq's low-latency inference API."),
        ("Key Insights Extraction",
         "A dedicated endpoint runs 'Extract 5 key insights from this paper in bullet points' "
         "through the RAG chain. The Insights page caches results in localStorage to avoid "
         "redundant API calls when navigating back to the page."),
        ("One-Line Summary",
         "Generates a concise single-sentence summary of the entire document, ideal for quick "
         "orientation before deep reading."),
        ("Terminology Helper",
         "Users type any technical term (e.g. In-Context Learning) and receive a plain-English "
         "definition drawn from the document's actual context via the RAG chain."),
        ("RAG Evaluation Dashboard",
         "Built-in evaluation computes 9 RAG quality metrics: context precision, recall, MRR, "
         "relevance, sufficiency, answer relevance, correctness, latency, and hallucination "
         "rate, using cosine similarity on sentence embeddings against 4 probe questions."),
    ]

    for title, desc in features:
        pdf.set_font("Helvetica", "B", 10)
        pdf.rgb(ACCENT)
        pdf.cell(0, 6, "  >> " + title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.rgb(WHITE)
        pdf.set_x(pdf.l_margin + 8)
        pdf.multi_cell(W - 8, 5.5, desc)
        pdf.ln(3)

    # ======================================================================
    # PAGE 3 - Architecture & Data Flow
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)

    pdf.section_header("3.  System Architecture")
    pdf.para("Contexta AI follows a three-tier architecture with an additional AI/ML layer:")

    layers = [
        ("Presentation Layer",
         "React 18 SPA (Vite build), React Router v7 for client-side routing, "
         "Framer Motion animations, Axios for HTTP calls."),
        ("API Layer",
         "FastAPI application (main.py) exposing RESTful endpoints. CORS is configurable "
         "via environment variable. Heavy computation is offloaded to the RAG layer."),
        ("RAG / AI Layer",
         "rag_pipeline.py: PyPDFLoader -> RecursiveCharacterTextSplitter -> "
         "HuggingFaceEmbeddings -> FAISS.from_documents -> LCEL chain "
         "(retriever | prompt | ChatGroq | StrOutputParser)."),
        ("Persistence Layer",
         "FAISS index saved under vector_store/. Only one document is active at a time; "
         "old uploads are purged automatically when a new file is uploaded."),
    ]

    for layer, desc in layers:
        by = pdf.get_y()
        pdf.fill(pdf.l_margin, by, W, 18, PANEL_BG)
        pdf.set_xy(pdf.l_margin + 4, by + 2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.rgb(PRIMARY)
        pdf.cell(0, 5, layer, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font("Helvetica", "", 9)
        pdf.rgb(WHITE)
        pdf.multi_cell(W - 4, 5, desc)
        pdf.ln(4)

    pdf.ln(3)
    pdf.section_header("3a.  Request Data Flow")

    flow_steps = [
        "User uploads PDF via React UploadPanel component",
        "POST /upload-paper  ->  FastAPI validates size and file type",
        "PyPDFLoader splits PDF into overlapping text chunks",
        "HuggingFace all-MiniLM-L6-v2 embeds each chunk locally on CPU",
        "FAISS index built and persisted to disk under vector_store/",
        "User types a question in the ChatWindow input bar",
        "POST /ask  ->  question embedded, then FAISS k-NN search runs",
        "Top-k chunks and question injected into LangChain prompt template",
        "Groq API calls Llama 3.3-70B and returns generated answer",
        "React renders the answer in a styled MessageBubble component",
    ]

    for i, step in enumerate(flow_steps):
        pdf.set_font("Helvetica", "B", 10)
        pdf.rgb(PRIMARY)
        pdf.set_x(pdf.l_margin + 3)
        pdf.cell(8, 6, str(i + 1) + ".")
        pdf.set_font("Helvetica", "", 10)
        pdf.rgb(WHITE)
        pdf.multi_cell(W - 11, 6, step)
        if i < len(flow_steps) - 1:
            pdf.set_x(pdf.l_margin + 10)
            pdf.rgb(MUTED)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 4, "|", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ======================================================================
    # PAGE 4 - Technology Stack
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)

    pdf.section_header("4.  Technology Stack")

    pdf.set_font("Helvetica", "B", 11)
    pdf.rgb(ACCENT)
    pdf.cell(0, 7, "  Backend (Python 3.x)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    backend_deps = [
        ("FastAPI + Uvicorn",       "REST API framework and ASGI server"),
        ("LangChain / LCEL",        "RAG orchestration, prompt templates, chain composition"),
        ("langchain-groq",          "Groq inference client for Llama 3.3-70B-versatile"),
        ("langchain-huggingface",   "HuggingFace embeddings wrapper"),
        ("FAISS-cpu",               "Approximate nearest-neighbour vector search"),
        ("sentence-transformers",   "all-MiniLM-L6-v2 embedding model (local, CPU)"),
        ("PyPDF2 / PyPDFLoader",    "PDF text extraction and page splitting"),
        ("python-dotenv",           "Environment variable management from .env files"),
        ("pyttsx3",                 "Offline text-to-speech synthesis"),
        ("NumPy",                   "Cosine similarity computation for evaluation metrics"),
    ]
    for pkg, desc in backend_deps:
        pdf.kv(pkg, desc)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.rgb(ACCENT)
    pdf.cell(0, 7, "  Frontend (JavaScript / React)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    frontend_deps = [
        ("React 18.2",              "Component-based UI library"),
        ("Vite 5.0",                "Lightning-fast HMR build tooling"),
        ("React Router v7",         "Client-side SPA routing"),
        ("Axios",                   "HTTP client for backend API calls"),
        ("Framer Motion",           "Page and component transition animations"),
        ("CSS Custom Properties",   "Dark-theme design system, no external CSS framework"),
        ("AppContext",              "React Context: global documentReady, documentName, reset"),
    ]
    for pkg, desc in frontend_deps:
        pdf.kv(pkg, desc)

    # ======================================================================
    # PAGE 5 - API Endpoints
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)

    pdf.section_header("5.  REST API Endpoints")

    endpoints = [
        ("POST",  "/upload-paper",
         "Upload PDF (multipart/form-data). Validates type and size (<50 MB), deletes prior "
         "uploads, chunks and embeds document, saves FAISS index. Returns {message, filename, pages, chunks}."),
        ("POST",  "/ask",
         "Body: {question}. Retrieves relevant chunks via FAISS, calls Groq LLM with RAG prompt. Returns {answer}."),
        ("GET",   "/insights",
         "Runs 'Extract 5 key insights' prompt through RAG chain. Returns {insights} string."),
        ("GET",   "/summary",
         "Runs 'Give a one-line summary' prompt. Returns {summary}."),
        ("POST",  "/define-term",
         "Body: {term}. Explains the term in simple words using document context. Returns {definition}."),
        ("GET",   "/metrics",
         "Computes 9 RAG evaluation metrics. Returns JSON object or null if no document is loaded."),
        ("POST",  "/reset",
         "Deletes FAISS index and all uploaded files, resetting the system to a clean state."),
        ("GET",   "/health",
         "Health-check endpoint. Returns {status: ok}."),
    ]

    COL_M = 16
    COL_P = 44
    COL_D = W - COL_M - COL_P

    pdf.fill(pdf.l_margin, pdf.get_y(), W, 7, PANEL_BG)
    pdf.set_font("Helvetica", "B", 9)
    pdf.rgb(MUTED)
    pdf.set_x(pdf.l_margin)
    pdf.cell(COL_M, 7, "Method")
    pdf.cell(COL_P, 7, "Endpoint")
    pdf.cell(COL_D, 7, "Description", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    mc = {"GET": ACCENT, "POST": PRIMARY, "DELETE": RED}

    for method, path, desc in endpoints:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.rgb(mc.get(method, WHITE))
        pdf.cell(COL_M, 6, method, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Courier", "B", 9)
        pdf.rgb(WHITE)
        pdf.cell(COL_P, 6, path, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9)
        pdf.rgb(MUTED)
        pdf.multi_cell(COL_D, 6, desc)
        pdf.set_draw_color(*BORDER)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(1)

    # ======================================================================
    # PAGE 6 - Frontend Pages & Evaluation
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)

    pdf.section_header("6.  Frontend Pages & Components")

    pages_info = [
        ("Chat  (default route /)",
         "Split layout: UploadPanel (collapsible drag-and-drop) + ChatWindow. "
         "Sends questions to /ask and renders styled message bubbles with user/assistant distinction."),
        ("Insights  (/insights)",
         "Calls /insights on document ready. Results cached in localStorage per document. "
         "Displays bullet-point key insights with Framer Motion fade-in animation."),
        ("Terminology  (/terminology)",
         "Search box for any technical term. On submit calls /define-term and displays a "
         "definition panel. All controls disabled until a document is uploaded."),
        ("About  (/about)",
         "Static page describing the project purpose, technology stack, and use cases."),
    ]

    for name, desc in pages_info:
        pdf.set_font("Helvetica", "B", 10)
        pdf.rgb(PRIMARY)
        pdf.cell(0, 6, "  " + name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.rgb(WHITE)
        pdf.set_x(pdf.l_margin + 8)
        pdf.multi_cell(W - 8, 5.5, desc)
        pdf.ln(3)

    pdf.para(
        "Shared components: Layout (Navbar + Sidebar wrapper), Navbar (brand, document status "
        "badge, reset button), Sidebar (navigation links with active-state highlighting), "
        "ChatWindow (message list + input bar), MessageBubble (styled user/assistant bubbles), "
        "UploadPanel (drag-and-drop upload with progress), and Loader (animated spinner)."
    )

    pdf.section_header("7.  RAG Evaluation Metrics")
    pdf.para(
        "The evaluation module (evaluate.py and the /metrics endpoint) measure pipeline quality "
        "using 4 generic probe questions. All 9 metrics are averaged across them:"
    )

    metrics = [
        ("Context Precision",   "Fraction of retrieved chunks above cosine-similarity threshold"),
        ("Context Recall",      "Query-term keyword overlap in retrieved context"),
        ("MRR",                 "Mean Reciprocal Rank of the first relevant chunk"),
        ("Context Relevance",   "Average cosine similarity of retrieved chunk embeddings"),
        ("Context Sufficiency", "Fraction of query keywords present in retrieved context"),
        ("Answer Relevance",    "Cosine similarity between question and generated answer"),
        ("Answer Correctness",  "Token-level F1 score between context tokens and answer"),
        ("Latency (ms)",        "End-to-end answer generation time in milliseconds"),
        ("Hallucination Rate",  "Fraction of answer words absent from retrieved context"),
    ]
    for metric, desc in metrics:
        pdf.kv(metric, desc)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.rgb(ACCENT)
    pdf.cell(0, 6, "  Clarity Checks (no ground truth needed):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for c in [
        "Answer is not empty",
        "Answer does not contain error strings",
        "Answer does not flatly refuse with only 'I don't know'",
        "Answer meets a minimum meaningful length",
    ]:
        pdf.bullet(c)
    pdf.ln(3)

    # ======================================================================
    # PAGE 7 - Setup, Security & Roadmap
    # ======================================================================
    pdf.add_page()
    pdf.fill(0, 0, pdf.w, pdf.h, DARK_BG)

    pdf.section_header("8.  Installation & Setup")

    setup = [
        "git clone https://github.com/Srivardhan04/Contexta-AI.git  &&  cd Contexta-AI",
        "python -m venv .venv  &&  .venv\\Scripts\\activate  (Windows)",
        "pip install -r requirements.txt",
        "Create .env file:  GROQ_API_KEY=your_api_key_here",
        "uvicorn main:app --reload --port 8000     (backend starts on :8000)",
        "cd frontend  &&  npm install  &&  npm run dev  (frontend starts on :5173)",
        "Open http://localhost:5173 in your browser",
    ]
    for i, step in enumerate(setup, 1):
        pdf.set_font("Helvetica", "B", 9)
        pdf.rgb(PRIMARY)
        pdf.set_x(pdf.l_margin + 3)
        pdf.cell(8, 6, str(i) + ".")
        pdf.set_font("Courier", "", 9)
        pdf.rgb(WHITE)
        pdf.multi_cell(W - 11, 6, step)

    pdf.ln(3)
    pdf.para(
        "Environment variables: GROQ_API_KEY (required), GROQ_MODEL (optional, defaults to "
        "llama-3.3-70b-versatile), ALLOWED_ORIGINS (optional, defaults to http://localhost:5173)."
    )

    pdf.section_header("9.  Security Considerations")
    for item in [
        "File type validation (MIME type + extension) blocks non-PDF uploads",
        "50 MB hard limit per upload prevents denial-of-service via large files",
        "CORS restricted to configurable origin list (no wildcard in production)",
        "GROQ_API_KEY loaded from .env file - never hardcoded in source",
        "FAISS allow_dangerous_deserialization acknowledged for local index only",
        "Old uploaded files purged on each new upload to prevent data leakage",
        "Pydantic BaseModel validates all incoming JSON request bodies",
    ]:
        pdf.bullet(item)
    pdf.ln(4)

    pdf.section_header("10.  Potential Enhancements / Roadmap")
    for item in [
        "Multi-document support with separate vector namespaces per document",
        "Streaming SSE responses for real-time answer token display",
        "User authentication and per-user document history",
        "Support for additional file formats: TXT, Markdown",
        "Graph-based knowledge extraction and visual concept maps",
        "Fine-tuned embedding model for academic domain specificity",
        "Docker Compose deployment configuration for easy self-hosting",
        "Cloud-hosted vector database (Pinecone / Weaviate) for scale",
    ]:
        pdf.bullet(item)
    pdf.ln(6)

    by = pdf.get_y()
    pdf.fill(pdf.l_margin, by, W, 16, PANEL_BG)
    pdf.set_xy(pdf.l_margin + 4, by + 3)
    pdf.set_font("Helvetica", "I", 10)
    pdf.rgb(MUTED)
    pdf.multi_cell(W - 4, 5.5,
        "Contexta AI is an end-to-end, production-ready RAG application demonstrating modern "
        "AI engineering practices: local vector search, LLM integration, quantitative evaluation "
        "metrics, and a polished React front-end - all in a single cohesive codebase."
    )

    pdf.output(OUTPUT_PATH)
    print("PDF written to: " + OUTPUT_PATH)


if __name__ == "__main__":
    build()
