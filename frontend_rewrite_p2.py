import os

frontend_dir = os.path.join("frontend", "src")

files = {
    "pages/Insights.jsx": """import { useEffect, useState, useContext } from "react";
import { AppContext } from "../context/AppContext";
import { getInsights } from "../services/api";
import { FileText, Loader2, AlertCircle } from "lucide-react";

export default function Insights() {
  const { documentReady, documentName } = useContext(AppContext);

  const [insights, setInsights] = useState(() => {
    try {
      const cached = localStorage.getItem(`insights_${documentName}`);
      return cached ? JSON.parse(cached) : null;
    } catch {
      return null;
    }
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (documentReady && !insights && !loading) {
      fetchInsights();
    }
  }, [documentReady, documentName]);

  const fetchInsights = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getInsights();
      setInsights(res);
      localStorage.setItem(`insights_${documentName}`, JSON.stringify(res));    
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || "Failed to generate insights. Please try again.";
      setError(msg);
      console.error("Failed to fetch insights", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-bg-dark w-full overflow-hidden">
      <div className="flex-shrink-0 bg-bg-panel border-b border-border p-6 md:px-10">
        <h2 className="text-2xl font-bold text-text-main">Smart Insights</h2>
        <p className="text-text-muted mt-1 text-sm">Automated summaries and key findings from your uploaded documents</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 md:p-10 space-y-6 max-w-5xl mx-auto w-full">
        {!documentReady ? (
          <div className="flex flex-col items-center justify-center h-full text-center opacity-70">
            <div className="w-16 h-16 bg-bg-panel rounded-full flex items-center justify-center mb-4 text-text-muted">
              <FileText className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-medium text-text-main mb-2">No Document Context</h3>
            <p className="text-text-muted text-sm max-w-sm">
              Please upload a document to generate and view automated insights.
            </p>
          </div>
        ) : loading ? (
          <div className="bg-bg-panel border border-border rounded-xl p-8 flex flex-col items-center justify-center text-center animate-pulse">
             <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
             <h3 className="text-lg font-medium text-text-main">Analyzing Document...</h3>
             <p className="text-text-muted text-sm mt-2">Extracting key findings and summaries. This may take a moment.</p>
          </div>
        ) : error ? (
          <div className="bg-red-400/5 border border-red-500/20 rounded-xl p-6 text-center">
            <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-3" />
            <p className="text-red-400 text-sm mb-4">{error}</p>
            <button
              onClick={fetchInsights}
              className="px-4 py-2 bg-red-500/10 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/20 transition-colors"
            >
              Retry
            </button>
          </div>
        ) : insights ? (
          <>
            {(insights.documents || []).map((doc, idx) => (
              <div
                className="bg-bg-panel border border-border rounded-xl p-6 md:p-8 animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both"
                style={{ animationDelay: `${idx * 100}ms` }}
                key={doc.source}
              >
                <div className="flex items-center gap-3 mb-6 pb-4 border-b border-border/50">
                  <FileText className="w-5 h-5 text-primary" />
                  <h3 className="text-lg font-semibold text-text-main truncate">{doc.source}</h3>
                </div>

                {doc.error ? (
                  <div className="text-red-400 text-sm bg-red-400/10 p-4 rounded-lg flex items-center gap-2">
                     <AlertCircle className="w-4 h-4 flex-shrink-0" />
                     {doc.error}
                  </div>
                ) : (
                  <div className="space-y-8">
                    {doc.summary && (
                      <div>
                        <h4 className="text-sm font-medium text-text-muted uppercase tracking-wider mb-3">Summary</h4>
                        <p className="text-[15px] text-text-main leading-relaxed">{doc.summary}</p>
                      </div>
                    )}
                    {doc.key_insights && (
                      <div>
                        <h4 className="text-sm font-medium text-text-muted uppercase tracking-wider mb-3">Key Insights</h4>
                        <div className="text-[15px] text-text-main leading-relaxed whitespace-pre-line prose prose-invert max-w-none">
                          {doc.key_insights}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {(!insights.documents || insights.documents.length === 0) && (
              <div className="bg-bg-panel border border-border rounded-xl p-8 text-center text-text-muted text-sm">
                 No insights were generated for the current documents.
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
""",
    "pages/Terminology.jsx": """import { useState, useContext } from "react";
import { defineTerm } from "../services/api";
import { AppContext } from "../context/AppContext";
import { Search, Book, AlertCircle } from "lucide-react";

export default function Terminology() {
  const { documentReady } = useContext(AppContext);

  const [term, setTerm] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!documentReady) {
      setError("Upload a document first.");
      return;
    }
    if (!term.trim()) return;

    setLoading(true);
    setError("");
    setResult("");

    try {
      const res = await defineTerm(term);
      setResult(res.definition || "No definition found.");
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || "Failed to retrieve definition.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-bg-dark w-full overflow-hidden">
      <div className="flex-shrink-0 p-6 md:px-10 max-w-3xl w-full mx-auto mt-[10vh]">
        <div className="w-12 h-12 bg-primary/10 text-primary rounded-xl flex items-center justify-center mb-6">
          <Book className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-text-main mb-2">Terminology Helper</h2>
        <p className="text-text-muted mb-8 text-lg">
          Search technical terms directly from the uploaded document's context.
        </p>

        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1 relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-text-muted" />
            </div>
            <input
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              placeholder={documentReady ? "Enter a technical term (e.g. In-Context Learning)" : "Upload a document first"}
              disabled={!documentReady || loading}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="block w-full pl-11 pr-4 py-3.5 bg-bg-panel border border-border rounded-xl text-text-main placeholder-text-muted/60 focus:border-primary focus:ring-1 focus:ring-primary/50 transition-all outline-none text-[15px]"
            />
          </div>
          <button 
            onClick={handleSearch} 
            disabled={!documentReady || loading || !term.trim()}
            className="px-6 py-3.5 bg-primary text-white font-medium rounded-xl hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap min-w-[120px] flex items-center justify-center"
          >
            {loading ? (
               <span className="flex items-center gap-2">
                 <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                 Searching
               </span>
            ) : "Define"}
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-400 bg-red-400/10 px-4 py-3 rounded-xl animate-in fade-in">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {result && (
          <div className="mt-8 bg-bg-panel border border-border rounded-xl p-6 md:p-8 animate-in fade-in slide-in-from-bottom-2 duration-300">
             <h4 className="text-sm font-medium text-text-muted uppercase tracking-wider mb-4 border-b border-border/50 pb-3">Definition</h4>
             <div className="text-[16px] text-text-main leading-relaxed prose prose-invert">
                {result}
             </div>
          </div>
        )}
      </div>
    </div>
  );
}
""",
    "pages/About.jsx": """import { Info, Github, Layers, Shield, Zap } from "lucide-react";

export default function About() {
  return (
    <div className="flex flex-col h-full bg-bg-dark w-full overflow-y-auto">
      <div className="max-w-4xl w-full mx-auto p-6 md:p-12 pb-20">
        
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 bg-primary/10 text-primary rounded-xl flex items-center justify-center">
            <Info className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-text-main">About Contexta AI</h2>
            <p className="text-text-muted text-sm mt-1">Version 1.0.0</p>
          </div>
        </div>

        <div className="prose prose-invert prose-p:text-text-muted prose-headings:text-text-main max-w-none mb-12">
          <p className="text-lg leading-relaxed">
            Contexta AI is an advanced study companion and document analysis platform designed to help students, researchers, and professionals quickly extract knowledge from complex documents like research papers, slides, and reports.
          </p>
          <p className="text-lg leading-relaxed mt-4">
            Unlike standard AI chatbots, Contexta operates directly against your exact documents, providing precise answers and citing its sources down to the exact page and section.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
           <div className="bg-bg-panel border border-border rounded-xl p-6">
             <Layers className="w-8 h-8 text-primary mb-4" />
             <h3 className="text-lg font-semibold text-text-main mb-2">RAG Pipeline</h3>
             <p className="text-text-muted text-sm leading-relaxed">
               Built on a robust Retrieval-Augmented Generation pipeline using FAISS and HuggingFace sentence transformers for ultra-fast, local vector retrieval.
             </p>
           </div>
           
           <div className="bg-bg-panel border border-border rounded-xl p-6">
             <Zap className="w-8 h-8 text-yellow-500 mb-4" />
             <h3 className="text-lg font-semibold text-text-main mb-2">Adaptive Chunking</h3>
             <p className="text-text-muted text-sm leading-relaxed">
               Smartly processes documents depending on format and density. Presentations get segmented slide-by-slide, while academic PDFs use content density markers.
             </p>
           </div>

           <div className="bg-bg-panel border border-border rounded-xl p-6">
             <Shield className="w-8 h-8 text-emerald-500 mb-4" />
             <h3 className="text-lg font-semibold text-text-main mb-2">Verifiable Answers</h3>
             <p className="text-text-muted text-sm leading-relaxed">
               Uses cross-encoder reranking to ensure high relevance and specifically references original files and page numbers directly in the UI to prevent hallucinations.
             </p>
           </div>

           <div className="bg-bg-panel border border-border rounded-xl p-6 flex flex-col justify-center">
             <h3 className="text-lg font-semibold text-text-main mb-2">Open Source</h3>
             <p className="text-text-muted text-sm leading-relaxed mb-4">
               Contexta AI is highly configurable. Review the codebase to modify embedding models or swap language model providers effortlessly.
             </p>
             <a href="#" className="flex items-center gap-2 text-sm text-primary hover:text-primary-hover font-medium transition-colors w-fit">
               <Github className="w-4 h-4" />
               View Repository
             </a>
           </div>
        </div>

        <div className="border-t border-border pt-8 text-center text-text-muted text-sm">
          <p>© {new Date().getFullYear()} Contexta AI. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}
"""
}

for filepath, content in files.items():
    full_path = os.path.join(frontend_dir, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

print("Rewrite part 2 complete successfully.")
