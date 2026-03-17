import os

content = """import { useState, useContext } from "react";
import { Info, Github, Layers, Shield, Zap, Activity } from "lucide-react";
import { AppContext } from "../context/AppContext";
import { getMetrics } from "../services/api";

const METRIC_INFO = [
  { key: "context_precision", label: "Context Precision", description: "Fraction of retrieved chunks that are semantically relevant to the query.", higherBetter: true },
  { key: "context_recall", label: "Context Recall", description: "How well the retrieved context covers the content of the generated answer.", higherBetter: true },
  { key: "mrr", label: "Mean Reciprocal Rank", description: "Retrieval ranking quality — 1 / rank of the first relevant chunk.", higherBetter: true },
  { key: "context_relevance", label: "Context Relevance", description: "Average semantic similarity between the question and the retrieved chunks.", higherBetter: true },
  { key: "context_sufficiency", label: "Context Sufficiency", description: "Whether the retrieved context is sufficient to produce a complete answer.", higherBetter: true },
  { key: "answer_relevance", label: "Answer Relevance", description: "Semantic similarity between the question and the generated answer.", higherBetter: true },
  { key: "answer_correctness", label: "Answer Correctness", description: "Fraction of question keywords present in the generated answer.", higherBetter: true },
  { key: "latency_ms", label: "Latency", description: "Average end-to-end time to generate a response.", higherBetter: false, isLatency: true },
  { key: "answer_hallucination", label: "Hallucination Rate", description: "Fraction of answer content not grounded in retrieved context. Lower is better.", higherBetter: false },
];

function getBarColor(value, higherBetter, isLatency) {
  if (isLatency) return "text-blue-400 bg-blue-400";
  const score = higherBetter ? value : 1 - value;
  if (score >= 0.7) return "text-emerald-400 bg-emerald-400";
  if (score >= 0.4) return "text-amber-400 bg-amber-400";
  return "text-red-400 bg-red-400";
}

function MetricCard({ info, value }) {
  const isLatency = !!info.isLatency;
  const pct = isLatency ? null : Math.round(value * 100);
  const colorClass = getBarColor(value, info.higherBetter, isLatency);
  const textColor = colorClass.split(' ')[0];
  const bgColorStyle = colorClass.split(' ')[1];

  return (
    <div className="bg-bg-panel border border-border rounded-xl p-5 flex flex-col gap-2 relative overflow-hidden transition-all hover:bg-white/5">
      <div className="text-[11px] text-text-muted uppercase tracking-wider font-semibold">
        {info.label}
      </div>
      <div className={`text-3xl font-bold ${textColor} leading-tight`}>
        {isLatency ? `${Math.round(value)} ms` : `${pct}%`}
      </div>
      {!isLatency && (
        <div className="h-1.5 bg-white/10 rounded-full w-full my-1 overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-1000 ease-out ${bgColorStyle}`} 
            style={{ width: `${pct}%` }} 
          />
        </div>
      )}
      <div className="text-xs text-text-muted leading-relaxed mt-1">
        {info.description}
      </div>
    </div>
  );
}

export default function About() {
  const { documentReady } = useContext(AppContext);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runEval = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Evaluation failed. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
           <div className="bg-bg-panel border border-border rounded-xl p-6 hover:border-primary/30 transition-colors">
             <Layers className="w-8 h-8 text-primary mb-4" />
             <h3 className="text-lg font-semibold text-text-main mb-2">RAG Pipeline</h3>
             <p className="text-text-muted text-sm leading-relaxed">
               Built on a robust Retrieval-Augmented Generation pipeline using FAISS and HuggingFace sentence transformers for ultra-fast, local vector retrieval.
             </p>
           </div>
           
           <div className="bg-bg-panel border border-border rounded-xl p-6 hover:border-yellow-500/30 transition-colors">
             <Zap className="w-8 h-8 text-yellow-500 mb-4" />
             <h3 className="text-lg font-semibold text-text-main mb-2">Adaptive Chunking</h3>
             <p className="text-text-muted text-sm leading-relaxed">
               Smartly processes documents depending on format and density. Presentations get segmented slide-by-slide, while academic PDFs use content density markers.
             </p>
           </div>

           <div className="bg-bg-panel border border-border rounded-xl p-6 hover:border-emerald-500/30 transition-colors">
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

        {/* --- SYSTEM EVALUATION --- */}
        <div className="border-t border-border pt-12 mb-12">
          <div className="flex items-center gap-3 mb-6">
            <Activity className="w-6 h-6 text-primary" />
            <h2 className="text-2xl font-bold text-text-main">System Evaluation (Accuracy)</h2>
          </div>
          
          <p className="text-text-muted text-sm mb-8 max-w-3xl leading-relaxed">
            Run an automated accuracy benchmark over the retrieved documents. This runs a rapid evaluation suite over the Context and Answers provided by the current LLM against your loaded FAISS index.
          </p>

          {!documentReady ? (
            <div className="bg-bg-panel/50 border border-border border-dashed rounded-xl p-6 text-center">
              <p className="text-text-muted text-sm">Please upload a document before running evaluation.</p>
            </div>
          ) : (
            <div className="bg-bg-panel border border-border rounded-xl p-8">
              <button
                onClick={runEval}
                disabled={loading}
                className={`flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold text-sm transition-all sm:w-auto w-full mb-6 ${
                  loading 
                    ? "bg-border text-text-muted cursor-not-allowed" 
                    : "bg-primary text-white hover:bg-primary-hover active:scale-95"
                }`}
              >
                {loading && <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>}
                {loading ? "Running evaluation... (~30-60s)" : metrics ? "Re-run Evaluation" : "Run Evaluation"}
              </button>

              {error && (
                <div className="bg-red-400/10 border border-red-500/20 text-red-400 p-4 rounded-lg text-sm mb-6 animate-in fade-in">
                  {error}
                </div>
              )}

              {metrics && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="flex items-center gap-4 mb-5 text-sm">
                    <span className="text-text-muted font-medium">Score legend:</span>
                    <span className="flex items-center gap-1.5 text-emerald-400"><div className="w-2 h-2 rounded-full bg-emerald-400"></div> Good (&ge;70%)</span>
                    <span className="flex items-center gap-1.5 text-amber-400"><div className="w-2 h-2 rounded-full bg-amber-400"></div> Fair (40-69%)</span>
                    <span className="flex items-center gap-1.5 text-red-400"><div className="w-2 h-2 rounded-full bg-red-400"></div> Poor (&lt;40%)</span>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {METRIC_INFO.map((info) => (
                       <MetricCard key={info.key} info={info} value={metrics[info.key] ?? 0} />
                    ))}
                  </div>

                  <p className="text-xs text-text-muted mt-6 leading-relaxed p-4 bg-white/5 rounded-lg border border-border/50">
                    * Metrics are approximated using token-overlap and embedding cosine similarity — not ground-truth labels. They are indicative of pipeline quality, not absolute scores.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-border pt-8 text-center text-text-muted text-sm">
          <p>© {new Date().getFullYear()} Contexta AI. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}
"""

with open(r"c:\Users\Harish\OneDrive\Desktop\PROJECTS\Contexta-AI\frontend\src\pages\About.jsx", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated About page successfully.")
