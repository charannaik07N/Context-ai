import { useEffect, useState, useContext } from "react";
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
