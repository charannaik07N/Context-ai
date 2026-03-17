import { useState, useContext } from "react";
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
