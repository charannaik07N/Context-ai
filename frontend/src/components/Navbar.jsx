import { useContext } from "react";
import { AppContext } from "../context/AppContext";
import { resetIndex } from "../services/api";
import { Trash2 } from "lucide-react";

export default function Navbar({ onUploadClick }) {
  const { documentReady, documentName, setDocumentReady, setDocumentNames, setMessages } = useContext(AppContext);

  const handleReset = async () => {
    try {
      await resetIndex();
      setDocumentReady(false);
      setDocumentNames([]);
      setMessages([]);
      localStorage.clear();
    } catch (e) {
      console.error("Reset failed", e);
    }
  };

  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-border bg-bg-panel z-10 flex-shrink-0">
      <div className="flex items-center gap-2">
        {documentReady ? (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-white/5 border border-border text-sm text-text-muted max-w-sm truncate whitespace-nowrap overflow-hidden">
            <span className="w-2 h-2 rounded-full bg-emerald-400 flex-shrink-0"></span>
            <span className="truncate">{documentName}</span>
          </div>
        ) : (
          <span className="text-sm font-medium text-text-muted">No document active</span>
        )}
      </div>

      <div className="flex items-center gap-3">
        {documentReady && (
          <button
            onClick={onUploadClick}
            className="text-sm px-4 py-2 border border-border bg-transparent text-text-main hover:bg-white/5 rounded-md transition-colors font-medium"
          >
            Upload New
          </button>
        )}
        {documentReady && (
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-sm px-3 py-2 border border-red-900/50 text-red-400 hover:bg-red-500/10 rounded-md transition-colors"
            title="Clear all documents"
          >
            <Trash2 className="w-4 h-4" />
            Reset
          </button>
        )}
      </div>
    </header>
  );
}
