import { useState, useContext, useRef } from "react";
import { uploadFiles } from "../services/api";
import { AppContext } from "../context/AppContext";
import {
  UploadCloud,
  FileType,
  CheckCircle2,
  AlertCircle,
  X,
} from "lucide-react";

export default function UploadPanel({ onComplete }) {
  const { setDocumentReady, setDocumentNames } = useContext(AppContext);
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files);
    if (selected.length > 0) {
      setFiles(selected);
      handleUpload(selected);
    }
  };

  const handleUpload = async (selectedFiles) => {
    const toUpload = selectedFiles ?? files;
    if (!toUpload.length) {
      setStatus("Please select a file.");
      return;
    }

    try {
      setLoading(true);
      setStatus(`Processing ${toUpload.length} document(s)...`);

      const response = await uploadFiles(toUpload);
      const processedNames = (response?.processed || []).map(
        (item) => item.filename,
      );

      if (processedNames.length > 0) {
        setDocumentNames((prev) => [...new Set([...prev, ...processedNames])]);
        setDocumentReady(true);
        setStatus("Ready!");
        setTimeout(() => {
          if (onComplete) onComplete();
        }, 800);
      } else {
        setStatus("No valid files processed.");
      }
    } catch (error) {
      console.error(error);
      setStatus("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center p-4">
      <div className="bg-bg-panel border border-border rounded-2xl shadow-xl w-full max-w-xl p-8 flex flex-col items-center text-center animate-in zoom-in-95 duration-300">
        <div className="w-16 h-16 bg-primary/10 text-primary rounded-2xl flex items-center justify-center mb-6">
          <UploadCloud className="w-8 h-8" />
        </div>

        <h2 className="text-2xl font-bold text-text-main mb-2">
          Upload your documents
        </h2>
        <p className="text-text-muted mb-8 max-w-md text-sm">
          Upload PDF files to analyze them, extract insights, and ask questions
          naturally.
        </p>

        <div className="w-full">
          <input
            type="file"
            multiple
            accept=".pdf"
            className="hidden"
            ref={fileInputRef}
            onChange={handleFileChange}
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className={`w-full group relative flex flex-col items-center justify-center gap-4 p-8 border-2 border-dashed rounded-xl transition-all ${
              loading
                ? "border-primary/50 bg-primary/5 cursor-not-allowed"
                : "border-border hover:border-primary/50 hover:bg-white/5 cursor-pointer"
            }`}
          >
            {loading ? (
              <>
                <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                <span className="text-sm font-medium text-primary">
                  {status}
                </span>
              </>
            ) : status === "Ready!" ? (
              <>
                <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                <span className="text-sm font-medium text-emerald-500">
                  Processed successfully
                </span>
              </>
            ) : (
              <>
                <div className="flex gap-4 text-text-muted group-hover:text-primary transition-colors">
                  <FileType className="w-8 h-8" />
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span className="text-sm font-semibold text-text-main">
                    Click to browse files
                  </span>
                  <span className="text-xs text-text-muted">PDF supported</span>
                </div>
              </>
            )}
          </button>
        </div>

        {status && !loading && status !== "Ready!" && (
          <div className="mt-6 flex items-center gap-2 text-sm text-red-400 bg-red-400/10 px-4 py-2 rounded-lg">
            <AlertCircle className="w-4 h-4" />
            {status}
          </div>
        )}

        <div className="mt-8 text-xs text-text-muted flex items-center gap-2">
          Secure private processing
        </div>
      </div>
    </div>
  );
}
