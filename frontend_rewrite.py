import os

frontend_dir = os.path.join("frontend", "src")

files = {
    "styles/index.css": """@import "tailwindcss";

@theme {
  --color-bg-dark: #0f1117;
  --color-bg-panel: #161b22;
  --color-primary: #3b82f6;
  --color-primary-hover: #2563eb;
  --color-text-main: #f3f4f6;
  --color-text-muted: #9ca3af;
  --color-border: #30363d;
}

body {
  @apply bg-bg-dark text-text-main font-sans antialiased;
  margin: 0;
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  @apply bg-border rounded-full hover:bg-gray-600;
}
""",
    "App.jsx": """import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Chat from "./pages/Chat";
import Insights from "./pages/Insights";
import Terminology from "./pages/Terminology";
import About from "./pages/About";
import { AppProvider } from "./context/AppContext";

export default function App() {
  return (
    <AppProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<Layout><Chat /></Layout>} />
        <Route path="/insights" element={<Layout><Insights /></Layout>} />
        <Route path="/terminology" element={<Layout><Terminology /></Layout>} />
        <Route path="/about" element={<Layout><About /></Layout>} />
      </Routes>
    </AppProvider>
  );
}
""",
    "components/Layout.jsx": """import Sidebar from "./Sidebar";

export default function Layout({ children }) {
  return (
    <div className="flex h-screen bg-bg-dark text-text-main overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        {children}
      </div>
    </div>
  );
}
""",
    "components/Sidebar.jsx": """import { NavLink } from "react-router-dom";
import { MessageSquare, BarChart2, BookOpen, Info, FileText } from "lucide-react";

export default function Sidebar() {
  const navItems = [
    { to: "/chat", label: "Chat", icon: <MessageSquare className="w-5 h-5" /> },
    { to: "/insights", label: "Insights", icon: <BarChart2 className="w-5 h-5" /> },
    { to: "/terminology", label: "Terminology", icon: <BookOpen className="w-5 h-5" /> },
    { to: "/documents", label: "Documents", icon: <FileText className="w-5 h-5" /> }, // Future use if needed
    { to: "/about", label: "About", icon: <Info className="w-5 h-5" /> },
  ];

  return (
    <aside className="w-64 bg-bg-panel border-r border-border flex flex-col p-4 flex-shrink-0">
      <div className="flex items-center gap-3 mb-8 px-2 font-bold text-xl text-text-main">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-white">
          <BookOpen className="w-5 h-5" />
        </div>
        Contexta AI
      </div>

      <nav className="flex flex-col gap-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-text-muted hover:text-text-main hover:bg-white/5"
              }`
            }
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
""",
    "components/Navbar.jsx": """import { useContext } from "react";
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
""",
    "components/ChatWindow.jsx": """import { useState, useContext, useRef, useEffect } from "react";
import { askQuestion } from "../services/api";
import MessageBubble from "./MessageBubble";
import { AppContext } from "../context/AppContext";
import { Send, Sparkles } from "lucide-react";

export default function ChatWindow() {
  const { documentReady, messages, setMessages } = useContext(AppContext);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const typingIntervalRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const typeAnswer = (fullText, sources = []) => {
    let index = 0;
    let currentText = "";
    if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);

    typingIntervalRef.current = setInterval(() => {
      if (index >= fullText.length) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
        return;
      }
      currentText += fullText[index];
      index++;
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "ai", text: currentText, sources };
        return updated;
      });
    }, 15);
  };

  const sendMessage = async (overrideInput) => {
    const messageText = overrideInput || input;
    if (!documentReady || !messageText.trim() || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: messageText }]);
    setLoading(true);

    try {
      const response = await askQuestion(messageText);
      const fullAnswer = response.answer || "No answer returned.";
      const sourceList = Array.isArray(response.sources) ? response.sources : [];
      setMessages((prev) => [...prev, { role: "ai", text: "", sources: sourceList }]);
      typeAnswer(fullAnswer, sourceList);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "ai", text: "⚠️ Error getting answer from backend.", sources: [] }]);
    } finally {
      setLoading(false);
    }
  };

  const suggestedQuestions = [
    "Summarize the key points",
    "What are the main findings?",
    "Explain the methodology"
  ];

  return (
    <div className="flex-1 flex flex-col bg-bg-dark h-[calc(100vh-64px)]">
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 w-full max-w-4xl mx-auto flex flex-col gap-6">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center max-w-2xl mx-auto h-full opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="w-12 h-12 bg-primary/10 text-primary rounded-xl flex items-center justify-center mb-6">
              <Sparkles className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold text-text-main mb-3">Ask about your document</h2>
            <p className="text-text-muted mb-8 text-sm max-w-md">
              Your document is ready. You can ask for a summary, extract specific details, or clarify complex topics.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full">
              {suggestedQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="px-4 py-3 rounded-xl border border-border bg-bg-panel text-sm text-text-main hover:border-primary hover:bg-primary/5 transition-all text-left"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, index) => (
              <MessageBubble key={index} role={msg.role} text={msg.text} sources={msg.sources || []} />
            ))}
            {loading && (
              <div className="flex w-full items-end justify-start animate-pulse">
                <div className="w-8 h-8 rounded-full bg-bg-panel border border-border mr-4 mt-2"></div>
                <div className="px-5 py-3.5 rounded-2xl bg-bg-panel border border-border text-text-muted text-sm shadow-sm rounded-bl-sm">
                  Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="p-4 bg-bg-dark/80 backdrop-blur-md border-t border-border flex justify-center sticky bottom-0 z-10 w-full">
        <div className="flex gap-2 w-full max-w-4xl bg-bg-panel border border-border rounded-xl p-1.5 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 transition-all shadow-lg">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about the document..."
            disabled={!documentReady || loading}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            className="flex-1 bg-transparent border-none text-text-main px-4 py-2 text-[15px] outline-none placeholder:text-text-muted/60 disabled:opacity-50"
            autoFocus
          />
          <button
            onClick={() => sendMessage()}
            disabled={!documentReady || loading || !input.trim()}
            className="bg-primary text-white border-none p-2.5 rounded-lg font-medium text-sm cursor-pointer transition-colors hover:bg-primary-hover disabled:bg-border disabled:text-text-muted/50 disabled:cursor-not-allowed flex items-center justify-center aspect-square"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
""",
    "components/MessageBubble.jsx": """import Markdown from "react-markdown";
import { FileText, Cpu, User } from "lucide-react";

export default function MessageBubble({ role, text, sources }) {
  const isUser = role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} group`}>
      {!isUser && (
        <div className="w-8 h-8 flex-shrink-0 rounded-full bg-primary/20 text-primary flex items-center justify-center mr-4 mt-1 border border-primary/20">
          <Cpu className="w-4 h-4" />
        </div>
      )}
      
      <div className="flex flex-col max-w-[85%]">
        <div
          className={`px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed shadow-sm ${
            isUser
              ? "bg-primary text-white rounded-br-sm"
              : "bg-bg-panel border border-border text-text-main rounded-bl-sm"
          }`}
        >
          <Markdown
            className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-bg-dark prose-pre:border prose-pre:border-border max-w-none text-current"
            components={{
              p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
              ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2 last:mb-0" {...props} />,
              li: ({node, ...props}) => <li className="mb-1" {...props} />
            }}
          >
            {text}
          </Markdown>
        </div>

        {!isUser && sources && sources.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-2">
            {sources.map((src, i) => (
              <div 
                key={i} 
                className="flex items-start gap-1.5 text-xs bg-bg-panel border border-border px-2 py-1.5 rounded-md text-text-muted hover:text-text-main hover:bg-white/5 transition-colors max-w-sm group-hover:border-primary/30"
                title={src.snippet}
              >
                <FileText className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-primary/70" />
                <div className="flex flex-col overflow-hidden">
                  <span className="font-medium truncate text-text-main/90">{src.file}</span>
                  <span className="text-[10px] opacity-70 truncate">
                    {src.page && `Page ${src.page}`}
                    {src.section && (src.page ? ` · ${src.section}` : src.section)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 flex-shrink-0 rounded-full bg-border text-text-muted flex items-center justify-center ml-4 mt-1">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  );
}
""",
    "components/UploadPanel.jsx": """import { useState, useContext, useRef } from "react";
import { uploadFiles } from "../services/api";
import { AppContext } from "../context/AppContext";
import { UploadCloud, FileType, CheckCircle2, AlertCircle, X } from "lucide-react";

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
      const processedNames = (response?.processed || []).map((item) => item.filename);
      
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

        <h2 className="text-2xl font-bold text-text-main mb-2">Upload your documents</h2>
        <p className="text-text-muted mb-8 max-w-md text-sm">
          Upload PDF files to analyze them, extract insights, and ask questions naturally.
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
                <span className="text-sm font-medium text-primary">{status}</span>
              </>
            ) : status === "Ready!" ? (
              <>
                <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                <span className="text-sm font-medium text-emerald-500">Processed successfully</span>
              </>
            ) : (
              <>
                <div className="flex gap-4 text-text-muted group-hover:text-primary transition-colors">
                  <FileType className="w-8 h-8" />
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span className="text-sm font-semibold text-text-main">Click to browse files</span>
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
""",
    "pages/Chat.jsx": """import { useContext, useState } from "react";
import UploadPanel from "../components/UploadPanel";
import ChatWindow from "../components/ChatWindow";
import Navbar from "../components/Navbar";
import { AppContext } from "../context/AppContext";
import { X } from "lucide-react";

export default function Chat() {
  const { documentReady } = useContext(AppContext);
  const [showUploadModal, setShowUploadModal] = useState(false);

  return (
    <div className="flex flex-col h-full w-full relative">
      <Navbar onUploadClick={() => setShowUploadModal(true)} />
      
      {documentReady ? (
        <ChatWindow />
      ) : (
        <UploadPanel />
      )}

      {/* Upload Modal for adding more docs when already ready */}
      {showUploadModal && documentReady && (
        <div className="absolute inset-0 bg-bg-dark/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="relative w-full max-w-xl">
             <button 
                onClick={() => setShowUploadModal(false)}
                className="absolute -top-12 right-0 text-text-muted hover:text-white p-2"
             >
                <X className="w-6 h-6" />
             </button>
             <UploadPanel onComplete={() => setShowUploadModal(false)} />
          </div>
        </div>
      )}
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

print("Rewrite complete successfully.")
