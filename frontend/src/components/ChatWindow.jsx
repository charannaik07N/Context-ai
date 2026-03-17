import { useState, useContext, useRef, useEffect } from "react";
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
