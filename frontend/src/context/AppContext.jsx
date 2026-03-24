import { createContext, useState, useEffect } from "react";
import { getIndexStatus } from "../services/api";

export const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [documentReady, setDocumentReady] = useState(
    localStorage.getItem("documentReady") === "true",
  );

  // Array of individual uploaded file names
  const [documentNames, setDocumentNames] = useState(() => {
    try {
      const saved = localStorage.getItem("documentNames");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Computed single-string name for backward compat (e.g. Insights cache key)
  const documentName =
    documentNames.length > 0 ? documentNames.join(", ") : null;

  // Keep legacy key in sync so Insights cache still works
  const setDocumentName = () => {}; // no-op – derive from documentNames

  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem("chatMessages");
      return saved ? JSON.parse(saved) : [];
    } catch {
      localStorage.removeItem("chatMessages");
      return [];
    }
  });

  // Persist document state
  useEffect(() => {
    localStorage.setItem("documentReady", documentReady);
  }, [documentReady]);

  useEffect(() => {
    localStorage.setItem("documentNames", JSON.stringify(documentNames));
    // Keep legacy key in sync for Insights cache
    if (documentNames.length > 0) {
      localStorage.setItem("documentName", documentNames.join(", "));
    }
  }, [documentNames]);

  // 🔹 Persist chat
  useEffect(() => {
    localStorage.setItem("chatMessages", JSON.stringify(messages));
  }, [messages]);

  // Keep local UI state aligned with backend namespace index readiness.
  useEffect(() => {
    let active = true;

    const syncIndexStatus = async () => {
      try {
        const result = await getIndexStatus();
        if (!active) return;
        const ready = Boolean(result?.ready);
        setDocumentReady(ready);
      } catch {
        // Ignore transient API errors here and preserve current UI state.
      }
    };

    syncIndexStatus();

    return () => {
      active = false;
    };
  }, []);

  return (
    <AppContext.Provider
      value={{
        documentReady,
        setDocumentReady,
        documentName,
        setDocumentName,
        documentNames,
        setDocumentNames,
        messages,
        setMessages,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
