import axios from "axios";

const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL || "").trim();
const configuredBackendUrl = (import.meta.env.VITE_BACKEND_URL || "").trim();
const defaultBackendUrl = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 60000;

const unique = (arr) => [...new Set(arr.filter(Boolean))];
const stripTrailingSlash = (value) => value.replace(/\/+$/, "");

// Try proxy path first, then direct backend paths so the app works
// whether Vite proxy is running or frontend is served standalone.
const API_BASE_CANDIDATES = unique([
  configuredBaseUrl || "/api",
  "/",
  configuredBackendUrl ? stripTrailingSlash(configuredBackendUrl) : "",
  stripTrailingSlash(defaultBackendUrl),
]);

const authHeaders = () => {
  const headers = {};
  const clientKey = import.meta.env.VITE_CLIENT_KEY;
  if (clientKey) {
    headers["X-Client-Key"] = clientKey;
  }
  return headers;
};

const normalizePath = (path) => {
  if (!path) return "/";
  return path.startsWith("/") ? path : `/${path}`;
};

const joinBaseAndPath = (base, path) => {
  const normalizedPath = normalizePath(path);
  if (base === "/") return normalizedPath;
  return `${stripTrailingSlash(base)}${normalizedPath}`;
};

const shouldTryNextBase = (error) => {
  // Network-level failures (e.g. ECONNREFUSED from Vite proxy) don't include a response.
  if (!error?.response) {
    return true;
  }
  const status = error?.response?.status;

  // Retry on route-level 404 only. Semantic API 404 responses should be surfaced directly.
  if (status === 404) {
    const detail = error?.response?.data?.detail;
    return detail === "Not Found";
  }

  // 502/503/504: proxy/backend bridge failure.
  return [502, 503, 504].includes(status);
};

const requestWithFallback = async (method, path, data, config = {}) => {
  let lastError;

  for (const base of API_BASE_CANDIDATES) {
    const url = joinBaseAndPath(base, path);
    try {
      const response = await axios({
        method,
        url,
        data,
        timeout: DEFAULT_TIMEOUT_MS,
        ...config,
      });
      return response.data;
    } catch (error) {
      lastError = error;
      if (!shouldTryNextBase(error)) {
        throw error;
      }
    }
  }

  throw lastError;
};

/* ================= UPLOAD FILE(S) ================= */
export const uploadFiles = async (files) => {
  if (!files || files.length === 0) {
    throw new Error(
      "Please select at least one supported file (.pdf, .docx, .txt, .html, .htm).",
    );
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  // Use batch endpoint for one or many files; backend handles both cases.
  return requestWithFallback("post", "/upload-papers", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      ...authHeaders(),
    },
  });
};

/* ================= RESET INDEX ================= */
export const resetIndex = async () => {
  return requestWithFallback("delete", "/reset-index", undefined, {
    headers: authHeaders(),
  });
};

/* ================= ASK QUESTION (POST — avoids URL length limits) ================= */
export const askQuestion = async (question) => {
  return requestWithFallback(
    "post",
    "/ask-question",
    { question },
    {
      headers: authHeaders(),
    },
  );
};

/* ================= DEFINE TERM (POST) ================= */
export const defineTerm = async (term) => {
  return requestWithFallback(
    "post",
    "/define-term",
    { term },
    {
      headers: authHeaders(),
    },
  );
};

/* ================= INSIGHTS ================= */
export const getInsights = async () => {
  return requestWithFallback("get", "/insights", undefined, {
    headers: authHeaders(),
  });
};

/* ================= METRICS ================= */
export const getMetrics = async () => {
  return requestWithFallback("get", "/metrics", undefined, {
    timeout: 120000,
    headers: authHeaders(),
  });
};

/* ================= INDEX STATUS ================= */
export const getIndexStatus = async () => {
  return requestWithFallback("get", "/index-status", undefined, {
    headers: authHeaders(),
  });
};
