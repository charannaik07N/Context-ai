import axios from "axios";

const API_BASE = "/api";

/* ================= UPLOAD FILE(S) ================= */
export const uploadFiles = async (files) => {
  if (!files || files.length === 0) {
    throw new Error("Please select at least one supported file (.pdf).");
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  // Use batch endpoint for one or many files; backend handles both cases.
  const response = await axios.post(`${API_BASE}/upload-papers`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};

/* ================= RESET INDEX ================= */
export const resetIndex = async () => {
  const response = await axios.delete(`${API_BASE}/reset-index`);
  return response.data;
};

/* ================= ASK QUESTION (POST — avoids URL length limits) ================= */
export const askQuestion = async (question) => {
  const response = await axios.post(`${API_BASE}/ask-question`, { question });
  return response.data;
};

/* ================= DEFINE TERM (POST) ================= */
export const defineTerm = async (term) => {
  const response = await axios.post(`${API_BASE}/define-term`, { term });
  return response.data;
};

/* ================= INSIGHTS ================= */
export const getInsights = async () => {
  const response = await axios.get(`${API_BASE}/insights`);
  return response.data;
};

/* ================= METRICS ================= */
export const getMetrics = async () => {
  const response = await axios.get(`${API_BASE}/metrics`, { timeout: 120000 });
  return response.data;
};
