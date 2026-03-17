import { Routes, Route, Navigate } from "react-router-dom";
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
