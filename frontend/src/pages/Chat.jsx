import { useContext, useState } from "react";
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
