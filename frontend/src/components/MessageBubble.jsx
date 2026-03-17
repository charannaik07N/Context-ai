import Markdown from "react-markdown";
import { FileText, Cpu, User } from "lucide-react";

export default function MessageBubble({ role, text, sources }) {
  const isUser = role === "user";
  const visibleSources = Array.isArray(sources) ? sources.slice(0, 1) : [];

  return (
    <div
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"} group`}
    >
      {!isUser && (
        <div className="w-8 h-8 flex-shrink-0 rounded-full bg-primary/20 text-primary flex items-center justify-center mr-4 mt-1 border border-primary/20">
          <Cpu className="w-4 h-4" />
        </div>
      )}

      <div className="flex flex-col max-w-[85%]">
        <div
          className={`px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed shadow-sm prose prose-invert prose-p:leading-relaxed prose-pre:bg-bg-dark prose-pre:border prose-pre:border-border max-w-none text-current ${
            isUser
              ? "bg-primary text-white rounded-br-sm"
              : "bg-bg-panel border border-border text-text-main rounded-bl-sm"
          }`}
        >
          <Markdown
            components={{
              p: ({ node, ...props }) => (
                <p className="mb-2 last:mb-0" {...props} />
              ),
              ul: ({ node, ...props }) => (
                <ul className="list-disc pl-4 mb-2 last:mb-0" {...props} />
              ),
              li: ({ node, ...props }) => <li className="mb-1" {...props} />,
            }}
          >
            {text}
          </Markdown>
        </div>

        {!isUser && visibleSources.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-2">
            {visibleSources.map((src, i) => {
              const locationParts = [];
              if (src.page != null && src.page !== "") {
                locationParts.push(`Page ${Number(src.page) + 1}`);
              }
              if (src.slide != null && src.slide !== "") {
                locationParts.push(`Slide ${src.slide}`);
              }
              if (src.section) {
                locationParts.push(String(src.section));
              }

              const fileName = src.file || src.source || "Unknown source";
              const shortName =
                fileName.length > 28 ? fileName.slice(0, 25) + "…" : fileName;

              return (
                <div
                  key={i}
                  className="flex items-start gap-1.5 text-xs bg-bg-panel border border-border px-2 py-1.5 rounded-md text-text-muted hover:text-text-main hover:bg-white/5 transition-colors max-w-sm group-hover:border-primary/30"
                  title={`${fileName}${src.snippet ? `\n\n${src.snippet}` : ""}`}
                >
                  <FileText className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-primary/70" />
                  <div className="flex flex-col overflow-hidden">
                    <span className="font-medium truncate text-text-main/90">
                      {shortName}
                    </span>
                    <span className="text-[10px] opacity-70 truncate">
                      {locationParts.length > 0
                        ? locationParts.join(" · ")
                        : "Reference"}
                    </span>
                  </div>
                </div>
              );
            })}
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
