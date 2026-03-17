import { NavLink } from "react-router-dom";
import {
  MessageSquare,
  BarChart2,
  BookOpen,
  Info,
  FileText,
} from "lucide-react";

export default function Sidebar() {
  const navItems = [
    { to: "/chat", label: "Chat", icon: <MessageSquare className="w-5 h-5" /> },
    {
      to: "/insights",
      label: "Insights",
      icon: <BarChart2 className="w-5 h-5" />,
    },
    {
      to: "/terminology",
      label: "Terminology",
      icon: <BookOpen className="w-5 h-5" />,
    },
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
