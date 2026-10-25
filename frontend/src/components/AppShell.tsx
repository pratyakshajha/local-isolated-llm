import type { ReactNode } from "react";
import { BookOpen, LockKeyhole, Plus, Sparkles } from "lucide-react";

interface AppShellProps {
  header: ReactNode;
  children: ReactNode;
  footer: ReactNode;
  onNewChat: () => void;
}

export function AppShell({ header, children, footer, onNewChat }: AppShellProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><Sparkles size={22} strokeWidth={2.5} /></div>
          <div>
            <div className="brand-name">WONDERDESK</div>
            <div className="brand-caption">your local idea machine</div>
          </div>
        </div>

        <button className="new-chat-button" type="button" onClick={onNewChat}>
          <Plus size={18} />
          <span>New chat</span>
          <span className="new-chat-sparkle" aria-hidden="true">✦</span>
        </button>

        <div className="sidebar-section-label">YOUR STUDIO</div>
        <div className="sidebar-empty">
          <BookOpen size={17} />
          <span>Chat history isn’t saved yet.</span>
        </div>

        <div className="sidebar-footer">
          <div className="privacy-stamp"><LockKeyhole size={16} /></div>
          <div>
            <strong>Made to stay local</strong>
            <span>Your model runs on this machine.</span>
          </div>
        </div>
      </aside>

      <main className="chat-main">
        <header className="topbar">{header}</header>
        {children}
        {footer}
      </main>
    </div>
  );
}
