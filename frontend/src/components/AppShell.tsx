import type { ReactNode } from "react";

interface AppShellProps {
  header: ReactNode;
  children: ReactNode;
  footer: ReactNode;
}

export function AppShell({ header, children, footer }: AppShellProps) {
  return (
    <div className="app-shell">
      <main className="chat-main">
        <header className="topbar">{header}</header>
        {children}
        {footer}
      </main>
    </div>
  );
}
