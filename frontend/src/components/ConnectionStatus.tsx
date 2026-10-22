import { RefreshCw } from "lucide-react";
import type { ConnectionState } from "../types";

interface ConnectionStatusProps {
  state: ConnectionState;
  onReconnect: () => void;
}

const labels: Record<ConnectionState, string> = {
  connecting: "Connecting",
  connected: "Online",
  disconnected: "Offline"
};

export function ConnectionStatus({ state, onReconnect }: ConnectionStatusProps) {
  return (
    <div className={`connection-status connection-status--${state}`}>
      <span className="connection-dot" aria-hidden="true" />
      <span>{labels[state]}</span>
      {state === "disconnected" && (
        <button
          className="icon-button reconnect-button"
          type="button"
          onClick={onReconnect}
          aria-label="Reconnect to the local server"
          title="Reconnect"
        >
          <RefreshCw size={15} />
        </button>
      )}
    </div>
  );
}
