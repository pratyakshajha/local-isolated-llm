export type ChatRole = "user" | "assistant" | "error";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
}

export type ConnectionState = "connecting" | "connected" | "disconnected";

export interface ModelOption {
  id: string;
  name: string;
  runtime: string;
  active: boolean;
}

export interface ModelsResponse {
  active_model: string;
  loaded: boolean;
  models: ModelOption[];
}

export type ServerEvent =
  | { type: "ready"; model: string }
  | { type: "assistant_message"; model: string; content: string }
  | { type: "error"; content: string }
  | { type: "generation_cancelled" };

export interface UserMessageEvent {
  type: "user_message";
  content: string;
}
