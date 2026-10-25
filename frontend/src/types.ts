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
  data: ModelOption[];
}
