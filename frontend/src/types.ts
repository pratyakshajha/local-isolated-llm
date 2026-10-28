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
  supports_chat: boolean;
  active: boolean;
}

export interface ModelsResponse {
  active_model: string;
  loaded: boolean;
  data: ModelOption[];
}

export interface ModelLoadStatus {
  state: "idle" | "downloading" | "loading" | "ready" | "failed";
  model_id: string;
  bytes_done: number;
  bytes_total: number | null;
  updated_at: number;
  error: string | null;
}

export interface CachedModelsResponse {
  active_model: string;
  models: ModelOption[];
}

export interface DetectedHardware {
  backend: string;
  gpu_name: string | null;
  gpu_vram_gb: number;
  ram_gb: number;
  disk_free_gb: number;
  notes: string[];
}

export interface ModelFitEntry {
  id: string;
  name: string;
  params_billion: number | null;
  download_gb: number | null;
  need_gb: number | null;
  verdict: "fits_gpu" | "fits_cpu" | "wont_fit" | "unknown_size";
  disk_ok: boolean | null;
  est_tokens_per_sec: number | null;
  downloaded: boolean;
}

export interface ModelFitResponse {
  detected: DetectedHardware;
  used: Pick<DetectedHardware, "gpu_vram_gb" | "ram_gb">;
  catalog_source: "hub" | "fallback";
  best_fit: string | null;
  models: ModelFitEntry[];
}
