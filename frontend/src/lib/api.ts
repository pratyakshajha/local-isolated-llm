import type { CachedModelsResponse, ModelFitResponse, ModelsResponse, ModelLoadStatus } from "../types";

export async function responseError(response: Response): Promise<Error> {
  try {
    const data = (await response.json()) as { detail?: string };
    if (data.detail) return new Error(data.detail);
  } catch {
    // Fall back to the HTTP status when the response body isn't JSON.
  }
  return new Error(`${response.status} ${response.statusText}`);
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch("/v1/models");
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as ModelsResponse;
}

export async function fetchModelLoadStatus(): Promise<ModelLoadStatus> {
  const response = await fetch("/api/models/active/status");
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as ModelLoadStatus;
}

export async function fetchCachedModels(): Promise<CachedModelsResponse> {
  const response = await fetch("/api/models/cached");
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as CachedModelsResponse;
}

export async function fetchModelFit(
  gpuVramGb?: number,
  ramGb?: number,
): Promise<ModelFitResponse> {
  const query = new URLSearchParams();
  if (gpuVramGb !== undefined) query.set("gpu_vram_gb", String(gpuVramGb));
  if (ramGb !== undefined) query.set("ram_gb", String(ramGb));
  const suffix = query.size ? `?${query}` : "";
  const response = await fetch(`/api/model-fit${suffix}`);
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as ModelFitResponse;
}

export async function selectActiveModel(modelId: string): Promise<string> {
  const response = await fetch("/v1/models/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId })
  });
  if (!response.ok) throw await responseError(response);
  const data = (await response.json()) as { active_model?: string };
  if (!data.active_model) throw new Error("The server did not return the selected model.");
  return data.active_model;
}
