import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Cpu, X } from "lucide-react";
import { fetchCachedModels, fetchModelFit } from "../lib/api";
import type { ModelFitEntry, ModelFitResponse, ModelOption } from "../types";

function formatGb(value: number | null) {
  return value === null ? "—" : `${value.toFixed(2)} GB`;
}

function verdictLabel(verdict: ModelFitEntry["verdict"]) {
  return {
    fits_gpu: "Fits GPU",
    fits_cpu: "Fits CPU",
    wont_fit: "Too large",
    unknown_size: "Unknown size",
  }[verdict];
}

function formatSpeed(value: number | null) {
  return value === null ? "n/a" : `~${value.toFixed(0)} tok/s`;
}

type ModelFitSortKey = "name" | "verdict" | "need_gb" | "download_gb" | "est_tokens_per_sec";
type ModelFitSort = { key: ModelFitSortKey; direction: "asc" | "desc" };

export function filterAndSortModels(
  models: ModelFitEntry[],
  search: string,
  sort: ModelFitSort | null,
  cachedModelIds: Set<string> = new Set(),
) {
  const query = search.trim().toLocaleLowerCase();
  const filtered = models.filter((model) => !model.downloaded && !cachedModelIds.has(model.id) && [
    model.name,
    verdictLabel(model.verdict),
    formatGb(model.need_gb),
    formatGb(model.download_gb),
    formatSpeed(model.est_tokens_per_sec),
    model.disk_ok === false ? "Low disk" : "",
  ].join(" ").toLocaleLowerCase().includes(query));

  if (!sort) return filtered;
  return filtered.sort((left, right) => {
    const value = (model: ModelFitEntry) => sort.key === "verdict"
      ? verdictLabel(model.verdict)
      : model[sort.key];
    const leftValue = value(left);
    const rightValue = value(right);
    if (leftValue === null) return rightValue === null ? 0 : 1;
    if (rightValue === null) return -1;
    const comparison = typeof leftValue === "string"
      ? leftValue.localeCompare(String(rightValue), undefined, { numeric: true, sensitivity: "base" })
      : Number(leftValue) - Number(rightValue);
    return sort.direction === "asc" ? comparison : -comparison;
  });
}

interface ModelFitTableProps {
  models: ModelFitEntry[];
  sort: ModelFitSort | null;
  onSort: (key: ModelFitSortKey) => void;
  onSelectModel: (modelId: string) => void;
  emptyMessage: string;
}

export function ModelFitTable({ models, sort, onSort, onSelectModel, emptyMessage }: ModelFitTableProps) {
  function sortHeader(key: ModelFitSortKey, label: string) {
    const active = sort?.key === key;
    return (
      <th key={key} scope="col" aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : undefined}>
        <button type="button" className="hardware-fit-sort" onClick={() => onSort(key)}>
          {label}<span aria-hidden="true">{active ? (sort.direction === "asc" ? "↑" : "↓") : "↕"}</span>
        </button>
      </th>
    );
  }

  return (
    <table className="hardware-fit-table">
      <thead>
        <tr>
          {sortHeader("name", "Model")}
          {sortHeader("verdict", "Fit")}
          {sortHeader("need_gb", "Memory")}
          {sortHeader("download_gb", "Download")}
          {sortHeader("est_tokens_per_sec", "Est. speed")}
          <th scope="col">Action</th>
        </tr>
      </thead>
      <tbody>
        {models.map((model) => (
          <tr key={model.id}>
            <td>{model.name}</td>
            <td>
              <span className={`hardware-fit-verdict hardware-fit-verdict--${model.verdict}`}>
                {verdictLabel(model.verdict)}
              </span>
              {model.disk_ok === false && <span className="hardware-fit-disk-warning">Low disk</span>}
            </td>
            <td>{formatGb(model.need_gb)}</td>
            <td>{formatGb(model.download_gb)}</td>
            <td>{formatSpeed(model.est_tokens_per_sec)}</td>
            <td><button type="button" className="hardware-fit-use" aria-label={`Use ${model.name}`} onClick={() => onSelectModel(model.id)}>Use</button></td>
          </tr>
        ))}
        {models.length === 0 && <tr><td colSpan={6}>{emptyMessage}</td></tr>}
      </tbody>
    </table>
  );
}

export function CachedModelsList({
  models,
  activeModelId,
  onSelectModel,
}: {
  models: ModelOption[];
  activeModelId: string;
  onSelectModel: (modelId: string) => void;
}) {
  const chatModels = models.filter((model) => model.supports_chat);
  if (chatModels.length === 0) {
    return <p className="hardware-fit-status">No downloaded chat models yet.</p>;
  }

  return (
    <ul className="hardware-fit-cached-list">
      {chatModels.map((model) => (
        <li className="hardware-fit-cached-model" key={model.id}>
          <div>
            <strong>{model.name}</strong>
            <code>{model.id}</code>
          </div>
          {model.id === activeModelId
            ? <span className="hardware-fit-active">Active</span>
            : (
              <button
                type="button"
                className="hardware-fit-use"
                aria-label={`Use ${model.name}`}
                onClick={() => onSelectModel(model.id)}
              >
                Use
              </button>
            )}
        </li>
      ))}
    </ul>
  );
}

export function HardwareFit({
  onSelectModel,
  activeModelId,
  activeModelName,
}: {
  onSelectModel: (modelId: string) => void;
  activeModelId: string;
  activeModelName?: string;
}) {
  const activeLabel = activeModelName || activeModelId;
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"device" | "find">("device");
  const [fit, setFit] = useState<ModelFitResponse | null>(null);
  const [cachedModels, setCachedModels] = useState<ModelOption[]>([]);
  const [gpuInput, setGpuInput] = useState("");
  const [ramInput, setRamInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingCached, setIsLoadingCached] = useState(false);
  const [error, setError] = useState("");
  const [cachedError, setCachedError] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<ModelFitSort | null>(null);
  const requestId = useRef(0);
  const cachedRequestId = useRef(0);
  const debounce = useRef<number | null>(null);

  const loadFit = useCallback(async (gpuVramGb?: number, ramGb?: number) => {
    const currentRequest = ++requestId.current;
    setIsLoading(true);
    setError("");
    try {
      const result = await fetchModelFit(gpuVramGb, ramGb);
      if (currentRequest === requestId.current) {
        setFit(result);
        setGpuInput(String(result.used.gpu_vram_gb));
        setRamInput(String(result.used.ram_gb));
      }
    } catch (loadError) {
      if (currentRequest === requestId.current) {
        setError(loadError instanceof Error ? loadError.message : "Could not estimate model fit.");
      }
    } finally {
      if (currentRequest === requestId.current) setIsLoading(false);
    }
  }, []);

  const loadCachedModels = useCallback(async () => {
    const currentRequest = ++cachedRequestId.current;
    setIsLoadingCached(true);
    setCachedError("");
    try {
      const result = await fetchCachedModels();
      if (currentRequest === cachedRequestId.current) setCachedModels(result.models);
    } catch (loadError) {
      if (currentRequest === cachedRequestId.current) {
        setCachedError(
          loadError instanceof Error ? loadError.message : "Could not load downloaded models.",
        );
      }
    } finally {
      if (currentRequest === cachedRequestId.current) setIsLoadingCached(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    void loadFit();
    void loadCachedModels();
    return () => {
      requestId.current += 1;
      cachedRequestId.current += 1;
      if (debounce.current !== null) window.clearTimeout(debounce.current);
    };
  }, [isOpen, loadCachedModels, loadFit]);

  function updateHardware(kind: "gpu" | "ram", value: string) {
    const nextGpu = kind === "gpu" ? value : gpuInput;
    const nextRam = kind === "ram" ? value : ramInput;
    if (kind === "gpu") setGpuInput(value);
    else setRamInput(value);
    if (!fit) return;

    const gpu = Number(nextGpu);
    const ram = Number(nextRam);
    const valid = nextGpu.trim() !== "" && nextRam.trim() !== ""
      && Number.isFinite(gpu) && gpu >= 0
      && Number.isFinite(ram) && ram >= 0;
    if (debounce.current !== null) window.clearTimeout(debounce.current);
    if (!valid) {
      requestId.current += 1;
      setIsLoading(false);
      return;
    }

    const gpuOverride = gpu === fit.detected.gpu_vram_gb ? undefined : gpu;
    const ramOverride = ram === fit.detected.ram_gb ? undefined : ram;
    debounce.current = window.setTimeout(() => {
      void loadFit(gpuOverride, ramOverride);
    }, 300);
  }

  const gpuInvalid = gpuInput.trim() === "" || !Number.isFinite(Number(gpuInput)) || Number(gpuInput) < 0;
  const ramInvalid = ramInput.trim() === "" || !Number.isFinite(Number(ramInput)) || Number(ramInput) < 0;
  const cachedModelIds = new Set(cachedModels.map((model) => model.id));
  const visibleModels = fit
    ? filterAndSortModels(fit.models, search, sort, cachedModelIds)
    : [];

  function sortBy(key: ModelFitSortKey) {
    setSort((current) => ({
      key,
      direction: current?.key === key && current.direction === "asc" ? "desc" : "asc",
    }));
  }

  return (
    <div className="model-selection-control">
      <button
        type="button"
        className="hardware-fit-trigger"
        aria-expanded={isOpen}
        aria-controls="model-selection-panel"
        aria-label={`Model selection${activeLabel ? `: ${activeLabel}` : ""}`}
        title={activeLabel || "Model selection"}
        onClick={() => setIsOpen((open) => !open)}
      >
        <Cpu size={15} aria-hidden="true" />
        <span>{activeLabel || "Model selection"}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {isOpen && (
        <section
          id="model-selection-panel"
          className="hardware-fit-panel"
          aria-labelledby="model-selection-title"
        >
          <header className="hardware-fit-heading">
            <div>
              <p className="eyebrow">MODEL SELECTION</p>
              <h2 id="model-selection-title">Choose a model</h2>
            </div>
            <button
              type="button"
              className="icon-button"
              aria-label="Close model selection"
              onClick={() => setIsOpen(false)}
            >
              <X size={18} aria-hidden="true" />
            </button>
          </header>

          <div className="model-selection-tabs" role="tablist" aria-label="Model source">
            <button
              type="button"
              role="tab"
              id="device-models-tab"
              aria-selected={activeTab === "device"}
              aria-controls="device-models-panel"
              onClick={() => setActiveTab("device")}
            >
              On this device
            </button>
            <button
              type="button"
              role="tab"
              id="find-models-tab"
              aria-selected={activeTab === "find"}
              aria-controls="find-models-panel"
              onClick={() => setActiveTab("find")}
            >
              Find models
            </button>
          </div>

          {activeTab === "device" && (
            <div id="device-models-panel" role="tabpanel" aria-labelledby="device-models-tab">
              {isLoadingCached && <p className="hardware-fit-status" role="status">Checking downloaded models…</p>}
              {cachedError && <p className="hardware-fit-error" role="alert">{cachedError}</p>}
              {!isLoadingCached && !cachedError && (
                <CachedModelsList
                  models={cachedModels}
                  activeModelId={activeModelId}
                  onSelectModel={onSelectModel}
                />
              )}
            </div>
          )}

          {activeTab === "find" && (
            <div id="find-models-panel" role="tabpanel" aria-labelledby="find-models-tab">
              {isLoading && <p className="hardware-fit-status" role="status">Checking model fit…</p>}
              {error && <p className="hardware-fit-error" role="alert">{error}</p>}
              {fit && (
                <>
              <div className="hardware-fit-summary">
                <span><strong>Backend</strong>{fit.detected.backend.toUpperCase()}</span>
                <span><strong>GPU</strong>{fit.detected.gpu_name ?? "Not detected"}</span>
                <span><strong>Free disk</strong>{formatGb(fit.detected.disk_free_gb)}</span>
              </div>

              <div className="hardware-fit-inputs">
                <div>
                  <label htmlFor="hardware-fit-vram">GPU VRAM (GB)</label>
                  <input
                    id="hardware-fit-vram"
                    type="number"
                    min="0"
                    step="any"
                    value={gpuInput}
                    aria-invalid={gpuInvalid}
                    aria-describedby={gpuInvalid ? "hardware-fit-vram-error" : undefined}
                    onChange={(event) => updateHardware("gpu", event.target.value)}
                  />
                  {gpuInvalid && <p id="hardware-fit-vram-error" className="hardware-fit-field-error">Enter a finite value of 0 or more.</p>}
                </div>
                <div>
                  <label htmlFor="hardware-fit-ram">System RAM (GB)</label>
                  <input
                    id="hardware-fit-ram"
                    type="number"
                    min="0"
                    step="any"
                    value={ramInput}
                    aria-invalid={ramInvalid}
                    aria-describedby={ramInvalid ? "hardware-fit-ram-error" : undefined}
                    onChange={(event) => updateHardware("ram", event.target.value)}
                  />
                  {ramInvalid && <p id="hardware-fit-ram-error" className="hardware-fit-field-error">Enter a finite value of 0 or more.</p>}
                </div>
              </div>

              {fit.best_fit && <p className="hardware-fit-best">Best fit: <strong>{fit.best_fit}</strong></p>}
              {fit.catalog_source === "fallback" && <p className="hardware-fit-source">Using built-in model list (Hub unavailable).</p>}
              {fit.detected.notes.map((note) => <p className="hardware-fit-note" key={note}>{note}</p>)}

              <label className="hardware-fit-search" htmlFor="hardware-fit-search">
                Search models
                <input
                  id="hardware-fit-search"
                  type="search"
                  value={search}
                  placeholder="Name, fit, memory, download, or speed"
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>

              <div className="hardware-fit-table-wrap">
                <ModelFitTable
                  models={visibleModels}
                  sort={sort}
                  onSort={sortBy}
                  onSelectModel={onSelectModel}
                  emptyMessage={fit.models.length === 0
                    ? "No model estimates available."
                    : fit.models.every((model) => model.downloaded || cachedModelIds.has(model.id))
                      ? "All suggested models are already on this device."
                      : "No models match your search."}
                />
              </div>
              <p className="hardware-fit-disclaimer">Approximate fit and speed estimates; actual throughput varies by model, quantization, and laptop power limits.</p>
                </>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
