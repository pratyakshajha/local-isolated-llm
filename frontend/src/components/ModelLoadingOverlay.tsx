import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import type { ModelLoadStatus } from "../types";

interface ModelLoadingOverlayProps {
  modelName: string;
  elapsedLabel: string;
  status: ModelLoadStatus | null;
}

function formatGb(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(1);
}

export function ModelLoadingOverlay({
  modelName,
  elapsedLabel,
  status,
}: ModelLoadingOverlayProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const progressPercent = status?.state === "downloading"
    && Number.isFinite(status.bytes_done)
    && status.bytes_done > 0
    && status.bytes_total !== null
    && Number.isFinite(status.bytes_total)
    && status.bytes_total > 0
    ? Math.min(100, Math.round((status.bytes_done / status.bytes_total) * 100))
    : null;
  const phaseLabel = {
    idle: "Preparing model",
    downloading: "Downloading model files",
    loading: "Loading model into memory",
    ready: "Finishing up",
    failed: "Model preparation failed",
  }[status?.state ?? "idle"];
  const isStalled = status?.state === "downloading"
    && Date.now() / 1000 - status.updated_at >= 30;
  const progressLabel = progressPercent !== null
    ? `${formatGb(status!.bytes_done)} / ${formatGb(status!.bytes_total!)} GB · ${progressPercent}%`
    : `Elapsed ${elapsedLabel}${status?.state === "downloading"
      && status.bytes_total !== null
      && Number.isFinite(status.bytes_total)
      && status.bytes_total > 0
      ? ` · Total ${formatGb(status.bytes_total)} GB`
      : ""}`;

  useEffect(() => dialogRef.current?.focus(), []);

  return (
    <div
      ref={dialogRef}
      className="model-loading-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-loading-title"
      aria-describedby="model-loading-description"
      tabIndex={-1}
      onKeyDown={(event) => {
        if (event.key === "Tab") event.preventDefault();
      }}
    >
      <div className="model-loading-card">
        <div className="model-loading-icon" aria-hidden="true">
          <Sparkles className="model-loading-sparkle" size={18} />
        </div>
        <p className="model-loading-eyebrow">Getting your studio ready</p>
        <h2 id="model-loading-title">Preparing your model</h2>
        <p className="model-loading-name">{modelName}</p>
        <p id="model-loading-description" className="model-loading-description">
          Downloading and loading the model. Large models can take several minutes on the first run.
        </p>
        <div className="model-loading-progress-meta" aria-live="off">
          <span aria-live="polite">{phaseLabel}</span>
          <span>{progressLabel}</span>
        </div>
        <div
          className={`model-loading-progress${progressPercent === null ? "" : " model-loading-progress--determinate"}`}
          role="progressbar"
          aria-label="Model preparation progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progressPercent ?? undefined}
        >
          <span style={progressPercent === null ? undefined : { width: `${progressPercent}%` }} />
        </div>
        {isStalled && (
          <p className="model-loading-stalled" role="status">
            No transfer updates yet. The download may still be running.
          </p>
        )}
        <p className="model-loading-local"><span /> Your model stays on this machine</p>
      </div>
    </div>
  );
}
