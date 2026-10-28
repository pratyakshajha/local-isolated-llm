import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ModelLoadingOverlay } from "./ModelLoadingOverlay";
import type { ModelLoadStatus } from "../types";

function makeStatus(overrides: Partial<ModelLoadStatus> = {}): ModelLoadStatus {
  return {
    state: "downloading",
    model_id: "google/gemma-3-1b-it",
    bytes_done: 0,
    bytes_total: null,
    updated_at: Date.now() / 1000,
    error: null,
    ...overrides,
  };
}

function renderOverlay(status: ModelLoadStatus) {
  return renderToStaticMarkup(
    <ModelLoadingOverlay
      modelName="Gemma 3 1B"
      elapsedLabel="0:42"
      status={status}
    />,
  );
}

describe("ModelLoadingOverlay", () => {
  it("shows indeterminate activity and total without zero-percent progress", () => {
    const html = renderOverlay(
      makeStatus({ bytes_total: 3.3 * 1024 ** 3 }),
    );

    expect(html).toContain("Downloading model files");
    expect(html).toContain("3.3 GB");
    expect(html).toContain("Elapsed 0:42");
    expect(html).toContain('role="progressbar"');
    expect(html).not.toContain("aria-valuenow");
    expect(html).not.toContain("0%");
    expect(html).not.toContain("model-loading-progress--determinate");
  });

  it("shows indeterminate activity when the total size is unknown", () => {
    const html = renderOverlay(makeStatus({ bytes_total: null }));

    expect(html).toContain("Downloading model files");
    expect(html).toContain("Elapsed 0:42");
    expect(html).not.toContain("NaN");
    expect(html).not.toContain("0%");
    expect(html).not.toContain("aria-valuenow");
  });

  it("shows determinate progress after bytes have transferred", () => {
    const html = renderOverlay(
      makeStatus({
        bytes_done: 1.2 * 1024 ** 3,
        bytes_total: 2 * 1024 ** 3,
        updated_at: Date.now() / 1000 - 31,
      }),
    );

    expect(html).toContain("1.2 / 2.0 GB · 60%");
    expect(html).toContain('aria-valuenow="60"');
    expect(html).toContain("No transfer updates yet");
  });

  it("keeps invalid byte counts indeterminate", () => {
    for (const counts of [
      { bytes_done: 1, bytes_total: Infinity },
      { bytes_done: Infinity, bytes_total: 2 * 1024 ** 3 },
      { bytes_done: 1, bytes_total: 0 },
    ]) {
      const html = renderOverlay(makeStatus(counts));

      expect(html).not.toContain("Infinity");
      expect(html).not.toContain("0%");
      expect(html).not.toContain("aria-valuenow");
      expect(html).not.toContain("model-loading-progress--determinate");
    }
  });

  it("shows the loading phase without stale download progress or hint", () => {
    const html = renderOverlay(
      makeStatus({
        state: "loading",
        bytes_done: 1.2 * 1024 ** 3,
        bytes_total: 2 * 1024 ** 3,
        updated_at: Date.now() / 1000 - 60,
      }),
    );

    expect(html).toContain("Loading model into memory");
    expect(html).toContain("Elapsed 0:42");
    expect(html).not.toContain("1.2 / 2.0 GB");
    expect(html).not.toContain("aria-valuenow");
    expect(html).not.toContain("No transfer updates yet");
    expect(html).toContain('aria-label="Model preparation progress"');
  });

  it("shows finishing up while the switch request is still pending", () => {
    const html = renderOverlay(makeStatus({ state: "ready" }));

    expect(html).toContain('role="dialog"');
    expect(html).toContain("Finishing up");
    expect(html).not.toContain("No transfer updates yet");
    expect(html).toContain('aria-label="Model preparation progress"');
  });

  it("shows the no-transfer hint only after 30 seconds of downloading", () => {
    const html = renderOverlay(
      makeStatus({ updated_at: Date.now() / 1000 - 30 }),
    );

    expect(html).toContain("No transfer updates yet. The download may still be running.");
    expect(html).not.toContain("The download appears stalled");
  });
});
