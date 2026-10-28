import { Children, isValidElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
// @ts-expect-error Vitest runs in Node, but the app intentionally omits Node types.
import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import type { ModelFitEntry, ModelOption } from "../types";
import {
  CachedModelsList,
  filterAndSortModels,
  HardwareFit,
  ModelFitTable,
} from "./HardwareFit";

const models: ModelFitEntry[] = [
  {
    id: "small",
    name: "Qwen 3B",
    params_billion: 3,
    download_gb: 2,
    need_gb: 4,
    verdict: "fits_gpu",
    disk_ok: true,
    est_tokens_per_sec: 64,
    downloaded: true,
  },
  {
    id: "large",
    name: "Gemma 9B",
    params_billion: 9,
    download_gb: 8,
    need_gb: 12,
    verdict: "wont_fit",
    disk_ok: false,
    est_tokens_per_sec: 21.3,
    downloaded: false,
  },
  {
    id: "unknown",
    name: "Unknown model",
    params_billion: null,
    download_gb: null,
    need_gb: null,
    verdict: "unknown_size",
    disk_ok: null,
    est_tokens_per_sec: null,
    downloaded: false,
  },
];

const cachedModels: ModelOption[] = [
  {
    id: "small",
    name: "Qwen 3B",
    runtime: "transformers",
    supports_chat: true,
    active: false,
  },
  {
    id: "code",
    name: "CodeGemma",
    runtime: "transformers",
    supports_chat: false,
    active: false,
  },
  {
    id: "other",
    name: "Qwen 7B",
    runtime: "transformers",
    supports_chat: true,
    active: false,
  },
];

function findButton(node: ReactNode, label: string): ReactNode | undefined {
  for (const child of Children.toArray(node)) {
    if (!isValidElement(child)) continue;
    if (child.type === "button" && child.props["aria-label"] === label) return child;
    const nested = findButton(child.props.children, label);
    if (nested) return nested;
  }
}

describe("filterAndSortModels", () => {
  it("searches model names and visible fit details without case sensitivity", () => {
    expect(filterAndSortModels(models, "GEMMA", null).map(({ id }) => id)).toEqual(["large"]);
    expect(filterAndSortModels(models, "low disk", null).map(({ id }) => id)).toEqual(["large"]);
    expect(filterAndSortModels(models, "n/a", null).map(({ id }) => id)).toEqual(["unknown"]);
  });

  it("sorts numeric values and leaves unknown values last in either direction", () => {
    expect(
      filterAndSortModels(models, "", { key: "est_tokens_per_sec", direction: "desc" })
        .map(({ id }) => id),
    ).toEqual(["large", "unknown"]);
  });

  it("excludes downloaded or cached models from model discovery", () => {
    expect(filterAndSortModels(models, "", null, new Set(["large"])).map(({ id }) => id))
      .toEqual(["unknown"]);
    expect(filterAndSortModels(models, "", null, new Set(models.map(({ id }) => id))))
      .toEqual([]);
  });

  it("renders a Use button that selects a discovered model", () => {
    const onSelectModel = vi.fn();
    const table = ModelFitTable({
      models: models.filter(({ downloaded }) => !downloaded),
      sort: null,
      onSort: vi.fn(),
      onSelectModel,
      emptyMessage: "",
    });
    const markup = renderToStaticMarkup(table);
    expect(markup).toContain(">Use</button>");

    const button = findButton(table, "Use Gemma 9B");
    if (!isValidElement(button)) throw new Error("Use button was not rendered");
    button.props.onClick();
    expect(onSelectModel).toHaveBeenCalledWith("large");
  });

  it("marks the active cached model and only offers Use for other models", () => {
    const onSelectModel = vi.fn();
    const list = CachedModelsList({
      models: cachedModels,
      activeModelId: "small",
      onSelectModel,
    });
    const markup = renderToStaticMarkup(list);

    expect(markup).toContain("Qwen 3B");
    expect(markup).toContain("Active");
    expect(markup).not.toContain("CodeGemma");
    expect(findButton(list, "Use Qwen 3B")).toBeUndefined();
    const button = findButton(list, "Use Qwen 7B");
    if (!isValidElement(button)) throw new Error("Inactive cached model Use button was not rendered");
    button.props.onClick();
    expect(onSelectModel).toHaveBeenCalledWith("other");
  });

  it("shows an empty state when there are no downloaded chat models", () => {
    const list = CachedModelsList({
      models: cachedModels.filter(({ supports_chat }) => !supports_chat),
      activeModelId: "code",
      onSelectModel: vi.fn(),
    });

    expect(renderToStaticMarkup(list)).toContain("No downloaded chat models yet.");
  });
});

describe("HardwareFit", () => {
  it("widens the discovery popup and keeps the model column compact", () => {
    const styles = readFileSync(new URL("../styles.css", import.meta.url), "utf8");
    expect(styles).toMatch(/\.hardware-fit-panel\s*\{[^}]*width:\s*min\(720px,\s*calc\(100vw - 48px\)\)/);
    expect(styles).toMatch(/\.hardware-fit-panel\s*\{[^}]*left:\s*50%;[^}]*transform:\s*translateX\(-50%\)/);
    expect(styles).toMatch(/\.hardware-fit-table th:first-child, \.hardware-fit-table td:first-child\s*\{\s*width:\s*160px;/);
    expect(styles).toMatch(/\.hardware-fit-table-wrap\s*\{[^}]*overflow-x:\s*auto/);
    expect(styles).toMatch(/\.hardware-fit-panel\s*\{\s*position:\s*fixed;[^}]*transform:\s*none/);
  });

  it("shows the active model name in the selection trigger", () => {
    const markup = renderToStaticMarkup(
      <HardwareFit
        activeModelId="Qwen/Qwen3-4B"
        activeModelName="Qwen 3B"
        onSelectModel={vi.fn()}
      />,
    );

    expect(markup).toContain(">Qwen 3B</span>");
    expect(markup).toContain('aria-label="Model selection: Qwen 3B"');
    expect(markup).not.toContain(">Model selection</span>");
  });
});
