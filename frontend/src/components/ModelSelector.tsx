import type { ModelOption } from "../types";

interface ModelSelectorProps {
  activeModel: string;
  isLoading: boolean;
  models: ModelOption[];
  onChange: (modelId: string) => void;
}

export function ModelSelector({
  activeModel,
  isLoading,
  models,
  onChange
}: ModelSelectorProps) {
  const options = activeModel && !models.some((model) => model.id === activeModel)
    ? [{ id: activeModel, name: activeModel, runtime: "transformers", active: true }, ...models]
    : models;

  return (
    <div className="model-control">
      <label htmlFor="model-select">MODEL</label>
      <select
        id="model-select"
        value={activeModel}
        onChange={(event) => onChange(event.target.value)}
        disabled={isLoading || options.length === 0}
        aria-label="Select active model"
      >
        {options.map((model) => (
          <option key={model.id} value={model.id}>{model.name}</option>
        ))}
      </select>
      {isLoading && <span className="switching-label">Loading…</span>}
    </div>
  );
}
