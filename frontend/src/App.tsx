import { useCallback, useEffect, useState } from "react";
import { Feather, Sparkles } from "lucide-react";
import { AppShell } from "./components/AppShell";
import { Composer } from "./components/Composer";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { HardwareFit } from "./components/HardwareFit";
import { MessageList } from "./components/MessageList";
import { ModelLoadingOverlay } from "./components/ModelLoadingOverlay";
import { fetchModelLoadStatus, fetchModels, selectActiveModel } from "./lib/api";
import { useChat } from "./hooks/useChat";
import type { ModelLoadStatus, ModelOption } from "./types";

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [activeModel, setActiveModel] = useState("");
  const {
    connection,
    cancelGeneration,
    isCancelling,
    isGenerating,
    messages,
    modelName,
    reconnect,
    sendMessage,
    startNewChat
  } = useChat(activeModel, models.find((model) => model.id === activeModel)?.name);
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [loadingModelName, setLoadingModelName] = useState("");
  const [modelLoadStartedAt, setModelLoadStartedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [modelError, setModelError] = useState("");
  const [modelLoadStatus, setModelLoadStatus] = useState<ModelLoadStatus | null>(null);

  const loadModels = useCallback(async () => {
    try {
      const data = await fetchModels();
      setModels(data.data);
      setActiveModel(data.active_model);
      setModelError("");
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Could not load models.");
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  useEffect(() => {
    if (!isSwitchingModel || modelLoadStartedAt === null) return;

    const updateElapsed = () => {
      setElapsedSeconds(Math.floor((Date.now() - modelLoadStartedAt) / 1000));
    };
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(interval);
  }, [isSwitchingModel, modelLoadStartedAt]);

  useEffect(() => {
    if (!isSwitchingModel) return;
    let active = true;
    const pollStatus = async () => {
      try {
        const status = await fetchModelLoadStatus();
        if (active) setModelLoadStatus(status);
      } catch {
        // Keep the elapsed-time fallback visible if status polling fails.
      }
    };
    void pollStatus();
    const interval = window.setInterval(() => void pollStatus(), 1000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [isSwitchingModel]);

  const elapsedLabel = `${Math.floor(elapsedSeconds / 60)}:${String(elapsedSeconds % 60).padStart(2, "0")}`;

  async function handleModelChange(modelId: string) {
    if (!modelId || modelId === activeModel) return;
    setIsSwitchingModel(true);
    setLoadingModelName(models.find((model) => model.id === modelId)?.name ?? modelId);
    setModelLoadStartedAt(Date.now());
    setElapsedSeconds(0);
    setModelLoadStatus(null);
    setModelError("");
    try {
      const selectedModel = await selectActiveModel(modelId);
      setActiveModel(selectedModel);
      startNewChat();
      await loadModels();
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Could not switch models.");
    } finally {
      setIsSwitchingModel(false);
      setLoadingModelName("");
      setModelLoadStartedAt(null);
    }
  }

  const header = (
    <div className="topbar-controls">
      <div className="topbar-brand"><Sparkles size={19} aria-hidden="true" /> WONDERDESK</div>
      <HardwareFit
        activeModelId={activeModel}
        activeModelName={models.find((model) => model.id === activeModel)?.name}
        onSelectModel={(modelId) => void handleModelChange(modelId)}
      />
      <ConnectionStatus state={connection} onReconnect={reconnect} />
    </div>
  );

  const footer = (
    <footer className="composer-dock">
      {models.some((model) => model.id === activeModel && model.supports_chat === false) && (
        <span className="model-capability-warning" role="status">
          The active model doesn&apos;t support chat. Use /v1/completions or choose a chat model.
        </span>
      )}
      <div className="composer-row">
        <Composer
          disabled={connection !== "connected" || isSwitchingModel}
          isCancelling={isCancelling}
          isGenerating={isGenerating}
          onSend={sendMessage}
          onStop={cancelGeneration}
        />
      </div>
      <p className="footer-note">
        <Feather size={12} />
        An AI model can make mistakes. Keep your curiosity and double-check the facts.
      </p>
    </footer>
  );

  return (
    <AppShell header={header} footer={footer}>
      {modelError && <div className="model-error" role="alert">{modelError}</div>}
      <MessageList
        messages={messages}
        isGenerating={isGenerating}
        modelName={modelName}
        onSuggestion={sendMessage}
      />
      {isSwitchingModel && (
        <ModelLoadingOverlay
          modelName={loadingModelName}
          elapsedLabel={elapsedLabel}
          status={modelLoadStatus}
        />
      )}
    </AppShell>
  );
}
