import { useCallback, useEffect, useRef, useState } from "react";
import { Feather, Plus, Sparkles } from "lucide-react";
import { AppShell } from "./components/AppShell";
import { Composer } from "./components/Composer";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { MessageList } from "./components/MessageList";
import { ModelSelector } from "./components/ModelSelector";
import { fetchModels, selectActiveModel } from "./lib/api";
import { useChat } from "./hooks/useChat";
import type { ModelOption } from "./types";

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
  const loadingDialogRef = useRef<HTMLDivElement>(null);

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
    if (isSwitchingModel) loadingDialogRef.current?.focus();
  }, [isSwitchingModel]);

  useEffect(() => {
    if (!isSwitchingModel || modelLoadStartedAt === null) return;

    const updateElapsed = () => {
      setElapsedSeconds(Math.floor((Date.now() - modelLoadStartedAt) / 1000));
    };
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(interval);
  }, [isSwitchingModel, modelLoadStartedAt]);

  const elapsedLabel = `${Math.floor(elapsedSeconds / 60)}:${String(elapsedSeconds % 60).padStart(2, "0")}`;

  async function handleModelChange(modelId: string) {
    if (!modelId || modelId === activeModel) return;
    setIsSwitchingModel(true);
    setLoadingModelName(models.find((model) => model.id === modelId)?.name ?? modelId);
    setModelLoadStartedAt(Date.now());
    setElapsedSeconds(0);
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
    <>
      <div className="mobile-brand"><Sparkles size={19} /> WONDERDESK</div>
      <button
        className="mobile-new-chat"
        type="button"
        onClick={startNewChat}
        aria-label="Start a new chat"
        title="New chat"
      >
        <Plus size={19} />
      </button>
      <ModelSelector
        activeModel={activeModel}
        isLoading={isSwitchingModel}
        models={models}
        onChange={(modelId) => void handleModelChange(modelId)}
      />
      <ConnectionStatus state={connection} onReconnect={reconnect} />
    </>
  );

  const footer = (
    <footer className="composer-dock">
      <Composer
        disabled={connection !== "connected" || isSwitchingModel}
        isCancelling={isCancelling}
        isGenerating={isGenerating}
        onSend={sendMessage}
        onStop={cancelGeneration}
      />
      <p className="footer-note">
        <Feather size={12} />
        An AI model can make mistakes. Keep your curiosity and double-check the facts.
      </p>
    </footer>
  );

  return (
    <AppShell header={header} footer={footer} onNewChat={startNewChat}>
      {modelError && <div className="model-error" role="alert">{modelError}</div>}
      <MessageList
        messages={messages}
        isGenerating={isGenerating}
        modelName={modelName}
        onSuggestion={sendMessage}
      />
      {isSwitchingModel && (
        <div
          ref={loadingDialogRef}
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
            <p className="model-loading-name">{loadingModelName}</p>
            <p id="model-loading-description" className="model-loading-description">
              Downloading and loading the model. Large models can take several minutes on the first run.
            </p>
            <div className="model-loading-progress-meta" aria-live="off">
              <span>Preparing model</span>
              <span>Elapsed {elapsedLabel}</span>
            </div>
            <div className="model-loading-progress" aria-hidden="true"><span /></div>
            <p className="model-loading-local"><span /> Your model stays on this machine</p>
          </div>
        </div>
      )}
    </AppShell>
  );
}
