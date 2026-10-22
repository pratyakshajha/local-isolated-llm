import { useCallback, useEffect, useRef, useState } from "react";
import { Feather, Plus, Sparkles } from "lucide-react";
import { AppShell } from "./components/AppShell";
import { Composer } from "./components/Composer";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { MessageList } from "./components/MessageList";
import { ModelSelector } from "./components/ModelSelector";
import { fetchModels, selectActiveModel } from "./lib/api";
import { useChatSocket } from "./hooks/useChatSocket";
import type { ModelOption } from "./types";

export default function App() {
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
  } = useChatSocket();
  const [models, setModels] = useState<ModelOption[]>([]);
  const [activeModel, setActiveModel] = useState("");
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [loadingModelName, setLoadingModelName] = useState("");
  const [modelError, setModelError] = useState("");
  const loadingDialogRef = useRef<HTMLDivElement>(null);

  const loadModels = useCallback(async () => {
    try {
      const data = await fetchModels();
      setModels(data.models);
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

  async function handleModelChange(modelId: string) {
    if (!modelId || modelId === activeModel) return;
    setIsSwitchingModel(true);
    setLoadingModelName(models.find((model) => model.id === modelId)?.name ?? modelId);
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
    }
  }

  const header = (
    <>
      <div className="mobile-brand"><Sparkles size={19} /> INK &amp; PIXEL</div>
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
        A local model can make mistakes. Keep your curiosity—and double-check the facts.
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
              <span className="model-loading-spinner" />
            </div>
            <p className="model-loading-eyebrow">Getting your studio ready</p>
            <h2 id="model-loading-title">Preparing your model</h2>
            <p className="model-loading-name">{loadingModelName}</p>
            <p id="model-loading-description" className="model-loading-description">
              Downloading and loading the model. This can take a little while—your chat will be ready as soon as it’s done.
            </p>
            <div className="model-loading-progress" aria-hidden="true"><span /></div>
            <p className="model-loading-local"><span /> Your model stays on this machine</p>
          </div>
        </div>
      )}
    </AppShell>
  );
}
