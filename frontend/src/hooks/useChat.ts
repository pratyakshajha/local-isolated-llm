import { useCallback, useEffect, useRef, useState } from "react";
import { responseError } from "../lib/api";
import { readChatCompletionStream } from "../lib/chatCompletion";
import type { ChatMessage, ConnectionState } from "../types";

function makeMessage(role: ChatMessage["role"], content: string): ChatMessage {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content
  };
}

export function useChat(modelId: string, modelName = "Local model") {
  const controllerRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  const [connectionKey, setConnectionKey] = useState(0);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    let active = true;
    setConnection("connecting");
    void fetch("/health/ready")
      .then((response) => {
        if (!response.ok) throw new Error("The local server is not ready.");
        if (active) setConnection("connected");
      })
      .catch(() => {
        if (active) setConnection("disconnected");
      });
    return () => {
      active = false;
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, [connectionKey]);

  const sendMessage = useCallback((content: string): boolean => {
    const message = content.trim();
    if (!message || !modelId || controllerRef.current) return false;

    const conversation = messagesRef.current
      .filter((item) => item.role !== "error")
      .map(({ role, content: text }) => ({ role, content: text }));
    conversation.push({ role: "user", content: message });
    const userMessage = makeMessage("user", message);
    messagesRef.current = [...messagesRef.current, userMessage];
    setMessages(messagesRef.current);
    setIsGenerating(true);
    setIsCancelling(false);

    const controller = new AbortController();
    controllerRef.current = controller;
    void (async () => {
      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: modelId, messages: conversation, stream: true }),
          signal: controller.signal
        });
        if (!response.ok) throw await responseError(response);

        const assistantMessage = makeMessage("assistant", "");
        messagesRef.current = [...messagesRef.current, assistantMessage];
        setMessages(messagesRef.current);
        await readChatCompletionStream(response, (chunk) => {
          messagesRef.current = messagesRef.current.map((item) =>
            item.id === assistantMessage.id
              ? { ...item, content: item.content + chunk }
              : item
          );
          setMessages(messagesRef.current);
        });
      } catch (error) {
        if (!controller.signal.aborted) {
          messagesRef.current = [
            ...messagesRef.current,
            makeMessage(
              "error",
              error instanceof Error ? error.message : "Chat completion failed."
            )
          ];
          setMessages(messagesRef.current);
        }
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
          setIsGenerating(false);
          setIsCancelling(false);
        }
      }
    })();
    return true;
  }, [modelId]);

  const cancelGeneration = useCallback(() => {
    if (!controllerRef.current || isCancelling) return;
    setIsCancelling(true);
    controllerRef.current.abort();
  }, [isCancelling]);

  const startNewChat = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    messagesRef.current = [];
    setMessages([]);
    setIsGenerating(false);
    setIsCancelling(false);
  }, []);

  const reconnect = useCallback(() => {
    setConnectionKey((key) => key + 1);
  }, []);

  return {
    connection,
    isGenerating,
    isCancelling,
    messages,
    modelName: modelName || "Local model",
    cancelGeneration,
    reconnect,
    sendMessage,
    startNewChat
  };
}
