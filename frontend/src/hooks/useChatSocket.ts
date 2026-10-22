import { useCallback, useEffect, useRef, useState } from "react";
import {
  makeCancelGenerationEvent,
  makeUserMessageEvent,
  makeWebSocketUrl,
  parseServerEvent
} from "../lib/chatProtocol";
import type { ChatMessage, ConnectionState, ServerEvent } from "../types";

function makeMessage(role: ChatMessage["role"], content: string): ChatMessage {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content
  };
}

export function useChatSocket() {
  const socketRef = useRef<WebSocket | null>(null);
  const [connectionKey, setConnectionKey] = useState(0);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [modelName, setModelName] = useState("Local model");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    let active = true;
    const socket = new WebSocket(makeWebSocketUrl(window.location));
    socketRef.current = socket;
    setConnection("connecting");

    socket.onopen = () => {
      if (active) setConnection("connected");
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      if (!active) return;
      let serverEvent: ServerEvent;
      try {
        serverEvent = parseServerEvent(event.data);
      } catch (error) {
        setMessages((current) => [
          ...current,
          makeMessage(
            "error",
            error instanceof Error ? error.message : "Received an invalid server response."
          )
        ]);
        setIsGenerating(false);
        setIsCancelling(false);
        return;
      }

      if (serverEvent.type === "ready") {
        setModelName(serverEvent.model);
      } else if (serverEvent.type === "assistant_message") {
        setModelName(serverEvent.model);
        setMessages((current) => [
          ...current,
          makeMessage("assistant", serverEvent.content)
        ]);
        setIsGenerating(false);
        setIsCancelling(false);
      } else if (serverEvent.type === "error") {
        setMessages((current) => [...current, makeMessage("error", serverEvent.content)]);
        setIsGenerating(false);
        setIsCancelling(false);
      } else if (serverEvent.type === "generation_cancelled") {
        setIsGenerating(false);
        setIsCancelling(false);
      }
    };

    socket.onerror = () => {
      if (active) setConnection("disconnected");
    };

    socket.onclose = () => {
      if (active) {
        setConnection("disconnected");
        setIsGenerating(false);
        setIsCancelling(false);
      }
    };

    return () => {
      active = false;
      socketRef.current = null;
      socket.close();
    };
  }, [connectionKey]);

  const sendMessage = useCallback((content: string): boolean => {
    const socket = socketRef.current;
    const message = content.trim();
    if (!message || !socket || socket.readyState !== WebSocket.OPEN || isGenerating) {
      return false;
    }

    setMessages((current) => [...current, makeMessage("user", message)]);
    setIsGenerating(true);
    setIsCancelling(false);
    try {
      socket.send(makeUserMessageEvent(message));
    } catch {
      setIsGenerating(false);
      setMessages((current) => [
        ...current,
        makeMessage("error", "Could not send the message. Reconnect and try again.")
      ]);
      return false;
    }
    return true;
  }, [isGenerating]);

  const cancelGeneration = useCallback(() => {
    const socket = socketRef.current;
    if (!isGenerating || isCancelling || !socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    try {
      socket.send(makeCancelGenerationEvent());
      setIsCancelling(true);
    } catch {
      setMessages((current) => [
        ...current,
        makeMessage("error", "Could not stop generation. Try reconnecting.")
      ]);
    }
  }, [isCancelling, isGenerating]);

  const startNewChat = useCallback(() => {
    setMessages([]);
    setIsGenerating(false);
    setIsCancelling(false);
    setConnectionKey((key) => key + 1);
  }, []);

  const reconnect = useCallback(() => {
    setConnectionKey((key) => key + 1);
  }, []);

  return {
    connection,
    isGenerating,
    isCancelling,
    messages,
    modelName,
    cancelGeneration,
    reconnect,
    sendMessage,
    startNewChat
  };
}
