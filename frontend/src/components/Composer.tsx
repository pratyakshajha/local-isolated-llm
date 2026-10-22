import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { ArrowUp, Square } from "lucide-react";

interface ComposerProps {
  disabled: boolean;
  isGenerating: boolean;
  isCancelling: boolean;
  onSend: (message: string) => boolean;
  onStop: () => void;
}

export function Composer({ disabled, isGenerating, isCancelling, onSend, onStop }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [value]);

  function submit() {
    if (disabled || isGenerating || !value.trim()) return;
    if (onSend(value)) {
      setValue("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="composer-wrap">
      <div className={`composer ${disabled ? "composer--disabled" : ""}`}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? "Connect to start chatting…" : "Write your next thought…"}
          aria-label="Message"
          rows={1}
          disabled={disabled}
        />
        <button
          className={`send-button ${isGenerating ? "send-button--stop" : ""}`}
          type="button"
          onClick={isGenerating ? onStop : submit}
          disabled={disabled || isCancelling || (!isGenerating && !value.trim())}
          aria-label={isGenerating ? (isCancelling ? "Stopping generation" : "Stop generation") : "Send message"}
          title={isGenerating ? (isCancelling ? "Stopping…" : "Stop generation") : "Send message"}
        >
          {isCancelling
            ? <span className="send-pulse" />
            : isGenerating
              ? <Square size={14} fill="currentColor" />
              : <ArrowUp size={20} strokeWidth={2.5} />}
        </button>
      </div>
      <p className="composer-hint">
        {isCancelling
          ? "Stopping the model…"
          : isGenerating
            ? "The model is thinking · Select stop to cancel"
            : "Enter to send · Shift + Enter for a new line"}
      </p>
    </div>
  );
}
