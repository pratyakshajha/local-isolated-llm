import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import { Bot, UserRound, Zap } from "lucide-react";
import type { ChatMessage } from "../types";

interface MessageListProps {
  messages: ChatMessage[];
  isGenerating: boolean;
  modelName: string;
  onSuggestion: (message: string) => void;
}

const prompts = [
  "Explain a big idea simply",
  "Write a comic-book origin story",
  "Help me brainstorm something fun"
];

function EmptyState({ onSuggestion }: Pick<MessageListProps, "onSuggestion">) {
  return (
    <div className="empty-state">
      <div className="empty-art" aria-hidden="true">
        <span className="art-sunburst">POW!</span>
        <div className="art-note">
          <Zap size={20} fill="currentColor" />
          <span>YOUR IDEAS<br />START HERE</span>
        </div>
      </div>
      <p className="eyebrow">A little imagination goes a long way</p>
      <h1>What are we<br /><span>making today?</span></h1>
      <p className="empty-description">Ask a question, explore an idea, or make something wonderfully weird.</p>
      <div className="prompt-suggestions" aria-label="Suggested prompts">
        {prompts.map((prompt) => (
          <button key={prompt} type="button" onClick={() => onSuggestion(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

export function MessageList({ messages, isGenerating, modelName, onSuggestion }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastMessage = messages[messages.length - 1];
  const hasAssistantMessage = lastMessage?.role === "assistant";

  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    bottomRef.current?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
  }, [messages, isGenerating]);

  return (
    <div className="message-scroll-area">
      {messages.length === 0 ? (
        <EmptyState onSuggestion={onSuggestion} />
      ) : (
        <div className="message-list">
          {messages.map((message) => (
            <article className={`message message--${message.role}`} key={message.id}>
              <div className="message-heading">
                <span className={`message-avatar message-avatar--${message.role}`} aria-hidden="true">
                  {message.role === "user" ? <UserRound size={16} /> : <Bot size={16} />}
                </span>
                <span>
                  {message.role === "user" ? "You" : message.role === "error" ? "Oops" : "Assistant"}
                </span>
                {message.role === "assistant" && <span className="message-model">· {modelName}</span>}
              </div>
              <div className="message-content">
                {message.role === "assistant" ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <p>{message.content}</p>
                )}
                {isGenerating && hasAssistantMessage && message.id === lastMessage.id && (
                  <div className="thinking-indicator" role="status" aria-label="Assistant is responding">
                    <span /><span /><span />
                  </div>
                )}
              </div>
            </article>
          ))}
          {isGenerating && !hasAssistantMessage && (
            <article className="message message--assistant" aria-live="polite">
              <div className="message-heading">
                <span className="message-avatar message-avatar--assistant" aria-hidden="true">
                  <Bot size={16} />
                </span>
                <span>Assistant</span>
                <span className="message-model">· {modelName}</span>
              </div>
              <div className="thinking-indicator"><span /><span /><span /></div>
            </article>
          )}
          <div ref={bottomRef} />
        </div>
      )}
      {messages.length === 0 && <div ref={bottomRef} />}
    </div>
  );
}
