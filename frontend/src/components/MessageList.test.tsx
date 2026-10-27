import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MessageList } from "./MessageList";

describe("MessageList", () => {
  it("shows streaming dots without adding a second assistant avatar", () => {
    const html = renderToStaticMarkup(
      <MessageList
        messages={[
          { id: "user", role: "user", content: "Question" },
          { id: "assistant", role: "assistant", content: "Partial answer" }
        ]}
        isGenerating
        modelName="Test model"
        onSuggestion={() => {}}
      />
    );

    expect(html.match(/class="message message--assistant"/g)).toHaveLength(1);
    expect(html.match(/class="message-avatar message-avatar--assistant"/g)).toHaveLength(1);
    expect(html).toContain('class="thinking-indicator"');
  });

  it("shows a loading avatar before the assistant message exists", () => {
    const html = renderToStaticMarkup(
      <MessageList
        messages={[{ id: "user", role: "user", content: "Question" }]}
        isGenerating
        modelName="Test model"
        onSuggestion={() => {}}
      />
    );

    expect(html.match(/class="message message--assistant"/g)).toHaveLength(1);
    expect(html.match(/class="message-avatar message-avatar--assistant"/g)).toHaveLength(1);
    expect(html).toContain('class="thinking-indicator"');
  });
});
